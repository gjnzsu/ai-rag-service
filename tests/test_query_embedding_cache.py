from importlib import import_module

import pytest


def cache(**kwargs):
    try:
        return import_module('app.query_embedding_cache').QueryEmbeddingCache(**kwargs)
    except ModuleNotFoundError:
        pytest.fail('Query embedding cache is not implemented')


def query(store, text='query', **kwargs):
    return store.embed(text, scope=('site', 'project'), model='model',
                       deployment='endpoint', dimensions=2, **kwargs)


def test_exact_hit_and_defensive_copy():
    store = cache()
    calls = []

    def load():
        calls.append(1)
        return [1., 2.]

    query(store, load=load)[0] = 99
    assert query(store, load=load) == [1., 2.]
    query(store, 'Query', load=load)
    query(store, 'query ', load=load)
    assert len(calls) == 3
    assert store.stats()['hits'] == 1
    assert store.stats()['misses'] == 3


def test_ttl_is_not_extended_by_hits_and_lru_is_bounded():
    now = [0.]
    store = cache(max_entries=2, ttl_seconds=10, clock=lambda: now[0])
    def load():
        return [1., 2.]
    query(store, 'a', load=load)
    query(store, 'b', load=load)
    now[0] = 9
    query(store, 'a', load=load)
    query(store, 'c', load=load)
    assert store.stats()['entries'] == 2
    assert store.stats()['evictions'] == 1
    now[0] = 10
    query(store, 'a', load=load)
    assert store.stats()['misses'] == 4
    assert store.stats()['expirations'] == 1


@pytest.mark.parametrize('field,value', [
    ('scope', ('other',)), ('model', 'other'), ('deployment', 'other'),
    ('dimensions', 3), ('preprocessing_version', 'v2'),
])
def test_key_isolation(field, value):
    store = cache()
    args = dict(scope=('site',), model='model', deployment='endpoint', dimensions=2,
                preprocessing_version='v1')
    store.embed('q', **args, load=lambda: [1., 2.])
    args[field] = value
    vector = [3.] * args['dimensions']
    assert store.embed('q', **args, load=lambda: vector) == vector
    assert store.stats()['misses'] == 2


def test_failure_is_not_cached():
    store = cache()

    def fail():
        raise RuntimeError('provider unavailable')

    with pytest.raises(RuntimeError):
        query(store, load=fail)
    assert query(store, load=lambda: [1., 2.]) == [1., 2.]
    assert store.stats()['misses'] == 2


@pytest.mark.parametrize('vector', [[], [float('nan'), 1.], [1.]])
def test_invalid_vectors_are_not_cached(vector):
    store = cache()
    query(store, load=lambda: vector)
    assert query(store, load=lambda: [1., 2.]) == [1., 2.]
    assert store.stats()['hits'] == 0


def test_cache_failure_falls_back_without_duplicate_provider_call(monkeypatch):
    store = cache()
    calls = []

    def broken(*args):
        raise RuntimeError('cache failed')

    monkeypatch.setattr(store, '_put', broken)
    assert query(store, load=lambda: calls.append(1) or [1., 2.]) == [1., 2.]
    assert calls == [1]
    assert store.stats()['errors'] == 1


def test_disabled_cache_always_calls_provider(monkeypatch):
    module = import_module('app.query_embedding_cache')
    monkeypatch.setattr(module.settings, 'query_embedding_cache_enabled', False)
    calls = []
    for _ in range(2):
        module.embed_query('q', scope=('disabled',), load=lambda: calls.append(1) or [1.])
    assert len(calls) == 2


def test_read_error_falls_back(monkeypatch):
    store = cache()
    monkeypatch.setattr(store, '_get', lambda key: (_ for _ in ()).throw(RuntimeError()))
    assert query(store, load=lambda: [1., 2.]) == [1., 2.]
    assert store.stats()['errors'] == 1


def test_concurrent_different_queries_do_not_block_on_provider():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    store = cache(max_entries=2)
    barrier = Barrier(4)

    def run(i):
        def load():
            barrier.wait(timeout=5)
            return [float(i), 2.]
        return query(store, str(i), load=load)

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(run, range(4))) == [[float(i), 2.] for i in range(4)]
    assert store.stats()['entries'] == 2


def test_graph_query_cache_survives_requests_and_snapshot_changes(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from app.api.graph import get_graph_retrieval_service
    from app.graph.models import GraphScope
    module = import_module('app.query_embedding_cache')
    monkeypatch.setattr(module.settings, 'query_embedding_cache_enabled', True)
    monkeypatch.setattr(module, 'query_embedding_cache', cache())
    monkeypatch.setattr(module.settings, 'graph_data_dir', str(tmp_path))
    calls, clients = [], []

    class Client:
        def __init__(self, **kwargs):
            self.embeddings = self
            self.http_client = kwargs['http_client']
            clients.append(self)

        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(data=[SimpleNamespace(index=0, embedding=[1.] * 1536)], usage=None)

        def close(self):
            self.http_client.close()

    monkeypatch.setattr('openai.OpenAI', Client)
    monkeypatch.setattr('app.graph.search.SnapshotSearch', lambda directory, scope, embed: embed)
    monkeypatch.setattr('app.graph.retrieval.GraphRetrievalService',
                        lambda *args, search_provider: SimpleNamespace(search=search_provider, close=lambda: None))
    for snapshot, account in [('one', 'a'), ('two', 'a'), ('two', 'b')]:
        manifest = SimpleNamespace(scope=GraphScope(site_id='site', project_key='TEST', snapshot_id=snapshot),
                                   account_scope=account,
                                   index_versions={'chroma': 'chroma/text-embedding-3-small'})
        dependency = get_graph_retrieval_service()
        try:
            assert next(dependency).search(manifest)(['hello']) == [[1.] * 1536]
        finally:
            dependency.close()
    assert len(calls) == 2
    assert len(clients) == 2
