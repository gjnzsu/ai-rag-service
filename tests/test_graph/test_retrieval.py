from datetime import datetime, timezone
from importlib import import_module

import pytest
from pydantic import ValidationError

from app.graph.models import Edge, GraphEvidenceResult, GraphPath, GraphScope, Issue, SnapshotManifest, TextEvidence
from app.graph.snapshots import SnapshotRepository, write_json
from app.graph.service import GraphNotFound, GraphUnavailable


def implementation():
    try:
        return import_module('app.graph.retrieval')
    except ModuleNotFoundError:
        pytest.fail('Graph retrieval orchestration is not implemented')


def request(**kwargs):
    cls = getattr(import_module('app.graph.api_models'), 'GraphRetrieveRequest', None)
    assert cls is not None, 'Graph retrieve request contract missing'
    return cls(project_key='AIPLAT', query='retrieval security', **kwargs)


class Search:
    diagnostics = []

    def __init__(self, evidence):
        self.evidence = evidence
        self.search_calls = []

    def search(self, query, top_k=5):
        self.search_calls.append(query)
        return self.evidence[2:3]

    def for_issues(self, issue_ids):
        return [item for item in self.evidence if item.issue_id in issue_ids]


class Store:
    def __init__(self, issues, edges):
        self.issues, self.edges, self.requests = issues, edges, []

    def list_issues(self, scope):
        return self.issues

    def list_edges(self, scope):
        return self.edges

    def traverse(self, req):
        self.requests.append(req)
        seed = next(x for x in self.issues if x.issue_id == req.issue_id)
        edge = self.edges[-1]
        if req.unresolved_pair or (req.direction == 'outbound' and seed.issue_id != '3'):
            return GraphEvidenceResult(scope=req.scope, seeds=[seed.issue_id], nodes=[seed])
        return GraphEvidenceResult(scope=req.scope, seeds=[seed.issue_id], nodes=self.issues[1:],
                                   paths=[GraphPath(issue_ids=['3','2'], edges=[edge])])

    def close(self):
        pass


@pytest.fixture
def context(tmp_path):
    scope = GraphScope(site_id='https://example.test', project_key='AIPLAT', snapshot_id='one')
    now = datetime.now(timezone.utc)
    issues = [Issue(scope=scope, issue_id=str(i), key=f'AIPLAT-{i}', issue_type=kind, title=kind,
                    status='Done', status_category='done', source_url=f'https://example.test/browse/AIPLAT-{i}',
                    document_id=f'doc-{i}', updated_at=now) for i, kind in [(1,'Epic'),(2,'Story'),(3,'Bug')]]
    edges = [Edge(scope=scope, source_issue_id=str(i), target_issue_id='1', relation_type='CHILD_OF',
                  origin_issue_id=str(i), source_field='parent', source_type='parent', source_direction='parent') for i in [2,3]]
    edges.append(Edge(scope=scope, source_issue_id='3', target_issue_id='2', relation_type='BLOCKS',
                      origin_issue_id='3', source_field='issuelinks', source_type='Blocks', source_direction='outward', link_id='42'))
    repo = SnapshotRepository(tmp_path)
    directory = repo.reserve(scope)
    write_json(directory/'build-report.json', {'issues':3,'edges':3})
    write_json(directory/'normalized.json', {'issues':[x.model_dump(mode='json') for x in issues]})
    repo.publish(SnapshotManifest(scope=scope, account_scope='hash', capture_started_at=now, capture_finished_at=now,
                                  source_checksum='hash', normalization_version='1', index_versions={k:'1' for k in ['graph','chroma','fts']},
                                  stages={k:'ready' for k in ['graph','chroma','fts']}))
    evidence = [TextEvidence(scope=scope, issue_id=x.issue_id, document_id=x.document_id, chunk_id='chunk-'+x.issue_id,
                             text='Description '+x.title, source_url=x.source_url) for x in issues]
    search = Search(evidence)
    store = Store(issues, edges)
    return repo, store, search, scope


def service(context):
    repo, store, search, scope = context
    return implementation().GraphRetrievalService(repo, lambda:store, scope.site_id, ('AIPLAT',),
                                                  search_provider=lambda manifest:search)


def test_contract_bounds_and_conflicting_anchors():
    for fields in [{'top_k':21},{'hops':3},{'intent':'arbitrary'},{'issue_key':'AIPLAT-3','epic_key':'AIPLAT-1'}, {'text_budget':0}]:
        with pytest.raises(ValidationError):
            request(**fields)


def test_hybrid_seed_reaches_related_evidence(context):
    result = service(context).retrieve(request(top_k=1, direction='outbound'))
    assert result.data.seeds == ['3']
    assert result.data.paths[0].edges[0].target_issue_id == '2'
    assert {x.issue_id for x in result.data.text_evidence} == {'2','3'}
    assert result.seed_method == 'hybrid'
    assert context[2].search_calls == ['retrieval security']


def test_explicit_anchor_skips_seed_embedding(context):
    result = service(context).retrieve(request(issue_key='AIPLAT-3'))
    assert result.data.seeds == ['3']
    assert context[2].search_calls == []
    assert result.seed_method == 'explicit'


def test_epic_anchor_traverses_children_with_separate_hop_budget(context):
    result = service(context).retrieve(request(epic_key='AIPLAT-1', direction='outbound', hops=2))
    assert set(result.data.seeds) == {'2','3'}
    assert len(result.data.paths) == 1
    assert all(x.hops == 2 for x in context[1].requests)
    assert {x.issue_id for x in result.data.nodes} == {'1','2','3'}


def test_text_budget_keeps_edge_and_marks_partial(context):
    result = service(context).retrieve(request(issue_key='AIPLAT-3', text_budget=1))
    assert result.data.paths
    assert sum(len(x.text) for x in result.data.text_evidence) <= 1
    assert result.data.coverage == 'partial' and result.data.truncated


def test_overview_counts_do_not_use_seed_top_k(context):
    result = service(context).retrieve(request(intent='overview', top_k=1))
    assert result.structure['counts']['total'] == 3
    assert context[2].search_calls == []


def test_epic_detail_returns_full_counts(context):
    result = service(context).retrieve(request(intent='epic_detail', epic_key='AIPLAT-1', top_k=1))
    assert result.structure['total'] == 2


def test_resolve_once_and_use_same_snapshot(context, monkeypatch):
    repo = context[0]
    original = repo.resolve
    calls = []
    def resolve(*args):
        calls.append(args)
        return original(*args)
    monkeypatch.setattr(repo, 'resolve', resolve)
    result = service(context).retrieve(request(issue_key='AIPLAT-3'))
    assert len(calls) == 1
    assert all(x.scope == result.scope for x in result.data.text_evidence)


def test_graph_failure_returns_same_snapshot_text_only(context):
    def fail(scope):
        raise RuntimeError('secret backend details')
    context[1].list_issues = fail
    result = service(context).retrieve(request())
    assert result.data.coverage == 'graph_unavailable'
    assert result.data.paths == [] and result.data.nodes == []
    assert result.data.text_evidence[0].scope == context[3]
    assert 'secret' not in result.model_dump_json()


def test_text_failure_preserves_explicit_graph_evidence(context):
    def fail(ids):
        raise RuntimeError('secret')
    context[2].for_issues = fail
    result = service(context).retrieve(request(issue_key='AIPLAT-3'))
    assert result.data.paths and result.data.coverage == 'partial'
    assert not result.data.text_evidence


def test_wrong_snapshot_text_never_returned(context):
    other = context[3].model_copy(update={'snapshot_id':'other'})
    context[2].evidence = [x.model_copy(update={'scope':other}) for x in context[2].evidence]
    with pytest.raises(GraphUnavailable):
        service(context).retrieve(request())


def test_unknown_anchor_and_snapshot_are_not_guessed(context):
    with pytest.raises(GraphNotFound):
        service(context).retrieve(request(issue_key='AIPLAT-99'))
    with pytest.raises(GraphNotFound):
        service(context).retrieve(request(snapshot_id='missing'))


def test_no_seeds_returns_empty_evidence(context):
    context[2].evidence = []
    result = service(context).retrieve(request())
    assert not result.data.seeds and not result.data.paths
    assert 'no_matching_seeds' in result.diagnostics


def test_node_budget_applies_across_seeds(context):
    result = service(context).retrieve(request(epic_key='AIPLAT-1', limit=2))
    assert len(result.data.nodes) <= 2
    assert result.data.truncated
    ids = {x.issue_id for x in result.data.nodes}
    assert all(set(x.issue_ids) <= ids for x in result.data.paths)


@pytest.fixture
def client(context, monkeypatch):
    from fastapi.testclient import TestClient
    from app.config import settings
    from app.main import create_app
    import app.api.graph as api
    monkeypatch.setattr(settings, 'graph_enabled', True)
    app = create_app()
    dependency = getattr(api, 'get_graph_retrieval_service', None)
    assert dependency is not None, 'Graph retrieval HTTP dependency missing'
    app.dependency_overrides[dependency] = lambda: service(context)
    with TestClient(app) as client:
        yield client


def test_retrieve_http_does_not_construct_answer_model(client, monkeypatch):
    from app.grounding.generator import GroundedAnswerGenerator
    monkeypatch.setattr(GroundedAnswerGenerator, '__init__', lambda *a, **k: pytest.fail('retrieve must not generate'))
    response = client.post('/graph/retrieve', json={'project_key':'AIPLAT','query':'security','issue_key':'AIPLAT-3'})
    assert response.status_code == 200
    assert response.json()['data']['paths'][0]['edges'][0]['link_id'] == '42'


@pytest.mark.parametrize('changes', [{'hops':3},{'intent':'agent'},{'top_k':21},{'query':'  '},{'snapshot_id':'../secret'}])
def test_retrieve_http_rejects_unsupported_inputs(client, changes):
    response = client.post('/graph/retrieve', json={'project_key':'AIPLAT','query':'security', **changes})
    assert response.status_code == 422


def test_query_failure_preserves_retrieved_paths(client, monkeypatch):
    from app.grounding.generator import GroundedAnswerGenerator
    def fail(*args):
        raise RuntimeError('secret model failure')
    monkeypatch.setattr(GroundedAnswerGenerator, 'generate', fail)
    response = client.post('/graph/query', json={'project_key':'AIPLAT','query':'security','issue_key':'AIPLAT-3'})
    assert response.status_code == 200
    assert response.json()['answer']['status'] == 'answer_unavailable'
    assert response.json()['data']['paths']
    assert 'secret model' not in response.text


def test_retrieve_http_unknown_project_404(client):
    assert client.post('/graph/retrieve', json={'project_key':'OTHER','query':'security'}).status_code == 404


def test_epic_membership_is_evidence_not_an_uncited_assumption(context):
    result = service(context).retrieve(request(epic_key='AIPLAT-1', direction='outbound'))
    assert result.structure is not None
    membership = result.structure['epic_membership']
    assert membership[0]['epic']['key'] == 'AIPLAT-1'
    assert {e['source_issue_id'] for e in membership[0]['edges']} == {'2','3'}
    assert all(e['relation_type'] == 'CHILD_OF' for e in membership[0]['edges'])


def test_query_construction_failure_preserves_evidence(client, monkeypatch):
    import app.graph.answer as module
    def fail(*args, **kwargs):
        raise RuntimeError('secret construction detail')
    monkeypatch.setattr(module, 'GraphAnswerer', fail)
    response = client.post('/graph/query', json={'project_key':'AIPLAT','query':'security','issue_key':'AIPLAT-3'})
    assert response.status_code == 200
    assert response.json()['answer']['status'] == 'answer_unavailable'
    assert response.json()['data']['paths']


def test_query_cleanup_failure_preserves_evidence(client, monkeypatch):
    import app.graph.answer as module
    monkeypatch.setattr(module.GraphAnswerer, 'answer', lambda *args: module.GraphAnswer(answer=None,status='answer_unavailable'))
    def fail(*args):
        raise RuntimeError('secret cleanup detail')
    monkeypatch.setattr(module.GraphAnswerer, 'close', fail)
    response = client.post('/graph/query', json={'project_key':'AIPLAT','query':'security','issue_key':'AIPLAT-3'})
    assert response.status_code == 200
    assert response.json()['data']['paths']


def test_multiple_epic_anchors_report_dropped_nodes(context):
    store=context[1]
    store.issues=[x.model_copy(update={'issue_type':'Epic'}) for x in store.issues]
    store.edges=[]
    write_json(context[0].directory(context[3])/'build-report.json', {'issues':3,'edges':0})
    req=request(limit=1).model_copy(update={'query':'AIPLAT-1 AIPLAT-2 AIPLAT-3'})
    result=service(context).retrieve(req)
    assert result.data.truncated and result.data.coverage == 'partial'


def test_fallback_does_not_replace_unknown_query_key_with_semantic_match(context):
    def fail(scope):
        raise RuntimeError('graph unavailable')
    context[1].list_issues=fail
    req=request().model_copy(update={'query':'AIPLAT-99 dependencies'})
    with pytest.raises(GraphNotFound):
        service(context).retrieve(req)


def test_fallback_query_key_skips_embedding(context):
    def fail(scope):
        raise RuntimeError('graph unavailable')
    context[1].list_issues=fail
    result=service(context).retrieve(request().model_copy(update={'query':'AIPLAT-2 dependencies'}))
    assert result.seed_method == 'explicit'
    assert {x.issue_id for x in result.data.text_evidence} == {'2'}
    assert not context[2].search_calls


def test_overview_answer_does_not_duplicate_visual_projection(context):
    result=service(context).retrieve(request(intent='overview'))
    assert 'graph' not in result.structure
