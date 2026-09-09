from importlib import import_module
from types import SimpleNamespace

import pytest


def embedder(client):
    try:
        return import_module("app.graph.embedding").OpenAIChunkEmbedder(client)
    except ModuleNotFoundError:
        pytest.fail("Graph embedding adapter is not implemented")


class Client:
    def __init__(self):
        self.embeddings = self
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, input))
        return SimpleNamespace(
            data=[SimpleNamespace(index=i, embedding=[float(i), 1.0]) for i in reversed(range(len(input)))],
            usage=SimpleNamespace(total_tokens=len(input)),
        )


def test_embedding_batches_order_and_usage():
    client = Client()
    adapter = embedder(client)
    vectors = adapter([str(i) for i in range(101)])
    assert [len(call[1]) for call in client.calls] == [100, 1]
    assert all(call[0] == "text-embedding-3-small" for call in client.calls)
    assert vectors[0] == [0.0, 1.0]
    assert vectors[99] == [99.0, 1.0]
    assert adapter.total_tokens == 101


def test_empty_embedding_does_not_call_remote():
    client = Client()
    assert embedder(client)([]) == []
    assert not client.calls


def test_duplicate_embedding_indexes_rejected():
    client = Client()
    client.create = lambda **kwargs: SimpleNamespace(
        data=[SimpleNamespace(index=0, embedding=[1.0])]*2, usage=None,
    )
    with pytest.raises(ValueError, match="indexes"):
        embedder(client)(["one", "two"])
