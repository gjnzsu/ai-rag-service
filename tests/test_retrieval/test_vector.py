from types import SimpleNamespace

import pytest

from app.retrieval.vector import ChromaVectorRetriever


class _Collection:
    def __init__(self, result=None, error=None):
        self.result = result or {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
        self.error = error
        self.calls = []

    def query(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.result


class _Chroma:
    def __init__(self, collection):
        self.collection = collection
        self.names = []

    def get_or_create_collection(self, *, name, metadata):
        self.names.append((name, metadata))
        return self.collection


def test_vector_retriever_embeds_and_maps_canonical_candidates():
    client = SimpleNamespace(
        embeddings=SimpleNamespace(
            create=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[0.25, 0.5])])
        )
    )
    collection = _Collection({
        "ids": [["chunk-1"]],
        "documents": [["trusted content"]],
        "metadatas": [[{
            "chunk_id": "chunk-1", "document_id": "jira:OPS-7", "source_type": "jira",
            "source_url": "https://jira.example/browse/OPS-7", "title": "Issue", "status": "Open",
        }]],
        "distances": [[0.25]],
    })
    chroma = _Chroma(collection)

    result = ChromaVectorRetriever(openai_client=client, chroma_client=chroma).search(
        "find OPS-7", 4, {"status": "Open"}, "alpha"
    )

    assert result[0].model_dump(include={"content", "chunk_id", "document_id", "source_type", "source_url", "title"}) == {
        "content": "trusted content", "chunk_id": "chunk-1", "document_id": "jira:OPS-7",
        "source_type": "jira", "source_url": "https://jira.example/browse/OPS-7", "title": "Issue",
    }
    assert result[0].score == 0.75
    assert result[0].method_scores == {"vector": 0.75}
    assert result[0].retrieval_methods == ["vector"]
    assert result[0].rank_by_method == {"vector": 1}
    assert collection.calls == [{
        "query_embeddings": [[0.25, 0.5]], "n_results": 4,
        "include": ["documents", "metadatas", "distances"], "where": {"status": "Open"},
    }]
    assert chroma.names == [("alpha", {"hnsw:space": "cosine"})]


def test_vector_retriever_returns_empty_results_and_propagates_upstream_errors():
    client = SimpleNamespace(
        embeddings=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(data=[SimpleNamespace(embedding=[1.0])]))
    )
    assert ChromaVectorRetriever(openai_client=client, chroma_client=_Chroma(_Collection())).search("q", 2, None, "alpha") == []

    failing_client = SimpleNamespace(embeddings=SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("embedding unavailable"))))
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        ChromaVectorRetriever(openai_client=failing_client, chroma_client=_Chroma(_Collection())).search("q", 2, None, "alpha")

    with pytest.raises(RuntimeError, match="chroma unavailable"):
        ChromaVectorRetriever(openai_client=client, chroma_client=_Chroma(_Collection(error=RuntimeError("chroma unavailable")))).search("q", 2, None, "alpha")
