import pytest

from app.reranking.base import Reranker
from app.reranking.noop import NoOpReranker
from app.retrieval.models import RetrievalCandidate


def _candidate(chunk_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        content=f"content {chunk_id}",
        document_id=f"doc:{chunk_id}",
        chunk_id=chunk_id,
        metadata={"nested": {"value": chunk_id}},
        rrf_score=0.5,
    )


def test_noop_preserves_order_truncates_and_does_not_mutate_inputs():
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
    before = [candidate.model_copy(deep=True) for candidate in candidates]

    result = NoOpReranker().rerank("query", candidates, top_k=2)

    assert isinstance(NoOpReranker(), Reranker)
    assert [candidate.chunk_id for candidate in result] == ["a", "b"]
    assert candidates == before
    assert result[0] is not candidates[0]
    assert result[0].metadata is not candidates[0].metadata


def test_noop_accepts_empty_candidates_and_zero_top_k():
    reranker = NoOpReranker()

    assert reranker.rerank("query", [], top_k=5) == []
    assert reranker.rerank("query", [_candidate("a")], top_k=0) == []


def test_noop_rejects_negative_top_k():
    with pytest.raises(ValueError, match="top_k"):
        NoOpReranker().rerank("query", [_candidate("a")], top_k=-1)
