from app.retrieval.fusion import ReciprocalRankFusion
from app.retrieval.models import RetrievalCandidate


def _candidate(chunk_id: str, method: str, rank: int, score: float, *, exact: bool = False):
    return RetrievalCandidate(
        content=chunk_id,
        chunk_id=chunk_id,
        document_id=f"doc:{chunk_id}",
        score=score,
        retrieval_methods=[method],
        rank_by_method={method: rank},
        exact_match=exact,
    )


def test_rrf_uses_one_based_ranks_and_deduplicates_by_chunk_id_without_mutation():
    vector_a = _candidate("a", "vector", 1, 0.9)
    bm25_b = _candidate("b", "bm25", 1, 8.0)
    bm25_a = _candidate("a", "bm25", 2, 7.0)

    result = ReciprocalRankFusion(k=60).fuse([[vector_a], [bm25_b, bm25_a]], top_k=10)

    assert [item.chunk_id for item in result] == ["a", "b"]
    assert result[0].rrf_score == (1 / 61) + (1 / 62)
    assert result[0].score == result[0].rrf_score
    assert result[0].retrieval_methods == ["vector", "bm25"]
    assert result[0].rank_by_method == {"vector": 1, "bm25": 2}
    assert result[0].method_scores == {"vector": 0.9, "bm25": 7.0}
    assert [item.fused_rank for item in result] == [1, 2]
    assert vector_a.retrieval_methods == ["vector"]
    assert vector_a.rrf_score == 0.0


def test_rrf_prioritizes_exact_candidates_then_keeps_stable_ties_and_truncates():
    ordinary_a = _candidate("a", "vector", 1, 0.1)
    ordinary_b = _candidate("b", "bm25", 1, 999.0)
    exact = _candidate("exact", "exact", 3, 0.0, exact=True)

    result = ReciprocalRankFusion().fuse([[ordinary_a, ordinary_b], [exact]], top_k=2)

    assert [item.chunk_id for item in result] == ["exact", "a"]
    assert result[0].exact_match is True
    assert result[0].retrieval_methods == ["exact"]
