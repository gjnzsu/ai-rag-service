import pytest

from app.grounding.evidence import EvidenceSelector
from app.retrieval.models import RetrievalCandidate


def _candidate(
    chunk_id: str,
    *,
    document_id: str | None = None,
    content: str | None = None,
    exact_match: bool = False,
    fused_rank: int | None = None,
    rerank_score: float | None = None,
) -> RetrievalCandidate:
    document_id = document_id or f"doc:{chunk_id}"
    return RetrievalCandidate(
        content=content or f"Useful passage for {chunk_id}",
        document_id=document_id,
        chunk_id=chunk_id,
        source_type="jira",
        source_url=f"https://trusted.example/{document_id}",
        title=f"Title {document_id}",
        metadata={"nested": {"chunk_id": chunk_id}},
        score=0.25,
        exact_match=exact_match,
        fused_rank=fused_rank,
        rerank_score=rerank_score,
    )


def test_selector_returns_empty_for_empty_candidates():
    assert EvidenceSelector().select("query", [], top_k=5) == []


@pytest.mark.parametrize("top_k", [-1, 0, 1, 4, 11, 20])
def test_selector_requires_final_top_k_between_five_and_ten(top_k):
    with pytest.raises(ValueError, match="between 5 and 10"):
        EvidenceSelector().select("query", [_candidate("one")], top_k=top_k)


@pytest.mark.parametrize("top_k", [5, 10])
def test_selector_accepts_final_top_k_bounds(top_k):
    assert len(EvidenceSelector().select("query", [_candidate("one")], top_k=top_k)) == 1


def test_selector_deduplicates_canonical_ids_and_normalized_near_duplicates():
    candidates = [
        _candidate("c1", document_id="d1", content="The service retries requests after a timeout."),
        _candidate("c1", document_id="d1", content="different duplicate-id content"),
        _candidate("c2", document_id="d2", content="  THE service retries requests after a timeout!  "),
        _candidate("c3", document_id="d3", content="The service stops immediately after a timeout."),
    ]

    result = EvidenceSelector().select("timeout", candidates, top_k=5)

    assert [item.candidate.chunk_id for item in result] == ["c1", "c3"]


def test_selector_prefers_exact_then_reranked_then_fused_rank_without_score_arithmetic():
    candidates = [
        _candidate("fused-first", fused_rank=1),
        _candidate("rerank-low", rerank_score=0.0, fused_rank=3),
        _candidate("rerank-high", rerank_score=2.0, fused_rank=4),
        _candidate("exact", exact_match=True, fused_rank=20),
        _candidate("fused-second", fused_rank=2),
    ]

    result = EvidenceSelector().select("query", candidates, top_k=5)

    assert [item.candidate.chunk_id for item in result] == [
        "exact", "rerank-high", "rerank-low", "fused-first", "fused-second"
    ]


def test_selector_promotes_document_diversity_then_fills_in_stable_rank_order():
    candidates = [
        _candidate("a1", document_id="a", fused_rank=1),
        _candidate("a2", document_id="a", fused_rank=2),
        _candidate("a3", document_id="a", fused_rank=3),
        _candidate("b1", document_id="b", fused_rank=4),
        _candidate("c1", document_id="c", fused_rank=5),
        _candidate("b2", document_id="b", fused_rank=6),
    ]

    result = EvidenceSelector().select("compare sources", candidates, top_k=5)

    assert [item.candidate.chunk_id for item in result] == ["a1", "b1", "c1", "a2", "a3"]


def test_selector_uses_trusted_source_identity_when_legacy_document_id_is_empty():
    candidates = [
        _candidate("a1", document_id="", fused_rank=1),
        _candidate("a2", document_id="", fused_rank=2),
        _candidate("b1", document_id="", fused_rank=3),
    ]
    for candidate in candidates:
        candidate.document_id = ""
    candidates[0].source_url = "https://trusted.example/source-a"
    candidates[1].source_url = "https://trusted.example/source-a"
    candidates[2].source_url = "https://trusted.example/source-b"

    result = EvidenceSelector().select("compare sources", candidates, top_k=5)

    assert [item.candidate.chunk_id for item in result] == ["a1", "b1", "a2"]


def test_selector_assigns_contiguous_ids_copies_trusted_fields_and_does_not_mutate_inputs():
    candidates = [
        _candidate("one", document_id="trusted-1", fused_rank=1),
        _candidate("two", document_id="trusted-2", fused_rank=2),
    ]
    before = [candidate.model_copy(deep=True) for candidate in candidates]

    result = EvidenceSelector(max_prompt_chars=12).select("query", candidates, top_k=5)

    assert [item.citation_id for item in result] == ["E1", "E2"]
    assert result[0].prompt_content == "Useful passa"
    assert result[0].candidate.document_id == "trusted-1"
    assert result[0].candidate.chunk_id == "one"
    assert result[0].candidate.source_url == "https://trusted.example/trusted-1"
    assert candidates == before
    assert result[0].candidate is not candidates[0]
    result[0].candidate.metadata["nested"]["chunk_id"] = "changed"
    assert candidates[0].metadata["nested"]["chunk_id"] == "one"


def test_selector_has_no_score_threshold_refusal_for_non_empty_candidates():
    candidate = _candidate("low", fused_rank=999, rerank_score=-999.0)
    candidate.score = -1000.0
    candidate.rrf_score = 0.0

    result = EvidenceSelector().select("query", [candidate], top_k=5)

    assert [item.candidate.chunk_id for item in result] == ["low"]


def test_selector_rejects_invalid_prompt_bound():
    with pytest.raises(ValueError, match="max_prompt_chars"):
        EvidenceSelector(max_prompt_chars=0)
