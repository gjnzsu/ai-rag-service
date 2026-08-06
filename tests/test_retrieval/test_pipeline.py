import pytest

from app.retrieval.models import RetrievalCandidate
from app.retrieval.pipeline import HybridRetrievalPipeline, RetrievalUnavailableError


def _candidate(chunk_id: str, method: str = "vector") -> RetrievalCandidate:
    return RetrievalCandidate(
        content=f"content {chunk_id}", chunk_id=chunk_id, document_id=f"doc:{chunk_id}",
        score=0.5, retrieval_methods=[method], rank_by_method={method: 1},
    )


class _Retriever:
    def __init__(self, result=None, error=None):
        self.result = result if result is not None else []
        self.error = error
        self.calls = []

    def search(self, query, top_k, filters, collection_name):
        self.calls.append((query, top_k, filters, collection_name))
        if self.error:
            raise self.error
        return self.result


def test_hybrid_retrieves_primary_candidates_and_prioritizes_exact_jira_matches():
    vector = _Retriever([_candidate("shared", "vector"), _candidate("vector-only", "vector")])
    lexical = _Retriever([_candidate("shared", "bm25"), _candidate("lexical-only", "bm25")])
    exact_calls = []

    def exact_lookup(key, collection_name, filters):
        exact_calls.append((key, collection_name, filters))
        return [_candidate("exact", "exact")]

    result = HybridRetrievalPipeline(vector, lexical, exact_lookup=exact_lookup).retrieve(
        "Tell me about proj-7", collection_name="alpha", filters={"status": "Open"}
    )

    assert [item.chunk_id for item in result.candidates] == ["exact", "shared", "vector-only", "lexical-only"]
    assert result.candidates[0].exact_match is True
    assert result.retrieval_mode == "hybrid"
    assert result.failures == []
    assert vector.calls == [("Tell me about proj-7", 30, {"status": "Open"}, "alpha")]
    assert lexical.calls == [("Tell me about proj-7", 30, {"status": "Open"}, "alpha")]
    assert exact_calls == [("PROJ-7", "alpha", {"status": "Open"})]
    assert result.diagnostics == {
        "configured_retrievers": ["vector", "lexical"], "successful_retrievers": ["vector", "lexical"],
        "empty_retrievers": [], "failed_retrievers": [], "exact_jira_keys": ["PROJ-7"],
    }


def test_pipeline_applies_default_and_caller_result_limits():
    vector = _Retriever([_candidate(f"v-{index}") for index in range(35)])
    lexical = _Retriever([])

    default = HybridRetrievalPipeline(vector, lexical).retrieve("ordinary")
    requested = HybridRetrievalPipeline(vector, lexical).retrieve("ordinary", top_k=2)

    assert vector.calls[0][1] == 30
    assert lexical.calls[0][1] == 30
    assert len(default.candidates) == 20
    assert len(requested.candidates) == 2


@pytest.mark.parametrize(
    ("mode", "vector_result", "lexical_result", "expected_calls", "expected_ids"),
    [
        ("vector", [_candidate("v")], [_candidate("ignored", "bm25")], (1, 0), ["v"]),
        ("lexical", [_candidate("ignored")], [_candidate("l", "bm25")], (0, 1), ["l"]),
    ],
)
def test_pipeline_supports_single_retriever_modes(mode, vector_result, lexical_result, expected_calls, expected_ids):
    vector = _Retriever(vector_result)
    lexical = _Retriever(lexical_result)

    result = HybridRetrievalPipeline(vector, lexical, mode=mode).retrieve("ordinary")

    assert [item.chunk_id for item in result.candidates] == expected_ids
    assert (len(vector.calls), len(lexical.calls)) == expected_calls
    assert result.retrieval_mode == mode


def test_pipeline_degrades_when_one_primary_retriever_fails_or_is_empty():
    vector = _Retriever(error=RuntimeError("vector service outage: secret question"))
    lexical = _Retriever([_candidate("l", "bm25")])
    result = HybridRetrievalPipeline(vector, lexical).retrieve("very secret question")

    assert [item.chunk_id for item in result.candidates] == ["l"]
    assert result.failures == ["vector"]
    assert result.diagnostics["failed_retrievers"] == ["vector"]
    assert result.diagnostics["empty_retrievers"] == []

    empty = HybridRetrievalPipeline(_Retriever([]), _Retriever([_candidate("l", "bm25")])).retrieve("ordinary")
    assert empty.failures == []
    assert empty.diagnostics["empty_retrievers"] == ["vector"]


def test_pipeline_raises_only_when_every_configured_primary_retriever_errors():
    pipeline = HybridRetrievalPipeline(
        _Retriever(error=RuntimeError("vector failed")),
        _Retriever(error=RuntimeError("lexical failed")),
        exact_lookup=lambda *args: [_candidate("exact", "exact")],
    )

    with pytest.raises(RetrievalUnavailableError):
        pipeline.retrieve("PROJ-7 contains private content")


def test_pipeline_returns_valid_no_evidence_when_all_primary_retrievers_are_empty():
    result = HybridRetrievalPipeline(_Retriever([]), _Retriever([])).retrieve("ordinary")

    assert result.candidates == []
    assert result.failures == []
    assert result.diagnostics["empty_retrievers"] == ["vector", "lexical"]


def test_pipeline_rejects_unknown_modes():
    with pytest.raises(ValueError, match="retrieval mode"):
        HybridRetrievalPipeline(_Retriever(), _Retriever(), mode="semantic")
