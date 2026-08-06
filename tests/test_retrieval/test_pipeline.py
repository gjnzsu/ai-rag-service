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
        "empty_retrievers": [], "failed_retrievers": [],
        "exact_lookup": {"status": "ok", "attempted_count": 1, "failure_count": 0, "match_count": 1},
        "reranker": {"provider": "none", "status": "disabled"},
    }


def test_pipeline_applies_default_and_caller_result_limits():
    vector = _Retriever([_candidate(f"v-{index}") for index in range(35)])
    lexical = _Retriever([])

    default = HybridRetrievalPipeline(vector, lexical).retrieve("ordinary")
    requested = HybridRetrievalPipeline(vector, lexical).retrieve("ordinary", top_k=2)

    assert vector.calls[0][1] == 30
    assert lexical.calls[0][1] == 30
    assert len(default.candidates) == 10
    assert len(requested.candidates) == 2


class _Reranker:
    def __init__(self, result=None, error=None, provider="test"):
        self.result = result
        self.error = error
        self.provider = provider
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, candidates, top_k))
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
        return list(reversed(candidates))[:top_k]


def test_pipeline_configured_noop_preserves_rrf_order_with_safe_diagnostics():
    candidates = [_candidate(f"v-{index}") for index in range(15)]

    result = HybridRetrievalPipeline(
        _Retriever(candidates), _Retriever([]), final_top_k=7,
    ).retrieve("private query with customer content")

    assert [item.chunk_id for item in result.candidates] == [f"v-{index}" for index in range(7)]
    assert result.diagnostics["reranker"] == {"provider": "none", "status": "disabled"}


def test_pipeline_passes_fused_top_twenty_to_successful_reranker_and_returns_configured_count():
    candidates = [_candidate(f"v-{index}") for index in range(30)]
    reranker = _Reranker()

    result = HybridRetrievalPipeline(
        _Retriever(candidates), _Retriever([]), reranker=reranker, final_top_k=7,
    ).retrieve("query")

    assert len(reranker.calls) == 1
    assert len(reranker.calls[0][1]) == 20
    assert reranker.calls[0][2] == 7
    assert [item.chunk_id for item in result.candidates] == [f"v-{index}" for index in range(19, 12, -1)]
    assert result.diagnostics["reranker"] == {"provider": "test", "status": "ok"}


@pytest.mark.parametrize("error", [TimeoutError("private timeout detail"), RuntimeError("password=secret")])
def test_pipeline_reranker_error_preserves_exact_rrf_order_and_redacts_diagnostics(error):
    query = "private query customer content"
    candidates = [_candidate(f"v-{index}") for index in range(15)]
    baseline = HybridRetrievalPipeline(
        _Retriever(candidates), _Retriever([]), final_top_k=7,
    ).retrieve(query)
    reranker = _Reranker(error=error, provider="custom-provider")

    result = HybridRetrievalPipeline(
        _Retriever(candidates), _Retriever([]), reranker=reranker, final_top_k=7,
    ).retrieve(query)

    assert result.candidates == baseline.candidates
    assert result.diagnostics["reranker"] == {
        "provider": "custom-provider", "status": "fallback", "error_type": type(error).__name__,
    }
    rendered = repr(result.diagnostics["reranker"])
    assert query not in rendered
    assert "customer content" not in rendered
    assert str(error) not in rendered


@pytest.mark.parametrize("final_top_k", [5, 10, 20])
def test_pipeline_final_count_respects_configuration_with_a_hard_twenty_cap(final_top_k):
    candidates = [_candidate(f"v-{index}") for index in range(30)]
    result = HybridRetrievalPipeline(
        _Retriever(candidates), _Retriever([]), final_top_k=final_top_k,
    ).retrieve("query", top_k=25)

    assert len(result.candidates) == final_top_k


@pytest.mark.parametrize(
    ("vector", "lexical", "expected"),
    [
        (_Retriever(error=RuntimeError("vector")), _Retriever([_candidate("l", "bm25")]), ["l"]),
        (_Retriever([_candidate("v")]), _Retriever(error=RuntimeError("lexical")), ["v"]),
    ],
)
def test_pipeline_primary_fallbacks_still_reach_reranking(vector, lexical, expected):
    reranker = _Reranker()

    result = HybridRetrievalPipeline(vector, lexical, reranker=reranker).retrieve("query")

    assert [item.chunk_id for item in result.candidates] == expected
    assert len(reranker.calls) == 1


def test_pipeline_rejects_a_configured_final_limit_above_twenty():
    with pytest.raises(ValueError, match="final_top_k"):
        HybridRetrievalPipeline(_Retriever(), _Retriever(), final_top_k=21)


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


@pytest.mark.parametrize(
    ("mode", "inactive_factory"),
    [
        ("vector", "app.retrieval.pipeline.SQLiteFTSIndex"),
        ("lexical", "app.retrieval.pipeline.ChromaVectorRetriever"),
    ],
)
def test_pipeline_constructs_only_backends_active_for_its_mode(monkeypatch, mode, inactive_factory):
    monkeypatch.setattr(inactive_factory, lambda: pytest.fail("constructed inactive backend"))
    if mode == "vector":
        monkeypatch.setattr("app.retrieval.pipeline.ChromaVectorRetriever", lambda: _Retriever([_candidate("v")]))
    else:
        monkeypatch.setattr("app.retrieval.pipeline.SQLiteFTSIndex", lambda: _Retriever([_candidate("l", "bm25")]))

    result = HybridRetrievalPipeline(mode=mode).retrieve("ordinary")

    assert len(result.candidates) == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"mode": ""}, "retrieval mode"),
        ({"candidate_top_k": 0}, "limits"),
        ({"candidate_top_k": -1}, "limits"),
        ({"final_top_k": 0}, "limits"),
    ],
)
def test_pipeline_rejects_explicit_empty_or_invalid_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HybridRetrievalPipeline(_Retriever(), _Retriever(), **kwargs)


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


def test_pipeline_reports_exact_lookup_failure_without_exposing_query_or_error_content(monkeypatch):
    log_entries = []

    class _Logger:
        def warning(self, event, **kwargs):
            log_entries.append((event, kwargs))

    monkeypatch.setattr("app.retrieval.pipeline.logger", _Logger())

    def unavailable_lookup(*args):
        raise RuntimeError("exact backend error: password=do-not-log")

    query = "Find PROJ-7 with private customer content"
    result = HybridRetrievalPipeline(
        _Retriever([_candidate("v")]), _Retriever([]), exact_lookup=unavailable_lookup
    ).retrieve(query)

    assert result.diagnostics["exact_lookup"] == {
        "status": "unavailable", "attempted_count": 1, "failure_count": 1, "match_count": 0,
    }
    rendered = repr((result.diagnostics, log_entries))
    assert "PROJ-7" not in rendered
    assert "private customer content" not in rendered
    assert "password=do-not-log" not in rendered


def test_pipeline_reports_no_exact_match_separately_from_an_exact_lookup_failure():
    result = HybridRetrievalPipeline(
        _Retriever([_candidate("v")]), _Retriever([]), exact_lookup=lambda *args: []
    ).retrieve("Find PROJ-7")

    assert result.diagnostics["exact_lookup"] == {
        "status": "no_match", "attempted_count": 1, "failure_count": 0, "match_count": 0,
    }


def test_pipeline_reports_partial_exact_failure_when_another_key_successfully_has_no_match():
    def mixed_lookup(key, *args):
        if key == "PROJ-7":
            raise RuntimeError("exact backend unavailable")
        return []

    result = HybridRetrievalPipeline(
        _Retriever([_candidate("v")]), _Retriever([]), exact_lookup=mixed_lookup
    ).retrieve("Find PROJ-7 and PROJ-8")

    assert result.diagnostics["exact_lookup"] == {
        "status": "partial_failure", "attempted_count": 2, "failure_count": 1, "match_count": 0,
    }


def test_pipeline_primary_warning_redacts_query_content_and_error_message(monkeypatch):
    log_entries = []

    class _Logger:
        def warning(self, event, **kwargs):
            log_entries.append((event, kwargs))

    monkeypatch.setattr("app.retrieval.pipeline.logger", _Logger())
    query = "private question with customer content"
    result = HybridRetrievalPipeline(
        _Retriever(error=RuntimeError("backend password=do-not-log")),
        _Retriever([_candidate("l", "bm25")]),
    ).retrieve(query)

    assert result.failures == ["vector"]
    rendered = repr(log_entries)
    assert "private question with customer content" not in rendered
    assert "password=do-not-log" not in rendered


def test_default_exact_lookup_combines_aliases_with_filters_and_maps_trusted_candidates(monkeypatch):
    calls = []
    canonical_metadata = {
        "chunk_id": "canonical", "document_id": "jira:PROJ-7", "source_type": "jira",
        "source_url": "https://jira.example/browse/PROJ-7", "title": "Trusted issue", "status": "Open",
    }
    legacy_metadata = {
        "chunk_id": "legacy", "document_id": "page:1", "type": "confluence",
        "source_url": "https://wiki.example/page/1", "title": "Related page", "status": "Open",
    }

    class _ExactCollection:
        def get(self, *, where, include):
            calls.append((where, include))
            base_where = where["$and"][0]
            if base_where == {"document_id": "jira:PROJ-7"}:
                return {"ids": ["canonical"], "documents": ["trusted body"], "metadatas": [canonical_metadata]}
            if base_where == {"related_jira": "PROJ-7"}:
                return {"ids": ["legacy"], "documents": ["related body"], "metadatas": [legacy_metadata]}
            return {"ids": [], "documents": [], "metadatas": []}

    monkeypatch.setattr("app.pipeline.store._get_collection", lambda collection_name: _ExactCollection())
    result = HybridRetrievalPipeline(_Retriever([]), _Retriever([])).retrieve(
        "PROJ-7", collection_name="alpha", filters={"status": "Open"}
    )

    assert [(item.chunk_id, item.content, item.source_type, item.title, item.exact_match) for item in result.candidates] == [
        ("canonical", "trusted body", "jira", "Trusted issue", True),
        ("legacy", "related body", "confluence", "Related page", True),
    ]
    assert all(where["$and"][1] == {"status": "Open"} for where, _ in calls)
    assert [where["$and"][0] for where, _ in calls] == [
        {"document_id": "jira:PROJ-7"}, {"document_id": "jira_issue:PROJ-7"},
        {"issue_key": "PROJ-7"}, {"key": "PROJ-7"}, {"related_jira": "PROJ-7"},
    ]


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
