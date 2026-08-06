from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import query as api_query
from app.main import app
from app.grounding.models import GeneratedAnswer, REFUSAL_ANSWER
from app.rag import query_engine
from app.rag.query_engine import QueryPipeline
from app.reranking.noop import NoOpReranker
from app.retrieval.models import RetrievalCandidate
from app.retrieval.pipeline import HybridRetrievalPipeline

client = TestClient(app)


def _legacy_result():
    return {
        "answer": "Legacy answer",
        "sources": [{
            "document_id": "d1",
            "source_type": "pdf",
            "title": "Legacy source",
            "excerpt": "Legacy excerpt",
            "score": 0.7,
        }],
        "model": "gpt-4o",
    }


def _additive_result(*, mode="hybrid", provider="qwen_local", reranker_status="ok"):
    return {
        **_legacy_result(),
        "citations": [{
            "citation_id": "E1",
            "document_id": "d1",
            "chunk_id": "c1",
            "source_url": "https://trusted.example/d1",
            "excerpt": "Trusted excerpt",
        }],
        "grounding": {"status": "supported"},
        "retrieval_metadata": {
            "mode": mode,
            "failures": [],
            "reranker": {"provider": provider, "status": reranker_status},
        },
    }


class _RoutedRetriever:
    def __init__(self, candidates=None, error=None):
        self.candidates = candidates if candidates is not None else []
        self.error = error

    def search(self, query, top_k, filters, collection_name):
        if self.error is not None:
            raise self.error
        return self.candidates


class _RoutedGenerator:
    model = "routed-model"

    def __init__(self):
        self.calls = []

    def generate(self, question, evidence):
        self.calls.append((question, evidence))
        return GeneratedAnswer(answer="Routed answer [E1]", citation_ids=["E1"])


def _routed_candidate(chunk_id, method):
    return RetrievalCandidate(
        content=f"Trusted {method} evidence",
        document_id=f"document-{chunk_id}",
        chunk_id=chunk_id,
        source_type="pdf",
        source_url=f"https://trusted.example/{chunk_id}",
        title=f"Title {chunk_id}",
        metadata={"document_type": "guide"},
        score=0.9,
        retrieval_methods=[method],
        method_scores={method: 0.9},
        rank_by_method={method: 1},
    )


def _install_routed_query_pipeline(monkeypatch, vector, lexical, *, reranker=None):
    retrieval = HybridRetrievalPipeline(
        vector,
        lexical,
        mode="hybrid",
        exact_lookup=lambda *args: [],
        final_top_k=5,
        reranker=reranker if reranker is not None else NoOpReranker(),
        reranker_provider="custom" if reranker is not None else "none",
    )
    generator = _RoutedGenerator()
    pipeline = QueryPipeline(
        retrieval_pipeline=retrieval,
        generator=generator,
        model=generator.model,
        evidence_top_k=5,
    )
    monkeypatch.setattr(query_engine, "get_default_query_pipeline", lambda: pipeline)
    return generator


def test_query_response_preserves_legacy_dictionary_with_additive_defaults():
    response = api_query.QueryResponse.model_validate(_legacy_result())

    assert response.answer == "Legacy answer"
    assert response.sources[0].document_type == ""
    assert response.model == "gpt-4o"
    assert response.citations == []
    assert response.grounding is None
    assert response.retrieval_metadata is None


def test_query_response_exposes_typed_additive_fields():
    response = api_query.QueryResponse.model_validate(_additive_result())

    assert isinstance(response.citations[0], api_query.CitationItem)
    assert isinstance(response.grounding, api_query.GroundingInfo)
    assert isinstance(response.retrieval_metadata, api_query.RetrievalMetadata)
    assert response.citations[0].citation_id == "E1"
    assert response.citations[0].source_url == "https://trusted.example/d1"
    assert response.grounding.status == "supported"
    assert response.retrieval_metadata.mode == "hybrid"
    assert response.retrieval_metadata.reranker.provider == "qwen_local"


@pytest.mark.parametrize(
    ("path", "invalid"),
    [
        (("grounding", "status"), "private-status"),
        (("retrieval_metadata", "mode"), "private-mode"),
        (("retrieval_metadata", "reranker", "provider"), "private-provider"),
        (("retrieval_metadata", "reranker", "status"), "private-status"),
        (("retrieval_metadata", "reranker", "error_type"), "credential-secret"),
    ],
)
def test_query_response_rejects_non_allowlisted_diagnostic_tokens(path, invalid):
    result = _additive_result()
    target = result
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid

    with pytest.raises(ValidationError):
        api_query.QueryResponse.model_validate(result)


def test_query_response_mutable_defaults_are_fresh():
    first = api_query.QueryResponse.model_validate(_legacy_result())
    second = api_query.QueryResponse.model_validate(_legacy_result())
    first.citations.append(api_query.CitationItem(
        citation_id="E1",
        document_id="d1",
        chunk_id="c1",
        source_url="",
        excerpt="excerpt",
    ))
    first_metadata = api_query.RetrievalMetadata(mode="hybrid")
    second_metadata = api_query.RetrievalMetadata(mode="hybrid")
    first_metadata.failures.append("vector")

    assert second.citations == []
    assert second_metadata.failures == []


@patch("app.api.query.query_engine.query")
def test_query_success(mock_query):
    mock_query.return_value = {
        "answer": "The USD to CNY rate is 7.25.",
        "sources": [
            {
                "document_id": "d1",
                "source_type": "fx",
                "title": "FX Rates",
                "excerpt": "1 USD = 7.25 CNY",
                "score": 0.95,
            }
        ],
        "model": "gpt-4o",
    }

    response = client.post("/query", json={"question": "What is USD to CNY?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The USD to CNY rate is 7.25."
    assert len(data["sources"]) == 1
    assert data["model"] == "gpt-4o"


@patch("app.api.query.query_engine.query")
def test_query_passes_top_k(mock_query):
    mock_query.return_value = {"answer": "ok", "sources": [], "model": "gpt-4o"}
    client.post("/query", json={"question": "test", "top_k": 3})
    call_kwargs = mock_query.call_args[1]
    assert call_kwargs.get("top_k") == 3


@patch("app.api.query.query_engine.query")
def test_query_forwards_existing_parameters_and_document_type_filter(mock_query):
    mock_query.return_value = _legacy_result()

    response = client.post("/query", json={
        "question": "status",
        "collection": "alpha",
        "top_k": 7,
        "document_type": "guide",
    })

    assert response.status_code == 200
    mock_query.assert_called_once_with(
        question="status",
        collection_name="alpha",
        top_k=7,
        document_type="guide",
    )


@patch("app.api.query.query_engine.query")
def test_query_additive_fields_survive_fastapi_response_filtering(mock_query):
    mock_query.return_value = _additive_result()

    response = client.post("/query", json={"question": "status"})

    assert response.status_code == 200
    assert response.json()["citations"][0]["chunk_id"] == "c1"
    assert response.json()["grounding"] == {"status": "supported"}
    assert response.json()["retrieval_metadata"]["reranker"] == {
        "provider": "qwen_local",
        "status": "ok",
        "error_type": None,
    }


@pytest.mark.parametrize(
    ("mode", "failures", "provider", "reranker_status"),
    [
        ("hybrid", ["lexical"], "none", "disabled"),
        ("hybrid", ["vector"], "none", "disabled"),
        ("hybrid", [], "openai", "fallback"),
        ("vector", [], "none", "disabled"),
        ("lexical", [], "none", "disabled"),
    ],
)
@patch("app.api.query.query_engine.query")
def test_query_returns_success_for_retrieval_and_reranker_fallbacks(
    mock_query, mode, failures, provider, reranker_status
):
    result = _additive_result(mode=mode, provider=provider, reranker_status=reranker_status)
    result["retrieval_metadata"]["failures"] = failures
    mock_query.return_value = result

    response = client.post("/query", json={"question": "status"})

    assert response.status_code == 200
    assert response.json()["retrieval_metadata"]["failures"] == failures
    assert response.json()["retrieval_metadata"]["reranker"]["status"] == reranker_status


@pytest.mark.parametrize(
    ("vector", "lexical", "failed_method"),
    [
        (
            _RoutedRetriever([_routed_candidate("vector", "vector")]),
            _RoutedRetriever(error=RuntimeError("lexical unavailable")),
            "lexical",
        ),
        (
            _RoutedRetriever(error=RuntimeError("vector unavailable")),
            _RoutedRetriever([_routed_candidate("bm25", "bm25")]),
            "vector",
        ),
    ],
)
def test_query_route_generates_from_single_available_primary(
    monkeypatch, vector, lexical, failed_method
):
    generator = _install_routed_query_pipeline(monkeypatch, vector, lexical)

    response = client.post("/query", json={"question": "status"})

    assert response.status_code == 200
    assert response.json()["answer"] == "Routed answer [E1]"
    assert response.json()["retrieval_metadata"]["failures"] == [failed_method]
    assert len(generator.calls) == 1


def test_query_route_does_not_generate_when_both_primaries_raise(monkeypatch):
    generator = _install_routed_query_pipeline(
        monkeypatch,
        _RoutedRetriever(error=RuntimeError("vector unavailable")),
        _RoutedRetriever(error=RuntimeError("lexical unavailable")),
    )

    response = client.post("/query", json={"question": "status"})

    assert response.status_code == 502
    assert generator.calls == []


def test_query_route_generates_when_reranker_falls_back(monkeypatch):
    class FailingReranker:
        def rerank(self, query, candidates, top_k):
            raise RuntimeError("reranker unavailable")

    generator = _install_routed_query_pipeline(
        monkeypatch,
        _RoutedRetriever([_routed_candidate("vector", "vector")]),
        _RoutedRetriever([]),
        reranker=FailingReranker(),
    )

    response = client.post("/query", json={"question": "status"})

    assert response.status_code == 200
    assert response.json()["retrieval_metadata"]["reranker"]["status"] == "fallback"
    assert len(generator.calls) == 1


@patch("app.api.query.query_engine.query")
def test_query_returns_exact_refusal_without_citations(mock_query):
    result = _additive_result()
    result.update({
        "answer": REFUSAL_ANSWER,
        "sources": [],
        "citations": [],
        "grounding": {"status": "insufficient_evidence"},
    })
    mock_query.return_value = result

    response = client.post("/query", json={"question": "unknown"})

    assert response.status_code == 200
    assert response.json()["answer"] == REFUSAL_ANSWER
    assert response.json()["citations"] == []
    assert response.json()["grounding"] == {"status": "insufficient_evidence"}


@patch("app.api.query.query_engine.query")
def test_query_does_not_expose_invalid_citation_mapping(mock_query):
    result = _additive_result()
    result["citations"] = []
    result["grounding"] = {"status": "validation_failed"}
    mock_query.return_value = result

    response = client.post("/query", json={"question": "status"})

    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert response.json()["grounding"] == {"status": "validation_failed"}


def test_query_missing_question():
    response = client.post("/query", json={})
    assert response.status_code == 422


@patch("app.api.query.query_engine.query")
def test_query_engine_error_returns_safe_502_and_log(mock_query, monkeypatch):
    secret = "ChromaDB unavailable for private question"
    hostile_error = type("Credential_secret_token", (Exception,), {})
    mock_query.side_effect = hostile_error(secret)
    logged = []
    monkeypatch.setattr(
        api_query,
        "logger",
        SimpleNamespace(
            info=lambda event, **values: logged.append((event, values)),
            error=lambda event, **values: logged.append((event, values)),
        ),
    )

    response = client.post("/query", json={"question": secret})

    assert response.status_code == 502
    assert secret not in response.text
    assert secret not in repr(logged)
    assert logged == [("query_error", {"error_type": "Exception"})]


@patch("app.api.query.query_engine.query")
def test_query_logs_safe_status_and_counts_only(mock_query, monkeypatch):
    mock_query.return_value = _additive_result()
    logged = []
    monkeypatch.setattr(
        api_query,
        "logger",
        SimpleNamespace(info=lambda event, **values: logged.append((event, values))),
    )

    response = client.post("/query", json={"question": "private customer question"})

    assert response.status_code == 200
    assert logged == [(
        "query_served",
        {"grounding_status": "supported", "source_count": 1, "citation_count": 1},
    )]
    assert "private customer question" not in repr(logged)
