from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.main import app, create_app
from app.rag import query_engine
from app.api import lifecycle
from app.api.lifecycle import (
    RetrievalResult,
    build_document_metadata,
    generate_document_id,
)
from app.config import settings
from app.pipeline.store import _format_result
from app.retrieval.models import RetrievalCandidate
from app.retrieval.pipeline import HybridRetrievalPipeline


client = TestClient(app)


def test_retrieval_result_preserves_legacy_dictionary_contract():
    legacy = {
        "content": "Legacy vector passage",
        "score": 0.91,
        "document_id": "legacy-document",
        "chunk_id": "legacy-chunk",
        "metadata": {"type": "pdf"},
    }

    serialized = RetrievalResult.model_validate(legacy).model_dump()

    assert {key: serialized[key] for key in legacy} == legacy
    assert serialized["retrieval_methods"] == []
    assert serialized["fused_rank"] is None
    assert serialized["rerank_score"] is None
    assert serialized["source_url"] == ""


def test_retrieval_result_exposes_additive_hybrid_fields():
    result = RetrievalResult.model_validate({
        "content": "Canonical passage",
        "score": 0.82,
        "document_id": "jira:PROJ-7",
        "chunk_id": "jira:PROJ-7:chunk:1",
        "metadata": {"type": "jira_issue"},
        "retrieval_methods": ["vector", "bm25"],
        "fused_rank": 1,
        "rerank_score": 0.97,
        "source_url": "https://jira.example/browse/PROJ-7",
    })

    assert result.retrieval_methods == ["vector", "bm25"]
    assert result.fused_rank == 1
    assert result.rerank_score == 0.97
    assert result.source_url == "https://jira.example/browse/PROJ-7"


def test_retrieval_result_mutable_defaults_are_fresh():
    first = RetrievalResult(
        content="first", score=1.0, document_id="d1", chunk_id="c1", metadata={}
    )
    second = RetrievalResult(
        content="second", score=1.0, document_id="d2", chunk_id="c2", metadata={}
    )

    first.retrieval_methods.append("vector")

    assert second.retrieval_methods == []


class _Retriever:
    def __init__(self, candidates=None, error=None):
        self.candidates = candidates if candidates is not None else []
        self.error = error
        self.calls = []

    def search(self, query, top_k, filters, collection_name):
        self.calls.append((query, top_k, filters, collection_name))
        if self.error is not None:
            raise self.error
        return self.candidates


def _candidate(chunk_id, method):
    return RetrievalCandidate(
        content=f"Canonical {method} passage",
        document_id=f"document-{chunk_id}",
        chunk_id=chunk_id,
        source_type="pdf",
        source_url=f"https://trusted.example/{chunk_id}",
        title=f"Title {chunk_id}",
        metadata={"document_type": "guide"},
        score=0.8,
        retrieval_methods=[method],
        method_scores={method: 0.8},
        rank_by_method={method: 1},
    )


def _install_retrieval_pipeline(monkeypatch, vector, lexical):
    retrieval_pipeline = HybridRetrievalPipeline(
        vector,
        lexical,
        mode="hybrid",
        exact_lookup=lambda *args: [],
        final_top_k=20,
    )
    application_pipeline = SimpleNamespace(
        retrieval_pipeline=retrieval_pipeline,
        retrieve=retrieval_pipeline.retrieve,
    )
    monkeypatch.setattr(
        query_engine,
        "get_default_query_pipeline",
        lambda: application_pipeline,
    )
    monkeypatch.setattr(
        lifecycle,
        "embed_text",
        lambda _query: pytest.fail("/retrieve must not embed in the API layer"),
    )
    return application_pipeline


@pytest.mark.parametrize(
    ("vector", "lexical", "expected_method"),
    [
        (_Retriever([_candidate("vector", "vector")]), _Retriever(error=RuntimeError("down")), "vector"),
        (_Retriever(error=RuntimeError("down")), _Retriever([_candidate("bm25", "bm25")]), "bm25"),
        (_Retriever([]), _Retriever([_candidate("nonempty", "bm25")]), "bm25"),
    ],
)
def test_retrieve_succeeds_with_degraded_or_empty_primary_path(
    monkeypatch, vector, lexical, expected_method
):
    _install_retrieval_pipeline(monkeypatch, vector, lexical)

    response = client.post("/retrieve", json={"query": "safe query"})

    assert response.status_code == 200
    assert response.json()["results"][0]["retrieval_methods"] == [expected_method]


def test_retrieve_uses_application_pipeline_without_generation_and_propagates_inputs(monkeypatch):
    calls = []

    class ApplicationPipeline:
        def retrieve(self, question, collection_name="default", top_k=None, filters=None):
            calls.append((question, collection_name, top_k, filters))
            return SimpleNamespace(candidates=[_candidate("shared", "vector")])

        def query(self, *args, **kwargs):
            pytest.fail("/retrieve must not generate or validate an answer")

    shared = ApplicationPipeline()
    monkeypatch.setattr(query_engine, "get_default_query_pipeline", lambda: shared)
    monkeypatch.setattr(
        lifecycle,
        "embed_text",
        lambda _query: pytest.fail("/retrieve must not embed in the API layer"),
    )

    response = client.post("/retrieve", json={
        "query": "project status",
        "collection": "alpha",
        "top_k": 1,
        "filters": {"document_type": "guide"},
    })

    assert response.status_code == 200
    assert calls == [("project status", "alpha", 1, {"document_type": "guide"})]
    assert response.json()["results"][0]["source_url"] == "https://trusted.example/shared"


def test_retrieve_accepts_legacy_top_k_above_design_cap(monkeypatch):
    calls = []

    class ApplicationPipeline:
        def retrieve(self, question, collection_name="default", top_k=None, filters=None):
            calls.append(top_k)
            return SimpleNamespace(candidates=[])

    monkeypatch.setattr(
        query_engine,
        "get_default_query_pipeline",
        lambda: ApplicationPipeline(),
    )

    response = client.post("/retrieve", json={"query": "legacy", "top_k": 50})

    assert response.status_code == 200
    assert response.json() == {"results": []}
    assert calls == [20]


def test_retrieve_returns_502_when_both_primary_retrievers_fail_without_leaking(monkeypatch):
    secret = "private question and credential"
    _install_retrieval_pipeline(
        monkeypatch,
        _Retriever(error=RuntimeError(secret)),
        _Retriever(error=RuntimeError(secret)),
    )
    logged = []
    monkeypatch.setattr(
        lifecycle,
        "logger",
        SimpleNamespace(error=lambda event, **values: logged.append((event, values))),
    )

    response = client.post("/retrieve", json={"query": secret})

    assert response.status_code == 502
    assert secret not in response.text
    assert secret not in repr(logged)
    assert logged == [("retrieve_error", {"error_type": "RetrievalUnavailableError"})]


def test_retrieve_error_log_rejects_hostile_exception_class_name(monkeypatch):
    hostile_error = type("Credential_secret_token", (Exception,), {})
    monkeypatch.setattr(
        query_engine,
        "retrieve",
        lambda *args, **kwargs: (_ for _ in ()).throw(hostile_error("private")),
    )
    logged = []
    monkeypatch.setattr(
        lifecycle,
        "logger",
        SimpleNamespace(error=lambda event, **values: logged.append((event, values))),
    )

    response = client.post("/retrieve", json={"query": "safe"})

    assert response.status_code == 502
    assert logged == [("retrieve_error", {"error_type": "Exception"})]
    assert "secret" not in repr(logged).lower()


def test_retrieve_backend_logs_do_not_include_caller_controlled_collection(monkeypatch):
    secret = "https://private.example/credential"
    _install_retrieval_pipeline(
        monkeypatch,
        _Retriever(error=RuntimeError("private backend message")),
        _Retriever([]),
    )
    logged = []
    monkeypatch.setattr(
        "app.retrieval.pipeline.logger",
        SimpleNamespace(warning=lambda event, **values: logged.append((event, values))),
    )

    response = client.post("/retrieve", json={"query": "safe", "collection": secret})

    assert response.status_code == 200
    assert secret not in repr(logged)
    assert "private backend message" not in repr(logged)
    assert logged == [(
        "retrieval_path_unavailable",
        {"retrieval_method": "vector", "error_type": "RuntimeError"},
    )]


def test_retrieve_returns_empty_200_when_retrievers_have_no_results(monkeypatch):
    _install_retrieval_pipeline(monkeypatch, _Retriever([]), _Retriever([]))

    response = client.post("/retrieve", json={"query": "no match"})

    assert response.status_code == 200
    assert response.json() == {"results": []}


def test_application_lifespan_closes_lazy_default_query_pipeline_once(monkeypatch):
    closes = []
    monkeypatch.setattr(
        query_engine,
        "close_default_query_pipeline",
        lambda: closes.append(True),
        raising=False,
    )

    with TestClient(create_app()) as lifespan_client:
        assert lifespan_client.get("/health").status_code == 200
        assert closes == []

    assert closes == [True]


def test_lifecycle_generated_ids_and_urls_are_canonical_and_server_owned(monkeypatch):
    monkeypatch.setattr(settings, "jira_url", "https://jira.example/")
    monkeypatch.setattr(settings, "confluence_url", "https://wiki.example/")

    jira_metadata = {"type": "jira_issue", "key": "PROJ-7", "url": "https://untrusted.example"}
    confluence_metadata = {"type": "confluence_page", "page_id": "12345"}

    assert generate_document_id(jira_metadata, "body") == "jira:PROJ-7"
    assert generate_document_id(confluence_metadata, "body") == "confluence:12345"
    built = build_document_metadata(document_id="jira:PROJ-7", content="body", metadata=jira_metadata)
    assert built["source_url"] == "https://jira.example/browse/PROJ-7"


@patch("app.api.lifecycle.index_documents")
def test_lifecycle_normalizes_caller_supplied_legacy_document_ids(mock_index):
    mock_index.return_value = {"document_id": "jira:PROJ-7", "chunk_count": 1, "created": True}

    response = client.post("/documents/upsert", json={
        "document_id": "jira_issue:PROJ-7",
        "content": "body",
        "metadata": {
            "type": "jira_issue", "key": "PROJ-7", "project_key": "PROJ", "title": "Title",
            "url": "https://untrusted.example", "status": "Open", "priority": "High",
        },
    })

    assert response.status_code == 200
    assert mock_index.call_args.args[0][0].id == "jira:PROJ-7"


@patch("app.api.lifecycle.index_documents")
def test_lifecycle_rejects_conflicting_jira_key_aliases(mock_index):
    mock_index.return_value = {"document_id": "jira:OTHER-9", "chunk_count": 1, "created": True}
    response = client.post("/documents/upsert", json={
        "content": "body",
        "metadata": {
            "type": "jira_issue", "key": "PROJ-7", "issue_key": "OTHER-9", "project_key": "PROJ",
            "title": "Title", "url": "https://untrusted.example", "status": "Open", "priority": "High",
        },
    })

    assert response.status_code == 422
    assert "issue_key" in response.text
    mock_index.assert_not_called()


@patch("app.api.lifecycle.index_documents")
def test_lifecycle_normalizes_matching_jira_key_aliases(mock_index):
    mock_index.return_value = {"document_id": "jira:PROJ-7", "chunk_count": 1, "created": True}
    response = client.post("/documents/upsert", json={
        "content": "body",
        "metadata": {
            "type": "jira_issue", "key": "PROJ-7", "issue_key": "PROJ-7", "project_key": "PROJ",
            "title": "Title", "url": "https://untrusted.example", "status": "Open", "priority": "High",
        },
    })

    assert response.status_code == 200
    metadata = mock_index.call_args.args[0][0].metadata
    assert metadata["key"] == metadata["issue_key"] == "PROJ-7"


def test_lifecycle_confluence_requires_page_id_for_canonical_identity():
    response = client.post("/documents/upsert", json={
        "document_id": "confluence_page:legacy", "content": "body",
        "metadata": {"type": "confluence_page", "title": "Title", "url": "https://untrusted.example", "space_key": "TEAM", "related_jira": "PROJ-7"},
    })

    assert response.status_code == 422
    assert "page_id" in response.text


def test_returned_source_url_never_falls_back_to_generic_url():
    result = _format_result(content="body", metadata={"url": "https://untrusted.example"}, distance=0.0)

    assert result["source_url"] == ""


def _fake_embeddings(count: int):
    return [[0.1] * 1536 for _ in range(count)]


@patch("app.api.lifecycle.index_documents")
def test_upsert_jira_document_preserves_business_metadata(mock_index):
    mock_index.return_value = {
        "document_id": "jira:PROJ-123",
        "chunk_count": 1,
        "created": True,
    }

    response = client.post(
        "/documents/upsert",
        json={
            "document_id": "jira_issue:PROJ-123",
            "content": "Summary: Add login auditing",
            "metadata": {
                "type": "jira_issue",
                "key": "PROJ-123",
                "project_key": "PROJ",
                "title": "Add login auditing",
                "url": "https://jira.example/browse/PROJ-123",
                "status": "To Do",
                "priority": "High",
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["document_id"] == "jira:PROJ-123"
    documents = mock_index.call_args.args[0]
    assert documents[0].id == "jira:PROJ-123"
    assert documents[0].metadata["document_id"] == "jira:PROJ-123"
    assert documents[0].metadata["type"] == "jira_issue"
    assert documents[0].metadata["key"] == "PROJ-123"
    assert documents[0].metadata["project_key"] == "PROJ"
    assert "content_hash" in documents[0].metadata
    assert "schema_version" in documents[0].metadata


@patch("app.api.lifecycle.query_engine.retrieve")
def test_retrieve_applies_metadata_filters(mock_retrieve):
    mock_retrieve.return_value = SimpleNamespace(candidates=[
        RetrievalCandidate(
            content="Login auditing requirement",
            score=0.98,
            document_id="jira_issue:PROJ-123",
            chunk_id="jira_issue:PROJ-123_chunk_0",
            metadata={
                "type": "jira_issue",
                "project_key": "PROJ",
            },
            source_url="https://jira.example/browse/PROJ-123",
            retrieval_methods=["vector"],
        )
    ], retrieval_mode="vector")

    response = client.post(
        "/retrieve",
        json={
            "query": "login audit",
            "top_k": 3,
            "filters": {
                "type": "jira_issue",
                "project_key": {"in": ["PROJ", "AUTH"]},
            },
        },
    )

    assert response.status_code == 200
    assert response.json()["results"][0]["document_id"] == "jira_issue:PROJ-123"
    assert mock_retrieve.call_args.kwargs["filters"] == {
        "type": "jira_issue",
        "project_key": {"in": ["PROJ", "AUTH"]},
    }


@patch("app.api.lifecycle.get_jira_key_context")
def test_jira_key_context_returns_issue_and_related_confluence(mock_context):
    mock_context.return_value = [
        {
            "content": "Issue text",
            "score": 1.0,
            "document_id": "jira_issue:PROJ-123",
            "chunk_id": "jira_issue:PROJ-123_chunk_0",
            "metadata": {"type": "jira_issue", "key": "PROJ-123"},
            "source_url": "https://jira.example/browse/PROJ-123",
        },
        {
            "content": "Design notes",
            "score": 1.0,
            "document_id": "confluence_page:abc",
            "chunk_id": "confluence_page:abc_chunk_0",
            "metadata": {"type": "confluence_page", "related_jira": "PROJ-123"},
            "source_url": "https://wiki.example/pages/abc",
        },
    ]

    response = client.get("/context/jira/PROJ-123")

    assert response.status_code == 200
    assert [item["document_id"] for item in response.json()["results"]] == [
        "jira_issue:PROJ-123",
        "confluence_page:abc",
    ]


def test_upsert_rejects_missing_required_jira_metadata():
    response = client.post(
        "/documents/upsert",
        json={
            "content": "Summary: Missing key",
            "metadata": {
                "type": "jira_issue",
                "project_key": "PROJ",
                "title": "Missing key",
                "url": "https://jira.example/browse/PROJ-123",
                "status": "To Do",
                "priority": "High",
            },
        },
    )

    assert response.status_code == 422
    assert "key" in response.text


def test_retrieve_rejects_unsupported_filter_operator():
    response = client.post(
        "/retrieve",
        json={
            "query": "login audit",
            "filters": {"project_key": {"contains": "PROJ"}},
        },
    )

    assert response.status_code == 422
    assert "contains" in response.text


@patch("app.api.lifecycle.delete_document")
def test_delete_document_by_id(mock_delete):
    mock_delete.return_value = True

    response = client.delete("/documents/jira_issue:PROJ-123")

    assert response.status_code == 200
    assert response.json() == {"document_id": "jira_issue:PROJ-123", "deleted": True}


@patch("app.api.lifecycle.refresh_jira_key")
def test_refresh_jira_key_returns_deleted_count(mock_refresh):
    mock_refresh.return_value = 2

    response = client.post("/context/jira/PROJ-123/reindex")

    assert response.status_code == 200
    assert response.json() == {"jira_key": "PROJ-123", "refreshed_documents": 2}
