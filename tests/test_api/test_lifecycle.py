from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, create_app
from app.rag import query_engine
from app.api.lifecycle import build_document_metadata, generate_document_id
from app.config import settings
from app.pipeline.store import _format_result


client = TestClient(app)


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


@patch("app.api.lifecycle.query_lifecycle_collection")
@patch("app.api.lifecycle.embed_text")
def test_retrieve_applies_metadata_filters(mock_embed_text, mock_query):
    mock_embed_text.return_value = [0.1] * 1536
    mock_query.return_value = [
        {
            "content": "Login auditing requirement",
            "score": 0.98,
            "document_id": "jira_issue:PROJ-123",
            "chunk_id": "jira_issue:PROJ-123_chunk_0",
            "metadata": {
                "type": "jira_issue",
                "project_key": "PROJ",
                "url": "https://jira.example/browse/PROJ-123",
            },
            "source_url": "https://jira.example/browse/PROJ-123",
        }
    ]

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
    assert mock_query.call_args.kwargs["filters"] == {
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
