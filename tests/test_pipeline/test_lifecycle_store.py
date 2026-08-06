import pytest
from unittest.mock import MagicMock

from app.pipeline.store import (
    delete_document,
    get_document_chunks,
    get_jira_key_context,
    query_lifecycle_collection,
    refresh_jira_key,
    upsert_document,
)
from app.retrieval.lexical import SQLiteFTSIndex


@pytest.fixture
def tmp_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.store.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr("app.pipeline.store.settings.lexical_db_path", str(tmp_path / "lexical.db"))
    return str(tmp_path)


def test_upsert_document_replaces_existing_chunks(tmp_chroma):
    first_chunks = [
        {
            "id": "jira_issue:PROJ-123_chunk_0",
            "content": "old content",
            "document_id": "jira_issue:PROJ-123",
            "chunk_id": "jira_issue:PROJ-123_chunk_0",
            "type": "jira_issue",
            "key": "PROJ-123",
            "title": "Old",
        }
    ]
    second_chunks = [
        {
            "id": "jira_issue:PROJ-123_chunk_0",
            "content": "new content",
            "document_id": "jira_issue:PROJ-123",
            "chunk_id": "jira_issue:PROJ-123_chunk_0",
            "type": "jira_issue",
            "key": "PROJ-123",
            "title": "New",
        }
    ]

    created = upsert_document(first_chunks, [[1.0] + [0.0] * 1535], "lifecycle")
    updated = upsert_document(second_chunks, [[1.0] + [0.0] * 1535], "lifecycle")

    assert created["created"] is True
    assert updated["created"] is False
    chunks = get_document_chunks("jira_issue:PROJ-123", "lifecycle")
    assert len(chunks) == 1
    assert chunks[0]["content"] == "new content"


def test_query_lifecycle_collection_applies_equality_and_in_filters(tmp_chroma):
    chunks = [
        {
            "id": "jira_issue:PROJ-123_chunk_0",
            "content": "login audit",
            "document_id": "jira_issue:PROJ-123",
            "chunk_id": "jira_issue:PROJ-123_chunk_0",
            "type": "jira_issue",
            "project_key": "PROJ",
            "title": "Login",
        },
        {
            "id": "jira_issue:AUTH-9_chunk_0",
            "content": "password reset",
            "document_id": "jira_issue:AUTH-9",
            "chunk_id": "jira_issue:AUTH-9_chunk_0",
            "type": "jira_issue",
            "project_key": "AUTH",
            "title": "Password",
        },
    ]
    upsert_document(
        chunks,
        [[1.0] + [0.0] * 1535, [0.0, 1.0] + [0.0] * 1534],
        "lifecycle",
    )

    results = query_lifecycle_collection(
        [1.0] + [0.0] * 1535,
        collection_name="lifecycle",
        top_k=5,
        filters={"type": "jira_issue", "project_key": {"in": ["PROJ"]}},
    )

    assert [result["document_id"] for result in results] == ["jira_issue:PROJ-123"]


def test_jira_key_context_returns_exact_issue_and_related_page(tmp_chroma):
    chunks = [
        {
            "id": "jira_issue:PROJ-123_chunk_0",
            "content": "issue",
            "document_id": "jira_issue:PROJ-123",
            "chunk_id": "jira_issue:PROJ-123_chunk_0",
            "type": "jira_issue",
            "key": "PROJ-123",
            "title": "Issue",
        },
        {
            "id": "confluence_page:abc_chunk_0",
            "content": "page",
            "document_id": "confluence_page:abc",
            "chunk_id": "confluence_page:abc_chunk_0",
            "type": "confluence_page",
            "related_jira": "PROJ-123",
            "title": "Page",
        },
    ]
    upsert_document(chunks, [[1.0] + [0.0] * 1535, [0.0, 1.0] + [0.0] * 1534], "lifecycle")

    results = get_jira_key_context("PROJ-123", "lifecycle")

    assert {result["document_id"] for result in results} == {
        "jira_issue:PROJ-123",
        "confluence_page:abc",
    }


def test_delete_and_refresh_remove_matching_chunks(tmp_chroma):
    chunks = [
        {
            "id": "jira_issue:PROJ-123_chunk_0",
            "content": "issue",
            "document_id": "jira_issue:PROJ-123",
            "chunk_id": "jira_issue:PROJ-123_chunk_0",
            "type": "jira_issue",
            "key": "PROJ-123",
            "title": "Issue",
        },
        {
            "id": "confluence_page:abc_chunk_0",
            "content": "page",
            "document_id": "confluence_page:abc",
            "chunk_id": "confluence_page:abc_chunk_0",
            "type": "confluence_page",
            "related_jira": "PROJ-999",
            "title": "Page",
        },
    ]
    upsert_document(chunks, [[1.0] + [0.0] * 1535, [0.0, 1.0] + [0.0] * 1534], "lifecycle")

    assert delete_document("confluence_page:abc", "lifecycle") is True
    assert get_document_chunks("confluence_page:abc", "lifecycle") == []
    assert refresh_jira_key("PROJ-123", "lifecycle") == 1
    assert get_jira_key_context("PROJ-123", "lifecycle") == []


def test_delete_document_removes_vector_and_lexical_chunks(tmp_chroma, monkeypatch):
    lexical = MagicMock()
    lexical.delete_document.return_value = True
    monkeypatch.setattr("app.pipeline.store.get_lexical_index", lambda: lexical)

    upsert_document(
        [{
            "id": "jira:PROJ-7:chunk:0",
            "content": "login audit",
            "document_id": "jira:PROJ-7",
            "chunk_id": "jira:PROJ-7:chunk:0",
            "source_type": "jira",
            "title": "Login",
        }],
        [[1.0] + [0.0] * 1535],
        "alpha",
    )

    assert delete_document("jira:PROJ-7", "alpha") is True
    lexical.delete_document.assert_called_once_with("jira:PROJ-7", "alpha")


def test_refresh_jira_key_removes_matching_lexical_rows(tmp_chroma):
    chunk = {
        "id": "jira:PROJ-7:chunk:0", "chunk_id": "jira:PROJ-7:chunk:0", "content": "login",
        "document_id": "jira:PROJ-7", "source_type": "jira", "title": "Login", "key": "PROJ-7",
        "metadata": {"key": "PROJ-7", "issue_key": "PROJ-7"},
    }
    upsert_document([chunk], [[1.0] + [0.0] * 1535], "alpha")
    lexical = SQLiteFTSIndex("".join([tmp_chroma, "\\lexical.db"]))
    lexical.upsert_document([chunk], "alpha")

    assert refresh_jira_key("PROJ-7", "alpha") == 1
    assert lexical.search("login", 10, None, "alpha") == []
