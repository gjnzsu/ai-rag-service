from unittest.mock import MagicMock, patch

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def pdf_bytes(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "RAG test content")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path.read_bytes()


@patch("app.api.ingest.embed_chunks")
@patch("app.api.ingest.upsert_chunks")
def test_ingest_pdf_success(mock_upsert, mock_embed, pdf_bytes):
    mock_embed.return_value = [[0.1] * 1536]
    mock_upsert.return_value = None

    response = client.post(
        "/ingest/pdf",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "ingested_chunks" in data
    assert "document_id" in data
    assert data["ingested_chunks"] >= 1


def test_ingest_pdf_wrong_content_type():
    response = client.post(
        "/ingest/pdf",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 422


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@patch("app.api.ingest.JiraConnector")
@patch("app.api.ingest.embed_chunks")
@patch("app.api.ingest.upsert_chunks")
def test_ingest_jira_success(mock_upsert, mock_embed, mock_jira_cls):
    from app.connectors.base import Document
    mock_instance = MagicMock()
    mock_jira_cls.return_value = mock_instance
    mock_instance.fetch.return_value = [
        Document(id="j1", content="Issue content", source_type="jira",
                 title="[PROJ-1] Bug", metadata={"issue_key": "PROJ-1"})
    ]
    mock_embed.return_value = [[0.1] * 1536]
    mock_upsert.return_value = None

    response = client.post("/ingest/jira", json={"project_key": "PROJ"})
    assert response.status_code == 200
    data = response.json()
    assert data["ingested_chunks"] >= 1
    assert data["issues_fetched"] == 1


@patch("app.api.ingest.ConfluenceConnector")
@patch("app.api.ingest.embed_chunks")
@patch("app.api.ingest.upsert_chunks")
def test_ingest_confluence_success(mock_upsert, mock_embed, mock_conf_cls):
    from app.connectors.base import Document
    mock_instance = MagicMock()
    mock_conf_cls.return_value = mock_instance
    mock_instance.fetch.return_value = [
        Document(id="c1", content="Page content", source_type="confluence",
                 title="My Page", metadata={"page_id": "123", "space_key": "TEAM"})
    ]
    mock_embed.return_value = [[0.1] * 1536]
    mock_upsert.return_value = None

    response = client.post("/ingest/confluence", json={"space_key": "TEAM"})
    assert response.status_code == 200
    data = response.json()
    assert data["ingested_chunks"] >= 1
    assert data["pages_fetched"] == 1


@patch("app.api.ingest.embed_chunks")
@patch("app.api.ingest.upsert_chunks")
def test_ingest_fx_success(mock_upsert, mock_embed):
    mock_embed.return_value = [[0.1] * 1536]
    mock_upsert.return_value = None

    response = client.post("/ingest/fx", json={"base_currency": "USD"})
    assert response.status_code == 200
    data = response.json()
    assert data["ingested_chunks"] >= 1
    assert data["rates_count"] > 0
