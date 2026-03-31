"""Integration test: PDF ingest -> ChromaDB -> query pipeline (mocks OpenAI only)."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def patch_openai():
    """Replace all OpenAI calls with deterministic mocks."""
    fake_embedding = [0.1] * 1536

    def fake_embed(**kwargs):
        n = len(kwargs["input"])
        return MagicMock(data=[MagicMock(embedding=fake_embedding) for _ in range(n)])

    def fake_chat(**kwargs):
        return MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="The USD to CNY rate is 7.25 according to the document."
            ))]
        )

    with patch("app.pipeline.embedder.OpenAI") as mock_emb_cls, \
         patch("app.rag.query_engine.OpenAI") as mock_q_cls:
        mock_emb = MagicMock()
        mock_emb.embeddings.create.side_effect = fake_embed
        mock_emb_cls.return_value = mock_emb

        mock_q = MagicMock()
        mock_q.embeddings.create.side_effect = fake_embed
        mock_q.chat.completions.create.side_effect = fake_chat
        mock_q_cls.return_value = mock_q
        yield


def test_ingest_pdf_then_query(sample_pdf_path, tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.store.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr("app.rag.query_engine.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr("app.rag.query_engine.settings.top_k", 5)

    # Ingest
    with open(sample_pdf_path, "rb") as f:
        resp = client.post(
            "/ingest/pdf",
            files={"file": ("sample.pdf", f, "application/pdf")},
            params={"collection": "integration_test"},
        )
    assert resp.status_code == 200
    assert resp.json()["ingested_chunks"] >= 1

    # Query
    resp = client.post(
        "/query",
        json={"question": "What is the USD to CNY rate?", "collection": "integration_test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["source_type"] == "pdf"
