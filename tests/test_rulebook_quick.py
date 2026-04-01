import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
from pathlib import Path
from app.main import app

client = TestClient(app)
RULEBOOK_PATH = Path("AI Engieering Rulebook.pdf")

@pytest.fixture(autouse=True)
def patch_openai():
    fake_embedding = [0.1] * 1536
    def fake_embed(**kwargs):
        n = len(kwargs["input"])
        return MagicMock(data=[MagicMock(embedding=fake_embedding) for _ in range(n)])

    def fake_chat(**kwargs):
        return MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="The authors of the 2025 Edition are Akshay Pachaar and Avi Chawla."
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

def test_rulebook_rag_flow(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.store.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr("app.rag.query_engine.settings.chroma_persist_dir", str(tmp_path))

    # 1. Ingest
    with open(RULEBOOK_PATH, "rb") as f:
        resp = client.post(
            "/ingest/pdf",
            files={"file": (RULEBOOK_PATH.name, f, "application/pdf")},
            params={"collection": "rulebook_test"},
        )
    assert resp.status_code == 200
    assert resp.json()["ingested_chunks"] > 10

    # 2. Query
    resp = client.post(
        "/query",
        json={
            "question": "Who are the authors?",
            "collection": "rulebook_test"
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "Akshay Pachaar" in data["answer"]
    assert data["sources"][0]["title"] == "AI Engieering Rulebook.pdf"
