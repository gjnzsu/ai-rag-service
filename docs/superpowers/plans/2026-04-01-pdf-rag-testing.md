# AI Engineering Rulebook RAG Testing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the RAG pipeline using the real-world `AI Engineering Rulebook.pdf` document to ensure correct ingestion, chunking, and retrieval.

**Architecture:** Integration test using `FastAPI`'s `TestClient` with mocked OpenAI services and a temporary ChromaDB instance.

**Tech Stack:** Python, pytest, FastAPI, pymupdf, langchain_text_splitters.

---

### Task 1: Test Setup and Ingestion Verification

**Files:**
- Create: `tests/test_rulebook_rag.py`

- [ ] **Step 1: Create the test file with fixtures and ingestion test**

```python
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

def test_ingest_rulebook(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.store.settings.chroma_persist_dir", str(tmp_path))

    with open(RULEBOOK_PATH, "rb") as f:
        resp = client.post(
            "/ingest/pdf",
            files={"file": (RULEBOOK_PATH.name, f, "application/pdf")},
            params={"collection": "rulebook_test"},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ingested_chunks"] > 10 # Expecting many chunks for a large PDF
    assert "pdf:AI_Engieering_Rulebook.pdf" in data["document_id"]
```

- [ ] **Step 2: Run test to verify ingestion fails (if file missing) or passes**

Run: `pytest tests/test_rulebook_rag.py::test_ingest_rulebook -v`
Expected: PASS (since the file exists in root)

- [ ] **Step 3: Commit**

```bash
git add tests/test_rulebook_rag.py
git commit -m "test: add rulebook ingestion test"
```

### Task 2: Query and Source Retrieval Verification

**Files:**
- Modify: `tests/test_rulebook_rag.py`

- [ ] **Step 1: Add query test case**

```python
def test_query_rulebook(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.store.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr("app.rag.query_engine.settings.chroma_persist_dir", str(tmp_path))

    # 1. Ingest
    with open(RULEBOOK_PATH, "rb") as f:
        client.post(
            "/ingest/pdf",
            files={"file": (RULEBOOK_PATH.name, f, "application/pdf")},
            params={"collection": "rulebook_test"},
        )

    # 2. Query
    resp = client.post(
        "/query",
        json={
            "question": "Who are the authors of the 2025 Edition?",
            "collection": "rulebook_test"
        },
    )

    assert resp.status_code == 200
    data = resp.json()
    assert "Akshay Pachaar" in data["answer"]
    assert "Avi Chawla" in data["answer"]
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["title"] == "AI Engineering Rulebook.pdf"
```

- [ ] **Step 2: Run test to verify query returns correct sources**

Run: `pytest tests/test_rulebook_rag.py::test_query_rulebook -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_rulebook_rag.py
git commit -m "test: add rulebook query and source retrieval test"
```