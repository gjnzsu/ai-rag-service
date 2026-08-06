"""Integration test: PDF ingest -> ChromaDB -> query pipeline (mocks OpenAI only)."""
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.grounding.models import GeneratedAnswer
from app.main import app, create_app
from app.rag import query_engine
from app.rag.query_engine import QueryPipeline
from app.retrieval.models import RetrievalCandidate
from app.retrieval.pipeline import RetrievalResult

client = TestClient(app)


class _InjectedRetrieval:
    def __init__(self, candidate):
        self.candidate = candidate
        self.calls = []

    def retrieve(self, query, collection_name="default", top_k=None, filters=None):
        self.calls.append((query, collection_name, top_k, filters))
        return RetrievalResult(
            candidates=[self.candidate],
            retrieval_mode="hybrid",
            failures=["lexical"],
            diagnostics={
                "reranker": {"provider": "none", "status": "disabled"},
                "private": {"question": query, "content": self.candidate.content},
            },
        )


class _InjectedGenerator:
    model = "injected-model"

    def __init__(self):
        self.calls = []

    def generate(self, question, evidence):
        self.calls.append((question, evidence))
        return GeneratedAnswer(answer="Grounded answer [E1]", citation_ids=["E1"])


def test_retrieve_and_query_share_injected_application_pipeline_but_only_query_generates(
    monkeypatch,
):
    candidate = RetrievalCandidate(
        content="Trusted integration passage",
        document_id="document-1",
        chunk_id="chunk-1",
        source_type="pdf",
        source_url="https://trusted.example/document-1",
        title="Integration source",
        metadata={"document_type": "guide"},
        score=0.9,
        retrieval_methods=["vector"],
        method_scores={"vector": 0.9},
        rank_by_method={"vector": 1},
        fused_rank=1,
    )
    retrieval = _InjectedRetrieval(candidate)
    generator = _InjectedGenerator()
    shared = QueryPipeline(
        retrieval_pipeline=retrieval,
        generator=generator,
        model=generator.model,
        evidence_top_k=5,
    )
    constructions = []
    query_engine.close_default_query_pipeline()

    def build_pipeline():
        constructions.append(True)
        return shared

    monkeypatch.setattr(query_engine, "QueryPipeline", build_pipeline)

    with TestClient(create_app()) as routed_client:
        retrieve_response = routed_client.post("/retrieve", json={
            "query": "integration question",
            "collection": "alpha",
            "top_k": 3,
            "filters": {"document_type": "guide"},
        })
        assert retrieve_response.status_code == 200
        assert retrieve_response.json()["results"][0]["source_url"] == candidate.source_url
        assert generator.calls == []

        query_response = routed_client.post("/query", json={
            "question": "integration question",
            "collection": "alpha",
            "top_k": 3,
            "document_type": "guide",
        })
        assert query_response.status_code == 200
        assert query_response.json()["citations"][0]["chunk_id"] == candidate.chunk_id

    assert constructions == [True]
    assert retrieval.calls == [
        ("integration question", "alpha", 3, {"document_type": "guide"}),
        ("integration question", "alpha", 3, {"document_type": "guide"}),
    ]
    assert len(generator.calls) == 1
    assert shared._closed is True


@pytest.fixture(autouse=True)
def patch_openai():
    """Replace all OpenAI calls with deterministic mocks."""
    query_engine.close_default_query_pipeline()
    fake_embedding = [0.1] * 1536

    def fake_embed(**kwargs):
        n = len(kwargs["input"])
        return MagicMock(data=[MagicMock(embedding=fake_embedding) for _ in range(n)])

    def fake_chat(**kwargs):
        return MagicMock(
            choices=[MagicMock(message=MagicMock(
                content=json.dumps({
                    "answer": "The USD to CNY rate is 7.25 according to the document [E1].",
                    "citation_ids": ["E1"],
                })
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
        query_engine.close_default_query_pipeline()


def test_ingest_pdf_then_query(sample_pdf_path, tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.store.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr("app.pipeline.store.settings.lexical_db_path", str(tmp_path / "lexical.db"))
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
