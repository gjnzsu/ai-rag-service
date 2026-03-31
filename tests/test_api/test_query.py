from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


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


def test_query_missing_question():
    response = client.post("/query", json={})
    assert response.status_code == 422


@patch("app.api.query.query_engine.query")
def test_query_engine_error_returns_502(mock_query):
    mock_query.side_effect = Exception("ChromaDB unavailable")
    response = client.post("/query", json={"question": "anything"})
    assert response.status_code == 502
