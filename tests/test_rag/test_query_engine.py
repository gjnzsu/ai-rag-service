from unittest.mock import MagicMock, patch

from app.rag.query_engine import query


def mock_chroma_results(docs, metas, distances):
    return {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


@patch("app.rag.query_engine.chromadb")
@patch("app.rag.query_engine.OpenAI")
def test_query_returns_answer_and_sources(mock_openai_cls, mock_chromadb):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="The rate is 7.25."))]
    )
    mock_chroma_client = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_chroma_client
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    mock_collection.query.return_value = mock_chroma_results(
        ["1 USD = 7.25 CNY"],
        [{"document_id": "d1", "source_type": "fx", "title": "FX Rates"}],
        [0.05],
    )

    result = query("What is USD to CNY?")
    assert result["answer"] == "The rate is 7.25."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_type"] == "fx"
    assert result["model"] == "gpt-4o"


@patch("app.rag.query_engine.chromadb")
@patch("app.rag.query_engine.OpenAI")
def test_query_empty_collection_returns_no_info(mock_openai_cls, mock_chromadb):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    mock_chroma_client = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_chroma_client
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    mock_collection.query.return_value = mock_chroma_results([], [], [])

    result = query("What is the answer?")
    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []
