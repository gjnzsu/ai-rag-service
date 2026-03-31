from unittest.mock import MagicMock, patch

from app.pipeline.embedder import embed_chunks


def make_chunks(n: int) -> list[dict]:
    return [{"id": f"chunk_{i}", "content": f"text {i}"} for i in range(n)]


def mock_embedding_response(texts):
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in texts]
    return mock_resp


@patch("app.pipeline.embedder.OpenAI")
def test_embed_chunks_returns_one_vector_per_chunk(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.side_effect = lambda **kwargs: mock_embedding_response(kwargs["input"])

    chunks = make_chunks(3)
    embeddings = embed_chunks(chunks)
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 1536


@patch("app.pipeline.embedder.OpenAI")
def test_embed_chunks_batches_correctly(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.side_effect = lambda **kwargs: mock_embedding_response(kwargs["input"])

    chunks = make_chunks(250)  # exceeds batch size of 100
    embeddings = embed_chunks(chunks)
    assert len(embeddings) == 250
    # 250 chunks / 100 batch size = 3 API calls
    assert mock_client.embeddings.create.call_count == 3
