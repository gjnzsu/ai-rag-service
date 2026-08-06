from unittest.mock import MagicMock, patch

import pytest

from app.connectors.base import Document


@patch("app.pipeline.indexer.upsert_document")
@patch("app.pipeline.indexer.embed_chunks")
def test_index_documents_chunks_embeds_and_upserts(mock_embed, mock_upsert):
    from app.pipeline.indexer import index_documents

    mock_embed.return_value = [[0.1] * 1536]
    mock_upsert.return_value = {
        "document_id": "doc-1",
        "chunk_count": 1,
        "created": True,
    }
    lexical = MagicMock()

    result = index_documents(
        [
            Document(
                id="doc-1",
                content="Shared indexing content",
                source_type="pdf",
                title="Shared Indexing",
                metadata={"document_type": "research_report"},
            )
        ],
        collection_name="research_reports",
        lexical_index=lexical,
    )

    assert result == {"document_id": "doc-1", "chunk_count": 1, "created": True}
    stored_chunks = mock_upsert.call_args.kwargs["chunks"]
    assert stored_chunks[0]["document_id"] == "doc-1"
    assert stored_chunks[0]["chunk_id"] == stored_chunks[0]["id"]
    assert stored_chunks[0]["chunk_index"] == 0
    assert mock_upsert.call_args.kwargs["collection_name"] == "research_reports"


@patch("app.pipeline.indexer.upsert_document")
@patch("app.pipeline.indexer.embed_chunks")
def test_index_documents_writes_the_same_chunks_to_vector_and_lexical_indexes(mock_embed, mock_vector):
    from app.pipeline.indexer import index_documents

    mock_embed.return_value = [[0.1] * 1536]
    mock_vector.return_value = {"document_id": "jira:PROJ-7", "chunk_count": 1, "created": True}
    lexical = MagicMock()

    index_documents(
        [Document(id="jira:PROJ-7", content="Login audit", source_type="jira", title="Login")],
        collection_name="alpha",
        lexical_index=lexical,
    )

    vector_chunks = mock_vector.call_args.kwargs["chunks"]
    lexical_chunks = lexical.upsert_document.call_args.args[0]
    assert vector_chunks is lexical_chunks
    assert lexical.upsert_document.call_args.args[1] == "alpha"


@patch("app.pipeline.indexer.logger")
@patch("app.pipeline.indexer.upsert_document")
@patch("app.pipeline.indexer.embed_chunks")
def test_partial_lexical_failure_logs_safely_and_repeated_upsert_repairs(mock_embed, mock_vector, mock_logger):
    from app.pipeline.indexer import index_documents

    mock_embed.return_value = [[0.1] * 1536]
    mock_vector.return_value = {"document_id": "jira:PROJ-7", "chunk_count": 1, "created": True}
    lexical = MagicMock()
    lexical.upsert_document.side_effect = [RuntimeError("lexical unavailable"), None]
    document = Document(id="jira:PROJ-7", content="secret login detail", source_type="jira", title="Login")

    with pytest.raises(RuntimeError, match="lexical unavailable"):
        index_documents([document], collection_name="alpha", lexical_index=lexical)
    index_documents([document], collection_name="alpha", lexical_index=lexical)

    mock_logger.error.assert_called_once_with(
        "dual_index_upsert_failed", collection_name="alpha", document_id="jira:PROJ-7",
    )
    assert mock_vector.call_count == 2
    assert lexical.upsert_document.call_count == 2
