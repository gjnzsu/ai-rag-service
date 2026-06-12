from unittest.mock import patch

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
    )

    assert result == {"document_id": "doc-1", "chunk_count": 1, "created": True}
    stored_chunks = mock_upsert.call_args.kwargs["chunks"]
    assert stored_chunks[0]["document_id"] == "doc-1"
    assert stored_chunks[0]["chunk_id"] == stored_chunks[0]["id"]
    assert stored_chunks[0]["chunk_index"] == 0
    assert mock_upsert.call_args.kwargs["collection_name"] == "research_reports"
