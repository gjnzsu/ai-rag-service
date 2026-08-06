import structlog

from app.connectors.base import Document
from app.pipeline.chunker import chunk_documents
from app.pipeline.embedder import embed_chunks
from app.pipeline.store import upsert_document
from app.retrieval.lexical import SQLiteFTSIndex

logger = structlog.get_logger()


def index_documents(
    documents: list[Document],
    collection_name: str = "default",
    lexical_index: SQLiteFTSIndex | None = None,
) -> dict:
    """Chunk, embed, and upsert documents through the shared indexing path."""
    if not documents:
        return {"document_id": "", "chunk_count": 0, "created": True}

    chunks = chunk_documents(documents)
    if not chunks:
        return {"document_id": documents[0].id, "chunk_count": 0, "created": True}

    embeddings = embed_chunks(chunks)
    lexical_index = lexical_index or SQLiteFTSIndex()
    results = []
    start = 0
    for document in documents:
        document_chunks = [chunk for chunk in chunks if chunk["document_id"] == document.id]
        if not document_chunks:
            continue
        document_embeddings = embeddings[start:start + len(document_chunks)]
        start += len(document_chunks)
        try:
            vector_result = upsert_document(
                chunks=document_chunks,
                embeddings=document_embeddings,
                collection_name=collection_name,
            )
            lexical_index.upsert_document(document_chunks, collection_name)
        except Exception:
            logger.error(
                "dual_index_upsert_failed",
                collection_name=collection_name,
                document_id=document.id,
            )
            raise
        results.append(vector_result)

    return {
        "document_id": documents[0].id,
        "chunk_count": len(chunks),
        "created": all(result["created"] for result in results),
    }


def _add_chunk_metadata(chunks: list[dict]) -> list[dict]:
    return chunks
