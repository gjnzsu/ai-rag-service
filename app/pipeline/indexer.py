from app.connectors.base import Document
from app.pipeline.chunker import chunk_documents
from app.pipeline.embedder import embed_chunks
from app.pipeline.store import upsert_document


def index_documents(
    documents: list[Document],
    collection_name: str = "default",
) -> dict:
    """Chunk, embed, and upsert documents through the shared indexing path."""
    if not documents:
        return {"document_id": "", "chunk_count": 0, "created": True}

    chunks = _add_chunk_metadata(chunk_documents(documents))
    if not chunks:
        return {"document_id": documents[0].id, "chunk_count": 0, "created": True}

    embeddings = embed_chunks(chunks)
    return upsert_document(
        chunks=chunks,
        embeddings=embeddings,
        collection_name=collection_name,
    )


def _add_chunk_metadata(chunks: list[dict]) -> list[dict]:
    for index, chunk in enumerate(chunks):
        chunk["chunk_id"] = chunk["id"]
        chunk["chunk_index"] = index
    return chunks
