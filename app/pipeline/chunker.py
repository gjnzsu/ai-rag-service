from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.connectors.base import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for doc in documents:
        texts = splitter.split_text(doc.content)
        for i, text in enumerate(texts):
            flat_meta = {k: str(v) for k, v in doc.metadata.items()}
            chunk_id = f"{doc.id}:chunk:{i}"
            chunks.append({
                "id": chunk_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "content": text,
                "document_id": doc.id,
                "source_type": doc.source_type,
                "source_url": str(doc.metadata.get("source_url", "")),
                "title": doc.title,
                "metadata": dict(doc.metadata),
                **flat_meta,
            })
    return chunks
