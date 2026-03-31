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
            chunks.append({
                "id": f"{doc.id}_chunk_{i}",
                "content": text,
                "document_id": doc.id,
                "source_type": doc.source_type,
                "title": doc.title,
                **flat_meta,
            })
    return chunks
