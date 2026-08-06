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
    reserved_fields = {
        "id", "chunk_id", "chunk_index", "content", "document_id", "source_type", "title", "source_url",
    }
    chunks = []
    for doc in documents:
        texts = splitter.split_text(doc.content)
        for i, text in enumerate(texts):
            flat_meta = {k: str(v) for k, v in doc.metadata.items() if k not in reserved_fields}
            chunk_id = f"{doc.id}:chunk:{i}"
            chunks.append({
                **flat_meta,
                "id": chunk_id,
                "chunk_id": chunk_id,
                "chunk_index": i,
                "content": text,
                "document_id": doc.id,
                "source_type": doc.source_type,
                "source_url": _trusted_source_url(doc),
                "title": doc.title,
                "metadata": dict(doc.metadata),
            })
    return chunks


def _trusted_source_url(doc: Document) -> str:
    if doc.source_type in {"jira", "jira_issue"} and doc.id.startswith("jira:") and settings.jira_url:
        return f"{settings.jira_url.rstrip('/')}/browse/{doc.id.removeprefix('jira:')}"
    if doc.source_type in {"confluence", "confluence_page"} and doc.id.startswith("confluence:") and settings.confluence_url:
        return f"{settings.confluence_url.rstrip('/')}/pages/{doc.id.removeprefix('confluence:')}"
    return ""
