from app.connectors.base import Document
from app.pipeline.chunker import chunk_documents


def make_doc(content: str, source_type: str = "pdf") -> Document:
    return Document(
        id="doc1",
        content=content,
        source_type=source_type,
        title="Test Doc",
        metadata={"filename": "test.pdf"},
    )


def test_short_doc_produces_one_chunk():
    doc = make_doc("Short content.")
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Short content."


def test_long_doc_produces_multiple_chunks():
    long_text = "word " * 300  # ~1500 chars
    doc = make_doc(long_text)
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_chunk_has_required_fields():
    doc = make_doc("Some content here.")
    chunk = chunk_documents([doc])[0]
    assert "id" in chunk
    assert "content" in chunk
    assert "document_id" in chunk
    assert chunk["document_id"] == "doc1"
    assert chunk["source_type"] == "pdf"
    assert chunk["title"] == "Test Doc"


def test_chunk_ids_are_unique():
    long_text = "word " * 300
    doc = make_doc(long_text)
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
