import pytest

from app.pipeline.store import upsert_chunks, query_collection


@pytest.fixture
def tmp_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.store.settings.chroma_persist_dir", str(tmp_path))
    return str(tmp_path)


def test_upsert_and_query(tmp_chroma):
    chunks = [
        {"id": "c1", "content": "the cat sat on the mat",
         "document_id": "doc1", "source_type": "pdf", "title": "Test"},
        {"id": "c2", "content": "dogs love to run in parks",
         "document_id": "doc1", "source_type": "pdf", "title": "Test"},
    ]
    # Use orthogonal vectors so cosine similarity is unambiguous
    emb_cat = [1.0] + [0.0] * 1535
    emb_dog = [0.0, 1.0] + [0.0] * 1534
    embeddings = [emb_cat, emb_dog]
    upsert_chunks(chunks, embeddings, collection_name="test_col")

    results = query_collection(
        query_embedding=emb_cat,  # should match chunk c1
        collection_name="test_col",
        top_k=1,
    )
    assert len(results["documents"][0]) == 1
    assert results["documents"][0][0] == "the cat sat on the mat"


def test_upsert_is_idempotent(tmp_chroma):
    chunks = [{"id": "c1", "content": "hello",
               "document_id": "doc1", "source_type": "pdf", "title": "T"}]
    embeddings = [[0.1] * 1536]
    upsert_chunks(chunks, embeddings, collection_name="test_col")
    upsert_chunks(chunks, embeddings, collection_name="test_col")  # should not raise

    results = query_collection([0.1] * 1536, collection_name="test_col", top_k=5)
    assert len(results["documents"][0]) == 1  # still one, not two
