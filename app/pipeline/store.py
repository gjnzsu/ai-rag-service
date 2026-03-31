import chromadb

from app.config import settings


def _get_collection(collection_name: str):
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection_name: str = "default",
) -> None:
    collection = _get_collection(collection_name)
    metadatas = [
        {k: str(v) for k, v in chunk.items() if k != "content"}
        for chunk in chunks
    ]
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["content"] for c in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_collection(
    query_embedding: list[float],
    collection_name: str = "default",
    top_k: int = 5,
) -> dict:
    collection = _get_collection(collection_name)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
