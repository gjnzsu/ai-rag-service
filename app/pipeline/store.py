import chromadb
import structlog

from app.config import settings
from app.retrieval.lexical import SQLiteFTSIndex

logger = structlog.get_logger()


def get_lexical_index() -> SQLiteFTSIndex:
    return SQLiteFTSIndex()


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


def _to_chroma_where(filters: dict | None) -> dict | None:
    if not filters:
        return None

    clauses = []
    for key, value in filters.items():
        if isinstance(value, dict):
            if "in" in value:
                clauses.append({key: {"$in": [str(item) for item in value["in"]]}})
            else:
                raise ValueError(f"Unsupported filter operator for {key}")
        else:
            clauses.append({key: str(value)})

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


def upsert_document(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection_name: str = "default",
) -> dict:
    if not chunks:
        return {"document_id": "", "chunk_count": 0, "created": True}

    document_id = chunks[0]["document_id"]
    collection = _get_collection(collection_name)
    existing = collection.get(where={"document_id": str(document_id)})
    existing_ids = existing.get("ids", [])
    if existing_ids:
        collection.delete(ids=existing_ids)

    upsert_chunks(chunks, embeddings, collection_name=collection_name)
    return {
        "document_id": document_id,
        "chunk_count": len(chunks),
        "created": not bool(existing_ids),
    }


def delete_document(document_id: str, collection_name: str = "default") -> bool:
    vector_deleted = False
    try:
        collection = _get_collection(collection_name)
        existing = collection.get(where={"document_id": str(document_id)})
        ids = existing.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            vector_deleted = True
        lexical_deleted = get_lexical_index().delete_document(document_id, collection_name)
    except Exception:
        logger.error(
            "dual_index_delete_failed",
            collection_name=collection_name,
            document_id=document_id,
        )
        raise
    return vector_deleted or lexical_deleted


def get_document_chunks(
    document_id: str,
    collection_name: str = "default",
) -> list[dict]:
    collection = _get_collection(collection_name)
    result = collection.get(
        where={"document_id": str(document_id)},
        include=["documents", "metadatas"],
    )
    return _format_get_results(result)


def query_lifecycle_collection(
    query_embedding: list[float],
    collection_name: str = "default",
    top_k: int = 5,
    filters: dict | None = None,
) -> list[dict]:
    collection = _get_collection(collection_name)
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    where = _to_chroma_where(filters)
    if where:
        query_params["where"] = where

    result = collection.query(**query_params)
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        _format_result(
            content=document,
            metadata=metadata,
            distance=distance,
        )
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


def get_jira_key_context(
    jira_key: str,
    collection_name: str = "default",
    filters: dict | None = None,
) -> list[dict]:
    collection = _get_collection(collection_name)
    results = []
    seen_chunk_ids = set()
    filter_where = _to_chroma_where(filters)
    for base_where in (
        {"document_id": f"jira:{jira_key}"},
        {"document_id": f"jira_issue:{jira_key}"},
        {"issue_key": jira_key},
        {"key": jira_key},
        {"related_jira": jira_key},
    ):
        where = base_where if not filter_where else {"$and": [base_where, filter_where]}
        result = collection.get(where=where, include=["documents", "metadatas"])
        for item in _format_get_results(result):
            chunk_id = item["chunk_id"]
            if chunk_id not in seen_chunk_ids:
                results.append(item)
                seen_chunk_ids.add(chunk_id)
    return results


def refresh_jira_key(jira_key: str, collection_name: str = "default") -> int:
    collection = _get_collection(collection_name)
    ids = []
    for where in (
        {"document_id": f"jira:{jira_key}"},
        {"document_id": f"jira_issue:{jira_key}"},
        {"key": jira_key},
        {"related_jira": jira_key},
    ):
        result = collection.get(where=where)
        ids.extend(result.get("ids", []))
    unique_ids = sorted(set(ids))
    if unique_ids:
        collection.delete(ids=unique_ids)
    try:
        get_lexical_index().delete_jira_key(jira_key, collection_name)
    except Exception:
        logger.error(
            "dual_index_refresh_failed",
            collection_name=collection_name,
            document_id=f"jira:{jira_key}",
        )
        raise
    return len(unique_ids)


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


def _format_get_results(result: dict) -> list[dict]:
    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])
    return [
        _format_result(
            content=document,
            metadata=metadata,
            chunk_id=chunk_id,
            distance=0.0,
        )
        for chunk_id, document, metadata in zip(ids, documents, metadatas)
    ]


def _format_result(
    *,
    content: str,
    metadata: dict,
    distance: float,
    chunk_id: str | None = None,
) -> dict:
    chunk_id = chunk_id or metadata.get("chunk_id") or metadata.get("id", "")
    return {
        "content": content,
        "score": round(1 - float(distance), 4),
        "document_id": metadata.get("document_id", ""),
        "chunk_id": chunk_id,
        "metadata": metadata,
        "source_type": metadata.get("source_type", metadata.get("type", "")),
        "source_url": metadata.get("source_url", ""),
        "title": metadata.get("title", ""),
    }
