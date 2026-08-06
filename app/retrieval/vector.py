"""Chroma-backed vector retrieval adapter."""

from typing import Any

import chromadb
from openai import OpenAI

from app.config import settings
from app.pipeline.store import _to_chroma_where
from app.retrieval.models import RetrievalCandidate


class ChromaVectorRetriever:
    """Embed one query and map Chroma's records to canonical candidates."""

    def __init__(self, openai_client: Any | None = None, chroma_client: Any | None = None) -> None:
        self.openai_client = openai_client if openai_client is not None else OpenAI(api_key=settings.openai_api_key)
        self.chroma_client = chroma_client if chroma_client is not None else chromadb.PersistentClient(path=settings.chroma_persist_dir)

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        collection_name: str,
    ) -> list[RetrievalCandidate]:
        if top_k <= 0:
            return []

        embedding = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[query],
        ).data[0].embedding
        collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        params: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        where = _to_chroma_where(filters)
        if where:
            params["where"] = where
        result = collection.query(**params)

        ids = result.get("ids", [[]])[0] or []
        documents = result.get("documents", [[]])[0] or []
        metadatas = result.get("metadatas", [[]])[0] or []
        distances = result.get("distances", [[]])[0] or []
        candidates: list[RetrievalCandidate] = []
        for rank, (chunk_id, content, metadata, distance) in enumerate(
            zip(ids, documents, metadatas, distances), start=1
        ):
            trusted_metadata = dict(metadata or {})
            score = round(1 - float(distance), 4)
            candidates.append(
                RetrievalCandidate(
                    content=content,
                    chunk_id=str(trusted_metadata.get("chunk_id") or chunk_id),
                    document_id=str(trusted_metadata.get("document_id", "")),
                    source_type=str(trusted_metadata.get("source_type") or trusted_metadata.get("type", "")),
                    source_url=str(trusted_metadata.get("source_url", "")),
                    title=str(trusted_metadata.get("title", "")),
                    metadata=trusted_metadata,
                    score=score,
                    method_scores={"vector": score},
                    retrieval_methods=["vector"],
                    rank_by_method={"vector": rank},
                    collection_name=collection_name,
                )
            )
        return candidates
