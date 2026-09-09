"""Snapshot-local Chroma and FTS indexes; never use default service collections."""

import math
import sqlite3
from collections.abc import Callable, Sequence
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.connectors.base import Document
from app.graph.models import GraphScope, Issue
from app.pipeline.chunker import chunk_documents
from app.retrieval.lexical import SQLiteFTSIndex

Embed = Callable[[list[str]], list[list[float]]]
COLLECTION = "chunks"


class SnapshotTextIndex:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        self.chroma = chromadb.PersistentClient(
            path=str(self.directory / "chroma"),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.lexical = SQLiteFTSIndex(self.directory / "lexical.db")

    def write(self, scope: GraphScope, issues: Sequence[Issue], texts: dict[str, str], embed: Embed) -> list[dict]:
        if any(issue.scope != scope for issue in issues):
            raise ValueError("text index scope mismatch")
        if len({issue.issue_id for issue in issues}) != len(issues):
            raise ValueError("duplicate issue identity")
        if len({issue.document_id for issue in issues}) != len(issues):
            raise ValueError("duplicate document identity")
        if set(texts) != {issue.issue_id for issue in issues}:
            raise ValueError("text and issue identities differ")
        chunks = []
        for issue in issues:
            if not texts[issue.issue_id].strip():
                raise ValueError("issue text must not be empty")
            metadata = {**scope.model_dump(), "issue_id": issue.issue_id, "issue_key": issue.key,
                        "key": issue.key, "issue_type": issue.issue_type, "status": issue.status,
                        "status_category": issue.status_category}
            document = Document(id=issue.document_id, content=texts[issue.issue_id],
                                source_type="jira_issue", title=issue.title, metadata=metadata)
            for chunk in chunk_documents([document]):
                # Stable document IDs are not mutable Jira keys. Use normalized source provenance.
                chunk["source_url"] = issue.source_url
                chunks.append(chunk)
        vectors = embed([chunk["content"] for chunk in chunks]) if chunks else []
        _validate_vectors(vectors, len(chunks))
        # A fresh snapshot has no collection. Refuse to mutate a previously built snapshot.
        collection = self.chroma.create_collection(
            COLLECTION, metadata={"hnsw:space": "cosine"}, embedding_function=None,
        )
        for offset in range(0, len(chunks), 100):
            batch = chunks[offset:offset + 100]
            collection.add(
                ids=[chunk["chunk_id"] for chunk in batch],
                embeddings=vectors[offset:offset + 100],
                documents=[chunk["content"] for chunk in batch],
                metadatas=[{**chunk["metadata"], "document_id": chunk["document_id"],
                            "chunk_id": chunk["chunk_id"], "source_url": chunk["source_url"],
                            "source_type": chunk["source_type"], "title": chunk["title"]} for chunk in batch],
            )
        for issue in issues:
            self.lexical.upsert_document(
                [chunk for chunk in chunks if chunk["document_id"] == issue.document_id], COLLECTION,
            )
        return chunks

    def verify(self, chunks: list[dict]) -> None:
        expected = {chunk["chunk_id"]: (chunk["content"], chunk["document_id"], chunk["source_url"])
                    for chunk in chunks}
        vector = self.chroma.get_collection(COLLECTION).get(include=["documents", "metadatas"])
        actual = {key: (text, meta["document_id"], meta["source_url"])
                  for key, text, meta in zip(vector["ids"], vector["documents"], vector["metadatas"])}
        with sqlite3.connect(self.lexical.db_path) as connection:
            rows = connection.execute(
                "SELECT c.chunk_id, f.content, c.document_id, c.source_url FROM lexical_chunks c "
                "JOIN lexical_fts f ON c.collection_name=f.collection_name AND c.chunk_id=f.chunk_id "
                "WHERE c.collection_name=?", (COLLECTION,),
            ).fetchall()
        lexical = {row[0]: tuple(row[1:]) for row in rows}
        if actual != expected or lexical != expected or len(rows) != len(expected):
            raise ValueError("text index verification failed")


def _validate_vectors(vectors: list[list[float]], count: int) -> None:
    if len(vectors) != count:
        raise ValueError("embedding count differs from chunks")
    dimension = len(vectors[0]) if vectors else 0
    for vector in vectors:
        if not dimension or len(vector) != dimension or any(not math.isfinite(value) for value in vector):
            raise ValueError("invalid embedding dimensions or values")
