"""Read-only hybrid search over one immutable graph snapshot."""

import json
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.graph.models import GraphScope, TextEvidence
from app.graph.text_index import COLLECTION, _validate_vectors
from app.retrieval.fusion import MAX_FUSED_CANDIDATES, ReciprocalRankFusion
from app.retrieval.lexical import SQLiteFTSIndex
from app.retrieval.models import RetrievalCandidate

Embed = Callable[[list[str]], list[list[float]]]
_JIRA_KEY = re.compile(r"\b[A-Z][A-Z0-9_]*-\d+\b", re.IGNORECASE)


class SnapshotSearch:
    """Search a pinned snapshot without consulting or creating global indexes."""

    def __init__(self, directory: Path, scope: GraphScope, embed: Embed):
        self.directory = Path(directory)
        self.scope = scope
        self.embed = embed
        self.diagnostics: list[str] = []
        required = (self.directory / "chroma", self.directory / "lexical.db", self.directory / "chunks.json")
        if not all(path.exists() for path in required):
            raise FileNotFoundError("snapshot text index is incomplete")

        self._chunks = self._read_chunks()
        self._by_chunk = {str(chunk.get("chunk_id", "")): chunk for chunk in self._chunks}
        if "" in self._by_chunk or len(self._by_chunk) != len(self._chunks):
            raise ValueError("snapshot contains invalid or duplicate chunk IDs")
        self.chroma = chromadb.PersistentClient(
            path=str(self.directory / "chroma"), settings=ChromaSettings(anonymized_telemetry=False),
        )
        try:
            self.collection = self.chroma.get_collection(COLLECTION)
        except Exception as error:
            raise FileNotFoundError("snapshot Chroma index or chunks collection is missing") from error
        # The path check above prevents SQLiteFTSIndex from manufacturing a missing database.
        self.lexical = SQLiteFTSIndex(self.directory / "lexical.db")
        self._validate_persisted_indexes()

    def search(self, query: str, top_k: int = 5) -> list[TextEvidence]:
        self.diagnostics = []
        if top_k < 0:
            raise ValueError("top_k must be non-negative")
        if top_k == 0 or not query.strip():
            return []
        if not self._chunks:
            return []
        limit = min(top_k, MAX_FUSED_CANDIDATES)
        candidate_limit = MAX_FUSED_CANDIDATES
        filters = self.scope.model_dump()
        exact = self._exact_candidates(query)
        if exact:
            # An explicit issue identity is already the strongest available signal.
            return [self._candidate_evidence(item) for item in exact[:limit]]

        result_sets: list[list[RetrievalCandidate]] = []
        try:
            vectors = self.embed([query])
            _validate_vectors(vectors, 1)
            result = self.collection.query(
                query_embeddings=vectors, n_results=min(candidate_limit, len(self._chunks)),
                where={"$and": [{key: value} for key, value in filters.items()]},
                include=["documents", "metadatas", "distances"],
            )
            result_sets.append(self._vector_candidates(result))
        except Exception as error:
            self.diagnostics.append(f"vector unavailable: {type(error).__name__}")

        try:
            result_sets.append(self.lexical.search(query, candidate_limit, filters, COLLECTION))
        except Exception as error:
            self.diagnostics.append(f"lexical unavailable: {type(error).__name__}")
            if not result_sets:
                raise RuntimeError("all snapshot retrieval indexes are unavailable") from error

        fused = ReciprocalRankFusion().fuse(result_sets, limit)
        return [self._candidate_evidence(item) for item in fused]

    def for_issues(self, issue_ids: list[str]) -> list[TextEvidence]:
        requested = set(issue_ids)
        known = {str(chunk["metadata"].get("issue_id", "")) for chunk in self._chunks}
        missing = requested - known
        if missing:
            raise ValueError(f"unknown issue IDs: {', '.join(sorted(missing))}")
        return [self._evidence(chunk, self.scope) for chunk in self._chunks
                if str(chunk["metadata"].get("issue_id", "")) in requested]

    def _read_chunks(self) -> list[dict]:
        value = json.loads((self.directory / "chunks.json").read_text(encoding="utf-8"))
        if not isinstance(value, list) or any(not isinstance(chunk, dict) for chunk in value):
            raise ValueError("snapshot chunks file is invalid")
        for chunk in value:
            self._validate_chunk(chunk)
        return value

    def _validate_persisted_indexes(self) -> None:
        vector = self.collection.get(include=["documents", "metadatas"])
        actual_vector = {
            str(chunk_id): (text, dict(metadata or {}))
            for chunk_id, text, metadata in zip(vector["ids"], vector["documents"], vector["metadatas"])
        }
        with sqlite3.connect(self.directory / "lexical.db") as connection:
            rows = connection.execute(
                "SELECT c.chunk_id, f.content, c.document_id, c.source_url, c.metadata_json "
                "FROM lexical_chunks c JOIN lexical_fts f ON c.collection_name=f.collection_name "
                "AND c.chunk_id=f.chunk_id WHERE c.collection_name=?", (COLLECTION,),
            ).fetchall()
        actual_lexical = {str(row[0]): row[1:] for row in rows}
        if set(actual_vector) != set(self._by_chunk) or set(actual_lexical) != set(self._by_chunk):
            raise ValueError("snapshot indexes differ from canonical chunks")
        for chunk_id, chunk in self._by_chunk.items():
            metadata = chunk["metadata"]
            vector_text, vector_meta = actual_vector[chunk_id]
            lexical_text, document_id, source_url, lexical_json = actual_lexical[chunk_id]
            lexical_meta = json.loads(lexical_json)
            self._validate_scope(vector_meta)
            self._validate_scope(lexical_meta)
            expected = (chunk["content"], chunk["document_id"], chunk["source_url"])
            if (vector_text, vector_meta.get("document_id"), vector_meta.get("source_url")) != expected:
                raise ValueError("snapshot vector provenance mismatch")
            if (lexical_text, document_id, source_url) != expected or lexical_meta != metadata:
                raise ValueError("snapshot lexical provenance mismatch")

    def _exact_candidates(self, query: str) -> list[RetrievalCandidate]:
        keys = {match.group(0).upper() for match in _JIRA_KEY.finditer(query)}
        matches = [chunk for chunk in self._chunks
                   if str(chunk["metadata"].get("issue_key") or chunk["metadata"].get("key", "")).upper() in keys]
        return [self._candidate(chunk, "exact", rank, exact=True) for rank, chunk in enumerate(matches, 1)]

    def _vector_candidates(self, result: dict) -> list[RetrievalCandidate]:
        values = zip(result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0])
        output = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(values, 1):
            chunk = self._canonical(str(chunk_id), text, metadata)
            output.append(self._candidate(chunk, "vector", rank, score=1 - float(distance)))
        return output

    def _candidate(self, chunk: dict, method: str, rank: int, score: float = 1.0,
                   exact: bool = False) -> RetrievalCandidate:
        return RetrievalCandidate(
            content=chunk["content"], document_id=chunk["document_id"], chunk_id=chunk["chunk_id"],
            source_type=chunk["source_type"], source_url=chunk["source_url"], title=chunk["title"],
            metadata=chunk["metadata"], score=score, method_scores={method: score},
            retrieval_methods=[method], rank_by_method={method: rank}, exact_match=exact,
            collection_name=COLLECTION,
        )

    def _canonical(self, chunk_id: str, text: str, metadata: dict) -> dict:
        chunk = self._by_chunk.get(chunk_id)
        if chunk is None:
            raise ValueError("retrieval returned an unknown chunk")
        self._validate_scope(dict(metadata or {}))
        if text != chunk["content"] or metadata.get("document_id") != chunk["document_id"]:
            raise ValueError("retrieval provenance mismatch")
        return chunk

    def _candidate_evidence(self, candidate: RetrievalCandidate) -> TextEvidence:
        chunk = self._canonical(candidate.chunk_id, candidate.content,
                                {**candidate.metadata, "document_id": candidate.document_id})
        return self._evidence(chunk, self.scope)

    def _validate_chunk(self, chunk: dict) -> None:
        required = ("chunk_id", "document_id", "content", "source_url", "metadata")
        if any(not chunk.get(key) for key in required) or not isinstance(chunk.get("metadata"), dict):
            raise ValueError("snapshot chunk provenance is incomplete")
        self._validate_scope(chunk["metadata"])
        if not chunk["metadata"].get("issue_id"):
            raise ValueError("snapshot chunk issue identity is missing")

    def _validate_scope(self, metadata: dict) -> None:
        if any(metadata.get(key) != value for key, value in self.scope.model_dump().items()):
            raise ValueError("snapshot text index scope mismatch")

    @staticmethod
    def _evidence(chunk: dict, scope: GraphScope) -> TextEvidence:
        return TextEvidence(
            scope=scope, issue_id=str(chunk["metadata"]["issue_id"]),
            document_id=str(chunk["document_id"]), chunk_id=str(chunk["chunk_id"]),
            text=str(chunk["content"]), source_url=str(chunk["source_url"]),
        )
