import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from app.config import settings
from app.retrieval.models import RetrievalCandidate, canonical_jira_key


class SQLiteFTSIndex:
    """A persistent, collection-scoped FTS5 index for canonical chunks."""

    def __init__(self, db_path: str | Path | None = None):
        self.db_path = Path(db_path or settings.lexical_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def upsert_document(self, chunks: list[dict], collection_name: str) -> None:
        if not chunks:
            return

        document_id = str(chunks[0]["document_id"])
        if any(str(chunk["document_id"]) != document_id for chunk in chunks):
            raise ValueError("All chunks in a document upsert must have the same document_id")

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM lexical_fts WHERE collection_name = ? "
                "AND chunk_id IN (SELECT chunk_id FROM lexical_chunks "
                "WHERE collection_name = ? AND document_id = ?)",
                (collection_name, collection_name, document_id),
            )
            connection.execute(
                "DELETE FROM lexical_chunks WHERE collection_name = ? AND document_id = ?",
                (collection_name, document_id),
            )
            for chunk in chunks:
                metadata = _chunk_metadata(chunk)
                issue_key = _jira_key(chunk, metadata)
                values = (
                    collection_name,
                    str(chunk["chunk_id"]),
                    document_id,
                    str(chunk.get("source_type", "")),
                    str(chunk.get("source_url", "")),
                    str(chunk.get("title", "")),
                    json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                    issue_key,
                    str(chunk.get("content", "")),
                )
                connection.execute(
                    "INSERT INTO lexical_chunks "
                    "(collection_name, chunk_id, document_id, source_type, source_url, title, metadata_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    values[:7],
                )
                connection.execute(
                    "INSERT INTO lexical_fts (collection_name, chunk_id, issue_key, title, content) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (values[0], values[1], values[7], values[5], values[8]),
                )

    def delete_document(self, document_id: str, collection_name: str) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM lexical_chunks "
                "WHERE collection_name = ? AND document_id = ?",
                (collection_name, document_id),
            ).fetchone()
            if not row["count"]:
                return False
            connection.execute(
                "DELETE FROM lexical_fts WHERE collection_name = ? "
                "AND chunk_id IN (SELECT chunk_id FROM lexical_chunks "
                "WHERE collection_name = ? AND document_id = ?)",
                (collection_name, collection_name, document_id),
            )
            connection.execute(
                "DELETE FROM lexical_chunks WHERE collection_name = ? AND document_id = ?",
                (collection_name, document_id),
            )
            return True

    def delete_jira_key(self, jira_key: str, collection_name: str) -> int:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT document_id FROM lexical_chunks WHERE collection_name = ? AND ("
                "document_id = ? OR CAST(json_extract(metadata_json, '$.issue_key') AS TEXT) = ? "
                "OR CAST(json_extract(metadata_json, '$.key') AS TEXT) = ? "
                "OR CAST(json_extract(metadata_json, '$.related_jira') AS TEXT) = ?)",
                (collection_name, f"jira:{jira_key}", jira_key, jira_key, jira_key),
            ).fetchall()
        document_ids = sorted({row["document_id"] for row in rows})
        for document_id in document_ids:
            self.delete_document(document_id, collection_name)
        return len(document_ids)

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        collection_name: str,
    ) -> list[RetrievalCandidate]:
        match_query = _safe_match_query(query)
        if not match_query or top_k <= 0:
            return []

        filter_sql, filter_values = _filter_sql(filters)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT chunks.chunk_id, chunks.document_id, chunks.source_type, chunks.source_url, "
                "chunks.title, chunks.metadata_json, fts.content, "
                "bm25(lexical_fts, 0.0, 0.0, ?, ?, ?) AS bm25_score "
                "FROM lexical_fts AS fts "
                "JOIN lexical_chunks AS chunks "
                "ON chunks.collection_name = fts.collection_name AND chunks.chunk_id = fts.chunk_id "
                "WHERE lexical_fts MATCH ? AND fts.collection_name = ?"
                f"{filter_sql} ORDER BY bm25_score ASC LIMIT ?",
                (
                    settings.lexical_issue_key_weight,
                    settings.lexical_title_weight,
                    settings.lexical_content_weight,
                    match_query,
                    collection_name,
                    *filter_values,
                    top_k,
                ),
            ).fetchall()

        return [
            RetrievalCandidate(
                content=row["content"],
                document_id=row["document_id"],
                chunk_id=row["chunk_id"],
                source_type=row["source_type"],
                source_url=row["source_url"],
                title=row["title"],
                metadata=json.loads(row["metadata_json"]),
                score=-float(row["bm25_score"]),
                retrieval_methods=["bm25"],
                rank_by_method={"bm25": rank},
                collection_name=collection_name,
            )
            for rank, row in enumerate(rows, start=1)
        ]

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS lexical_chunks ("
                "collection_name TEXT NOT NULL, chunk_id TEXT NOT NULL, document_id TEXT NOT NULL, "
                "source_type TEXT NOT NULL, source_url TEXT NOT NULL, title TEXT NOT NULL, "
                "metadata_json TEXT NOT NULL, PRIMARY KEY (collection_name, chunk_id))"
            )
            connection.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS lexical_fts USING fts5("
                "collection_name UNINDEXED, chunk_id UNINDEXED, issue_key, title, content, "
                "tokenize='unicode61')"
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection


def _chunk_metadata(chunk: dict) -> dict[str, Any]:
    metadata = chunk.get("metadata")
    if isinstance(metadata, dict):
        return metadata
    return {
        key: value
        for key, value in chunk.items()
        if key
        not in {
            "id", "chunk_id", "chunk_index", "content", "document_id", "source_type",
            "source_url", "title", "metadata",
        }
    }


def _jira_key(chunk: dict, metadata: dict[str, Any]) -> str:
    return canonical_jira_key({**metadata, **{key: chunk[key] for key in ("issue_key", "key") if key in chunk}})


def _safe_match_query(query: str) -> str:
    tokens = re.findall(r"[^\W_]+", query, flags=re.UNICODE)
    return " AND ".join(f'"{token}"' for token in tokens)


def _filter_sql(filters: dict[str, Any] | None) -> tuple[str, list[str]]:
    if not filters:
        return "", []

    clauses: list[str] = []
    values: list[str] = []
    for field, value in filters.items():
        column = {"document_id": "document_id", "source_type": "source_type", "title": "title"}.get(field)
        items = value.get("in") if isinstance(value, dict) and "in" in value else [value]
        if not isinstance(items, list):
            raise ValueError(f"Filter operator 'in' for {field} requires a list")
        if not items:
            clauses.append("1 = 0")
            continue
        if column:
            placeholders = ", ".join("?" for _ in items)
            clauses.append(f"chunks.{column} IN ({placeholders})")
            values.extend(_filter_value(item) for item in items)
            continue
        path = f"$.{field}"
        item_clauses = []
        for item in items:
            item_clauses.append(
                "(json_type(chunks.metadata_json, ?) = ? "
                "AND CAST(json_extract(chunks.metadata_json, ?) AS TEXT) = ?)"
            )
            values.extend((path, _json_type(item), path, _filter_value(item)))
        clauses.append("(" + " OR ".join(item_clauses) + ")")
    return " AND " + " AND ".join(clauses), values


def _filter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "real"
    return "text"
