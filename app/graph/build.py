"""Build and verify isolated indexes before publishing a visible-project snapshot."""

import hashlib
import json
import time
from collections import Counter
from importlib.metadata import version
from uuid import uuid4

from app.graph.jira import CapturedProject, normalize
from app.graph.models import GraphScope, SnapshotManifest
from app.graph.snapshots import SnapshotRepository, write_json
from app.graph.text_index import Embed, SnapshotTextIndex


def build_snapshot(repository: SnapshotRepository, graph_store, capture: CapturedProject,
                   site_url: str, project_key: str, embed: Embed) -> SnapshotManifest:
    site_id = site_url.rstrip("/")
    started = time.monotonic()
    with repository.build_lock(site_id, project_key):
        scope = GraphScope(site_id=site_id, project_key=project_key, snapshot_id=uuid4().hex)
        directory = repository.reserve(scope)
        phase = "normalize"
        try:
            normalized = normalize(scope, capture, site_url)
            raw = {"issues": capture.issues, "account_scope": capture.account_scope,
                   "started_at": capture.started_at.isoformat(), "finished_at": capture.finished_at.isoformat()}
            write_json(directory / "raw.json", raw)
            write_json(directory / "normalized.json", {
                "issues": [issue.model_dump(mode="json") for issue in normalized.issues],
                "edges": [edge.model_dump(mode="json") for edge in normalized.edges],
                "texts": normalized.texts, "diagnostics": normalized.diagnostics,
            })
            phase = "graph"
            graph_store.write_snapshot(scope, normalized.issues, normalized.edges)
            graph_store.verify_snapshot(scope, normalized.issues, normalized.edges)
            phase = "text"
            index = SnapshotTextIndex(directory)
            chunks = index.write(scope, normalized.issues, normalized.texts, embed)
            index.verify(chunks)
            write_json(directory / "chunks.json", chunks)
            manifest = SnapshotManifest(
                scope=scope, account_scope=capture.account_scope,
                capture_started_at=capture.started_at, capture_finished_at=capture.finished_at,
                source_checksum=hashlib.sha256(json.dumps(capture.issues, sort_keys=True).encode()).hexdigest(),
                normalization_version="jira-graph-v1",
                index_versions={"graph": "neo4j-5.26.12/schema-1",
                                "chroma": f"chromadb-{version('chromadb')}/text-embedding-3-small",
                                "fts": "sqlite-fts5/schema-1"},
                stages={"graph": "ready", "chroma": "ready", "fts": "ready"},
                diagnostics=[*normalized.diagnostics, "capture_is_not_a_jira_transactional_snapshot"],
            )
            phase = "publish"
            write_json(directory / "build-report.json", {
                "snapshot_id": scope.snapshot_id, "project": project_key,
                "scope": "configured_account_visible_project", "issues": len(normalized.issues),
                "issue_types": dict(Counter(issue.issue_type for issue in normalized.issues)),
                "edges": len(normalized.edges),
                "relation_types": dict(Counter(edge.relation_type for edge in normalized.edges)),
                "chunks": len(chunks), "embedding_tokens": getattr(embed, "total_tokens", None),
                "build_seconds_before_publish": round(time.monotonic() - started, 2),
                "capture_seconds": (capture.finished_at - capture.started_at).total_seconds(),
                "captured_at": manifest.capture_finished_at.isoformat(), "diagnostics": manifest.diagnostics,
            })
            repository.publish(manifest)
            return manifest
        except Exception:
            # Only safe phase information; exception messages may contain upstream credentials/content.
            write_json(directory / "failed.json", {"scope": scope.model_dump(), "phase": phase})
            raise
