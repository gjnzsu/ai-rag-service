"""Local immutable snapshot directories and an atomically replaced active pointer."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from app.graph.models import GraphScope, SnapshotManifest


def _digest(*values: str) -> str:
    # Compact opaque path components keep native Chroma files below Windows path limits.
    # Full identity is checked in manifests; an existing directory is never overwritten.
    return hashlib.sha256(json.dumps(values).encode()).hexdigest()[:24]


def write_json(path: Path, value: dict | list) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class SnapshotRepository:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()

    def _project(self, site_id: str, project_key: str) -> Path:
        return self.root / _digest(site_id, project_key)

    def directory(self, scope: GraphScope) -> Path:
        return self._project(scope.site_id, scope.project_key) / "snapshots" / _digest(scope.snapshot_id)

    def reserve(self, scope: GraphScope) -> Path:
        directory = self.directory(scope)
        directory.mkdir(parents=True, exist_ok=False)
        return directory

    @contextmanager
    def build_lock(self, site_id: str, project_key: str):
        project = self._project(site_id, project_key)
        project.mkdir(parents=True, exist_ok=True)
        path = project / "build.lock"
        # Fail immediately rather than queue an unbounded build behind another writer.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(str(os.getpid()))
        try:
            yield
        finally:
            path.unlink()

    def publish(self, manifest: SnapshotManifest) -> None:
        if not manifest.ready:
            raise ValueError("snapshot must be ready before publication")
        active_path = self._project(manifest.scope.site_id, manifest.scope.project_key) / "active.json"
        if active_path.exists():
            active = self.resolve(manifest.scope.site_id, manifest.scope.project_key)
            if manifest.capture_started_at < active.capture_started_at:
                raise ValueError("older capture cannot replace the active snapshot")
        directory = self.directory(manifest.scope)
        if (directory / "manifest.json").exists():
            raise FileExistsError("snapshot manifest is immutable")
        write_json(directory / "manifest.json", manifest.model_dump(mode="json"))
        write_json(active_path, manifest.scope.model_dump())

    def resolve(self, site_id: str, project_key: str, snapshot_id: str | None = None) -> SnapshotManifest:
        if snapshot_id is None:
            scope = GraphScope.model_validate_json(
                (self._project(site_id, project_key) / "active.json").read_text(encoding="utf-8"),
            )
            if scope.site_id != site_id or scope.project_key != project_key:
                raise ValueError("active pointer scope mismatch")
        else:
            scope = GraphScope(site_id=site_id, project_key=project_key, snapshot_id=snapshot_id)
        manifest = SnapshotManifest.model_validate_json(
            (self.directory(scope) / "manifest.json").read_text(encoding="utf-8"),
        )
        if manifest.scope != scope or not manifest.ready:
            raise ValueError("snapshot scope or readiness mismatch")
        return manifest
