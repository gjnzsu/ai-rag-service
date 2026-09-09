"""Request-level scope pinning and deterministic structural graph operations."""

import json
from collections.abc import Callable

from app.graph.api_models import DependenciesResponse, EpicDetailResponse, OverviewResponse
from app.graph.models import SnapshotManifest, TraversalRequest
from app.graph.snapshots import SnapshotRepository
from app.graph import structure


class GraphNotFound(LookupError):
    pass


class GraphUnavailable(RuntimeError):
    pass


class GraphService:
    def __init__(self, repository: SnapshotRepository, store_provider: Callable,
                 site_id: str, allowed_projects: tuple[str, ...]):
        self.repository = repository
        self.store_provider = store_provider
        self.site_id = site_id.rstrip("/")
        self.allowed_projects = allowed_projects
        self._store = None

    @property
    def store(self):
        if self._store is None:
            self._store = self.store_provider()
        return self._store

    def close(self):
        if self._store is not None:
            try:
                self._store.close()
            except Exception:
                pass  # Cleanup must not replace an already produced response.

    def _resolve(self, project_key: str, snapshot_id: str | None):
        if project_key not in self.allowed_projects:
            raise GraphNotFound("Project not in configured scope")
        if not self.site_id:
            raise GraphUnavailable("Graph site not configured")
        try:
            manifest = self.repository.resolve(self.site_id, project_key, snapshot_id)
        except FileNotFoundError:
            if snapshot_id is not None:
                raise GraphNotFound("Snapshot not found") from None
            raise GraphUnavailable("No active snapshot") from None
        except Exception:
            raise GraphUnavailable("Snapshot unavailable") from None
        return manifest

    def _records(self, manifest):
        try:
            issues = self.store.list_issues(manifest.scope)
            edges = self.store.list_edges(manifest.scope)
            report = json.loads((self.repository.directory(manifest.scope) / "build-report.json").read_text(encoding="utf-8"))
            if len(issues) != report["issues"] or len(edges) != report["edges"]:
                raise ValueError("graph records differ from published build")
            return manifest, issues, edges
        except Exception:
            raise GraphUnavailable("Graph snapshot unavailable") from None

    def _load(self, project_key: str, snapshot_id: str | None):
        return self._records(self._resolve(project_key, snapshot_id))

    @staticmethod
    def _metadata(manifest: SnapshotManifest) -> dict:
        return {"snapshot_id": manifest.scope.snapshot_id, "captured_at": manifest.capture_finished_at,
                "scope": manifest.scope, "completeness": manifest.completeness,
                "diagnostics": manifest.diagnostics}

    def overview(self, project_key: str, snapshot_id: str | None = None) -> OverviewResponse:
        manifest, issues, edges = self._load(project_key, snapshot_id)
        try:
            return OverviewResponse(**self._metadata(manifest), data=structure.overview(manifest.scope, issues, edges))
        except Exception:
            raise GraphUnavailable("Graph structure unavailable") from None

    def epic_detail(self, project_key: str, epic_key: str, snapshot_id: str | None = None,
                    offset: int = 0, limit: int = 50) -> EpicDetailResponse:
        manifest, issues, edges = self._load(project_key, snapshot_id)
        try:
            data = structure.epic_detail(manifest.scope, issues, edges, epic_key, offset, limit)
        except LookupError:
            raise GraphNotFound("Epic not found in snapshot") from None
        except Exception:
            raise GraphUnavailable("Graph structure unavailable") from None
        return EpicDetailResponse(**self._metadata(manifest), data=data)

    def dependencies(self, project_key: str, issue_key: str, snapshot_id: str | None = None,
                     direction: str = "both", hops: int = 1, limit: int = 100,
                     relation_type: str = "BLOCKS", unresolved_pair: bool = False) -> DependenciesResponse:
        manifest, issues, edges = self._load(project_key, snapshot_id)
        try:
            # Validate the same complete graph used to resolve the key, before traversal.
            structure.overview(manifest.scope, issues, edges)
            issue = next((node for node in issues if node.key == issue_key), None)
            if issue is None:
                raise GraphNotFound("Issue not found in snapshot")
            request = TraversalRequest(scope=manifest.scope, issue_id=issue.issue_id, direction=direction,
                                       hops=hops, limit=limit, relation_types=(relation_type,), unresolved_pair=unresolved_pair)
            result = self.store.traverse(request)
            if result.scope != manifest.scope:
                raise ValueError("traversal scope mismatch")
            metadata = self._metadata(manifest)
            metadata["diagnostics"] = [*manifest.diagnostics, *result.diagnostics]
            if result.truncated:
                metadata["completeness"] = "partial"
            return DependenciesResponse(**metadata, data=result)
        except GraphNotFound:
            raise
        except Exception:
            raise GraphUnavailable("Graph traversal unavailable") from None
