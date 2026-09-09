"""Storage boundary; no Neo4j imports or raw query strings escape this interface."""

from collections.abc import Sequence
from typing import Protocol

from app.graph.models import Edge, GraphEvidenceResult, GraphScope, Issue, TraversalRequest


class GraphStore(Protocol):
    def write_snapshot(self, scope: GraphScope, issues: Sequence[Issue], edges: Sequence[Edge]) -> None:
        """Write an isolated staging snapshot, without changing the active manifest."""
        ...

    def get_issue(self, scope: GraphScope, issue_id: str) -> Issue | None:
        """Resolve a stable Jira ID inside exactly one scope."""
        ...

    def list_issues(self, scope: GraphScope) -> list[Issue]:
        """Return scoped issues for deterministic full-snapshot aggregation."""
        ...

    def list_edges(self, scope: GraphScope) -> list[Edge]:
        """Return all canonical scoped edges for full-snapshot structural validation."""
        ...

    def verify_snapshot(self, scope: GraphScope, issues: Sequence[Issue], edges: Sequence[Edge]) -> None:
        """Fail when persisted records differ from the staging snapshot."""
        ...

    def traverse(self, request: TraversalRequest) -> GraphEvidenceResult:
        """Return bounded paths with preserved relationship provenance."""
        ...
