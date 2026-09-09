"""Pure aggregation of complete, scoped Jira graph snapshots."""

import re
from collections import Counter

from app.graph.models import Contract, Edge, GraphScope, Issue


class Counts(Contract):
    total: int
    by_type: dict[str, int]
    by_status: dict[str, int]
    by_status_category: dict[str, int]


class EpicSummary(Contract):
    issue: Issue
    children_counts: Counts


class ProjectGraph(Contract):
    nodes: list[Issue]
    edges: list[Edge]
    total_nodes: int
    total_edges: int
    truncated: bool


class OverviewData(Contract):
    graph: ProjectGraph
    counts: Counts
    epics: list[EpicSummary]
    unassigned: list[Issue]
    other: list[Issue]


class EpicDetailData(Contract):
    epic: Issue
    counts: Counts
    children: list[Issue]
    offset: int
    limit: int
    total: int
    has_more: bool


def _validate(scope: GraphScope, issues: list[Issue], edges: list[Edge]) -> None:
    if any(record.scope != scope for record in (*issues, *edges)):
        raise ValueError("snapshot contains a different scope")
    ids = {issue.issue_id for issue in issues}
    if len(ids) != len(issues) or len({issue.key for issue in issues}) != len(issues):
        raise ValueError("snapshot contains duplicate issue IDs or keys")
    identities = set()
    for edge in edges:
        if not {edge.source_issue_id, edge.target_issue_id, edge.origin_issue_id} <= ids:
            raise ValueError("edge references an issue outside the snapshot")
        identity = (edge.source_issue_id, edge.target_issue_id, edge.relation_type,
                    edge.source_type, edge.link_id)
        if identity in identities:
            raise ValueError("snapshot contains duplicate edges")
        identities.add(identity)


def _order(issue: Issue) -> tuple:
    return tuple((1, int(part)) if part.isdigit() else (0, part.casefold())
                 for part in re.split(r"(\d+)", issue.key))


def _counts(issues: list[Issue]) -> Counts:
    return Counts(total=len(issues),
                  by_type=dict(Counter(issue.issue_type for issue in issues)),
                  by_status=dict(Counter(issue.status for issue in issues)),
                  by_status_category=dict(Counter(issue.status_category for issue in issues)))


def _children(issues: list[Issue], edges: list[Edge], parent_id: str) -> list[Issue]:
    ids = {edge.source_issue_id for edge in edges
           if edge.relation_type == "CHILD_OF" and edge.target_issue_id == parent_id}
    return sorted((issue for issue in issues if issue.issue_id in ids), key=_order)


def overview(scope: GraphScope, issues: list[Issue], edges: list[Edge]) -> OverviewData:
    """Count all issues; keep direct child counts separate from each Epic's status.

    Other types are intentionally also present in unassigned when they lack a
    CHILD_OF parent. The two categories are not mutually exclusive totals.
    """
    _validate(scope, issues, edges)
    ordered = sorted(issues, key=_order)
    assigned = {edge.source_issue_id for edge in edges if edge.relation_type == "CHILD_OF"}
    epics = [EpicSummary(issue=issue, children_counts=_counts(_children(issues, edges, issue.issue_id)))
             for issue in ordered if issue.issue_type.casefold() == "epic"]
    # The drawing is bounded independently from full-snapshot statistics.
    map_nodes = sorted(ordered, key=lambda node: (node.issue_type.casefold() != "epic", _order(node)))[:100]
    visible = {node.issue_id for node in map_nodes}
    map_edges = [edge for edge in edges
                 if {edge.source_issue_id, edge.target_issue_id, edge.origin_issue_id} <= visible][:500]
    return OverviewData(
        graph=ProjectGraph(nodes=map_nodes, edges=map_edges, total_nodes=len(issues), total_edges=len(edges),
                           truncated=len(map_nodes) != len(issues) or len(map_edges) != len(edges)),
        counts=_counts(issues), epics=epics,
        unassigned=[issue for issue in ordered
                    if issue.issue_type.casefold() != "epic" and issue.issue_id not in assigned],
        other=[issue for issue in ordered if issue.issue_type.casefold() not in {"epic", "story", "bug"}],
    )


def epic_detail(scope: GraphScope, issues: list[Issue], edges: list[Edge], epic_key: str,
                offset: int = 0, limit: int = 50) -> EpicDetailData:
    """Return direct children with full counts independent of the requested page."""
    if offset < 0 or not 1 <= limit <= 100:
        raise ValueError("offset must be nonnegative and limit between 1 and 100")
    _validate(scope, issues, edges)
    epic = next((issue for issue in issues
                 if issue.key == epic_key and issue.issue_type.casefold() == "epic"), None)
    if epic is None:
        raise LookupError("Epic not found in snapshot")
    children = _children(issues, edges, epic.issue_id)
    return EpicDetailData(epic=epic, counts=_counts(children), children=children[offset:offset + limit],
                          offset=offset, limit=limit, total=len(children),
                          has_more=offset + limit < len(children))
