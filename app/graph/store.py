"""Neo4j staging snapshots; publication is owned by the snapshot coordinator."""

from collections import Counter
from collections.abc import Sequence

from app.graph.models import (Edge, GraphEvidenceResult, GraphPath, GraphScope, Issue,
                              TraversalRequest)

_NODE = "{site_id: $site_id, project_key: $project_key, snapshot_id: $snapshot_id}"
_PATTERNS = {"outbound": "-[r]->", "inbound": "<-[r]-", "both": "-[r]-"}
_PATH_CAP = 1000
# Relationship identifiers are code constants, never caller-supplied Cypher.
_EDGE_WRITES = {
    kind: (
        "UNWIND $rows AS row "
        f"MATCH (a:Issue {_NODE}), (b:Issue {_NODE}) "
        "WHERE a.issue_id = row.source_issue_id AND b.issue_id = row.target_issue_id "
        f"CREATE (a)-[r:{kind}]->(b) SET r = row"
    )
    for kind in ("CHILD_OF", "BLOCKS", "RELATED_TO", "UNKNOWN")
}


def _validate(scope: GraphScope, issues: Sequence[Issue], edges: Sequence[Edge]) -> None:
    if any(record.scope != scope for record in (*issues, *edges)):
        raise ValueError("snapshot contains a different scope")
    ids = {issue.issue_id for issue in issues}
    if len(ids) != len(issues) or len({issue.key for issue in issues}) != len(issues):
        raise ValueError("snapshot contains duplicate issue IDs or keys")
    identities = set()
    for edge in edges:
        if not {edge.source_issue_id, edge.target_issue_id, edge.origin_issue_id} <= ids:
            raise ValueError("edge references an issue outside the snapshot")
        if edge.relation_type not in _EDGE_WRITES:
            raise ValueError("unsupported relationship type")
        identity = (edge.source_issue_id, edge.target_issue_id, edge.relation_type,
                    edge.source_type, edge.link_id)
        if identity in identities:
            raise ValueError("snapshot contains duplicate edges")
        identities.add(identity)


def _properties(record: Issue | Edge) -> dict:
    properties = record.model_dump(mode="json", exclude={"scope"})
    properties.update(record.scope.model_dump())
    properties["payload"] = record.model_dump_json()
    return properties


class Neo4jGraphStore:
    """Replace a staging scope atomically, preserving all other scopes.

    Callers must serialize writes to the same staging scope. Published snapshots
    are immutable by coordinator convention; this adapter does not activate them.
    """

    def __init__(self, driver, database: str = "neo4j"):
        self.driver = driver
        self.database = database

    @classmethod
    def connect(cls, uri: str, user: str, password: str, database: str = "neo4j"):
        from neo4j import GraphDatabase

        return cls(GraphDatabase.driver(uri, auth=(user, password), connection_timeout=10,
                                        max_transaction_retry_time=0), database)

    def write_snapshot(self, scope: GraphScope, issues: Sequence[Issue], edges: Sequence[Edge]) -> None:
        issues, edges = list(issues), list(edges)
        _validate(scope, issues, edges)
        params = scope.model_dump()

        def replace(tx):
            tx.run(f"MATCH (n:Issue {_NODE}) DETACH DELETE n", **params).consume()
            tx.run("UNWIND $rows AS row CREATE (n:Issue) SET n = row",
                   rows=[_properties(issue) for issue in issues]).consume()
            for kind, query in _EDGE_WRITES.items():
                rows = [_properties(edge) for edge in edges if edge.relation_type == kind]
                if rows:
                    tx.run(query, **params, rows=rows).consume()

        with self.driver.session(database=self.database) as session:
            session.execute_write(replace)

    def _read(self, query: str, scope: GraphScope, **params) -> list:
        with self.driver.session(database=self.database) as session:
            return session.execute_read(
                lambda tx: [row["payload"] for row in tx.run(query, **scope.model_dump(), **params)]
            )

    def get_issue(self, scope: GraphScope, issue_id: str) -> Issue | None:
        records = self._read(f"MATCH (n:Issue {_NODE}) WHERE n.issue_id = $issue_id "
                             "RETURN n.payload AS payload", scope, issue_id=issue_id)
        if len(records) > 1:
            raise ValueError("duplicate stored issue identity")
        return Issue.model_validate_json(records[0]) if records else None

    def list_issues(self, scope: GraphScope) -> list[Issue]:
        records = self._read(f"MATCH (n:Issue {_NODE}) RETURN n.payload AS payload "
                             "ORDER BY n.issue_id", scope)
        return [Issue.model_validate_json(record) for record in records]

    def list_edges(self, scope: GraphScope) -> list[Edge]:
        records = self._read(f"MATCH (a:Issue {_NODE})-[r]->(b:Issue {_NODE}) "
                             "WHERE r.site_id = $site_id AND r.project_key = $project_key "
                             "AND r.snapshot_id = $snapshot_id RETURN r.payload AS payload",
                             scope)
        return [Edge.model_validate_json(record) for record in records]

    def _neighbors(self, request: TraversalRequest, issue_id: str) -> list[tuple[Issue, Edge]]:
        """A scoped, ordered one-hop fetch with one overflow row, never an all-graph scan."""
        query = (
            f"MATCH (a:Issue {_NODE}){_PATTERNS[request.direction]}(b:Issue {_NODE}) "
            "WHERE a.issue_id = $issue_id AND type(r) IN $relation_types "
            "AND r.site_id = $site_id AND r.project_key = $project_key "
            "AND r.snapshot_id = $snapshot_id "
            "AND (NOT $unresolved_pair OR (type(r) = 'BLOCKS' "
            "AND toLower(a.status_category) IN ['new', 'indeterminate'] "
            "AND toLower(b.status_category) IN ['new', 'indeterminate'])) "
            "RETURN {node: b.payload, edge: r.payload} AS payload "
            "ORDER BY b.issue_id, r.payload LIMIT $row_limit"
        )
        records = self._read(query, request.scope, issue_id=issue_id,
                             relation_types=list(request.relation_types),
                             unresolved_pair=request.unresolved_pair, row_limit=_PATH_CAP + 1)
        return [(Issue.model_validate_json(row["node"]), Edge.model_validate_json(row["edge"]))
                for row in records]

    def traverse(self, request: TraversalRequest) -> GraphEvidenceResult:
        """Return bounded path evidence, including the seed in the node limit.

        At most 100 nodes and 1000 paths (thus 2000 edge occurrences) are returned.
        Each node's adjacency is fetched once, with a 1001-row overflow probe.
        Cyclic closing edges are evidence but a repeated node is never re-expanded
        on that path. Alternative paths to a shared node are preserved.
        """
        request = TraversalRequest.model_validate(request.model_dump())
        seed = self.get_issue(request.scope, request.issue_id)
        if seed is None:
            raise LookupError("issue not found in selected snapshot")
        nodes = {seed.issue_id: seed}
        paths: list[GraphPath] = []
        frontier = [GraphPath(issue_ids=[seed.issue_id])]
        cache = {}
        diagnostics = set()
        for _ in range(request.hops):
            following = []
            for path in frontier:
                current_id = path.issue_ids[-1]
                if current_id not in cache:
                    rows = self._neighbors(request, current_id)
                    if len(rows) > _PATH_CAP:
                        diagnostics.add("adjacency limit reached (1000); results are partial")
                    cache[current_id] = sorted(rows[:_PATH_CAP],
                                               key=lambda row: (row[0].issue_id, row[1].model_dump_json()))
                for neighbor, relation in cache[current_id]:
                    if neighbor.scope != request.scope or relation.scope != request.scope:
                        raise ValueError("traversal evidence has a different scope")
                    if relation.relation_type not in request.relation_types:
                        continue
                    if request.unresolved_pair and (
                        relation.relation_type != "BLOCKS" or any(
                            node.status_category.casefold() not in {"new", "indeterminate"}
                            for node in (nodes[current_id], neighbor)
                        )
                    ):
                        continue
                    if neighbor.issue_id not in nodes and len(nodes) >= request.limit:
                        diagnostics.add("node limit reached; results are partial")
                        continue
                    if len(paths) >= _PATH_CAP:
                        diagnostics.add("path limit reached (1000); results are partial")
                        break
                    expanded = GraphPath(issue_ids=[*path.issue_ids, neighbor.issue_id],
                                         edges=[*path.edges, relation])
                    nodes[neighbor.issue_id] = neighbor
                    paths.append(expanded)
                    if neighbor.issue_id not in path.issue_ids:
                        following.append(expanded)
                if len(paths) >= _PATH_CAP and diagnostics:
                    break
            frontier = following
            if not frontier or (len(paths) >= _PATH_CAP and diagnostics):
                break
        return GraphEvidenceResult(
            scope=request.scope, nodes=[nodes[key] for key in sorted(nodes)],
            seeds=[seed.issue_id], paths=paths, truncated=bool(diagnostics),
            coverage="partial" if diagnostics else "complete", diagnostics=sorted(diagnostics),
        )

    def verify_snapshot(self, scope: GraphScope, issues: Sequence[Issue], edges: Sequence[Edge]) -> None:
        _validate(scope, issues, edges)
        expected = (Counter(item.model_dump_json() for item in issues),
                    Counter(item.model_dump_json() for item in edges))
        actual = (Counter(item.model_dump_json() for item in self.list_issues(scope)),
                  Counter(item.model_dump_json() for item in self.list_edges(scope)))
        if actual != expected:
            raise ValueError("stored snapshot mismatch")

    def close(self) -> None:
        self.driver.close()
