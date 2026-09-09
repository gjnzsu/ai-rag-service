"""Bounded traversal behavior and real Neo4j direction/scope verification."""

import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.graph.models import TraversalRequest
from app.graph.store import Neo4jGraphStore
from tests.test_graph.test_store import edge, issue, scope


def graph(monkeypatch, pairs):
    value = scope()
    nodes = {str(i): issue(value, str(i)).model_copy(update={"status_category": "indeterminate"})
             for i in range(1, 110)}
    edges = [edge(value).model_copy(update={"source_issue_id": a, "target_issue_id": b,
                                          "link_id": f"{a}-{b}"}) for a, b in pairs]
    store = Neo4jGraphStore(None)
    monkeypatch.setattr(store, "get_issue", lambda _, identity: nodes.get(identity))

    def neighbors(request, identity):
        result = []
        for relation in edges:
            a, b = relation.source_issue_id, relation.target_issue_id
            if a == identity and request.direction in ("outbound", "both"):
                result.append((nodes[b], relation))
            elif b == identity and request.direction in ("inbound", "both"):
                result.append((nodes[a], relation))
        return result

    monkeypatch.setattr(store, "_neighbors", neighbors, raising=False)
    return store, nodes


@pytest.mark.parametrize("direction,expected", [("outbound", ["1", "2"]),
                                               ("inbound", ["1", "3"]),
                                               ("both", ["1", "2", "3"])])
def test_direction_and_canonical_provenance(monkeypatch, direction, expected):
    store, _ = graph(monkeypatch, [("1", "2"), ("3", "1")])
    result = store.traverse(TraversalRequest(scope=scope(), issue_id="1", direction=direction))
    assert [n.issue_id for n in result.nodes] == expected
    assert all(p.edges[0].source_field == "issuelinks" for p in result.paths)
    if direction == "inbound":
        assert result.paths[0].edges[0].source_issue_id == "3"


def test_two_hops_cycle_safe_and_retains_alternative_paths(monkeypatch):
    store, _ = graph(monkeypatch, [("1", "2"), ("1", "3"), ("2", "4"),
                                 ("3", "4"), ("2", "1")])
    result = store.traverse(TraversalRequest(scope=scope(), issue_id="1", hops=2,
                                           direction="outbound"))
    paths = [p.issue_ids for p in result.paths]
    assert ["1", "2", "4"] in paths and ["1", "3", "4"] in paths
    assert ["1", "2", "1"] in paths
    assert len(result.nodes) == 4
    assert not result.truncated


def test_node_limit_reports_partial_without_dangling_paths(monkeypatch):
    store, _ = graph(monkeypatch, [("1", str(i)) for i in range(2, 109)])
    result = store.traverse(TraversalRequest(scope=scope(), issue_id="1", limit=3))
    assert len(result.nodes) == 3
    assert result.truncated and result.coverage == "partial" and result.diagnostics
    assert all(set(p.issue_ids) <= {n.issue_id for n in result.nodes} for p in result.paths)


@pytest.mark.parametrize("category,count", [("Done", 0), ("unknown", 0), ("indeterminate", 1)])
def test_unresolved_requires_known_non_done_status(monkeypatch, category, count):
    store, nodes = graph(monkeypatch, [("1", "2")])
    nodes["2"] = nodes["2"].model_copy(update={"status_category": category})
    result = store.traverse(TraversalRequest(scope=scope(), issue_id="1", unresolved_pair=True))
    assert len(result.paths) == count


def test_unknown_origin_and_mismatched_evidence_fail(monkeypatch):
    store, nodes = graph(monkeypatch, [("1", "2")])
    with pytest.raises(LookupError):
        store.traverse(TraversalRequest(scope=scope(), issue_id="missing"))
    nodes["2"] = nodes["2"].model_copy(update={"scope": scope("other")})
    with pytest.raises(ValueError, match="scope"):
        store.traverse(TraversalRequest(scope=scope(), issue_id="1"))


def test_unresolved_never_promotes_other_relation_types(monkeypatch):
    store, nodes = graph(monkeypatch, [])
    relation = edge(scope(), "RELATED_TO")
    monkeypatch.setattr(store, "_neighbors", lambda *_: [(nodes["2"], relation)])
    result = store.traverse(TraversalRequest(scope=scope(), issue_id="1", unresolved_pair=True,
                                           relation_types=("RELATED_TO",)))
    assert not result.paths


def test_overflow_is_explicit_and_each_frontier_fetched_once(monkeypatch):
    store, nodes = graph(monkeypatch, [])
    calls = []

    def neighbors(_, identity):
        calls.append(identity)
        if identity != "1":
            return []
        return [(nodes["2"], edge(scope()).model_copy(update={"link_id": str(i)}))
                for i in range(1001)]

    monkeypatch.setattr(store, "_neighbors", neighbors)
    result = store.traverse(TraversalRequest(scope=scope(), issue_id="1", hops=2))
    assert len(result.paths) == 1000 and result.truncated
    assert len(calls) == len(set(calls))


@pytest.fixture
def traversal_store():
    if os.environ.get("GRAPH_INTEGRATION") != "1":
        pytest.skip("set GRAPH_INTEGRATION=1 for local Neo4j integration")
    from dotenv import dotenv_values
    config = dotenv_values(Path(".env.graph"))
    password = config.get("NEO4J_PASSWORD") or config.get("GRAPH_NEO4J_PASSWORD")
    assert password, "local graph password is not configured"
    store = Neo4jGraphStore.connect("bolt://localhost:7687", "neo4j", password)
    yield store
    store.close()


def test_real_direction_multihop_and_scope_isolation(traversal_store):
    unique = f"traversal-{uuid4().hex}"
    scopes = [scope(unique), scope(unique + "-other"), scope(unique, project="OTHER"),
              scope(unique, site="other.invalid")]
    try:
        for value in scopes:
            nodes = [issue(value, str(i)).model_copy(update={"status_category": "indeterminate"})
                     for i in range(1, 5)]
            edges = [edge(value).model_copy(update={"source_issue_id": a, "target_issue_id": b,
                                                   "link_id": f"{a}-{b}"})
                     for a, b in [("1", "2"), ("2", "3"), ("3", "1")]]
            traversal_store.write_snapshot(value, nodes, edges)
        value = scopes[0]
        request = TraversalRequest(scope=value, issue_id="1", direction="outbound", hops=2)
        result = traversal_store.traverse(request)
        assert [p.issue_ids for p in result.paths] == [["1", "2"], ["1", "2", "3"]]
        inbound = traversal_store.traverse(request.model_copy(update={"direction": "inbound"}))
        assert [p.issue_ids for p in inbound.paths] == [["1", "3"], ["1", "3", "2"]]
        assert all(node.scope == value for node in result.nodes)
        assert len(traversal_store.traverse(request.model_copy(update={"unresolved_pair": True})).paths) == 2
        # The selected-scope relationship must not allow traversal to a foreign endpoint.
        # Likewise a foreign-scope edge between selected-scope nodes is excluded.
        with traversal_store.driver.session(database="neo4j") as session:
            session.run(
                "MATCH (a:Issue {site_id:$site_id, project_key:$project_key, snapshot_id:$snapshot_id, issue_id:'1'}), "
                "(b:Issue {site_id:$site_id, project_key:$project_key, snapshot_id:$other, issue_id:'4'}) "
                "CREATE (a)-[r:BLOCKS]->(b) SET r.site_id=$site_id, r.project_key=$project_key, "
                "r.snapshot_id=$snapshot_id, r.payload='invalid'", **value.model_dump(),
                other=scopes[1].snapshot_id).consume()
            session.run(
                "MATCH (a:Issue {site_id:$site_id, project_key:$project_key, snapshot_id:$snapshot_id, issue_id:'1'}), "
                "(b:Issue {site_id:$site_id, project_key:$project_key, snapshot_id:$snapshot_id, issue_id:'4'}) "
                "CREATE (a)-[r:BLOCKS]->(b) SET r.site_id=$site_id, r.project_key=$project_key, "
                "r.snapshot_id=$other, r.payload='invalid'", **value.model_dump(),
                other=scopes[1].snapshot_id).consume()
        assert traversal_store.traverse(request) == result
        assert traversal_store.traverse(request.model_copy(update={"limit": 1})).truncated
        selected = traversal_store.list_issues(value)
        selected[1] = selected[1].model_copy(update={"status_category": "Done"})
        # Replace exact test scope, removing deliberately malformed edges.
        traversal_store.write_snapshot(value, selected, [edge(value)])
        assert not traversal_store.traverse(request.model_copy(update={"unresolved_pair": True})).paths
    finally:
        for value in scopes:
            traversal_store.write_snapshot(value, [], [])
