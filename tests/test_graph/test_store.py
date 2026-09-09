"""Contract tests and opt-in local Neo4j snapshot integration."""

import importlib.util
import os
from pathlib import Path
from uuid import uuid4

import pytest

from app.graph.models import Edge, GraphScope, Issue


def store_class():
    assert importlib.util.find_spec("app.graph.store"), "snapshot store is not implemented"
    from app.graph.store import Neo4jGraphStore
    return Neo4jGraphStore


def scope(snapshot="one", project="TEST", site="test.invalid"):
    return GraphScope(site_id=site, project_key=project, snapshot_id=snapshot)


def issue(value, identity):
    return Issue(scope=value, issue_id=identity, key=f"TEST-{identity}",
                 issue_type="Story", title="A 'parameterized' title", status="Done",
                 status_category="done", source_url=f"https://test.invalid/{identity}",
                 updated_at="2026-09-08T00:00:00Z", document_id=f"doc-{identity}")


def edge(value, relation="BLOCKS"):
    return Edge(scope=value, source_issue_id="1", target_issue_id="2",
                relation_type=relation, origin_issue_id="1", source_field="issuelinks",
                source_type="Blocks", source_direction="outward", link_id="123")


class NoDatabase:
    def session(self, **kwargs):
        pytest.fail("invalid input must be rejected before database access")


@pytest.mark.parametrize("case", ["scope", "duplicate", "key", "endpoint", "origin", "edge_scope", "edge_duplicate"])
def test_invalid_snapshot_rejected_before_database(case):
    value = scope()
    nodes = [issue(value, "1"), issue(value, "2")]
    edges = [edge(value)]
    if case == "scope":
        nodes[0] = issue(scope("other"), "1")
    elif case == "duplicate":
        nodes.append(nodes[0])
    elif case == "key":
        nodes[1] = nodes[1].model_copy(update={"key": nodes[0].key})
    elif case == "endpoint":
        nodes.pop()
    elif case == "origin":
        edges[0] = edges[0].model_copy(update={"origin_issue_id": "missing"})
    elif case == "edge_scope":
        edges[0] = edge(scope("other"))
    else:
        edges.append(edges[0])
    with pytest.raises(ValueError):
        store_class()(NoDatabase()).write_snapshot(value, nodes, edges)


def test_verify_rejects_extra_and_missing_records():
    value = scope()
    nodes = [issue(value, "1"), issue(value, "2")]
    store = store_class()(NoDatabase())
    store.list_issues = lambda _: nodes
    store.list_edges = lambda _: [edge(value)]
    store.verify_snapshot(value, nodes, [edge(value)])
    with pytest.raises(ValueError, match="mismatch"):
        store.verify_snapshot(value, nodes, [])
    store.list_edges = lambda _: []
    with pytest.raises(ValueError, match="mismatch"):
        store.verify_snapshot(value, nodes, [edge(value)])
    store.list_edges = lambda _: [edge(value)]
    store.list_issues = lambda _: nodes[:1]
    with pytest.raises(ValueError, match="mismatch"):
        store.verify_snapshot(value, nodes, [edge(value)])


def test_different_unknown_source_types_are_distinct():
    value = scope()
    nodes = [issue(value, "1"), issue(value, "2")]
    edges = [edge(value, "UNKNOWN").model_copy(update={"source_type": kind, "link_id": None})
             for kind in ("Clones", "Duplicates")]
    store = store_class()(NoDatabase())
    store.list_issues = lambda _: nodes
    store.list_edges = lambda _: edges
    store.verify_snapshot(value, nodes, edges)


def test_connect_has_bounded_connection_and_retry_time(monkeypatch):
    from neo4j import GraphDatabase

    calls = []
    marker = object()

    def driver(uri, **kwargs):
        calls.append((uri, kwargs))
        return marker

    monkeypatch.setattr(GraphDatabase, "driver", driver)
    store = store_class().connect("bolt://example.invalid:7687", "user", "test-only")
    assert store.driver is marker
    assert calls == [("bolt://example.invalid:7687", {
        "auth": ("user", "test-only"), "connection_timeout": 10,
        "max_transaction_retry_time": 0,
    })]


@pytest.fixture
def local_store():
    if os.environ.get("GRAPH_INTEGRATION") != "1":
        pytest.skip("set GRAPH_INTEGRATION=1 for local Neo4j integration")
    from dotenv import dotenv_values
    config = dotenv_values(Path(".env.graph"))
    password = config.get("NEO4J_PASSWORD") or config.get("GRAPH_NEO4J_PASSWORD")
    if not password:
        pytest.fail("local graph password is not configured")
    store = store_class().connect("bolt://localhost:7687", "neo4j", password)
    yield store
    store.close()


def test_local_round_trip_replacement_and_scope_isolation(local_store):
    unique = f"store-test-{uuid4().hex}"
    scopes = [scope(unique), scope(unique + "-other"), scope(unique, project="OTHER"),
              scope(unique, site="other.invalid")]
    try:
        for value in scopes:
            nodes = [issue(value, "1"), issue(value, "2")]
            edges = [edge(value, relation) for relation in ("CHILD_OF", "BLOCKS", "RELATED_TO", "UNKNOWN")]
            local_store.write_snapshot(value, nodes, edges)
            local_store.verify_snapshot(value, nodes, edges)
            assert local_store.get_issue(value, "1") == nodes[0]
            assert local_store.get_issue(value, "missing") is None
        value = scopes[0]
        local_store.write_snapshot(value, [issue(value, "1")], [])
        local_store.verify_snapshot(value, [issue(value, "1")], [])
        for other in scopes[1:]:
            assert len(local_store.list_issues(other)) == 2
            assert len(local_store.list_edges(other)) == 4
    finally:
        for value in scopes:
            local_store.write_snapshot(value, [], [])


def test_local_failed_replacement_rolls_back(local_store, monkeypatch):
    from app.graph import store as store_module

    value = scope(f"store-rollback-{uuid4().hex}")
    nodes = [issue(value, "1"), issue(value, "2")]
    edges = [edge(value)]
    try:
        local_store.write_snapshot(value, nodes, edges)
        # Inject a database-side error after delete/create to test real rollback.
        with monkeypatch.context() as patch:
            patch.setitem(store_module._EDGE_WRITES, "BLOCKS", "INVALID CYPHER")
            from neo4j.exceptions import CypherSyntaxError
            with pytest.raises(CypherSyntaxError):
                local_store.write_snapshot(value, nodes, edges)
        local_store.verify_snapshot(value, nodes, edges)
    finally:
        local_store.write_snapshot(value, [], [])
