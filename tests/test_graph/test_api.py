from datetime import datetime, timezone
from importlib import import_module

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.graph.models import Edge, GraphEvidenceResult, GraphPath, GraphScope, Issue, SnapshotManifest
from app.graph.snapshots import SnapshotRepository, write_json
from app.main import create_app


def api_module():
    try:
        return import_module("app.api.graph")
    except ModuleNotFoundError:
        pytest.fail("Graph structural API is not implemented")


def test_enabled_app_registers_structural_routes(monkeypatch):
    monkeypatch.setattr(settings, "graph_enabled", True)
    paths = create_app().openapi()["paths"]
    assert "/graph/projects/{project_key}/overview" in paths


class Store:
    def __init__(self, scope, issues, edges):
        self.scope, self.issues, self.edges = scope, issues, edges
        self.calls = []

    def list_issues(self, scope):
        self.calls.append(scope)
        return self.issues

    def list_edges(self, scope):
        self.calls.append(scope)
        return self.edges

    def traverse(self, request):
        self.calls.append(request.scope)
        bug, story = self.issues[2], self.issues[1]
        if request.unresolved_pair:
            return GraphEvidenceResult(scope=request.scope, nodes=[bug], seeds=[bug.issue_id])
        return GraphEvidenceResult(scope=request.scope, nodes=[bug, story], seeds=[bug.issue_id],
                                   paths=[GraphPath(issue_ids=[bug.issue_id, story.issue_id], edges=[self.edges[-1]])])

    def close(self):
        pass


@pytest.fixture
def setup(tmp_path, monkeypatch):
    module = api_module()
    from app.graph.service import GraphService
    scope = GraphScope(site_id="https://example.test", project_key="AIPLAT", snapshot_id="one")
    now = datetime.now(timezone.utc)
    issues = [Issue(scope=scope, issue_id=str(i), key=f"AIPLAT-{i}", issue_type=kind,
                    title=kind, status="Done", status_category="done", source_url=f"https://example.test/browse/AIPLAT-{i}",
                    updated_at=now, document_id=f"doc-{i}") for i, kind in [(1, "Epic"), (2, "Story"), (3, "Bug")]]
    edges = [Edge(scope=scope, source_issue_id=str(i), target_issue_id="1", relation_type="CHILD_OF",
                  origin_issue_id=str(i), source_field="parent", source_type="parent", source_direction="parent") for i in [2, 3]]
    edges.append(Edge(scope=scope, source_issue_id="3", target_issue_id="2", relation_type="BLOCKS",
                      origin_issue_id="3", source_field="issuelinks", source_type="Blocks", source_direction="outward", link_id="42"))
    repo = SnapshotRepository(tmp_path)
    directory = repo.reserve(scope)
    write_json(directory / "build-report.json", {"issues": 3, "edges": 3})
    repo.publish(SnapshotManifest(scope=scope, account_scope="hash", capture_started_at=now, capture_finished_at=now,
                                  source_checksum="hash", normalization_version="1",
                                  index_versions={k: "1" for k in ["graph", "chroma", "fts"]},
                                  stages={k: "ready" for k in ["graph", "chroma", "fts"]}))
    store = Store(scope, issues, edges)
    service = GraphService(repo, lambda: store, scope.site_id, ("AIPLAT",))
    monkeypatch.setattr(settings, "graph_enabled", True)
    app = create_app()
    app.dependency_overrides[module.get_graph_service] = lambda: service
    with TestClient(app) as client:
        yield client, store, repo, service


def test_overview_has_full_counts_and_snapshot_metadata(setup):
    client, *_ = setup
    response = client.get("/graph/projects/AIPLAT/overview")
    assert response.status_code == 200
    body = response.json()
    assert body["snapshot_id"] == "one"
    assert body["scope"]["project_key"] == "AIPLAT"
    assert body["completeness"] == "visible_project"
    assert body["data"]["counts"]["total"] == 3
    assert body["data"]["epics"][0]["children_counts"]["total"] == 2


def test_epic_pagination_does_not_change_totals(setup):
    client, *_ = setup
    response = client.get("/graph/projects/AIPLAT/epics/AIPLAT-1?limit=1")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["total"] == data["counts"]["total"] == 2
    assert len(data["children"]) == 1
    assert data["has_more"] is True


def test_dependency_status_and_provenance_returned(setup):
    client, *_ = setup
    response = client.get("/graph/projects/AIPLAT/issues/AIPLAT-3/dependencies")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["paths"][0]["edges"][0]["link_id"] == "42"
    assert all(node["status_category"] == "done" for node in data["nodes"])
    filtered = client.get("/graph/projects/AIPLAT/issues/AIPLAT-3/dependencies?unresolved_pair=true")
    assert filtered.status_code == 200
    assert filtered.json()["data"]["paths"] == []


@pytest.mark.parametrize("url", [
    "/graph/projects/OTHER/overview", "/graph/projects/AIPLAT/epics/AIPLAT-99",
    "/graph/projects/AIPLAT/epics/AIPLAT-2", "/graph/projects/AIPLAT/issues/AIPLAT-99/dependencies",
    "/graph/projects/AIPLAT/overview?snapshot_id=missing",
])
def test_unknown_scopes_and_items_are_404(setup, url):
    assert setup[0].get(url).status_code == 404


@pytest.mark.parametrize("query", ["hops=0", "hops=3", "limit=101", "limit=0", "direction=arbitrary", "relation_type=EXECUTE"])
def test_invalid_traversal_input_is_422(setup, query):
    client, store, *_ = setup
    assert client.get("/graph/projects/AIPLAT/issues/AIPLAT-3/dependencies?" + query).status_code == 422
    assert store.calls == []


def test_graph_failure_is_safe_503(setup):
    client, store, *_ = setup

    def fail(_):
        raise RuntimeError("upstream secret detail")

    store.list_issues = fail
    response = client.get("/graph/projects/AIPLAT/overview")
    assert response.status_code == 503
    assert "upstream secret" not in response.text


def test_missing_active_is_unavailable_not_empty_success(setup):
    client, _, repo, _ = setup
    next(repo.root.rglob("active.json")).unlink()
    assert client.get("/graph/projects/AIPLAT/overview").status_code == 503


def test_missing_graph_records_are_unavailable_not_zero_counts(setup):
    client, store, *_ = setup
    store.issues = []
    assert client.get("/graph/projects/AIPLAT/overview").status_code == 503


def test_request_resolves_snapshot_once(setup, monkeypatch):
    client, store, repo, _ = setup
    resolve = repo.resolve
    resolutions = []

    def pinned(*args):
        resolutions.append(args)
        return resolve(*args)

    monkeypatch.setattr(repo, "resolve", pinned)
    assert client.get("/graph/projects/AIPLAT/overview").status_code == 200
    assert len(resolutions) == 1
    assert all(scope.snapshot_id == "one" for scope in store.calls)


def test_graph_routes_keep_legacy_contracts(setup):
    paths = setup[0].get("/openapi.json").json()["paths"]
    assert "/graph/projects/{project_key}/overview" in paths
    assert "/graph/retrieve" in paths
    assert "/graph/query" in paths
    assert "/query" in paths and "/retrieve" in paths
