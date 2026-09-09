from datetime import datetime, timezone
from importlib import import_module

import pytest

from app.graph.jira import CapturedProject
from app.graph.snapshots import SnapshotRepository


def capture():
    now = datetime.now(timezone.utc)
    return CapturedProject([
        {"id": "1", "key": "AIPLAT-1", "fields": {
            "project": {"key": "AIPLAT"}, "summary": "Scope", "description": "Knowledge retrieval",
            "issuetype": {"name": "Epic"}, "status": {"name": "Done", "statusCategory": {"key": "done"}},
            "updated": now.isoformat(), "issuelinks": [],
        }},
    ], now, now, "account-hash", [])


def builder():
    try:
        return import_module("app.graph.build").build_snapshot
    except ModuleNotFoundError:
        pytest.fail("Snapshot build coordinator is not implemented")


class Store:
    def __init__(self, fail=False):
        self.fail = fail
        self.verified = False

    def write_snapshot(self, scope, issues, edges):
        self.data = (scope, issues, edges)

    def verify_snapshot(self, scope, issues, edges):
        if self.fail:
            raise ValueError("graph verification failed")
        assert (scope, issues, edges) == self.data
        self.verified = True


def test_build_publishes_only_after_verifying_all_indexes(tmp_path):
    repo = SnapshotRepository(tmp_path)
    store = Store()
    result = builder()(repo, store, capture(), "https://example.test", "AIPLAT",
                       lambda texts: [[0.1, 0.9] for _ in texts])
    assert store.verified
    active = repo.resolve("https://example.test", "AIPLAT")
    assert active == result
    assert active.ready
    assert (repo.directory(active.scope) / "raw.json").exists()
    assert (repo.directory(active.scope) / "normalized.json").exists()
    assert (repo.directory(active.scope) / "build-report.json").exists()


def test_failed_rebuild_does_not_replace_active(tmp_path):
    repo = SnapshotRepository(tmp_path)
    first = builder()(repo, Store(), capture(), "https://example.test", "AIPLAT",
                      lambda texts: [[0.1, 0.9] for _ in texts])
    with pytest.raises(ValueError, match="embedding"):
        builder()(repo, Store(), capture(), "https://example.test", "AIPLAT", lambda _: [])
    assert repo.resolve("https://example.test", "AIPLAT") == first
    assert len(list(tmp_path.rglob("failed.json"))) == 1


def test_graph_verification_failure_prevents_embedding_and_activation(tmp_path):
    repo = SnapshotRepository(tmp_path)
    with pytest.raises(ValueError, match="graph verification"):
        builder()(repo, Store(fail=True), capture(), "https://example.test", "AIPLAT",
                  lambda _: pytest.fail("must not embed after graph failure"))
    with pytest.raises(FileNotFoundError):
        repo.resolve("https://example.test", "AIPLAT")
