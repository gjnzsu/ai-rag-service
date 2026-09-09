from datetime import datetime, timezone
from importlib import import_module

import pytest
from pydantic import ValidationError


def models():
    try:
        return import_module("app.graph.models")
    except ModuleNotFoundError:
        pytest.fail("Graph contracts must be available without a Neo4j driver")


def scope(snapshot="snapshot-1", project="AIPLAT"):
    return models().GraphScope(site_id="site-1", project_key=project, snapshot_id=snapshot)


def issue(issue_id="46", **changes):
    values = dict(scope=scope(), issue_id=issue_id, key=f"AIPLAT-{issue_id}",
                  issue_type="Bug", title="Example", status="Done",
                  status_category="Done", source_url="https://example.test/browse/AIPLAT-46",
                  updated_at=datetime.now(timezone.utc), document_id=f"jira-{issue_id}")
    values.update(changes)
    return models().Issue(**values)


def test_stable_identity_ignores_mutable_key_and_separates_snapshots():
    first = issue()
    renamed = issue(key="MOVED-46")
    rebuilt = issue(scope=scope("snapshot-2"))
    assert first.business_identity == renamed.business_identity == ("site-1", "46")
    assert first.storage_identity != rebuilt.storage_identity


@pytest.mark.parametrize("values", [{"hops": 0}, {"hops": 3}, {"limit": 0},
                                    {"limit": 101}, {"direction": "arbitrary"}])
def test_traversal_rejects_unbounded_or_invalid_input(values):
    with pytest.raises(ValidationError):
        models().TraversalRequest(scope=scope(), issue_id="46", **values)


def test_traversal_defaults_are_bounded_and_explicit():
    request = models().TraversalRequest(scope=scope(), issue_id="46")
    assert (request.hops, request.limit, request.direction) == (1, 100, "both")
    assert request.relation_types == ("BLOCKS",)


def test_manifest_requires_complete_index_stages_before_ready():
    values = dict(scope=scope(), account_scope="service-account-visible",
                  capture_started_at=datetime.now(timezone.utc),
                  capture_finished_at=datetime.now(timezone.utc), source_checksum="sha256:abc",
                  normalization_version="1", index_versions={"graph": "1", "chroma": "1", "fts": "1"},
                  stages={"graph": "ready", "chroma": "pending", "fts": "ready"})
    manifest = models().SnapshotManifest(**values)
    assert not manifest.ready
    values["stages"]["chroma"] = "ready"
    assert models().SnapshotManifest(**values).ready
    values["capture_finished_at"] = datetime(2000, 1, 1, tzinfo=timezone.utc)
    with pytest.raises(ValidationError):
        models().SnapshotManifest(**values)


def test_result_rejects_nodes_from_another_scope():
    with pytest.raises(ValidationError, match="scope"):
        models().GraphEvidenceResult(scope=scope(), nodes=[issue(scope=scope("other"))])


def test_path_preserves_edge_provenance_independently_of_text():
    edge = models().Edge(scope=scope(), source_issue_id="46", target_issue_id="16",
                         relation_type="BLOCKS", source_field="issuelinks",
                         origin_issue_id="46", link_id="123", source_type="Blocks",
                         source_direction="outward")
    path = models().GraphPath(issue_ids=["46", "16"], edges=[edge])
    result = models().GraphEvidenceResult(scope=scope(), nodes=[issue(), issue("16")],
                                          paths=[path], coverage="partial")
    assert result.text_evidence == []
    assert result.paths[0].edges[0].link_id == "123"
    with pytest.raises(ValidationError, match="scope"):
        models().GraphEvidenceResult(scope=scope("other"), paths=[path])


def test_path_rejects_disconnected_edges():
    edge = models().Edge(scope=scope(), source_issue_id="46", target_issue_id="16",
                         relation_type="BLOCKS", source_field="issuelinks",
                         origin_issue_id="46", source_type="Blocks", source_direction="outward")
    with pytest.raises(ValidationError):
        models().GraphPath(issue_ids=["46", "99"], edges=[edge])
