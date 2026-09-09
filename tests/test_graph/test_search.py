import json
from datetime import datetime, timezone

import pytest

from app.graph.models import GraphScope, Issue
from app.graph.search import SnapshotSearch
from app.graph.text_index import SnapshotTextIndex


def _scope(snapshot_id="one"):
    return GraphScope(site_id="site", project_key="AIPLAT", snapshot_id=snapshot_id)


def _issue(scope, issue_id, key, title):
    return Issue(
        scope=scope, issue_id=issue_id, key=key, issue_type="Bug", title=title,
        status="Open", status_category="new", source_url=f"https://example.test/browse/{key}",
        updated_at=datetime.now(timezone.utc), document_id=f"document-{issue_id}",
    )


def _snapshot(directory):
    scope = _scope()
    issues = [_issue(scope, "101", "AIPLAT-101", "Authentication"),
              _issue(scope, "202", "AIPLAT-202", "Vector seed")]
    texts = {"101": "Authentication token expires unexpectedly.",
             "202": "A quasar illuminates the retrieval pipeline."}
    vectors = {texts["101"]: [1.0, 0.0], texts["202"]: [0.0, 1.0]}
    chunks = SnapshotTextIndex(directory).write(scope, issues, texts, lambda values: [vectors[v] for v in values])
    (directory / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    return scope, chunks


def test_hybrid_search_can_find_vector_only_seed(tmp_path):
    scope, chunks = _snapshot(tmp_path)
    found = SnapshotSearch(tmp_path, scope, lambda _: [[0.0, 1.0]]).search("celestial", top_k=1)
    assert found == [SnapshotSearch._evidence(chunks[1], scope)]


def test_exact_issue_key_does_not_embed_and_returns_matching_chunk(tmp_path):
    scope, _ = _snapshot(tmp_path)
    found = SnapshotSearch(tmp_path, scope, lambda _: pytest.fail("exact match must not embed")).search(
        "please inspect AIPLAT-101", top_k=5,
    )
    assert [item.issue_id for item in found] == ["101"]


def test_lexical_match_is_fused_with_vector_results(tmp_path):
    scope, _ = _snapshot(tmp_path)
    found = SnapshotSearch(tmp_path, scope, lambda _: [[0.0, 1.0]]).search("authentication", top_k=2)
    assert [item.issue_id for item in found] == ["101", "202"]


def test_vector_failure_degrades_to_lexical_and_diagnostics_are_per_query(tmp_path):
    scope, _ = _snapshot(tmp_path)
    search = SnapshotSearch(tmp_path, scope, lambda _: (_ for _ in ()).throw(RuntimeError("offline")))
    assert [item.issue_id for item in search.search("authentication")] == ["101"]
    assert search.diagnostics == ["vector unavailable: RuntimeError"]
    search.embed = lambda _: [[1.0, 0.0]]
    search.search("authentication")
    assert search.diagnostics == []


def test_scope_mismatch_in_persisted_metadata_is_rejected(tmp_path):
    scope, _ = _snapshot(tmp_path)
    collection = SnapshotTextIndex(tmp_path).chroma.get_collection("chunks")
    record = collection.get(ids=[collection.get()["ids"][0]], include=["documents", "embeddings", "metadatas"])
    metadata = dict(record["metadatas"][0])
    metadata["snapshot_id"] = "other"
    collection.update(ids=record["ids"], embeddings=record["embeddings"], documents=record["documents"], metadatas=[metadata])
    with pytest.raises(ValueError, match="scope"):
        SnapshotSearch(tmp_path, scope, lambda _: [[1.0, 0.0]]).search("authentication")


def test_missing_indexes_fail_closed_without_creating_defaults(tmp_path):
    with pytest.raises(FileNotFoundError, match="snapshot.*index"):
        SnapshotSearch(tmp_path, _scope(), lambda _: [[1.0, 0.0]])
    assert list(tmp_path.iterdir()) == []


def test_for_issues_reads_all_chunks_in_snapshot_order_and_validates_ids(tmp_path):
    scope, chunks = _snapshot(tmp_path)
    search = SnapshotSearch(tmp_path, scope, lambda _: pytest.fail("must not embed"))
    found = search.for_issues(["202", "101", "202"])
    assert [item.chunk_id for item in found] == [chunk["chunk_id"] for chunk in chunks]
    with pytest.raises(ValueError, match="unknown issue"):
        search.for_issues(["missing"])


def test_for_issues_rejects_tampered_chunk_provenance(tmp_path):
    scope, chunks = _snapshot(tmp_path)
    chunks[0]["metadata"]["site_id"] = "other"
    (tmp_path / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    with pytest.raises(ValueError, match="scope"):
        SnapshotSearch(tmp_path, scope, lambda _: []).for_issues(["101"])
