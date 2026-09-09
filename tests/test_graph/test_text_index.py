from datetime import datetime, timezone
from importlib import import_module

import pytest

from app.graph.models import GraphScope, Issue


def module():
    try:
        return import_module("app.graph.text_index")
    except ModuleNotFoundError:
        pytest.fail("Snapshot text index is not implemented")


def sample():
    scope = GraphScope(site_id="site", project_key="AIPLAT", snapshot_id="one")
    issue = Issue(scope=scope, issue_id="123", key="AIPLAT-46", issue_type="Bug",
                  title="Retrieval security", status="Done", status_category="done",
                  source_url="https://example.test/browse/AIPLAT-46",
                  updated_at=datetime.now(timezone.utc), document_id="stable-document-123")
    return scope, [issue], {"123": "Unauthorized retrieval security chunks"}


def test_both_indexes_preserve_identity_and_source(tmp_path):
    scope, issues, texts = sample()
    index = module().SnapshotTextIndex(tmp_path)
    chunks = index.write(scope, issues, texts, lambda texts: [[0.1, 0.9] for _ in texts])
    index.verify(chunks)
    found = index.lexical.search("security", 5, None, "chunks")
    assert len(found) == 1
    assert found[0].metadata["issue_id"] == "123"
    assert found[0].metadata["snapshot_id"] == "one"
    assert found[0].source_url == issues[0].source_url
    vector = index.chroma.get_collection("chunks").query(query_embeddings=[[0.1, 0.9]], n_results=1)
    assert vector["metadatas"][0][0]["document_id"] == "stable-document-123"
    assert vector["ids"][0] == [chunks[0]["chunk_id"]]


@pytest.mark.parametrize("vectors", [[], [[float("nan"), 1]], [[]]])
def test_invalid_embedding_rejects_build(tmp_path, vectors):
    scope, issues, texts = sample()
    with pytest.raises(ValueError, match="embedding"):
        module().SnapshotTextIndex(tmp_path).write(scope, issues, texts, lambda _: vectors)


def test_verification_detects_missing_lexical_content(tmp_path):
    scope, issues, texts = sample()
    index = module().SnapshotTextIndex(tmp_path)
    chunks = index.write(scope, issues, texts, lambda texts: [[1.0, 0.0] for _ in texts])
    index.lexical.delete_document(issues[0].document_id, "chunks")
    with pytest.raises(ValueError, match="index verification"):
        index.verify(chunks)


def test_scope_mismatch_rejected_before_embedding(tmp_path):
    scope, issues, texts = sample()
    other = scope.model_copy(update={"snapshot_id": "other"})
    with pytest.raises(ValueError, match="scope"):
        module().SnapshotTextIndex(tmp_path).write(other, issues, texts, lambda _: pytest.fail("must not embed"))


def test_empty_snapshot_indexes_are_valid(tmp_path):
    scope, _, _ = sample()
    index = module().SnapshotTextIndex(tmp_path)
    chunks = index.write(scope, [], {}, lambda _: pytest.fail("must not embed empty input"))
    assert chunks == []
    index.verify(chunks)
