from datetime import datetime, timedelta, timezone
from importlib import import_module

import pytest

from app.graph.models import GraphScope, SnapshotManifest


def repository(path):
    try:
        return import_module("app.graph.snapshots").SnapshotRepository(path)
    except ModuleNotFoundError:
        pytest.fail("Snapshot repository is not implemented")


def manifest(snapshot="one", ready=True):
    now = datetime.now(timezone.utc)
    return SnapshotManifest(
        scope=GraphScope(site_id="site", project_key="AIPLAT", snapshot_id=snapshot),
        account_scope="visible", capture_started_at=now, capture_finished_at=now,
        source_checksum="hash", normalization_version="1",
        index_versions={key: "1" for key in ("graph", "chroma", "fts")},
        stages={"graph": "ready", "chroma": "ready", "fts": "ready" if ready else "failed"},
    )


def test_publish_requires_all_indexes_ready(tmp_path):
    repo = repository(tmp_path)
    good = manifest()
    repo.reserve(good.scope)
    repo.publish(good)
    failed = manifest("two", ready=False)
    repo.reserve(failed.scope)
    with pytest.raises(ValueError, match="ready"):
        repo.publish(failed)
    assert repo.resolve("site", "AIPLAT").scope == good.scope


def test_failed_pointer_replace_preserves_old_snapshot(tmp_path, monkeypatch):
    repo = repository(tmp_path)
    first, second = manifest(), manifest("two")
    repo.reserve(first.scope)
    repo.publish(first)
    pinned = repo.resolve("site", "AIPLAT")
    repo.reserve(second.scope)
    import os
    replace = os.replace

    def fail_active(source, target):
        if str(target).endswith("active.json"):
            raise OSError("simulated pointer write failure")
        return replace(source, target)

    monkeypatch.setattr("app.graph.snapshots.os.replace", fail_active)
    with pytest.raises(OSError):
        repo.publish(second)
    assert repo.resolve("site", "AIPLAT").scope == pinned.scope


def test_successful_switch_retains_pinned_version(tmp_path):
    repo = repository(tmp_path)
    first, second = manifest(), manifest("two")
    for item in (first, second):
        repo.reserve(item.scope)
        repo.publish(item)
    assert repo.resolve("site", "AIPLAT").scope == second.scope
    assert repo.resolve("site", "AIPLAT", "one").scope == first.scope


def test_scope_names_cannot_escape_root(tmp_path):
    repo = repository(tmp_path)
    scope = GraphScope(site_id="../../site", project_key="../project", snapshot_id="../../../outside")
    assert repo.reserve(scope).is_relative_to(tmp_path.resolve())


def test_missing_active_never_creates_a_default_snapshot(tmp_path):
    repo = repository(tmp_path)
    with pytest.raises(FileNotFoundError):
        repo.resolve("site", "AIPLAT")


def test_existing_snapshot_cannot_be_reserved_again(tmp_path):
    repo = repository(tmp_path)
    repo.reserve(manifest().scope)
    with pytest.raises(FileExistsError):
        repo.reserve(manifest().scope)


def test_project_build_lock_excludes_parallel_writers_and_releases(tmp_path):
    repo = repository(tmp_path)
    with pytest.raises(RuntimeError, match="test failure"):
        with repo.build_lock("site", "AIPLAT"):
            with pytest.raises(FileExistsError):
                with repo.build_lock("site", "AIPLAT"):
                    pytest.fail("second writer entered")
            raise RuntimeError("test failure")
    with repo.build_lock("site", "AIPLAT"):
        pass


def test_older_capture_cannot_replace_newer_active(tmp_path):
    repo = repository(tmp_path)
    newer = manifest("newer")
    repo.reserve(newer.scope)
    repo.publish(newer)
    older = manifest("older")
    older.capture_started_at = newer.capture_started_at - timedelta(minutes=1)
    repo.reserve(older.scope)
    with pytest.raises(ValueError, match="older capture"):
        repo.publish(older)
    assert repo.resolve("site", "AIPLAT").scope == newer.scope
