from importlib import import_module

import pytest


def cli():
    try:
        return import_module("app.graph.cli")
    except ModuleNotFoundError:
        pytest.fail("Snapshot CLI is not implemented")


def test_cli_reports_safe_errors_without_upstream_details(monkeypatch, capsys):
    module = cli()

    def fail(args):
        raise ValueError("private upstream token must not be printed")

    monkeypatch.setattr(module, "run", fail)
    assert module.main(["--env-file", "unused", "--project", "AIPLAT"]) == 1
    output = capsys.readouterr()
    assert "private upstream" not in output.err + output.out
    assert "ValueError" in output.err


def test_cli_rejects_out_of_scope_project_before_execution(monkeypatch):
    module = cli()
    monkeypatch.setattr(module, "run", lambda _: pytest.fail("must not execute"))
    with pytest.raises(SystemExit) as exc:
        module.main(["--env-file", "unused", "--project", "OTHER"])
    assert exc.value.code == 2


def test_cli_reports_completed_snapshot(monkeypatch, capsys):
    module = cli()
    monkeypatch.setattr(module, "run", lambda _: {"snapshot_id": "one", "issues": 35})
    assert module.main(["--env-file", "unused", "--project", "AIPLAT"]) == 0
    assert '"issues": 35' in capsys.readouterr().out


def test_post_publication_report_failure_keeps_published_status(tmp_path):
    from types import SimpleNamespace
    module = cli()
    assert hasattr(module, "published_report"), "published reporting must preserve commit status"
    repo = SimpleNamespace(directory=lambda _: tmp_path)
    manifest = SimpleNamespace(scope=SimpleNamespace(snapshot_id="committed"))
    report = module.published_report(repo, manifest)
    assert report["status"] == "published"
    assert report["snapshot_id"] == "committed"
    assert report["report_status"] == "unavailable"


def test_client_cleanup_error_does_not_hide_published_snapshot(tmp_path, monkeypatch):
    from types import SimpleNamespace
    module = cli()
    (tmp_path / "source.env").write_text(
        "JIRA_URL=https://example.test\nJIRA_EMAIL=test\nJIRA_API_TOKEN=test\nOPENAI_API_KEY=test-key\n",
    )
    (tmp_path / "graph.env").write_text("GRAPH_NEO4J_PASSWORD=test-password\n")
    manifest = SimpleNamespace(scope=SimpleNamespace(snapshot_id="committed"))

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise OSError("cleanup failure")

    monkeypatch.setattr("openai.OpenAI", lambda **_: Client())
    monkeypatch.setattr("app.graph.jira.JiraSnapshotReader.capture", lambda *_: None)
    monkeypatch.setattr("app.graph.build.build_snapshot", lambda *_: manifest)
    monkeypatch.setattr("app.graph.store.Neo4jGraphStore.connect", lambda *_: SimpleNamespace(close=lambda: None))
    args = SimpleNamespace(env_file=tmp_path / "source.env", graph_env_file=tmp_path / "graph.env",
                           data_dir=tmp_path / "data", project="AIPLAT")
    report = module.run(args)
    assert report["status"] == "published"
    assert report["snapshot_id"] == "committed"
