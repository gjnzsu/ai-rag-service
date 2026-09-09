"""Local read-only Jira snapshot build. No HTTP mutation endpoint is exposed."""

import argparse
import json
import os
from pathlib import Path
import sys
from urllib.parse import urlsplit


def run(args) -> dict:
    from dotenv import dotenv_values

    credentials = dotenv_values(args.env_file)
    graph_credentials = dotenv_values(args.graph_env_file)
    required = ("JIRA_URL", "JIRA_EMAIL", "JIRA_API_TOKEN", "OPENAI_API_KEY")
    if any(not credentials.get(key) for key in required) or not graph_credentials.get("GRAPH_NEO4J_PASSWORD"):
        raise ValueError("Required local credentials unavailable")
    site_url = credentials["JIRA_URL"].rstrip("/")
    url = urlsplit(site_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.path or url.query or url.fragment:
        raise ValueError("Jira origin must be an HTTPS site origin")
    # Legacy text-index modules require global settings at import. All remote calls below
    # receive explicit credentials; real secrets never enter test/default settings.
    os.environ.setdefault("OPENAI_API_KEY", "test-key")
    import httpx
    from openai import OpenAI

    from app.graph.build import build_snapshot
    from app.graph.embedding import OpenAIChunkEmbedder
    from app.graph.jira import JiraSnapshotReader
    from app.graph.snapshots import SnapshotRepository
    from app.graph.store import Neo4jGraphStore

    repository = SnapshotRepository(Path(args.data_dir))
    store = Neo4jGraphStore.connect("bolt://127.0.0.1:7687", "neo4j", graph_credentials["GRAPH_NEO4J_PASSWORD"])
    report = None
    try:
        with httpx.Client(auth=(credentials["JIRA_EMAIL"], credentials["JIRA_API_TOKEN"]),
                          timeout=30, follow_redirects=False) as jira_client:
            captured = JiraSnapshotReader(jira_client, site_url, ("AIPLAT",)).capture(args.project)
        with OpenAI(api_key=credentials["OPENAI_API_KEY"], timeout=30, max_retries=0,
                    http_client=httpx.Client()) as client:
            embedder = OpenAIChunkEmbedder(client)
            manifest = build_snapshot(repository, store, captured, site_url, args.project, embedder)
            report = published_report(repository, manifest)
        return report
    except Exception:
        if report is None:
            raise
        report["cleanup_status"] = "failed"
        return report
    finally:
        try:
            store.close()
        except Exception:
            if report is None:
                raise
            report["cleanup_status"] = "failed"


def published_report(repository, manifest) -> dict:
    outcome = {"status": "published", "snapshot_id": manifest.scope.snapshot_id}
    try:
        report = json.loads((repository.directory(manifest.scope) / "build-report.json").read_text(encoding="utf-8"))
        return {**report, **outcome}
    except Exception:
        return {**outcome, "report_status": "unavailable"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=["AIPLAT"], default="AIPLAT")
    parser.add_argument("--env-file", required=True, help="Explicit source credential file; never copied")
    parser.add_argument("--graph-env-file", default=".env.graph")
    parser.add_argument("--data-dir", default="data/graph-poc")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__}), file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
