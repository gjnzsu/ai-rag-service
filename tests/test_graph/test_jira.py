from datetime import datetime, timezone

import httpx
import pytest

from app.graph.jira import CapturedProject, JiraCaptureError, JiraSnapshotReader, normalize
from app.graph.models import GraphScope


def issue(number, kind="Story", parent=None, links=None):
    fields = {
        "project": {"key": "AIPLAT"}, "issuetype": {"name": kind},
        "summary": f"Title {number}", "description": None,
        "status": {"name": "Done", "statusCategory": {"key": "done"}},
        "updated": "2026-09-08T00:00:00.000+0000", "issuelinks": links or [],
    }
    if parent:
        fields["parent"] = {"id": str(parent)}
    return {"id": str(number), "key": f"AIPLAT-{number}", "fields": fields}


def reader(pages):
    calls = []

    def respond(request):
        calls.append(request)
        if request.url.path.endswith("myself"):
            return httpx.Response(200, json={"accountId": "opaque-user", "emailAddress": "private"})
        result = pages.pop(0)
        return httpx.Response(result if isinstance(result, int) else 200,
                              json={} if isinstance(result, int) else result)

    client = httpx.Client(transport=httpx.MockTransport(respond))
    return JiraSnapshotReader(client, "https://example.atlassian.net", ("AIPLAT",)), calls


def test_capture_paginates_and_hashes_account():
    source, calls = reader([
        {"issues": [issue(1)], "isLast": False, "nextPageToken": "next"},
        {"issues": [issue(2)], "isLast": True},
    ])
    result = source.capture("AIPLAT")
    assert len(result.issues) == 2
    assert result.started_at <= result.finished_at
    assert "opaque-user" not in result.account_scope and "private" not in repr(result)
    searches = [r for r in calls if r.url.path.endswith("search/jql")]
    assert searches[1].url.params["nextPageToken"] == "next"
    assert searches[0].url.params["jql"] == 'project = "AIPLAT" ORDER BY id ASC'


@pytest.mark.parametrize("pages", [
    [{"issues": []}],
    [{"issues": [], "isLast": False}],
    [{"issues": [], "isLast": False, "nextPageToken": "x"}] * 2,
    [{"issues": [issue(1)], "isLast": False, "nextPageToken": "x"}, 403],
])
def test_incomplete_capture_fails(pages):
    source, _ = reader(pages.copy())
    with pytest.raises(JiraCaptureError):
        source.capture("AIPLAT")


def test_allowed_project_checked_before_network():
    source, calls = reader([])
    with pytest.raises(JiraCaptureError):
        source.capture('AIPLAT" OR project = "SECRET')
    assert calls == []


def test_wrong_project_and_inconsistent_duplicate_fail():
    wrong = issue(1)
    wrong["fields"]["project"]["key"] = "SECRET"
    changed = issue(1)
    changed["fields"]["summary"] = "changed"
    for records in ([wrong], [issue(1), changed]):
        source, _ = reader([{"issues": records, "isLast": True}])
        with pytest.raises(JiraCaptureError):
            source.capture("AIPLAT")


def test_normalize_deduplicates_direction_and_retains_boundary():
    relation = {"name": "Blocks", "outward": "blocks", "inward": "is blocked by"}
    bug = issue(46, "Bug", 13, [
        {"id": "link", "type": relation, "outwardIssue": {"id": "16"}},
        {"id": "boundary", "type": relation, "outwardIssue": {"id": "999", "key": "SECRET-1"}},
    ])
    story = issue(16, parent=13, links=[
        {"id": "link", "type": relation, "inwardIssue": {"id": "46"}},
    ])
    story["fields"]["description"] = {"type": "doc", "content": [
        {"type": "paragraph", "content": [{"type": "text", "text": "First"}]},
        {"type": "paragraph", "content": [{"type": "text", "text": "Second"}]},
    ]}
    now = datetime.now(timezone.utc)
    captured = CapturedProject([issue(13, "Epic"), story, bug, issue(50)], now, now, "hash", [])
    result = normalize(GraphScope(site_id="site", project_key="AIPLAT", snapshot_id="s"),
                       captured, "https://example.atlassian.net")
    blocks = [edge for edge in result.edges if edge.relation_type == "BLOCKS"]
    assert len(blocks) == 1
    assert (blocks[0].source_issue_id, blocks[0].target_issue_id) == ("46", "16")
    assert len(result.edges) == 3
    assert "First\nSecond" in result.texts["16"]
    assert result.texts["46"] == "Title 46"
    assert any("boundary" in d for d in result.diagnostics)
    assert any("unassigned" in d for d in result.diagnostics)
    assert "SECRET" not in repr(result)
    assert all(i.status_category == "done" for i in result.issues)


def test_unknown_link_preserved_and_identity_stable_across_key_change():
    first = issue(1, links=[{"id": "l", "type": {"name": "Custom", "outward": "foo"},
                           "outwardIssue": {"id": "2"}}])
    now = datetime.now(timezone.utc)
    scope = GraphScope(site_id="site", project_key="AIPLAT", snapshot_id="s")
    capture = CapturedProject([first, issue(2)], now, now, "hash", [])
    before = normalize(scope, capture, "https://example.atlassian.net")
    first["key"] = "AIPLAT-99"
    after = normalize(scope, capture, "https://example.atlassian.net")
    assert before.issues[0].document_id == after.issues[0].document_id
    assert before.edges[0].relation_type == "UNKNOWN"
    assert before.edges[0].source_type == "Custom"


def test_malformed_project_fails_with_safe_error():
    broken = issue(1)
    broken["fields"]["project"] = None
    source, _ = reader([{"issues": [broken], "isLast": True}])
    with pytest.raises(JiraCaptureError, match="boundary"):
        source.capture("AIPLAT")


def test_timeout_does_not_expose_request_details():
    def fail(request):
        raise httpx.ReadTimeout("secret-token", request=request)

    source = JiraSnapshotReader(httpx.Client(transport=httpx.MockTransport(fail)),
                                "https://example.atlassian.net", ("AIPLAT",))
    with pytest.raises(JiraCaptureError) as error:
        source.capture("AIPLAT")
    assert "secret-token" not in str(error.value)


def test_missing_parent_endpoint_has_no_invented_relationship():
    now = datetime.now(timezone.utc)
    capture = CapturedProject([issue(1, parent=999)], now, now, "hash", [])
    result = normalize(GraphScope(site_id="site", project_key="AIPLAT", snapshot_id="s"),
                       capture, "https://example.atlassian.net")
    assert result.edges == []
    assert result.diagnostics == ["boundary: parent of 1 is outside visible snapshot"]
