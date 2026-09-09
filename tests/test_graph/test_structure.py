from datetime import datetime, timezone
from importlib import import_module

import pytest

from app.graph.models import Edge, GraphScope, Issue


SCOPE = GraphScope(site_id="site", project_key="P", snapshot_id="s")


def api():
    try:
        return import_module("app.graph.structure")
    except ModuleNotFoundError:
        pytest.fail("structured graph aggregation is not implemented")


def issue(number, kind="Story", **updates):
    values = dict(scope=SCOPE, issue_id=str(number), key=f"P-{number}", issue_type=kind,
                  title="Example", status="Done", status_category="done",
                  source_url=f"https://example.test/P-{number}",
                  updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc), document_id=str(number))
    values.update(updates)
    return Issue(**values)


def parent(child, parent_id, **updates):
    values = dict(scope=SCOPE, source_issue_id=str(child), target_issue_id=str(parent_id),
                  relation_type="CHILD_OF", origin_issue_id=str(child), source_field="parent",
                  source_type="parent", source_direction="parent")
    values.update(updates)
    return Edge(**values)


def dataset():
    return ([issue(10, "Epic"), issue(2, "ePiC", status="Open", status_category="new"),
             issue(11), issue(3, "Bug"), issue(4, "Task"), issue(5), issue(6, "Subtask")],
            [parent(11, 2), parent(3, 2), parent(6, 11)])


def test_overview_counts_direct_children_and_preserves_epic_status():
    issues, edges = dataset()
    result = api().overview(SCOPE, issues, edges)
    assert result.counts.total == 7
    assert result.counts.by_type["Bug"] == 1
    assert result.counts.by_status == {"Done": 6, "Open": 1}
    assert result.counts.by_status_category == {"done": 6, "new": 1}
    assert [e.issue.key for e in result.epics] == ["P-2", "P-10"]
    assert result.epics[0].issue.status == "Open"
    assert result.epics[0].children_counts.total == 2
    assert result.epics[0].children_counts.by_type == {"Bug": 1, "Story": 1}
    assert [i.key for i in result.unassigned] == ["P-4", "P-5"]
    assert [i.key for i in result.other] == ["P-4", "P-6"]


def test_epic_pagination_does_not_change_counts():
    issues, edges = dataset()
    first = api().epic_detail(SCOPE, issues, edges, "P-2", limit=1)
    second = api().epic_detail(SCOPE, issues, edges, "P-2", offset=1, limit=1)
    assert first.counts == second.counts
    assert first.total == 2 and first.has_more and not second.has_more
    assert first.offset == 0 and first.limit == 1
    assert [i.key for i in first.children] == ["P-3"]
    assert [i.key for i in second.children] == ["P-11"]
    assert api().epic_detail(SCOPE, issues, edges, "P-2", offset=99).children == []


@pytest.mark.parametrize("key", ["P-404", "P-3"])
def test_missing_or_non_epic_is_not_found(key):
    with pytest.raises(LookupError):
        api().epic_detail(SCOPE, *dataset(), key)


@pytest.mark.parametrize("offset,limit", [(-1, 1), (0, 0), (0, 101)])
def test_invalid_pagination(offset, limit):
    with pytest.raises(ValueError):
        api().epic_detail(SCOPE, *dataset(), "P-2", offset, limit)


@pytest.mark.parametrize("bad", ["scope", "edge_scope", "id", "key", "dangling", "origin", "edge"])
def test_inconsistent_snapshot_is_rejected(bad):
    issues, edges = dataset()
    other_scope = SCOPE.model_copy(update={"snapshot_id": "other"})
    if bad == "scope":
        issues[0] = issues[0].model_copy(update={"scope": other_scope})
    elif bad == "edge_scope":
        edges[0] = edges[0].model_copy(update={"scope": other_scope})
    elif bad == "id":
        issues.append(issue(2, key="P-99"))
    elif bad == "key":
        issues.append(issue(99, key="P-2"))
    elif bad == "dangling":
        edges.append(parent(5, 99))
    elif bad == "origin":
        edges.append(parent(5, 2, origin_issue_id="99"))
    else:
        edges.append(edges[0])
    with pytest.raises(ValueError):
        api().overview(SCOPE, issues, edges)
    with pytest.raises(ValueError):
        api().epic_detail(SCOPE, issues, edges, "P-2")


def test_empty_project_returns_empty_complete_counts():
    result = api().overview(SCOPE, [], [])
    assert result.counts.total == 0
    assert result.counts.by_type == {}
    assert result.epics == result.unassigned == result.other == []


def test_overview_graph_contains_real_nodes_and_edges_without_inferred_links():
    issues, edges=dataset()
    result=api().overview(SCOPE,issues,edges)
    assert hasattr(result,'graph'), 'Overview is missing a graph projection'
    assert {n.issue_id for n in result.graph.nodes} == {n.issue_id for n in issues}
    assert result.graph.edges == edges
    assert result.graph.total_nodes == 7 and result.graph.total_edges == 3
    assert result.graph.truncated is False


def test_overview_graph_node_limit_preserves_full_counts():
    issues=[issue(1,'Epic'),*[issue(i) for i in range(2,105)]]
    edges=[parent(i,1) for i in range(2,105)]
    result=api().overview(SCOPE,issues,edges)
    assert hasattr(result,'graph'), 'Overview is missing a bounded graph projection'
    assert len(result.graph.nodes) == 100
    assert result.graph.truncated is True
    assert result.counts.total == 104 and result.epics[0].children_counts.total == 103
    ids={n.issue_id for n in result.graph.nodes}
    assert all({e.source_issue_id,e.target_issue_id,e.origin_issue_id} <= ids for e in result.graph.edges)


def test_overview_graph_edge_limit_is_explicit():
    issues=[issue(1,'Epic'),issue(2)]
    edges=[parent(2,1,link_id=str(i)) for i in range(501)]
    result=api().overview(SCOPE,issues,edges)
    assert hasattr(result,'graph'), 'Overview is missing a bounded graph projection'
    assert len(result.graph.edges) == 500
    assert result.graph.total_edges == 501 and result.graph.truncated
