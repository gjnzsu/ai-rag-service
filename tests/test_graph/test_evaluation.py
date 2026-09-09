from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from app.graph.models import GraphScope, Issue, Edge, TextEvidence, GraphEvidenceResult, GraphPath


def fixture():
    scope = GraphScope(site_id='https://example.test', project_key='AIPLAT', snapshot_id='one')
    issues = [Issue(scope=scope, issue_id=str(i), key=f'AIPLAT-{i}', issue_type=kind,
                    title=kind, status='Done', status_category='done', source_url=f'https://example.test/{i}',
                    document_id=str(i), updated_at=datetime.now(timezone.utc))
              for i, kind in [(1, 'Epic'), (2, 'Story'), (3, 'Bug')]]
    edges = [Edge(scope=scope, source_issue_id='2', target_issue_id='1', relation_type='CHILD_OF',
                  origin_issue_id='2', source_field='parent', source_type='parent', source_direction='parent'),
             Edge(scope=scope, source_issue_id='3', target_issue_id='2', relation_type='BLOCKS',
                  origin_issue_id='3', source_field='issuelinks', source_type='Blocks', source_direction='outward')]
    text = [TextEvidence(scope=scope, issue_id=x.issue_id, document_id=x.document_id,
                         chunk_id=x.issue_id, text='AIPLAT-3 BLOCKS AIPLAT-2 ' * 1000,
                         source_url=x.source_url) for x in issues]
    class Search:
        diagnostics = []
        calls = []
        def search(self, query, top_k):
            self.calls.append((query, top_k))
            return text[2:]
        def for_issues(self, ids):
            return [x for x in text if x.issue_id in ids]
    class Service:
        def retrieve(self, request):
            assert request.snapshot_id == 'one' and request.top_k == 5
            assert request.limit == 100 and request.text_budget == 16000
            return SimpleNamespace(scope=scope, snapshot_id='one', diagnostics=[], structure=None,
                data=GraphEvidenceResult(scope=scope, nodes=issues[1:], text_evidence=text[1:],
                    paths=[GraphPath(issue_ids=['2','3'], edges=[edges[1]])]))
    return SimpleNamespace(scope=scope), issues, edges, Search(), Service()


def run(case, data=None):
    from app.graph.evaluation import evaluate_cases
    manifest, issues, edges, search, service = data or fixture()
    return evaluate_cases([case], manifest, issues, edges, search, service)['cases'][0]['methods']


def case(**updates):
    return dict(id='x', question='which issues?', intent='dependencies', issue_key='AIPLAT-3',
                expected_issue_keys=['AIPLAT-2','AIPLAT-3'],
                expected_edges=[dict(source_key='AIPLAT-3', target_key='AIPLAT-2', relation_type='BLOCKS')],
                **updates)


def test_h_prose_is_not_a_structured_edge_and_g_preserves_canonical_direction():
    rows = run(case())
    assert rows['H']['structured_edge_coverage'] == 0
    assert rows['H']['evidence_node_coverage'] == .5
    assert rows['S']['returned_issue_keys'] == ['AIPLAT-3']
    assert rows['G']['structured_edge_coverage'] == 1
    assert rows['G']['returned_edges'][0]['source_key'] == 'AIPLAT-3'
    assert all(x['text_characters'] <= 16000 for x in rows.values())


def test_expected_labels_never_enter_search():
    data = fixture()
    run(case(notes='SECRET_LABEL'), data)
    assert data[3].calls == [('which issues?', 5)]


def test_scope_mismatch_rejected_before_execution():
    data = fixture()
    data[1][0] = data[1][0].model_copy(update={'scope': data[0].scope.model_copy(update={'snapshot_id':'other'})})
    with pytest.raises(ValueError):
        run(case(), data)


def test_failure_is_not_a_successful_empty_negative():
    data = fixture()
    def fail(*args, **kwargs):
        raise RuntimeError('credential text must not leak')
    data[3].search = fail
    rows = run(dict(id='none',question='missing',intent='dependencies',expected_issue_keys=[],expected_edges=[]), data)
    assert rows['H']['error'] == 'RuntimeError'
    assert rows['H']['evidence_node_coverage'] is None
    assert rows['H']['status'] == 'error'
    assert rows['G']['structured_edge_coverage'] is None


def test_structural_overview_counts_are_exact():
    rows = run(dict(id='overview',question='overview',intent='overview',expected_issue_keys=['AIPLAT-1'],
                    expected_edges=[],expected_counts={'total':3,'by_type':{'Epic':1,'Story':1,'Bug':1},'by_status':{'Done':3},'by_status_category':{'done':3}}))
    assert rows['S']['counts_match'] is True
    assert rows['S']['evidence_node_coverage'] == 1


def test_structural_parent_expansion_has_no_block_traversal():
    item = case()
    item.pop('issue_key')
    item['epic_key'] = 'AIPLAT-1'
    rows = run(item)
    assert rows['S']['returned_issue_keys'] == ['AIPLAT-1', 'AIPLAT-2']
    assert rows['S']['returned_edges'] == [dict(source_key='AIPLAT-2', target_key='AIPLAT-1', relation_type='CHILD_OF')]


def test_text_scope_mismatch_is_an_error():
    data = fixture()
    original = data[3].search('x', 5)[0]
    foreign = original.model_copy(update={'scope': original.scope.model_copy(update={'snapshot_id': 'foreign'})})
    data[3].search = lambda *args, **kwargs: [foreign]
    rows = run(case(), data)
    assert rows['H']['status'] == 'error'
    assert rows['H']['evidence_node_coverage'] is None


def test_g_foreign_response_is_an_error():
    data = fixture()
    original = data[4].retrieve
    def retrieve(request):
        response = original(request)
        response.snapshot_id = 'foreign'
        return response
    data[4].retrieve = retrieve
    assert run(case(), data)['G']['error'] == 'ValueError'


def test_case_cannot_override_fixed_budgets_and_successful_empty_scores_are_null():
    rows = run(dict(id='empty', question='none', intent='dependencies', issue_key='AIPLAT-3',
                    top_k=19, limit=1, text_budget=1, expected_issue_keys=[], expected_edges=[]))
    assert rows['G']['status'] == 'ok'
    assert rows['G']['evidence_node_coverage'] is None
    assert rows['G']['structured_edge_coverage'] is None
    assert rows['G']['text_characters'] == 16000
    assert rows['G']['truncated'] is True

def test_g_does_not_fill_missing_relations_from_snapshot():
    data = fixture()
    original = data[4].retrieve
    def retrieve(request):
        response = original(request)
        response.data.nodes = data[1]
        response.structure = {'epic_membership': [{'epic': data[1][0].model_dump(mode='json'), 'edges': []}]}
        return response
    data[4].retrieve = retrieve
    rows = run(case(), data)
    assert [e['relation_type'] for e in rows['G']['returned_edges']] == ['BLOCKS']


def test_negative_case_rejects_spurious_recorded_block():
    item = case()
    item.update(answerability='no_recorded_relation', expected_edges=[])
    rows = run(item)
    assert rows['G']['negative_check'] is False
    assert rows['S']['negative_check'] is None
    assert rows['H']['negative_check'] is None


def test_absent_anchor_is_expected_not_found_not_generic_error():
    item = case()
    item.update(issue_key='AIPLAT-999', question='AIPLAT-999', answerability='not_in_snapshot', expected_issue_keys=[], expected_edges=[])
    rows = run(item)
    assert rows['S']['negative_check'] is True
    assert rows['H']['negative_check'] is None
