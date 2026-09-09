"""Retrieval-only, same-snapshot comparison; gold labels are used only for scoring."""

from time import perf_counter

from app.graph import structure
from app.graph.api_models import GraphRetrieveRequest
from app.graph.retrieval import GraphRetrievalService
from app.graph.models import Edge
from app.retrieval.query_hints import extract_query_hints

BUDGETS = {'top_k': 5, 'limit': 100, 'text_budget': 16000}


def _edge(edge, by_id):
    return {'source_key': by_id[edge.source_issue_id].key,
            'target_key': by_id[edge.target_issue_id].key, 'relation_type': edge.relation_type}


def _identity(edge):
    return edge['source_key'], edge['target_key'], edge['relation_type']


def _request(case, scope):
    fields = {key: case[key] for key in ('intent', 'issue_key', 'epic_key', 'direction', 'hops',
                                        'unresolved_pair', 'relation_type') if key in case}
    return GraphRetrieveRequest(project_key=scope.project_key, snapshot_id=scope.snapshot_id,
                                query=case['question'], **BUDGETS, **fields)


def _anchors(request, issues, search, scope, by_id):
    explicit = request.issue_key or request.epic_key
    keys = [explicit] if explicit else extract_query_hints(request.query).jira_keys
    if keys:
        anchors = [next((node for node in issues if node.key == key), None) for key in keys]
        if any(node is None for node in anchors):
            raise LookupError('issue not found')
        if request.epic_key and anchors[0].issue_type.casefold() != 'epic':
            raise LookupError('epic not found')
        return anchors
    hits = search.search(request.query, top_k=request.top_k)
    GraphRetrievalService._validate_text(hits, scope, by_id)
    return list({hit.issue_id: by_id[hit.issue_id] for hit in hits}.values())


def _structural(request, issues, edges, search, scope, by_id):
    payload = None
    if request.intent == 'overview':
        payload = structure.overview(scope, issues, edges).model_dump(mode='json', exclude={'graph'})
        nodes = issues
    else:
        anchors = _anchors(request, issues, search, scope, by_id)
        if request.intent == 'epic_detail':
            epic = next((x for x in anchors if x.issue_type.casefold() == 'epic'), None)
            if epic is None:
                parents = {e.target_issue_id for e in edges if e.relation_type == 'CHILD_OF'
                           and e.source_issue_id in {x.issue_id for x in anchors}}
                epic = next((x for x in issues if x.issue_id in parents and x.issue_type.casefold() == 'epic'), None)
            if epic is None:
                nodes = []
            else:
                detail = structure.epic_detail(scope, issues, edges, epic.key, limit=request.limit - 1)
                payload = detail.model_dump(mode='json')
                nodes = [epic, *detail.children]
        else:
            nodes = []
            for anchor in anchors:
                nodes.append(anchor)
                if anchor.issue_type.casefold() == 'epic':
                    nodes.extend(structure._children(issues, edges, anchor.issue_id))
    nodes = list({x.issue_id: x for x in nodes}.values())
    truncated = len(nodes) > request.limit or bool(payload and payload.get('has_more'))
    nodes = nodes[:request.limit]
    selected = {x.issue_id for x in nodes}
    memberships = [e for e in edges if e.relation_type == 'CHILD_OF'
                   and {e.source_issue_id, e.target_issue_id} <= selected]
    return nodes, memberships, payload, truncated


def evaluate_cases(cases, manifest, issues, edges, search, service):
    """Run H (hybrid text), S (direct parent/filter), G (graph retrieval).

    No answer generation or expected-label inputs reach retrieval. Coverage scores
    describe returned evidence, not answer correctness or text relation reasoning.
    Latencies include retrieval and normalization, exclude scoring, and are local
    measurements whose cache conditions must be recorded by the caller.
    """
    scope = manifest.scope
    structure._validate(scope, issues, edges)
    by_id = {node.issue_id: node for node in issues}
    report = {'snapshot_id': scope.snapshot_id, 'scope': scope.model_dump(),
              'budgets': dict(BUDGETS), 'cases': []}
    for case in cases:
        request = _request(case, scope)
        result = {'id': case['id'], 'question': case['question'],
                  'answerability': case.get('answerability'), 'methods': {}}
        for method in ('H', 'S', 'G'):
            started = perf_counter()
            row = {'status': 'ok', 'error': None, 'returned_issue_keys': [], 'text_issue_keys': [],
                   'returned_edges': [], 'diagnostics': [], 'text_characters': 0, 'truncated': False,
                   'structure': None, 'evidence_node_coverage': None, 'structured_edge_coverage': None,
                   'expected_counts': case.get('expected_counts'), 'actual_counts': None, 'counts_match': None}
            try:
                if method == 'H':
                    text = search.search(request.query, top_k=request.top_k)
                    GraphRetrievalService._validate_text(text, scope, by_id)
                    nodes = list({x.issue_id: by_id[x.issue_id] for x in text}.values())[:request.limit]
                    links = []
                elif method == 'S':
                    nodes, links, row['structure'], row['truncated'] = _structural(
                        request, issues, edges, search, scope, by_id)
                    text = search.for_issues([x.issue_id for x in nodes])
                else:
                    response = service.retrieve(request)
                    if response.scope != scope or response.snapshot_id != scope.snapshot_id or response.data.scope != scope:
                        raise ValueError('response scope mismatch')
                    nodes, text = response.data.nodes, response.data.text_evidence
                    if any(by_id.get(x.issue_id) != x for x in nodes):
                        raise ValueError('response node mismatch')
                    links = [e for path in response.data.paths for e in path.edges]
                    row['structure'] = response.structure
                    if response.structure:
                        for membership in response.structure.get('epic_membership', []):
                            links.extend(Edge.model_validate(e) for e in membership.get('edges', []))
                        if request.intent == 'epic_detail' and response.structure.get('epic'):
                            # A returned Epic/children structure explicitly establishes parent membership.
                            parent_id = response.structure['epic']['issue_id']
                            child_ids = {x['issue_id'] for x in response.structure.get('children', [])}
                            links += [e for e in edges if e.relation_type == 'CHILD_OF'
                                      and e.target_issue_id == parent_id and e.source_issue_id in child_ids]
                    row['diagnostics'] = list(response.diagnostics)
                    row['truncated'] = response.data.truncated
                    if response.data.coverage != 'complete':
                        row['status'] = response.data.coverage
                GraphRetrievalService._validate_text(text, scope, by_id)
                remaining = request.text_budget
                selected_text = []
                for item in text:
                    take = min(len(item.text), remaining)
                    if take:
                        selected_text.append(item)
                    row['text_characters'] += take
                    remaining -= take
                    if take < len(item.text):
                        row['truncated'] = True
                        if 'text_budget_exceeded' not in row['diagnostics']:
                            row['diagnostics'].append('text_budget_exceeded')
                if method == 'H':
                    nodes = [by_id[x.issue_id] for x in selected_text]
                row['returned_issue_keys'] = sorted({x.key for x in nodes})
                row['text_issue_keys'] = sorted({by_id[x.issue_id].key for x in selected_text})
                if any(e not in edges for e in links):
                    raise ValueError('unrecorded returned edge')
                unique = {_identity(_edge(e, by_id)): _edge(e, by_id) for e in links}
                row['returned_edges'] = [unique[key] for key in sorted(unique)]
                row['diagnostics'] = list(dict.fromkeys([*row['diagnostics'], *getattr(search, 'diagnostics', [])]))
                if row['structure']:
                    row['actual_counts'] = row['structure'].get('counts')
            except Exception as exc:
                row.update(status='error', error=type(exc).__name__)
                row['returned_issue_keys'], row['text_issue_keys'], row['returned_edges'] = [], [], []
            row['latency_ms'] = round((perf_counter() - started) * 1000, 3)
            if row['status'] != 'error':
                expected_nodes = set(case.get('expected_issue_keys', []))
                expected_edges = {_identity(e) for e in case.get('expected_edges', [])}
                if expected_nodes:
                    row['evidence_node_coverage'] = len(expected_nodes & set(row['returned_issue_keys'])) / len(expected_nodes)
                if expected_edges:
                    row['structured_edge_coverage'] = len(expected_edges & {_identity(e) for e in row['returned_edges']}) / len(expected_edges)
                if row['expected_counts'] is not None:
                    row['counts_match'] = row['actual_counts'] == row['expected_counts']
            row['negative_check'] = None
            if case.get('answerability') == 'not_in_snapshot' and method in ('S', 'G'):
                row['negative_check'] = row['status'] == 'error' and row['error'] == ('LookupError' if method == 'S' else 'GraphNotFound')
            elif case.get('answerability') == 'no_recorded_relation' and method == 'G':
                # H/S do not traverse BLOCKS; their empty paths cannot establish absence.
                row['negative_check'] = (row['status'] == 'ok' and not row['truncated']
                                         and not any(e['relation_type'] == 'BLOCKS' for e in row['returned_edges']))
            result['methods'][method] = row
        report['cases'].append(result)
    return report
