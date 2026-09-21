"""Code-verified findings; report-wide evidence IDs assigned after retrieval."""

from app.agentic.models import AnalysisResponse, Coverage, EvidenceRecord, Finding
from app.agentic.tools import public_issue


def build_report(tools, requested, required, execution):
    facts = tools.facts()
    scope = {'project_key': tools.request.project_key, 'epic_key': tools.epic.key,
             'snapshot_id': tools.scope.snapshot_id, 'population': 'direct_story_children'}
    coverage = {}
    for item in ('completion', 'bugs'):
        coverage[item] = Coverage(
            status=('complete' if tools.backlog_complete else 'partial') if item in requested else 'not_requested',
            limitations=list(tools.backlog_limitations) if item in requested else [])
    expected = sorted(f'{k}:{d}' for k, d in required)
    checked = sorted(f'{k}:{d}' for k, d in required if tools.ledger.get((k, d)) == 'complete')
    failed = sorted(f'{k}:{d}' for k, d in required if tools.ledger.get((k, d)) == 'failed')
    truncated = sorted(f'{k}:{d}' for k, d in required if tools.ledger.get((k, d)) == 'partial')
    complete = tools.backlog_complete and len(checked) == len(expected)
    coverage['blockers'] = Coverage(
        status=('complete' if complete else 'partial') if 'blockers' in requested else 'not_requested',
        expected=expected, checked=checked, failed=failed, truncated=truncated,
        limitations=[] if complete or 'blockers' not in requested else ['dependency_scope_not_fully_checked'])
    if 'unsupported' in requested or not requested:
        coverage['unsupported'] = Coverage(status='unavailable', limitations=['requested_analysis_not_supported_or_not_resolved'])

    evidence = []

    def add(kind, content, urls):
        identifier = f'E{len(evidence)+1}'
        evidence.append(EvidenceRecord(id=identifier, kind=kind, content=content, source_urls=urls))
        return identifier

    aggregate = {k: v for k, v in facts.items() if k not in ('stories', 'direct_bugs', 'associated_bugs')}
    aggregate['backlog_complete'] = tools.backlog_complete
    stats_id = add('aggregate', {**scope, **aggregate}, [tools.epic.source_url])
    issue_ids = {}
    for key, item in sorted(tools.known.items()):
        issue_ids[key] = add('issue', public_issue(item), [item.source_url])
    coverage_id = add('coverage', {k: v.model_dump() for k, v in coverage.items()}, [tools.epic.source_url])
    findings = []

    def finding(code, statement, references, extra_scope=None, limitations=None):
        findings.append(Finding(code=code, statement=statement, scope={**scope, **(extra_scope or {})},
                                evidence_ids=references, limitations=limitations or []))

    if 'completion' in requested and tools.backlog_complete:
        count, done = facts['story_count'], facts['done_count']
        if not count:
            finding('no_stories', '当前快照中，该 Epic 没有直属 Story，完成率不适用。', [stats_id])
        else:
            code = 'stories_all_done' if done == count else 'stories_not_all_done'
            finding(code, f'直属 Story 共 {count} 项，Done {done} 项，按数量完成率为 {done/count:.1%}。', [stats_id])
    if 'bugs' in requested:
        count = len(facts['direct_bugs'])
        prefix = '完整清单中' if tools.backlog_complete else '已读取的部分清单中'
        finding('direct_bugs_observed', f'{prefix}有 {count} 个直属 Bug。',
                [stats_id, coverage_id, *[issue_ids[i['key']] for i in facts['direct_bugs']]],
                limitations=tools.backlog_limitations)

    by_id = {i.issue_id: i for i in tools.known.values()}
    unresolved = 0
    unknown = False
    for _, edge in sorted(tools.edges.items()):
        source, target = by_id[edge.source_issue_id], by_id[edge.target_issue_id]
        content = {'source': source.key, 'target': target.key, 'type': 'BLOCKS',
                   'source_status': source.status, 'target_status': target.status,
                   'source_category': source.status_category, 'target_category': target.status_category,
                   'provenance': edge.model_dump(mode='json', exclude={'scope'})}
        reference = add('relationship', content, [by_id[edge.origin_issue_id].source_url])
        relevant = (target.key, 'inbound') in required or (source.key, 'outbound') in required
        if 'blockers' not in requested or not relevant:
            continue
        categories = {source.status_category.casefold(), target.status_category.casefold()}
        active = categories <= {'new', 'indeterminate'}
        unknown |= not categories <= {'new', 'indeterminate', 'done'}
        unresolved += active
        finding('recorded_unresolved_blocker_found' if active else 'recorded_blocking_relation',
                f'当前快照记录 {source.key} BLOCKS {target.key}，两端状态分别为 {source.status} 和 {target.status}'
                + ('；双方均为已知未完成状态。' if active else '；该关系不被归类为双方已知未完成的阻塞。'),
                [reference], {'source': source.key, 'target': target.key, 'relation_type': 'BLOCKS'},
                [] if complete else ['other_requested_dependencies_not_fully_checked'])
    if 'blockers' in requested and complete and not unresolved and not unknown:
        finding('no_recorded_unresolved_blocker_in_checked_scope',
                '在当前快照的完整已查范围内，未发现双方均为已知未完成状态的 BLOCKS 关系。',
                [coverage_id, *[e.id for e in evidence if e.kind == 'relationship']],
                {'checked': checked, 'hops': 1, 'relation_type': 'BLOCKS'})
    if unknown and 'blockers' in requested:
        coverage['blockers'].limitations.append('unknown_endpoint_status')
        # Keep the coverage evidence synchronized with the final ledger.
        evidence[int(coverage_id[1:])-1].content = {k: v.model_dump() for k, v in coverage.items()}
    success = requested and all(c.status in ('complete', 'not_requested') for c in coverage.values())
    execution.status = 'completed' if success and execution.stop_reason == 'finished' else 'partial'
    return AnalysisResponse(subject={'project_key': tools.request.project_key, 'epic_key': tools.epic.key},
        snapshot={'id': tools.scope.snapshot_id, 'captured_at': tools.captured_at.isoformat(),
                  'population': 'direct_story_children'}, facts=facts, requested=sorted(requested),
        findings=findings, evidence=evidence, coverage=coverage, execution=execution)
