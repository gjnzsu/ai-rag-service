"""Read-only live API acceptance against a local immutable normalized snapshot."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import platform
import statistics
import time

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--normalized', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--base', default='http://127.0.0.1:8001')
    args = parser.parse_args()
    source = json.loads(Path(args.normalized).read_text(encoding='utf-8'))
    issues, edges = source['issues'], source['edges']
    scope = issues[0]['scope']
    checks, timings = {}, {}
    with httpx.Client(base_url=args.base, trust_env=False, timeout=30) as client:
        def get(path, **params):
            response = client.get(path, params={'snapshot_id': scope['snapshot_id'], **params})
            response.raise_for_status()
            result = response.json()
            assert result['scope'] == scope
            return result

        prefix = f"/graph/projects/{scope['project_key']}"
        overview = get(prefix + '/overview')['data']
        def canonical_node(node):
            return json.dumps({**node, 'updated_at': datetime.fromisoformat(node['updated_at']).isoformat()}, sort_keys=True)
        checks['all_nodes_exact'] = sorted(map(canonical_node, overview['graph']['nodes'])) == sorted(map(canonical_node, issues))
        checks['all_edges_exact'] = sorted(map(lambda x: json.dumps(x, sort_keys=True), overview['graph']['edges'])) == sorted(map(lambda x: json.dumps(x, sort_keys=True), edges))
        checks['full_type_counts'] = overview['counts']['by_type'] == dict(Counter(x['issue_type'] for x in issues))
        for epic in (x for x in issues if x['issue_type'] == 'Epic'):
            children = {x['source_issue_id'] for x in edges if x['relation_type'] == 'CHILD_OF' and x['target_issue_id'] == epic['issue_id']}
            detail = get(prefix + '/epics/' + epic['key'], limit=100)['data']
            checks[epic['key'] + '_children_exact'] = {x['issue_id'] for x in detail['children']} == children and detail['total'] == len(children)
            checks[epic['key'] + '_status_exact'] = detail['epic']['status'] == epic['status']
        relation = get(prefix + '/issues/AIPLAT-46/dependencies', direction='outbound')['data']
        checks['canonical_blocks'] = [e for p in relation['paths'] for e in p['edges']] == [e for e in edges if e['relation_type'] == 'BLOCKS']
        unresolved = get(prefix + '/issues/AIPLAT-46/dependencies', unresolved_pair='true')['data']
        checks['done_pair_excluded'] = not unresolved['paths']
        for name, path in [('overview', '/overview'), ('detail', '/epics/AIPLAT-13'), ('dependencies', '/issues/AIPLAT-46/dependencies')]:
            get(prefix + path)  # One warm-up; sequential, one local client.
            samples = []
            for _ in range(10):
                start = time.perf_counter()
                get(prefix + path)
                samples.append(round((time.perf_counter() - start) * 1000, 3))
            timings[name] = {'samples_ms': samples, 'p50_ms': statistics.median(samples), 'p95_nearest_rank_ms': sorted(samples)[9]}
        response = client.post('/graph/query', json={'project_key': scope['project_key'], 'snapshot_id': scope['snapshot_id'], 'query': 'What recorded issue blocks AIPLAT-16?', 'issue_key': 'AIPLAT-16', 'intent': 'dependencies', 'direction': 'inbound'})
        response.raise_for_status()
        answer = response.json()
        checks['query_preserves_evidence'] = bool(answer['data']['paths']) and answer['snapshot_id'] == scope['snapshot_id']
        report = {'scope': scope, 'checks': checks, 'all_passed': all(checks.values()), 'timings': timings,
                  'environment': {'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': __import__('os').cpu_count(), 'concurrency': 1, 'warmup_per_endpoint': 1, 'samples_per_endpoint': 10},
                  'live_query_answer_status': answer['answer']['status'], 'live_query_diagnostics': answer['answer']['diagnostics']}
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'all_passed': report['all_passed'], 'checks': len(checks), 'answer_status': report['live_query_answer_status']}))
    return 0 if report['all_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
