"""Reproduce the frozen, paired Agent / deterministic-workflow experiment.

Run from the repository root with its Python environment. Credentials stay local;
raw responses should be written to the ignored data/graph-poc directory.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

from dotenv import dotenv_values

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agentic.coordinator import FixedDecider  # noqa: E402
from app.agentic.models import AnalysisRequest  # noqa: E402
from app.api import agentic  # noqa: E402
from app.config import Settings  # noqa: E402


def check(result, oracle):
    expected = sorted(f'{key}:{oracle["direction"]}' for key in oracle['required_seeds'])
    actual_edges = sorted((f.scope['source'], f.scope['target'], f.code == 'recorded_unresolved_blocker_found')
                          for f in result.findings if f.code in {
                              'recorded_unresolved_blocker_found', 'recorded_blocking_relation'})
    expected_edges = sorted((e['source'], e['target'], e['unresolved']) for e in oracle['edges'])
    identifiers = {e.id for e in result.evidence}
    return {
        'facts_correct': (result.facts['story_count'] == oracle['story_count']
                          and result.facts['done_count'] == oracle['done_count']
                          and sorted(i['key'] for i in result.facts['direct_bugs']) == sorted(oracle['direct_bugs'])),
        'routing_correct': set(result.requested) == set(oracle['requested']),
        'coverage_correct': (result.coverage['blockers'].expected == expected
                             and result.coverage['blockers'].checked == expected
                             and all(result.coverage[k].status == 'complete' for k in oracle['requested'])),
        'relations_correct': actual_edges == expected_edges,
        'finding_references_valid': all(set(f.evidence_ids) <= identifiers for f in result.findings),
        'generation_supported': result.generation_status == 'supported',
        'necessary_backend_calls_only': result.execution.backend_calls == 1 + len(oracle['required_seeds']),
    }


def execute(case, repeat, arm):
    dependency = agentic.get_coordinator(AnalysisRequest.model_validate(case['request']))
    try:
        coordinator = next(dependency)
        if arm == 'baseline':
            oracle = case['oracle']
            coordinator.decider = FixedDecider(oracle['requested'], oracle['population'], oracle['direction'])
        result = coordinator.run()
        return {'case': case['id'], 'repeat': repeat, 'arm': arm,
                'checks': check(result, case['oracle']), 'result': result.model_dump(mode='json')}
    except Exception as error:
        # Do not serialize exception messages, settings, prompts or credentials.
        return {'case': case['id'], 'repeat': repeat, 'arm': arm, 'error_type': type(error).__name__}
    finally:
        dependency.close()


def pair(case, repeat):
    arms = ('agent', 'baseline') if repeat % 2 else ('baseline', 'agent')
    return [execute(case, repeat, arm) for arm in arms]


def summarize(records):
    summaries = {}
    for arm in ('agent', 'baseline'):
        rows = [r for r in records if r['arm'] == arm]
        successful = [r for r in rows if 'result' in r]
        executions = [r['result']['execution'] for r in successful]
        latency = sorted(e['elapsed_ms'] for e in executions)
        summaries[arm] = {
            'requests': len(rows), 'errors': len(rows)-len(successful),
            'checks_passed': {key: sum(r['checks'][key] for r in successful)
                              for key in successful[0]['checks']} if successful else {},
            'totals': {key: sum(e[key] for e in executions) for key in (
                'tool_calls', 'backend_calls', 'database_queries', 'decision_calls', 'generation_calls')},
            'recorded_tokens': sum(u['total_tokens'] or 0 for e in executions for u in e['model_usage']),
            'calls_without_usage': sum(e['decision_calls'] + e['generation_calls']
                                       - sum(u.get('total_tokens') is not None for u in e['model_usage'])
                                       for e in executions),
            'median_ms': statistics.median(latency) if latency else None,
            'p95_ms': latency[min(len(latency)-1, int(len(latency)*.95))] if latency else None,
        }
    return summaries


def public_record(row):
    result = {k: v for k, v in row.items() if k != 'result'}
    if 'result' in row:
        result['execution'] = row['result']['execution']
        result['generation_status'] = row['result']['generation_status']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', default='.env')
    parser.add_argument('--graph-env-file', default='.env.graph')
    parser.add_argument('--data-dir', default='data/graph-poc')
    parser.add_argument('--cases', default='docs/evaluation/agentic-epic-questions.json')
    parser.add_argument('--output', default='data/graph-poc/agentic-evaluation')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--probe', action='store_true', help='Run one Agent combined-analysis request only')
    args = parser.parse_args()
    if args.repeats < 1 or not 1 <= args.workers <= 3:
        parser.error('repeats must be positive; workers must be 1..3')
    # Take Gateway credentials only
    # from the explicitly selected main env file, never the graph-only overlay.
    provider = dotenv_values(args.env_file)
    agentic.settings = Settings(_env_file=(args.env_file, args.graph_env_file), graph_data_dir=args.data_dir,
                                ai_gateway_base_url=provider['AI_GATEWAY_BASE_URL'],
                                ai_gateway_api_key=provider['AI_GATEWAY_API_KEY'])
    fixture = json.loads(Path(args.cases).read_text(encoding='utf-8'))
    implementation = hashlib.sha256()
    for path in sorted([*Path('app/agentic').glob('*.py'), Path('app/api/agentic.py'), Path(__file__)]):
        implementation.update(path.name.encode())
        implementation.update(path.read_bytes())
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    raw_path = output / ('probe.jsonl' if args.probe else 'responses.jsonl')
    with raw_path.open('w', encoding='utf-8') as stream:
        if args.probe:
            record = execute(fixture['cases'][-1], 1, 'agent')
            records.append(record)
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            print(json.dumps({k: v for k, v in record.items() if k != 'result'}), flush=True)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                jobs = [pool.submit(pair, case, repeat) for repeat in range(1, args.repeats+1)
                        for case in fixture['cases']]
                for job in as_completed(jobs):
                    for record in job.result():
                        records.append(record)
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                        stream.flush()
                        print(json.dumps({k: v for k, v in record.items() if k != 'result'}), flush=True)
    summary = {'recorded_at': datetime.now(timezone.utc).isoformat(), 'snapshot_id': fixture['snapshot_id'],
               'implementation_sha256': implementation.hexdigest(),
               'model': agentic.settings.answer_openai_model, 'reasoning_effort': 'low',
               'cache': 'No embedding, retrieval or answer cache used by this structural flow; provider prompt caching uncontrolled.',
               'workers': args.workers, 'summaries': summarize(records),
               'cases': [public_record(row) for row in records]}
    (output / ('probe-summary.json' if args.probe else 'summary.json')).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary['summaries']), flush=True)


if __name__ == '__main__':
    main()
