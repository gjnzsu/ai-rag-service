"""Run a frozen H/S/G retrieval comparison; no generation or Jira calls."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('OPENAI_API_KEY', 'test-key')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', required=True, help='Existing credential file, read only')
    parser.add_argument('--graph-env-file', default='.env.graph')
    parser.add_argument('--questions', default='docs/evaluation/jira-graph-questions.json')
    parser.add_argument('--output', default='docs/evaluation/jira-graph-comparison.json')
    args = parser.parse_args()
    from dotenv import dotenv_values
    import httpx
    from openai import OpenAI
    from app.graph.embedding import OpenAIChunkEmbedder
    from app.graph.evaluation import evaluate_cases
    from app.graph.models import Issue, Edge
    from app.graph.retrieval import GraphRetrievalService
    from app.graph.search import SnapshotSearch
    from app.graph.snapshots import SnapshotRepository
    from app.graph.store import Neo4jGraphStore

    dataset = json.loads(Path(args.questions).read_text(encoding='utf-8'))
    provenance = dataset['provenance']
    raw = Path(provenance['normalized_path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != provenance['normalized_sha256']:
        raise ValueError('Frozen oracle checksum mismatch')
    source = json.loads(raw)
    repository = SnapshotRepository(Path('data/graph-poc'))
    manifest = repository.resolve(provenance['site_id'], provenance['project_key'], dataset['snapshot_id'])
    if not manifest.index_versions['chroma'].endswith('/text-embedding-3-small'):
        raise ValueError('Unsupported frozen embedding model')
    credentials = dotenv_values(args.env_file)
    graph_credentials = dotenv_values(args.graph_env_file)
    issues = [Issue.model_validate(x) for x in source['issues']]
    edges = [Edge.model_validate(x) for x in source['edges']]
    store = Neo4jGraphStore.connect('bolt://127.0.0.1:7687', 'neo4j', graph_credentials['GRAPH_NEO4J_PASSWORD'])
    try:
        with OpenAI(api_key=credentials['OPENAI_API_KEY'], timeout=20, max_retries=0, http_client=httpx.Client()) as client:
            embedder = OpenAIChunkEmbedder(client)
            questions = list(dict.fromkeys(case['question'] for case in dataset['cases']))
            started = time.perf_counter()
            vectors = embedder(questions)
            embedding_ms = round((time.perf_counter() - started) * 1000, 3)
            cache = dict(zip(questions, vectors, strict=True))
            search = SnapshotSearch(repository.directory(manifest.scope), manifest.scope,
                                    lambda texts: [cache[text] for text in texts])
            service = GraphRetrievalService(repository, lambda: store, manifest.scope.site_id,
                                            (manifest.scope.project_key,), search_provider=lambda _: search)
            # Equalize query embedding and index caches, outside recorded retrieval timings.
            for question in questions:
                search.search(question, top_k=5)
            service.overview(manifest.scope.project_key, manifest.scope.snapshot_id)
            report = evaluate_cases(dataset['cases'], manifest, issues, edges, search, service)
            report['dataset_provenance'] = {**provenance, 'questions_sha256': hashlib.sha256(Path(args.questions).read_bytes()).hexdigest()}
            report['execution'] = {'kind': 'real_snapshot_retrieval_only', 'embedding_model': 'text-embedding-3-small',
                                   'answer_model': 'gpt-5-2025-08-07', 'generation_enabled': False,
                                   'query_embedding_tokens': embedder.total_tokens, 'query_embedding_batch_ms': embedding_ms,
                                   'embedded_questions': len(questions), 'embedding_cache': 'prewarmed for all arms',
                                   'concurrency': 1, 'runs_per_case_per_method': 1, 'actual_billed_cost': None,
                                   'limitations': ['No successful answer-model generation; no human-rated answer correctness.',
                                                   'S filters canonical records in memory; G uses Neo4j. Not a database performance comparison.',
                                                   'Evidence coverage is not answer accuracy. Structured edges cannot measure H reasoning over prose.',
                                                   'Warm varied-query timings exclude embedding, index initialization and scoring.']}
            report['summary'] = {}
            for method in ('H', 'S', 'G'):
                rows = [case['methods'][method] for case in report['cases']]
                durations = sorted(row['latency_ms'] for row in rows)
                report['summary'][method] = {
                    'cases': len(rows), 'errors': sum(row['status'] == 'error' for row in rows),
                    'mean_evidence_node_coverage': statistics.mean(row['evidence_node_coverage'] for row in rows if row['evidence_node_coverage'] is not None),
                    'mean_structured_edge_coverage': statistics.mean(row['structured_edge_coverage'] for row in rows if row['structured_edge_coverage'] is not None),
                    'exact_count_cases': sum(row['counts_match'] is True for row in rows),
                    'annotated_count_cases': sum(row['expected_counts'] is not None for row in rows),
                    'p50_ms': statistics.median(durations), 'p95_nearest_rank_ms': durations[math.ceil(.95 * len(durations)) - 1],
                    'negative_checks_passed': sum(row['negative_check'] is True for row in rows),
                    'negative_checks_scored': sum(row['negative_check'] is not None for row in rows),
                }
            Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
            print(json.dumps({'summary': report['summary'], 'embedding_tokens': embedder.total_tokens}))
    finally:
        store.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'failed', 'error_type': type(error).__name__}), file=sys.stderr)
        raise SystemExit(1) from None
