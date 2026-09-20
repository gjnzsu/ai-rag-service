"""Compare uncached/cold/warm query embedding calls; no retrieval or generation."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.query_embedding_cache import DIMENSIONS, MODEL, QueryEmbeddingCache  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='Use configured OpenAI credentials (4 paid calls)')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    client = None
    if args.live:
        import httpx
        from openai import OpenAI
        from app.config import settings
        client = OpenAI(api_key=settings.openai_api_key, timeout=15, max_retries=0,
                        http_client=httpx.Client())
    store = QueryEmbeddingCache()
    calls = 0

    def load():
        nonlocal calls
        calls += 1
        if client is not None:
            return client.embeddings.create(model=MODEL, input=['Publish approved model to platform catalog']).data[0].embedding
        time.sleep(0.02)
        return [0.25] * DIMENSIONS

    results = {}
    try:
        for enabled in (False, True):
            before = calls
            timings, vectors = [], []
            for _ in range(3):
                start = time.perf_counter()
                vector = store.embed(
                    'Publish approved model to platform catalog', scope=('experiment',),
                    model=MODEL, dimensions=DIMENSIONS, deployment='experiment', load=load,
                ) if enabled else load()
                timings.append(round((time.perf_counter() - start) * 1000, 3))
                vectors.append(vector)
            results['enabled' if enabled else 'disabled'] = {
                'requests': 3, 'provider_calls': calls-before, 'elapsed_ms': timings,
                'median_ms': statistics.median(timings),
                'repeated_vectors_equal': vectors[0] == vectors[1] == vectors[2],
            }
        results.update(mode='live' if args.live else 'simulated_20ms_provider',
                       boundary='embedding only; not end-to-end RAG latency', cache=store.stats())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(results, indent=2))
    finally:
        if client is not None:
            client.close()


if __name__ == '__main__':
    main()
