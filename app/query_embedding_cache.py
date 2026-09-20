"""Process-local, exact-input query vectors. Never cache retrieval or answers."""

from collections import OrderedDict
from hashlib import sha256
import json
import math
import os
from threading import Lock
from time import monotonic, perf_counter
from typing import Callable

import structlog

from app.config import settings

logger = structlog.get_logger()
MODEL = 'text-embedding-3-small'
DIMENSIONS = 1536


class QueryEmbeddingCache:
    def __init__(self, max_entries=1024, ttl_seconds=900, clock=monotonic):
        if max_entries <= 0 or not math.isfinite(ttl_seconds) or ttl_seconds <= 0:
            raise ValueError('Cache capacity and TTL must be positive and finite')
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._entries = OrderedDict()
        self._lock = Lock()
        self._counts = dict(hits=0, misses=0, evictions=0, expirations=0, errors=0)

    def embed(self, text: str, *, scope: tuple[str, ...], model: str,
              deployment: str, dimensions: int, load: Callable[[], list[float]],
              preprocessing_version: str = 'identity-v1') -> list[float]:
        started = perf_counter()
        key = sha256(json.dumps(
            [scope, deployment, model, dimensions, preprocessing_version, text],
            ensure_ascii=False, separators=(',', ':'),
        ).encode('utf-8')).digest()
        try:
            vector = self._get(key)
        except Exception:
            self._error()
            vector = None
        if vector is not None:
            logger.info('query_embedding_cache', status='hit', elapsed_ms=(perf_counter()-started)*1000)
            return list(vector)

        # Outside both the cache lock and cache error handlers: provider failures propagate.
        vector = load()
        try:
            if len(vector) == dimensions and all(math.isfinite(v) for v in vector):
                self._put(key, tuple(vector))
        except Exception:
            self._error()
        logger.info('query_embedding_cache', status='miss', elapsed_ms=(perf_counter()-started)*1000)
        return vector

    def _get(self, key):
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                expires, vector = entry
                if self._clock() < expires:
                    self._entries.move_to_end(key)
                    self._counts['hits'] += 1
                    return vector
                del self._entries[key]
                self._counts['expirations'] += 1
            self._counts['misses'] += 1
            return None

    def _put(self, key, vector):
        with self._lock:
            now = self._clock()
            expired = [k for k, (expires, _) in self._entries.items() if expires <= now]
            for old_key in expired:
                del self._entries[old_key]
            self._counts['expirations'] += len(expired)
            self._entries[key] = (now + self.ttl_seconds, vector)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
                self._counts['evictions'] += 1

    def _error(self):
        with self._lock:
            self._counts['errors'] += 1
        logger.warning('query_embedding_cache_error')

    def stats(self):
        with self._lock:
            return {**self._counts, 'entries': len(self._entries)}


query_embedding_cache = QueryEmbeddingCache(
    max_entries=settings.query_embedding_cache_max_entries,
    ttl_seconds=settings.query_embedding_cache_ttl_seconds,
)


def embed_query(text: str, *, scope: tuple[str, ...], load: Callable[[], list[float]],
                client=None) -> list[float]:
    if not settings.query_embedding_cache_enabled:
        return load()
    endpoint = str(getattr(client, 'base_url', os.getenv('OPENAI_BASE_URL') or 'https://api.openai.com/v1'))
    credential = getattr(client, 'api_key', settings.openai_api_key)
    if not isinstance(credential, str):
        credential = settings.openai_api_key
    deployment = json.dumps([endpoint.rstrip('/'), sha256(credential.encode()).hexdigest()])
    return query_embedding_cache.embed(
        text, scope=scope, model=MODEL, deployment=deployment, dimensions=DIMENSIONS, load=load,
    )
