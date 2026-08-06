"""Deterministic selection of a small, diverse evidence set."""

import re
from collections.abc import Iterable

from app.config import settings
from app.grounding.models import Evidence
from app.retrieval.models import RetrievalCandidate

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


class EvidenceSelector:
    def __init__(self, *, max_prompt_chars: int | None = None) -> None:
        self.max_prompt_chars = (
            settings.grounding_prompt_max_chars if max_prompt_chars is None else max_prompt_chars
        )
        if self.max_prompt_chars <= 0:
            raise ValueError("max_prompt_chars must be positive")

    def select(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[Evidence]:
        del query  # Reserved for calibrated query-aware selection rules.
        if not 5 <= top_k <= 10:
            raise ValueError("top_k must be between 5 and 10")
        if not candidates:
            return []

        ranked = sorted(enumerate(candidates), key=_ranking_key)
        unique = _deduplicate(candidate for _, candidate in ranked)
        diverse = _diversify(unique, top_k)
        return [
            Evidence(
                citation_id=f"E{index}",
                candidate=candidate.model_copy(deep=True),
                prompt_content=candidate.content[: self.max_prompt_chars],
            )
            for index, candidate in enumerate(diverse, start=1)
        ]


def _ranking_key(item: tuple[int, RetrievalCandidate]) -> tuple[float | int, ...]:
    position, candidate = item
    has_rerank = candidate.rerank_score is not None
    return (
        0 if candidate.exact_match else 1,
        0 if has_rerank else 1,
        -(candidate.rerank_score or 0.0) if has_rerank else 0.0,
        candidate.fused_rank if candidate.fused_rank is not None else position + 1,
        position,
    )


def _deduplicate(candidates: Iterable[RetrievalCandidate]) -> list[RetrievalCandidate]:
    selected: list[RetrievalCandidate] = []
    chunk_ids: set[str] = set()
    fingerprints: list[tuple[str, frozenset[str]]] = []
    for candidate in candidates:
        if candidate.chunk_id in chunk_ids:
            continue
        normalized, tokens = _fingerprint(candidate.content)
        if any(_near_duplicate(normalized, tokens, old_text, old_tokens) for old_text, old_tokens in fingerprints):
            continue
        chunk_ids.add(candidate.chunk_id)
        fingerprints.append((normalized, tokens))
        selected.append(candidate)
    return selected


def _fingerprint(content: str) -> tuple[str, frozenset[str]]:
    words = [word.casefold() for word in _WORD_PATTERN.findall(content)]
    return " ".join(words), frozenset(words)


def _near_duplicate(
    text: str,
    tokens: frozenset[str],
    old_text: str,
    old_tokens: frozenset[str],
) -> bool:
    if text == old_text:
        return True
    if min(len(tokens), len(old_tokens)) < 5:
        return False
    union = tokens | old_tokens
    return bool(union) and len(tokens & old_tokens) / len(union) >= 0.9


def _diversify(candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
    first_by_document: list[RetrievalCandidate] = []
    remainder: list[RetrievalCandidate] = []
    seen_documents: set[str] = set()
    for candidate in candidates:
        document_key = (
            candidate.document_id
            or candidate.source_url
            or candidate.title
            or candidate.chunk_id
        )
        if document_key in seen_documents:
            remainder.append(candidate)
        else:
            seen_documents.add(document_key)
            first_by_document.append(candidate)
    return (first_by_document + remainder)[:top_k]
