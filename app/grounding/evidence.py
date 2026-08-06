"""Deterministic selection of a small, diverse evidence set."""

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.config import settings
from app.grounding.models import Evidence
from app.retrieval.models import RetrievalCandidate

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_ENTITY_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_-]*\b")
_CROSS_DOCUMENT_PATTERN = re.compile(
    r"\b(?:compare|comparison|contrast|across|cross[- ]document|"
    r"multiple\s+(?:sources|documents)|different\s+(?:sources|documents)|"
    r"between\s+(?:sources|documents))\b",
    re.IGNORECASE,
)
_NEGATIONS = frozenset({"no", "not", "never", "without", "cannot", "neither", "nor"})
_NUMBER_WORDS = frozenset({
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety", "hundred", "thousand", "million", "billion",
})
_SIMILARITY_TOKEN_LIMIT = 256
_NEAR_DUPLICATE_RATIO = 0.92


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
        if not 5 <= top_k <= 10:
            raise ValueError("top_k must be between 5 and 10")
        if not candidates:
            return []

        ranked = sorted(enumerate(candidates), key=_ranking_key)
        unique = _deduplicate(candidate for _, candidate in ranked)
        selected = _diversify(unique, top_k) if _should_diversify(query, unique) else unique[:top_k]
        return [
            Evidence(
                citation_id=f"E{index}",
                candidate=candidate.model_copy(deep=True),
                prompt_content=candidate.content[: self.max_prompt_chars],
            )
            for index, candidate in enumerate(selected, start=1)
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


@dataclass(frozen=True)
class _ContentFingerprint:
    normalized_digest: str
    tokens: tuple[str, ...]
    negations: tuple[str, ...]
    numbers: tuple[str, ...]
    entities: tuple[str, ...]
    bounded: bool


def _deduplicate(candidates: Iterable[RetrievalCandidate]) -> list[RetrievalCandidate]:
    selected: list[RetrievalCandidate] = []
    chunk_ids: set[str] = set()
    fingerprints: list[_ContentFingerprint] = []
    for candidate in candidates:
        if candidate.chunk_id in chunk_ids:
            continue
        fingerprint = _fingerprint(candidate.content)
        if any(_near_duplicate(fingerprint, old) for old in fingerprints):
            continue
        chunk_ids.add(candidate.chunk_id)
        fingerprints.append(fingerprint)
        selected.append(candidate)
    return selected


def _fingerprint(content: str) -> _ContentFingerprint:
    contraction_normalized = re.sub(r"n['’]t\b", " not", content, flags=re.IGNORECASE)
    all_tokens = tuple(word.casefold() for word in _WORD_PATTERN.findall(contraction_normalized))
    normalized = " ".join(all_tokens)
    bounded = len(all_tokens) <= _SIMILARITY_TOKEN_LIMIT
    tokens = all_tokens if bounded else all_tokens[:_SIMILARITY_TOKEN_LIMIT]
    return _ContentFingerprint(
        normalized_digest=hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        tokens=tokens,
        negations=tuple(token for token in tokens if token in _NEGATIONS),
        numbers=tuple(token for token in tokens if token.isdigit() or token in _NUMBER_WORDS),
        entities=tuple(match.group(0).casefold() for match in _ENTITY_PATTERN.finditer(content)),
        bounded=bounded,
    )


def _near_duplicate(
    fingerprint: _ContentFingerprint,
    old: _ContentFingerprint,
) -> bool:
    if fingerprint.normalized_digest == old.normalized_digest:
        return True
    if not fingerprint.bounded or not old.bounded:
        return False
    if min(len(fingerprint.tokens), len(old.tokens)) < 5:
        return False
    if (
        fingerprint.negations != old.negations
        or fingerprint.numbers != old.numbers
        or fingerprint.entities != old.entities
    ):
        return False
    matcher = SequenceMatcher(
        None,
        fingerprint.tokens,
        old.tokens,
        autojunk=False,
    )
    if matcher.ratio() < _NEAR_DUPLICATE_RATIO:
        return False
    changes = {tag for tag, *_ in matcher.get_opcodes() if tag != "equal"}
    return "replace" not in changes and changes != {"insert", "delete"}


def _cross_document_intent(query: str) -> bool:
    return bool(_CROSS_DOCUMENT_PATTERN.search(query))


def _should_diversify(query: str, candidates: list[RetrievalCandidate]) -> bool:
    if not _cross_document_intent(query):
        return False
    exact_documents = {
        _source_identity(candidate)
        for candidate in candidates
        if candidate.exact_match
    }
    return len(exact_documents) != 1


def _diversify(candidates: list[RetrievalCandidate], top_k: int) -> list[RetrievalCandidate]:
    if not candidates:
        return []
    first_rank = candidates[0].fused_rank if candidates[0].fused_rank is not None else 1
    rank_window = max(5, top_k)
    eligible: list[RetrievalCandidate] = []
    ineligible: list[RetrievalCandidate] = []
    for position, candidate in enumerate(candidates):
        within_position_window = position < top_k * 2
        within_fused_window = (
            candidate.fused_rank is None
            or candidate.fused_rank - first_rank <= rank_window
        )
        target = eligible if within_position_window and within_fused_window else ineligible
        target.append(candidate)

    first_by_document: list[RetrievalCandidate] = []
    remainder: list[RetrievalCandidate] = []
    seen_documents: set[str] = set()
    for candidate in eligible:
        document_key = _source_identity(candidate)
        if document_key in seen_documents:
            remainder.append(candidate)
        else:
            seen_documents.add(document_key)
            first_by_document.append(candidate)
    return (first_by_document + remainder + ineligible)[:top_k]


def _source_identity(candidate: RetrievalCandidate) -> str:
    return (
        candidate.document_id
        or candidate.source_url
        or candidate.title
        or candidate.chunk_id
    )
