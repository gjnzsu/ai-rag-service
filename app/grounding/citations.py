"""Server-side citation validation boundary."""

import re

from app.config import settings
from app.grounding.models import (
    Evidence,
    GeneratedAnswer,
    GroundedAnswerResult,
    REFUSAL_ANSWER,
    TrustedCitation,
)


class CitationValidator:
    def __init__(self, *, excerpt_max_chars: int | None = None) -> None:
        self.excerpt_max_chars = (
            settings.grounding_excerpt_max_chars
            if excerpt_max_chars is None
            else excerpt_max_chars
        )
        if self.excerpt_max_chars <= 0:
            raise ValueError("excerpt_max_chars must be positive")

    def validate(
        self,
        generated: GeneratedAnswer,
        evidence: list[Evidence],
    ) -> GroundedAnswerResult:
        if not evidence:
            return GroundedAnswerResult(
                answer=REFUSAL_ANSWER,
                citations=[],
                status="insufficient_evidence",
            )

        if not _valid_evidence_ids(evidence):
            return _validation_failure()
        if (
            not isinstance(generated.answer, str)
            or not generated.answer.strip()
            or not isinstance(generated.citation_ids, list)
        ):
            return _validation_failure()

        known = {item.citation_id: item for item in evidence}
        if generated.answer == REFUSAL_ANSWER:
            if generated.citation_ids:
                return _validation_failure()
            return GroundedAnswerResult(
                answer=REFUSAL_ANSWER,
                citations=[],
                status="insufficient_evidence",
            )

        if not _valid_citation_binding(generated.answer, generated.citation_ids, set(known)):
            return _validation_failure()

        selected = [known[citation_id] for citation_id in generated.citation_ids]
        citations = [self._trusted_citation(item) for item in selected]
        return GroundedAnswerResult(
            answer=generated.answer,
            citations=citations,
            status="supported",
        )

    def _trusted_citation(self, evidence: Evidence) -> TrustedCitation:
        candidate = evidence.candidate
        return TrustedCitation(
            citation_id=evidence.citation_id,
            document_id=candidate.document_id,
            chunk_id=candidate.chunk_id,
            source_url=candidate.source_url,
            excerpt=candidate.content[: self.excerpt_max_chars],
        )


def _valid_evidence_ids(evidence: list[Evidence]) -> bool:
    return [item.citation_id for item in evidence] == [
        f"E{index}" for index in range(1, len(evidence) + 1)
    ]


_URL_PATTERN = re.compile(r"(?i)(?:https?://|www\.)")
_CITATION_PATTERN = re.compile(r"\[(E\d+)\]")
_CLAIM_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n+")


def _valid_citation_binding(answer: str, citation_ids: list[str], known_ids: set[str]) -> bool:
    if _URL_PATTERN.search(answer):
        return False
    if len(citation_ids) != len(set(citation_ids)):
        return False
    if any(citation_id not in known_ids for citation_id in citation_ids):
        return False

    inline_ids = _CITATION_PATTERN.findall(answer)
    if set(inline_ids) != set(citation_ids) or not inline_ids:
        return False
    claims = [claim.strip() for claim in _CLAIM_BOUNDARY_PATTERN.split(answer) if claim.strip()]
    return bool(claims) and all(_CITATION_PATTERN.search(claim) for claim in claims)


def _validation_failure() -> GroundedAnswerResult:
    return GroundedAnswerResult(
        answer=REFUSAL_ANSWER,
        citations=[],
        status="validation_failed",
    )
