"""Server-side citation validation boundary."""

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
        seen: set[str] = set()
        selected: list[Evidence] = []
        invalid_reference = False
        for citation_id in generated.citation_ids:
            if not isinstance(citation_id, str) or citation_id not in known:
                invalid_reference = True
                continue
            if citation_id not in seen:
                seen.add(citation_id)
                selected.append(known[citation_id])

        if generated.answer == REFUSAL_ANSWER:
            if generated.citation_ids:
                return _validation_failure(answer=REFUSAL_ANSWER)
            return GroundedAnswerResult(
                answer=REFUSAL_ANSWER,
                citations=[],
                status="insufficient_evidence",
            )

        citations = [self._trusted_citation(item) for item in selected]
        if invalid_reference:
            status = "validation_failed"
        elif citations:
            status = "supported"
        else:
            status = "partially_supported"
        return GroundedAnswerResult(
            answer=generated.answer,
            citations=citations,
            status=status,
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


def _validation_failure(answer: str = REFUSAL_ANSWER) -> GroundedAnswerResult:
    return GroundedAnswerResult(
        answer=answer,
        citations=[],
        status="validation_failed",
    )
