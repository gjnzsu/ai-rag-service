from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.retrieval.models import RetrievalCandidate

REFUSAL_ANSWER = "I don't have enough information to answer this question."
GroundingStatus = Literal[
    "supported",
    "partially_supported",
    "insufficient_evidence",
    "validation_failed",
]


class Evidence(BaseModel):
    """A server-assigned reference to one canonical retrieval candidate."""

    model_config = ConfigDict(frozen=True)

    citation_id: str
    candidate: RetrievalCandidate
    prompt_content: str


class GeneratedAnswer(BaseModel):
    """Untrusted structured fields returned by the generation boundary.

    ``None`` fields are the fail-closed signal for an unusable model response.
    """

    answer: str | None
    citation_ids: list[str] | None


class TrustedCitation(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: str
    source_url: str
    excerpt: str


class GroundedAnswerResult(BaseModel):
    answer: str
    citations: list[TrustedCitation] = Field(default_factory=list)
    status: GroundingStatus
