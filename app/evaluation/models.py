"""Typed, offline-only data contracts for evaluation cases and observations."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryType(str, Enum):
    """Supported synthetic or manually-labelled evaluation case categories."""

    EXACT_FACT = "exact_fact"
    CROSS_DOCUMENT = "cross_document"
    HARD_NEGATIVE = "hard_negative"
    UNANSWERABLE = "unanswerable"


class EvaluationCase(BaseModel):
    """A manually-labelled question with document labels and optional chunk refinement."""

    model_config = ConfigDict(frozen=True)

    question: str
    query_type: QueryType
    relevant_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    relevant_chunk_ids: tuple[str, ...] = Field(default_factory=tuple)
    expected_facts: tuple[str, ...] = Field(default_factory=tuple)
    should_abstain: bool = False

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value

    @field_validator("relevant_document_ids", "relevant_chunk_ids", "expected_facts")
    @classmethod
    def validate_nonempty_values(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("labels and expected facts must not contain empty values")
        if len(set(value)) != len(value):
            raise ValueError("labels and expected facts must not contain duplicates")
        return value


class RankedEvaluationResult(BaseModel):
    """One configuration's ranked and grounded outcome for one evaluation case."""

    model_config = ConfigDict(frozen=True)

    ranked_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    ranked_chunk_ids: tuple[str, ...] = Field(default_factory=tuple)
    selected_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    selected_chunk_ids: tuple[str, ...] = Field(default_factory=tuple)
    cited_document_ids: tuple[str, ...] = Field(default_factory=tuple)
    cited_chunk_ids: tuple[str, ...] = Field(default_factory=tuple)
    refused: bool = False
    latency_ms: float = 0.0
    answer_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    local_cpu_percent: float | None = None
    local_memory_mb: float | None = None
    human_citation_correctness: bool | None = None

    @field_validator("latency_ms", "answer_latency_ms", "local_cpu_percent", "local_memory_mb")
    @classmethod
    def validate_nonnegative_numbers(cls, value: float | None) -> float | None:
        if value is not None and value < 0:
            raise ValueError("timing and local resource observations must be non-negative")
        return value

    @field_validator("input_tokens", "output_tokens")
    @classmethod
    def validate_nonnegative_tokens(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError("token usage must be non-negative")
        return value
