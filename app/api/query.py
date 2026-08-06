from typing import Literal

import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.rag import query_engine

logger = structlog.get_logger()
router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    collection: str = "default"
    top_k: int = 5
    document_type: str | None = None


class SourceItem(BaseModel):
    document_id: str
    source_type: str
    title: str
    document_type: str = ""
    excerpt: str
    score: float


class CitationItem(BaseModel):
    citation_id: str
    document_id: str
    chunk_id: str
    source_url: str
    excerpt: str


class GroundingInfo(BaseModel):
    status: Literal[
        "supported",
        "partially_supported",
        "insufficient_evidence",
        "validation_failed",
    ]


class RerankerInfo(BaseModel):
    provider: Literal["none", "openai", "qwen_local", "custom"]
    status: Literal["ok", "fallback", "disabled"]
    error_type: str | None = None

    @field_validator("error_type")
    @classmethod
    def validate_error_type(cls, value: str | None) -> str | None:
        if value is not None and value not in query_engine.SAFE_ERROR_TYPES:
            raise ValueError("Unsupported error type")
        return value


class ExactLookupInfo(BaseModel):
    status: Literal[
        "not_requested",
        "unavailable",
        "partial_failure",
        "ok",
        "no_match",
    ]
    attempted_count: int = Field(ge=0)
    failure_count: int = Field(ge=0)
    match_count: int = Field(ge=0)


class RetrievalMetadata(BaseModel):
    mode: Literal["vector", "lexical", "hybrid"]
    failures: list[Literal["vector", "lexical"]] = Field(default_factory=list)
    reranker: RerankerInfo | None = None
    exact_lookup: ExactLookupInfo | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    model: str
    citations: list[CitationItem] = Field(default_factory=list)
    grounding: GroundingInfo | None = None
    retrieval_metadata: RetrievalMetadata | None = None


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = query_engine.query(
            question=request.question,
            collection_name=request.collection,
            top_k=request.top_k,
            document_type=request.document_type,
        )
        response = QueryResponse(**_normalize_query_result(result))
        logger.info(
            "query_served",
            grounding_status=(
                response.grounding.status if response.grounding is not None else "unavailable"
            ),
            source_count=len(response.sources),
            citation_count=len(response.citations),
        )
        return response
    except Exception as error:
        logger.error("query_error", error_type=query_engine.safe_error_type(error))
        raise HTTPException(status_code=502, detail="Query service unavailable")


def _normalize_query_result(result: dict) -> dict:
    """Normalize the engine's safe empty-reranker sentinel for the typed API model."""
    normalized = dict(result)
    retrieval_metadata = normalized.get("retrieval_metadata")
    if isinstance(retrieval_metadata, dict):
        retrieval_metadata = dict(retrieval_metadata)
        if retrieval_metadata.get("reranker") == {}:
            retrieval_metadata["reranker"] = None
        normalized["retrieval_metadata"] = retrieval_metadata
    return normalized
