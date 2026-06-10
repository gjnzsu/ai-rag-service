import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

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


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    model: str


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = query_engine.query(
            question=request.question,
            collection_name=request.collection,
            top_k=request.top_k,
            document_type=request.document_type,
        )
        logger.info("query_served", question=request.question[:80], document_type=request.document_type)
        return QueryResponse(**result)
    except Exception as e:
        logger.error("query_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
