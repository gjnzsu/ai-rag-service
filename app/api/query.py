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


class SourceItem(BaseModel):
    document_id: str
    source_type: str
    title: str
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
        )
        logger.info("query_served", question=request.question[:80])
        return QueryResponse(**result)
    except Exception as e:
        logger.error("query_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
