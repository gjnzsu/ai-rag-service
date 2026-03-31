import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.connectors.confluence import ConfluenceConnector
from app.connectors.fx import FXConnector
from app.connectors.jira import JiraConnector
from app.connectors.pdf import PDFConnector
from app.pipeline.chunker import chunk_documents
from app.pipeline.embedder import embed_chunks
from app.pipeline.store import upsert_chunks

logger = structlog.get_logger()
router = APIRouter()


# --- PDF ---

class IngestResponse(BaseModel):
    ingested_chunks: int
    document_id: str


@router.post("/pdf", response_model=IngestResponse)
def ingest_pdf(
    file: UploadFile = File(...),
    collection: str = "default",
):
    if not (file.filename or "").endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        connector = PDFConnector()
        documents = connector.fetch(file_path=tmp_path)
        for doc in documents:
            doc.metadata["filename"] = file.filename
        chunks = chunk_documents(documents)
        embeddings = embed_chunks(chunks)
        upsert_chunks(chunks, embeddings, collection_name=collection)
        logger.info("pdf_ingested", filename=file.filename, chunks=len(chunks))
        return IngestResponse(ingested_chunks=len(chunks), document_id=documents[0].id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error("pdf_ingest_error", error=str(e))
        raise HTTPException(status_code=422, detail=f"PDF parsing failed: {e}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)


# --- Jira ---

class JiraIngestRequest(BaseModel):
    project_key: str
    max_results: int = 100
    collection: str = "default"


class JiraIngestResponse(BaseModel):
    ingested_chunks: int
    issues_fetched: int


@router.post("/jira", response_model=JiraIngestResponse)
def ingest_jira(request: JiraIngestRequest):
    try:
        connector = JiraConnector()
        documents = connector.fetch(
            project_key=request.project_key,
            max_results=request.max_results,
        )
        chunks = chunk_documents(documents)
        embeddings = embed_chunks(chunks)
        upsert_chunks(chunks, embeddings, collection_name=request.collection)
        logger.info("jira_ingested", project=request.project_key, issues=len(documents))
        return JiraIngestResponse(ingested_chunks=len(chunks), issues_fetched=len(documents))
    except Exception as e:
        logger.error("jira_ingest_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))


# --- Confluence ---

class ConfluenceIngestRequest(BaseModel):
    space_key: str
    max_pages: int = 50
    collection: str = "default"


class ConfluenceIngestResponse(BaseModel):
    ingested_chunks: int
    pages_fetched: int


@router.post("/confluence", response_model=ConfluenceIngestResponse)
def ingest_confluence(request: ConfluenceIngestRequest):
    try:
        connector = ConfluenceConnector()
        documents = connector.fetch(
            space_key=request.space_key,
            max_pages=request.max_pages,
        )
        chunks = chunk_documents(documents)
        embeddings = embed_chunks(chunks)
        upsert_chunks(chunks, embeddings, collection_name=request.collection)
        logger.info("confluence_ingested", space=request.space_key, pages=len(documents))
        return ConfluenceIngestResponse(ingested_chunks=len(chunks), pages_fetched=len(documents))
    except Exception as e:
        logger.error("confluence_ingest_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))


# --- FX ---

class FXIngestRequest(BaseModel):
    base_currency: str = "USD"
    date_str: str | None = None
    collection: str = "default"


class FXIngestResponse(BaseModel):
    ingested_chunks: int
    rates_count: int


@router.post("/fx", response_model=FXIngestResponse)
def ingest_fx(request: FXIngestRequest):
    try:
        connector = FXConnector()
        documents = connector.fetch(
            base_currency=request.base_currency,
            date_str=request.date_str,
        )
        chunks = chunk_documents(documents)
        embeddings = embed_chunks(chunks)
        upsert_chunks(chunks, embeddings, collection_name=request.collection)
        rates_count = int(documents[0].metadata.get("rates_count", 0))
        logger.info("fx_ingested", currency=request.base_currency, chunks=len(chunks))
        return FXIngestResponse(ingested_chunks=len(chunks), rates_count=rates_count)
    except Exception as e:
        logger.error("fx_ingest_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
