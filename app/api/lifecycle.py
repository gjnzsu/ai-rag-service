import hashlib
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field, model_validator

from app.config import settings
from app.connectors.base import Document
from app.pipeline.indexer import index_documents
from app.pipeline.store import (
    delete_document,
    get_document_chunks,
    get_jira_key_context,
    query_lifecycle_collection,
    refresh_jira_key,
)

logger = structlog.get_logger()
router = APIRouter()

SCHEMA_VERSION = "platform-rag-lifecycle/v1"
EMBEDDING_MODEL = "text-embedding-3-small"
SUPPORTED_FILTER_OPERATORS = {"in"}
REQUIRED_METADATA = {
    "jira_issue": {"type", "key", "project_key", "title", "url", "status", "priority"},
    "confluence_page": {"type", "title", "url", "space_key", "related_jira"},
}


FilterValue = str | int | float | bool | dict[str, list[str | int | float | bool]]


class UpsertDocumentRequest(BaseModel):
    content: str = Field(min_length=1)
    metadata: dict[str, Any]
    document_id: str | None = None
    collection: str = "default"

    @model_validator(mode="after")
    def validate_metadata(self):
        doc_type = self.metadata.get("type")
        required = REQUIRED_METADATA.get(doc_type)
        if required:
            missing = sorted(field for field in required if not self.metadata.get(field))
            if missing:
                raise ValueError(
                    f"Missing required metadata fields for {doc_type}: {', '.join(missing)}"
                )
        return self


class UpsertDocumentResponse(BaseModel):
    document_id: str
    ingested_chunks: int
    created: bool


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    collection: str = "default"
    top_k: int = Field(default=5, ge=1, le=50)
    filters: dict[str, FilterValue] | None = None

    @model_validator(mode="after")
    def validate_filters(self):
        validate_filters(self.filters)
        return self


class RetrievalResult(BaseModel):
    content: str
    score: float
    document_id: str
    chunk_id: str
    metadata: dict[str, Any]
    source_url: str = ""


class RetrieveResponse(BaseModel):
    results: list[RetrievalResult]


class DocumentLookupResponse(BaseModel):
    document_id: str
    results: list[RetrievalResult]


class DeleteDocumentResponse(BaseModel):
    document_id: str
    deleted: bool


class ReindexJiraResponse(BaseModel):
    jira_key: str
    refreshed_documents: int


def validate_filters(filters: dict[str, FilterValue] | None) -> None:
    if not filters:
        return
    for field, value in filters.items():
        if isinstance(value, dict):
            unsupported = set(value) - SUPPORTED_FILTER_OPERATORS
            if unsupported:
                raise ValueError(
                    f"Unsupported filter operator for {field}: {', '.join(sorted(unsupported))}"
                )
            if "in" in value and not isinstance(value["in"], list):
                raise ValueError(f"Filter operator 'in' for {field} requires a list")


def embed_text(text: str) -> list[float]:
    client = OpenAI(api_key=settings.openai_api_key)
    return client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
    ).data[0].embedding


@router.post("/documents/upsert", response_model=UpsertDocumentResponse)
def upsert_lifecycle_document(request: UpsertDocumentRequest):
    try:
        document_id = request.document_id or generate_document_id(
            request.metadata,
            request.content,
        )
        document = Document(
            id=document_id,
            content=request.content,
            source_type=str(request.metadata.get("type", "unknown")),
            title=str(request.metadata.get("title", document_id)),
            metadata=build_document_metadata(
                document_id=document_id,
                content=request.content,
                metadata=request.metadata,
            ),
        )
        result = index_documents([document], collection_name=request.collection)
        return UpsertDocumentResponse(
            document_id=result["document_id"],
            ingested_chunks=result["chunk_count"],
            created=result["created"],
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        logger.error("document_upsert_error", error=str(error))
        raise HTTPException(status_code=502, detail=str(error))


@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest):
    try:
        query_embedding = embed_text(request.query)
        results = query_lifecycle_collection(
            query_embedding=query_embedding,
            collection_name=request.collection,
            top_k=request.top_k,
            filters=request.filters,
        )
        return RetrieveResponse(results=results)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except Exception as error:
        logger.error("retrieve_error", error=str(error))
        raise HTTPException(status_code=502, detail=str(error))


@router.get("/documents/{document_id}", response_model=DocumentLookupResponse)
def get_document(document_id: str, collection: str = "default"):
    try:
        return DocumentLookupResponse(
            document_id=document_id,
            results=get_document_chunks(document_id, collection_name=collection),
        )
    except Exception as error:
        logger.error("document_lookup_error", error=str(error))
        raise HTTPException(status_code=502, detail=str(error))


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_lifecycle_document(document_id: str, collection: str = "default"):
    try:
        return DeleteDocumentResponse(
            document_id=document_id,
            deleted=delete_document(document_id, collection_name=collection),
        )
    except Exception as error:
        logger.error("document_delete_error", error=str(error))
        raise HTTPException(status_code=502, detail=str(error))


@router.get("/context/jira/{jira_key}", response_model=RetrieveResponse)
def jira_key_context(jira_key: str, collection: str = "default"):
    try:
        return RetrieveResponse(
            results=get_jira_key_context(jira_key, collection_name=collection),
        )
    except Exception as error:
        logger.error("jira_context_error", error=str(error))
        raise HTTPException(status_code=502, detail=str(error))


@router.post("/context/jira/{jira_key}/reindex", response_model=ReindexJiraResponse)
def reindex_jira_key(jira_key: str, collection: str = "default"):
    try:
        return ReindexJiraResponse(
            jira_key=jira_key,
            refreshed_documents=refresh_jira_key(jira_key, collection_name=collection),
        )
    except Exception as error:
        logger.error("jira_reindex_error", error=str(error))
        raise HTTPException(status_code=502, detail=str(error))


def build_document_metadata(
    *,
    document_id: str,
    content: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    return {
        **metadata,
        "issue_key": metadata.get("issue_key") or metadata.get("key", ""),
        "source_url": _trusted_source_url(metadata),
        "document_id": document_id,
        "content_hash": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "ingested_at": now,
        "updated_at": now,
        "embedding_model": EMBEDDING_MODEL,
        "schema_version": SCHEMA_VERSION,
    }


def generate_document_id(metadata: dict[str, Any], content: str) -> str:
    doc_type = str(metadata.get("type", "document"))
    if doc_type == "jira_issue" and metadata.get("key"):
        return f"jira:{metadata['key']}"
    if doc_type == "confluence_page":
        if metadata.get("page_id"):
            return f"confluence:{metadata['page_id']}"
        if metadata.get("url"):
            return f"confluence:{_short_hash(str(metadata['url']))}"
    title = str(metadata.get("title", "untitled"))
    return f"{doc_type}:{_short_hash(f'{title}:{content}')}"


def _short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _trusted_source_url(metadata: dict[str, Any]) -> str:
    doc_type = str(metadata.get("type", ""))
    if doc_type == "jira_issue" and metadata.get("key"):
        return f"{settings.jira_url.rstrip('/')}/browse/{metadata['key']}"
    if doc_type == "confluence_page" and metadata.get("page_id"):
        return f"{settings.confluence_url.rstrip('/')}/pages/{metadata['page_id']}"
    return ""
