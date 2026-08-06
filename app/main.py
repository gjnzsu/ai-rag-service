from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.api.lifecycle import router as lifecycle_router
from app.api.query import router as query_router
from app.rag import query_engine

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    del app
    try:
        yield
    finally:
        query_engine.close_default_query_pipeline()


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Service",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
    app.include_router(lifecycle_router, tags=["lifecycle"])
    app.include_router(query_router, tags=["query"])

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
