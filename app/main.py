import structlog
from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.api.query import router as query_router

logger = structlog.get_logger()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Service", version="0.1.0")
    app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
    app.include_router(query_router, tags=["query"])

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
