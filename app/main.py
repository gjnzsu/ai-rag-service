from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.api.lifecycle import router as lifecycle_router
from app.api.query import router as query_router
from app.config import settings
from app.rag import query_engine
from app.model_access import CONTEXT_HEADERS, request_context
from uuid import uuid4

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

    @app.middleware('http')
    async def correlate_model_requests(request, call_next):
        context = {key: request.headers[key] for key in CONTEXT_HEADERS if key in request.headers}
        context.setdefault('x-request-id', str(uuid4()))
        token = request_context.set(context)
        try:
            response = await call_next(request)
            response.headers['x-request-id'] = context['x-request-id']
            return response
        finally:
            request_context.reset(token)

    app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
    app.include_router(lifecycle_router, tags=["lifecycle"])
    app.include_router(query_router, tags=["query"])
    if settings.agentic_rag_enabled:
        from app.api.agentic import router as agentic_router
        app.include_router(agentic_router, prefix="/agentic", tags=["agentic"])
    if settings.graph_enabled:
        from app.api.graph import router as graph_router
        app.include_router(graph_router, prefix="/graph", tags=["graph"])
        if settings.graph_demo_enabled:
            from app.graph.demo import router as demo_router
            app.include_router(demo_router, prefix="/graph")

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
