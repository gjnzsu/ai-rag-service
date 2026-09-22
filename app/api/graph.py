"""Optional read-only graph API; only query performs answer generation."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query

from app.config import settings
from app.model_access import gateway_options, gateway_http_client
from app.graph.api_models import (DependenciesResponse, EpicDetailResponse, OverviewResponse,
                                  GraphRetrieveRequest, GraphRetrieveResponse, GraphQueryResponse)
from app.graph.retrieval import GraphRetrievalService
from app.graph.models import Direction, RelationType
from app.graph.service import GraphNotFound, GraphService, GraphUnavailable
from app.graph.snapshots import SnapshotRepository
from app.query_embedding_cache import embed_query

router = APIRouter()
SnapshotQuery = Annotated[str | None, Query(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]


def get_graph_service():
    def store_provider():
        from app.graph.store import Neo4jGraphStore
        return Neo4jGraphStore.connect(settings.graph_neo4j_uri, settings.graph_neo4j_user,
                                       settings.graph_neo4j_password.get_secret_value())

    service = GraphService(SnapshotRepository(Path(settings.graph_data_dir)), store_provider,
                           settings.graph_site_url or settings.jira_url, settings.graph_allowed_projects)
    try:
        yield service
    finally:
        service.close()


Service = Annotated[GraphService, Depends(get_graph_service)]


def _respond(call):
    try:
        return call()
    except GraphNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except GraphUnavailable:
        raise HTTPException(status_code=503, detail="Graph snapshot unavailable") from None


@router.get("/projects/{project_key}/overview", response_model=OverviewResponse)
def overview(project_key: str, service: Service, snapshot_id: SnapshotQuery = None):
    return _respond(lambda: service.overview(project_key, snapshot_id))


@router.get("/projects/{project_key}/epics/{epic_key}", response_model=EpicDetailResponse)
def epic_detail(project_key: str, epic_key: str, service: Service, snapshot_id: SnapshotQuery = None,
                offset: Annotated[int, Query(ge=0)] = 0, limit: Annotated[int, Query(ge=1, le=100)] = 50):
    return _respond(lambda: service.epic_detail(project_key, epic_key, snapshot_id, offset, limit))


@router.get("/projects/{project_key}/issues/{issue_key}/dependencies", response_model=DependenciesResponse)
def dependencies(project_key: str, issue_key: str, service: Service, snapshot_id: SnapshotQuery = None,
                 direction: Direction = "both", hops: Annotated[int, Query(ge=1, le=2)] = 1,
                 limit: Annotated[int, Query(ge=1, le=100)] = 100, relation_type: RelationType = "BLOCKS",
                 unresolved_pair: bool = False):
    return _respond(lambda: service.dependencies(project_key, issue_key, snapshot_id, direction,
                                                 hops, limit, relation_type, unresolved_pair))


def get_graph_retrieval_service():
    from app.graph.embedding import OpenAIChunkEmbedder
    from app.graph.retrieval import GraphRetrievalService
    from app.graph.search import SnapshotSearch
    from app.graph.store import Neo4jGraphStore
    clients = []
    repository = SnapshotRepository(Path(settings.graph_data_dir))

    def search_provider(manifest):
        def embed(texts):
            # Match the frozen index, not the default service embedding configuration.
            if not manifest.index_versions['chroma'].endswith('/text-embedding-3-small'):
                raise ValueError('Unsupported snapshot embedding model')
            def load(text):
                if not clients:
                    from openai import OpenAI
                    clients.append(OpenAI(**gateway_options(settings), timeout=15, max_retries=0,
                                          http_client=gateway_http_client()))
                return OpenAIChunkEmbedder(clients[0])([text])[0]

            return [embed_query(
                text, scope=('graph', manifest.scope.site_id, manifest.scope.project_key,
                             manifest.account_scope), load=lambda text=text: load(text),
            ) for text in texts]
        return SnapshotSearch(repository.directory(manifest.scope), manifest.scope, embed)

    service = GraphRetrievalService(repository,
        lambda: Neo4jGraphStore.connect(settings.graph_neo4j_uri, settings.graph_neo4j_user,
                                        settings.graph_neo4j_password.get_secret_value()),
        settings.graph_site_url or settings.jira_url, settings.graph_allowed_projects,
        search_provider=search_provider)
    try:
        yield service
    finally:
        service.close()
        for client in clients:
            try:
                client.close()
            except Exception:
                pass


RetrievalService = Annotated[GraphRetrievalService, Depends(get_graph_retrieval_service)]


@router.post('/retrieve', response_model=GraphRetrieveResponse)
def retrieve(request: GraphRetrieveRequest, service: RetrievalService):
    return _respond(lambda: service.retrieve(request))


@router.post('/query', response_model=GraphQueryResponse)
def query(request: GraphRetrieveRequest, service: RetrievalService):
    from app.graph.answer import GraphAnswerer, GraphAnswer
    evidence = _respond(lambda: service.retrieve(request))
    answerer = None
    try:
        answerer = GraphAnswerer()
        answer = answerer.answer(request.query, evidence.data, evidence.structure)
    except Exception:
        answer = GraphAnswer(answer=None, status='answer_unavailable', diagnostics=['generation_failed'])
    finally:
        if answerer is not None:
            try:
                answerer.close()
            except Exception:
                pass
    return GraphQueryResponse(**evidence.model_dump(), answer=answer)
