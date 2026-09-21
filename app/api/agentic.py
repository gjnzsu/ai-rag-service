"""Optional platform API. The agent is an internal retrieval strategy."""

from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from app.agentic.coordinator import Coordinator
from app.agentic.model import ModelDecider, ReportGenerator
from app.agentic.models import AnalysisRequest, AnalysisResponse
from app.agentic.tools import EpicTools, ReadBudget, TimedGraphStore
from app.config import settings
from app.graph.service import GraphNotFound, GraphService, GraphUnavailable
from app.graph.snapshots import SnapshotRepository

router = APIRouter()


def get_coordinator(request: AnalysisRequest):
    import httpx
    from neo4j import GraphDatabase
    from openai import OpenAI

    budget = ReadBudget()

    def store_provider():
        budget.check()
        driver = GraphDatabase.driver(settings.graph_neo4j_uri,
            auth=(settings.graph_neo4j_user, settings.graph_neo4j_password.get_secret_value()),
            connection_timeout=min(5., budget.remaining()), connection_acquisition_timeout=min(5., budget.remaining()),
            max_transaction_retry_time=0)
        return TimedGraphStore(driver, budget)

    service = GraphService(SnapshotRepository(Path(settings.graph_data_dir)), store_provider,
                           settings.graph_site_url or settings.jira_url, settings.graph_allowed_projects)
    client = None
    try:
        client = OpenAI(api_key=settings.openai_api_key, max_retries=0, timeout=15,
                        http_client=httpx.Client())
        yield Coordinator(EpicTools(service, request, budget=budget),
                          ModelDecider(client, settings.answer_openai_model),
                          ReportGenerator(client, settings.answer_openai_model))
    finally:
        service.close()
        if client is not None:
            client.close()


@router.post('/query', response_model=AnalysisResponse)
def query(coordinator: Annotated[Coordinator, Depends(get_coordinator)]):
    try:
        return coordinator.run()
    except GraphNotFound as error:
        raise HTTPException(status_code=404, detail=str(error)) from None
    except (GraphUnavailable, TimeoutError):
        raise HTTPException(status_code=503, detail='Epic snapshot unavailable') from None
