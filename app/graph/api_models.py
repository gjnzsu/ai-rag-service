"""Public structural response envelopes; every result identifies its pinned snapshot."""

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from app.graph.models import Contract, Direction, RelationType, GraphEvidenceResult, GraphScope
from app.graph.structure import EpicDetailData, OverviewData
from app.graph.answer import GraphAnswer


class GraphEnvelope(Contract):
    snapshot_id: str
    captured_at: AwareDatetime
    scope: GraphScope
    completeness: Literal["visible_project", "partial"]
    diagnostics: list[str] = Field(default_factory=list)


class OverviewResponse(GraphEnvelope):
    data: OverviewData


class EpicDetailResponse(GraphEnvelope):
    data: EpicDetailData


class DependenciesResponse(GraphEnvelope):
    data: GraphEvidenceResult

class GraphRetrieveRequest(Contract):
    project_key: str = Field(min_length=1, max_length=64, pattern=r'^[A-Z][A-Z0-9_]*$')
    query: str = Field(min_length=1, max_length=4000)
    snapshot_id: str | None = Field(default=None, min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    intent: Literal['overview', 'epic_detail', 'dependencies'] = 'dependencies'
    epic_key: str | None = Field(default=None, pattern=r'^[A-Z][A-Z0-9_]*-[0-9]+$')
    issue_key: str | None = Field(default=None, pattern=r'^[A-Z][A-Z0-9_]*-[0-9]+$')
    top_k: int = Field(default=5, ge=1, le=20)
    direction: Direction = 'both'
    hops: int = Field(default=1, ge=1, le=2)
    limit: int = Field(default=100, ge=1, le=100)
    relation_type: RelationType = 'BLOCKS'
    unresolved_pair: bool = False
    text_budget: int = Field(default=16000, ge=1, le=32000)

    @model_validator(mode='after')
    def validate_anchors(self):
        if not self.query.strip():
            raise ValueError('query must not be blank')
        if self.epic_key and self.issue_key:
            raise ValueError('provide only one anchor')
        if self.intent == 'overview' and (self.epic_key or self.issue_key):
            raise ValueError('overview does not take an issue anchor')
        return self


class GraphRetrieveResponse(GraphEnvelope):
    data: GraphEvidenceResult
    intent: Literal['overview', 'epic_detail', 'dependencies']
    seed_method: Literal['explicit', 'hybrid', 'project']
    structure: dict | None = None



class GraphQueryResponse(GraphRetrieveResponse):
    answer: GraphAnswer
