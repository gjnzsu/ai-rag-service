"""Strict external requests, model actions and evidence-bearing responses."""

from typing import Literal

from pydantic import Field, model_validator

from app.graph.models import Contract, Direction
from app.grounding.models import TrustedCitation

Analysis = Literal['completion', 'bugs', 'blockers', 'unsupported']


class AnalysisRequest(Contract):
    project_key: str = Field(pattern=r'^[A-Z][A-Z0-9_]*$', max_length=64)
    epic_key: str = Field(pattern=r'^[A-Z][A-Z0-9_]*-[0-9]+$', max_length=80)
    question: str = Field(min_length=1, max_length=4000)
    snapshot_id: str | None = Field(default=None, pattern=r'^[A-Za-z0-9_-]+$', max_length=128)

    @model_validator(mode='after')
    def validate_request(self):
        if not self.epic_key.startswith(self.project_key + '-') or not self.question.strip():
            raise ValueError('Epic must belong to the project and question must be nonblank')
        return self


class Decision(Contract):
    action: Literal['get_issue_dependencies', 'finish']
    requested: list[Analysis] = Field(min_length=1, max_length=4)
    blocker_population: Literal['all_stories', 'unfinished_stories']
    blocker_direction: Direction
    issue_keys: list[str] = Field(max_length=20)
    direction: Direction

    @model_validator(mode='after')
    def validate_action(self):
        if len(set(self.issue_keys)) != len(self.issue_keys) or len(set(self.requested)) != len(self.requested):
            raise ValueError('Duplicate action fields')
        if (self.action == 'finish') != (not self.issue_keys):
            raise ValueError('Lookup needs keys; finish must not have keys')
        if self.action != 'finish' and 'blockers' not in self.requested:
            raise ValueError('Dependency lookup requires blocker analysis')
        return self


class Coverage(Contract):
    status: Literal['complete', 'partial', 'unavailable', 'not_requested']
    expected: list[str] = Field(default_factory=list)
    checked: list[str] = Field(default_factory=list)
    failed: list[str] = Field(default_factory=list)
    truncated: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class EvidenceRecord(Contract):
    id: str
    kind: Literal['aggregate', 'issue', 'relationship', 'coverage']
    content: dict
    source_urls: list[str]


class Finding(Contract):
    code: str
    statement: str
    scope: dict
    evidence_ids: list[str]
    limitations: list[str] = Field(default_factory=list)


class Execution(Contract):
    model_config = {'extra': 'forbid', 'protected_namespaces': ()}
    status: Literal['completed', 'partial', 'failed'] = 'partial'
    stop_reason: str = 'not_started'
    tool_calls: int = 0
    backend_calls: int = 0
    database_queries: int = 0
    decision_calls: int = 0
    decision_rounds: int = 0
    generation_calls: int = 0
    elapsed_ms: float = 0
    model_usage: list[dict] = Field(default_factory=list)
    trace: list[dict] = Field(default_factory=list)


class AnalysisResponse(Contract):
    subject: dict
    snapshot: dict
    facts: dict
    requested: list[Analysis]
    findings: list[Finding]
    evidence: list[EvidenceRecord]
    coverage: dict[str, Coverage]
    execution: Execution
    answer: str | None = None
    citations: list[TrustedCitation] = Field(default_factory=list)
    generation_status: Literal['not_requested', 'supported', 'unavailable'] = 'not_requested'
