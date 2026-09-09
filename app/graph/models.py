"""Driver-independent contracts for isolated Jira graph snapshots."""

from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

Identifier = Annotated[str, Field(min_length=1)]
RelationType = Literal["CHILD_OF", "BLOCKS", "RELATED_TO", "UNKNOWN"]
Direction = Literal["inbound", "outbound", "both"]
StageStatus = Literal["pending", "ready", "failed"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GraphScope(Contract):
    model_config = ConfigDict(extra="forbid", frozen=True)
    site_id: Identifier
    project_key: Identifier
    snapshot_id: Identifier


class SnapshotManifest(Contract):
    scope: GraphScope
    account_scope: Identifier
    capture_started_at: AwareDatetime
    capture_finished_at: AwareDatetime
    source_checksum: Identifier
    normalization_version: Identifier
    index_versions: dict[Literal["graph", "chroma", "fts"], Identifier]
    stages: dict[Literal["graph", "chroma", "fts"], StageStatus]
    completeness: Literal["visible_project", "partial"] = "visible_project"
    diagnostics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_window(self) -> Self:
        if self.capture_finished_at < self.capture_started_at:
            raise ValueError("capture window ends before it starts")
        return self

    @property
    def ready(self) -> bool:
        return self.completeness == "visible_project" and all(
            self.stages.get(index) == "ready" and bool(self.index_versions.get(index))
            for index in ("graph", "chroma", "fts")
        )


class Issue(Contract):
    scope: GraphScope
    issue_id: Identifier
    key: Identifier
    issue_type: Identifier
    title: str
    status: Identifier
    status_category: Identifier
    source_url: Identifier
    updated_at: AwareDatetime
    document_id: Identifier

    @property
    def business_identity(self) -> tuple[str, str]:
        return self.scope.site_id, self.issue_id

    @property
    def storage_identity(self) -> tuple[str, str, str]:
        return self.scope.site_id, self.issue_id, self.scope.snapshot_id


class Edge(Contract):
    scope: GraphScope
    source_issue_id: Identifier
    target_issue_id: Identifier
    relation_type: RelationType
    origin_issue_id: Identifier
    source_field: Identifier
    source_type: Identifier
    source_direction: Literal["inward", "outward", "parent"]
    link_id: Identifier | None = None


class TraversalRequest(Contract):
    scope: GraphScope
    issue_id: Identifier
    direction: Direction = "both"
    hops: int = Field(default=1, ge=1, le=2)
    limit: int = Field(default=100, ge=1, le=100)
    relation_types: tuple[RelationType, ...] = Field(default=("BLOCKS",), min_length=1)
    unresolved_pair: bool = False


class GraphPath(Contract):
    issue_ids: list[Identifier] = Field(min_length=1)
    edges: list[Edge] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_chain(self) -> Self:
        if len(self.edges) != len(self.issue_ids) - 1:
            raise ValueError("path must contain one edge per hop")
        for index, edge in enumerate(self.edges):
            # Inbound traversal follows the same fact in the reverse direction.
            if {edge.source_issue_id, edge.target_issue_id} != set(self.issue_ids[index:index + 2]):
                raise ValueError("path edge does not connect adjacent issues")
        return self


class TextEvidence(Contract):
    scope: GraphScope
    issue_id: Identifier
    document_id: Identifier
    chunk_id: Identifier
    text: str
    source_url: Identifier


class GraphEvidenceResult(Contract):
    scope: GraphScope
    nodes: list[Issue] = Field(default_factory=list)
    seeds: list[Identifier] = Field(default_factory=list)
    paths: list[GraphPath] = Field(default_factory=list)
    text_evidence: list[TextEvidence] = Field(default_factory=list)
    coverage: Literal["complete", "partial", "graph_unavailable"] = "complete"
    truncated: bool = False
    diagnostics: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        records = [*self.nodes, *self.text_evidence,
                   *(edge for path in self.paths for edge in path.edges)]
        if any(record.scope != self.scope for record in records):
            raise ValueError("all evidence must belong to the result scope")
        return self
