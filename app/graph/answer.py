"""Grounded answer adapter for canonical graph traversal evidence."""

import json
from typing import Literal

from pydantic import BaseModel, Field

from app.graph.models import Edge, GraphEvidenceResult, Issue
from app.grounding.citations import CitationValidator
from app.grounding.generator import GroundedAnswerGenerator
from app.grounding.models import Evidence, REFUSAL_ANSWER, TrustedCitation
from app.retrieval.models import RetrievalCandidate

GraphAnswerStatus = Literal[
    "supported", "partially_supported", "insufficient_evidence", "answer_unavailable"
]
_MAX_TEXT_CHARS = 24_000
_GRAPH_RULES = """Use these Jira graph semantics when answering:
- BLOCKS means a recorded relationship in the snapshot, not proof of a current risk.
- Done / Done relationships are historical unless text evidence explicitly establishes current impact.
- Do not invent a cause or explanation from an edge; causal explanations require text evidence.
- Do not make exhaustive claims when graph coverage is partial or graph_unavailable.
- Distinguish issue status from status category. Preserve the recorded edge direction.
"""


class GraphAnswer(BaseModel):
    answer: str | None
    citations: list[TrustedCitation] = Field(default_factory=list)
    status: GraphAnswerStatus
    diagnostics: list[str] = Field(default_factory=list)


class GraphAnswerer:
    """Translate graph results to the legacy grounding boundary without pruning edges."""

    def __init__(self, generator=None) -> None:
        self._owns_generator = generator is None
        self.generator = generator or GroundedAnswerGenerator()
        self.validator = CitationValidator()

    def answer(
        self,
        query: str,
        result: GraphEvidenceResult,
        structure: dict | None = None,
    ) -> GraphAnswer:
        diagnostics = list(result.diagnostics)
        if result.coverage == "graph_unavailable":
            return GraphAnswer(answer=REFUSAL_ANSWER, citations=[],
                               status="insufficient_evidence",
                               diagnostics=[*diagnostics, "graph_unavailable"])

        evidence, adapter_diagnostics = _evidence(result, structure)
        diagnostics.extend(adapter_diagnostics)
        if not evidence:
            return GraphAnswer(answer=REFUSAL_ANSWER, citations=[],
                               status="insufficient_evidence", diagnostics=diagnostics)

        controlled_query = f"{_GRAPH_RULES}\nGraph coverage: {result.coverage}.\nQUESTION:\n{query}"
        try:
            generated = self.generator.generate(controlled_query, evidence)
            if generated.answer is None or generated.citation_ids is None:
                return GraphAnswer(answer=None, citations=[], status='answer_unavailable',
                                   diagnostics=[*diagnostics, 'generation_unavailable'])
            validated = self.validator.validate(generated, evidence)
        except Exception:
            return GraphAnswer(answer=None, citations=[], status="answer_unavailable",
                               diagnostics=[*diagnostics, "generation_failed"])

        if validated.status == "validation_failed":
            return GraphAnswer(answer=None, citations=[], status="answer_unavailable",
                               diagnostics=[*diagnostics, "citation_validation_failed"])
        if validated.status == "insufficient_evidence":
            return GraphAnswer(answer=validated.answer, citations=[],
                               status="insufficient_evidence", diagnostics=diagnostics)
        status: GraphAnswerStatus = (
            "partially_supported" if (result.coverage == "partial" or result.truncated
                                      or "text_evidence_truncated" in diagnostics)
            else "supported"
        )
        return GraphAnswer(answer=validated.answer, citations=validated.citations,
                           status=status, diagnostics=diagnostics)

    def close(self) -> None:
        """Close a lazily-created OpenAI client owned by this adapter, if present."""
        if not self._owns_generator:
            return
        client = getattr(self.generator, "_client", None)
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def _candidate(*, content: str, document_id: str, chunk_id: str,
               source_url: str, source_type: str, metadata: dict) -> RetrievalCandidate:
    return RetrievalCandidate(content=content, document_id=document_id, chunk_id=chunk_id,
                              source_url=source_url, source_type=source_type,
                              metadata=metadata)


def _item(candidate: RetrievalCandidate, prompt: str, index: int) -> Evidence:
    return Evidence(citation_id=f"E{index}", candidate=candidate, prompt_content=prompt)


def _node_prompt(node: Issue) -> str:
    return (f"Issue {node.key}: issue_id={node.issue_id}; title={node.title!r}; type={node.issue_type}; "
            f"status={node.status}; status_category={node.status_category}.")


def _edge_prompt(edge: Edge, nodes: dict[str, Issue]) -> str:
    source = nodes.get(edge.source_issue_id)
    target = nodes.get(edge.target_issue_id)
    source_name = source.key if source else edge.source_issue_id
    target_name = target.key if target else edge.target_issue_id
    source_state = f" ({source.status} / {source.status_category})" if source else ""
    target_state = f" ({target.status} / {target.status_category})" if target else ""
    origin = nodes.get(edge.origin_issue_id)
    origin_name = origin.key if origin else edge.origin_issue_id
    return (f"Canonical directed Jira edge: {source_name}{source_state} "
            f"{edge.relation_type} {target_name}{target_state}; provenance: origin issue "
            f"{origin_name}, field {edge.source_field}, source type {edge.source_type}, "
            f"recorded direction {edge.source_direction}.")


def _evidence(result: GraphEvidenceResult, structure: dict | None) -> tuple[list[Evidence], list[str]]:
    candidates: list[tuple[RetrievalCandidate, str]] = []
    diagnostics: list[str] = []
    nodes = {node.issue_id: node for node in result.nodes}
    for node in result.nodes:
        prompt = _node_prompt(node)
        candidates.append((_candidate(content=prompt, document_id=node.document_id,
                                      chunk_id=f"graph-node:{node.issue_id}",
                                      source_url=node.source_url, source_type="jira_graph_node",
                                      metadata={"issue_id": node.issue_id}), prompt))

    edges: list[Edge] = []
    identities: set[tuple] = set()
    for path in result.paths:
        for edge in path.edges:
            identity = (edge.source_issue_id, edge.target_issue_id, edge.relation_type,
                        edge.origin_issue_id, edge.source_field, edge.source_type,
                        edge.source_direction, edge.link_id)
            if identity not in identities:
                identities.add(identity)
                edges.append(edge)
    for position, edge in enumerate(edges, start=1):
        prompt = _edge_prompt(edge, nodes)
        origin = nodes.get(edge.origin_issue_id)
        candidates.append((_candidate(
            content=prompt, document_id=origin.document_id if origin else f"graph:{edge.origin_issue_id}",
            chunk_id=f"graph-edge:{position}", source_url=origin.source_url if origin else "",
            source_type="jira_graph_edge",
            metadata={"source_issue_id": edge.source_issue_id,
                      "target_issue_id": edge.target_issue_id,
                      "relation_type": edge.relation_type,
                      "origin_issue_id": edge.origin_issue_id}), prompt))

    remaining = _MAX_TEXT_CHARS
    for text in result.text_evidence:
        if remaining <= 0:
            diagnostics.append("text_evidence_character_budget_exhausted")
            break
        bounded = text.text[:remaining]
        remaining -= len(bounded)
        if len(bounded) < len(text.text):
            diagnostics.append("text_evidence_truncated")
        candidates.append((_candidate(content=text.text, document_id=text.document_id,
                                      chunk_id=f"text:{text.chunk_id}", source_url=text.source_url,
                                      source_type="jira_text", metadata={"issue_id": text.issue_id,
                                                                         "chunk_id": text.chunk_id}),
                           bounded))

    if structure is not None:
        prompt = "Deterministic structural aggregate: " + json.dumps(
            structure, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        project_url = (f"{result.scope.site_id.rstrip('/')}"
                       f"/jira/software/projects/{result.scope.project_key}")
        candidates.append((_candidate(content=prompt, document_id=f"graph:{result.scope.snapshot_id}",
                                      chunk_id="graph-structure", source_url=project_url,
                                      source_type="jira_graph_structure",
                                      metadata={"site_id": result.scope.site_id,
                                                "project_key": result.scope.project_key,
                                                "snapshot_id": result.scope.snapshot_id}), prompt))
    return [_item(candidate, prompt, index)
            for index, (candidate, prompt) in enumerate(candidates, start=1)], diagnostics
