from typing import Any

from pydantic import BaseModel, Field


def canonical_jira_key(metadata: dict[str, Any]) -> str:
    """Return the one canonical Jira-key accessor while preserving legacy names."""
    return str(metadata.get("issue_key") or metadata.get("key") or "")


class RetrievalCandidate(BaseModel):
    content: str
    document_id: str
    chunk_id: str
    source_type: str = ""
    source_url: str = ""
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0
    retrieval_methods: list[str] = Field(default_factory=list)
    rank_by_method: dict[str, int] = Field(default_factory=dict)
    fused_rank: int | None = None
    rerank_score: float | None = None
    exact_match: bool = False
    collection_name: str = ""
