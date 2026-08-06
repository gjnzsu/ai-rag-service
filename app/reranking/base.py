"""Shared reranker interface."""

from typing import Protocol, runtime_checkable

from app.retrieval.models import RetrievalCandidate


@runtime_checkable
class Reranker(Protocol):
    """Order fused retrieval candidates by relevance to a query."""

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]: ...


def validate_top_k(top_k: int) -> None:
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
