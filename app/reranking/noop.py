"""No-op reranker used by default."""

from app.reranking.base import validate_top_k
from app.retrieval.models import RetrievalCandidate


class NoOpReranker:
    """Preserve reciprocal-rank-fusion order."""

    provider = "none"

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        del query
        validate_top_k(top_k)
        return [candidate.model_copy(deep=True) for candidate in candidates[:top_k]]
