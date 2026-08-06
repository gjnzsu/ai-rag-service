"""Rank-only reciprocal-rank fusion for heterogeneous retrievers."""

from app.retrieval.models import RetrievalCandidate


class ReciprocalRankFusion:
    """Fuse ranked candidate sets without comparing their raw scores."""

    def __init__(self, k: int = 60) -> None:
        if k < 0:
            raise ValueError("RRF k must be non-negative")
        self.k = k

    def fuse(
        self,
        result_sets: list[list[RetrievalCandidate]],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        if top_k <= 0:
            return []

        candidates: dict[str, RetrievalCandidate] = {}
        encounter_order: dict[str, int] = {}
        for result_set in result_sets:
            for position, source in enumerate(result_set, start=1):
                candidate = candidates.get(source.chunk_id)
                if candidate is None:
                    candidate = source.model_copy(deep=True)
                    candidate.retrieval_methods = []
                    candidate.rank_by_method = {}
                    candidate.method_scores = {}
                    candidate.rrf_score = 0.0
                    candidate.fused_rank = None
                    candidates[source.chunk_id] = candidate
                    encounter_order[source.chunk_id] = len(encounter_order)

                methods = source.retrieval_methods or ["unknown"]
                for method in methods:
                    rank = source.rank_by_method.get(method, position)
                    if rank < 1:
                        raise ValueError("RRF ranks must be one-based positive integers")
                    candidate.rrf_score += 1 / (self.k + rank)
                    if method not in candidate.retrieval_methods:
                        candidate.retrieval_methods.append(method)
                    candidate.rank_by_method[method] = rank
                    candidate.method_scores[method] = source.method_scores.get(method, source.score)
                    if method == "exact":
                        candidate.exact_match = True
                candidate.exact_match = candidate.exact_match or source.exact_match

        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                not item.exact_match,
                -item.rrf_score,
                encounter_order[item.chunk_id],
            ),
        )
        for rank, candidate in enumerate(ordered[:top_k], start=1):
            candidate.score = candidate.rrf_score
            candidate.fused_rank = rank
        return ordered[:top_k]
