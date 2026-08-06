"""Hybrid retrieval orchestration with safe, deterministic degradation."""

from typing import Any, Callable, Literal

import structlog
from pydantic import BaseModel, Field

from app.config import settings
from app.pipeline.store import get_jira_key_context
from app.retrieval.fusion import MAX_FUSED_CANDIDATES, ReciprocalRankFusion
from app.retrieval.interfaces import LexicalRetriever, VectorRetriever
from app.retrieval.lexical import SQLiteFTSIndex
from app.retrieval.models import RetrievalCandidate
from app.retrieval.query_hints import extract_query_hints
from app.retrieval.vector import ChromaVectorRetriever

logger = structlog.get_logger()

RetrievalMode = Literal["vector", "lexical", "hybrid"]
ExactLookup = Callable[[str, str, dict[str, Any] | None], list[RetrievalCandidate] | list[dict[str, Any]]]


class RetrievalUnavailableError(RuntimeError):
    """Every configured primary retrieval backend failed by exception."""


class RetrievalResult(BaseModel):
    candidates: list[RetrievalCandidate] = Field(default_factory=list)
    retrieval_mode: RetrievalMode
    failures: list[str] = Field(default_factory=list)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class HybridRetrievalPipeline:
    """Retrieve from one or both primary indexes plus exact Jira context."""

    def __init__(
        self,
        vector_retriever: VectorRetriever | None = None,
        lexical_retriever: LexicalRetriever | None = None,
        *,
        mode: str | None = None,
        exact_lookup: ExactLookup | None = None,
        candidate_top_k: int | None = None,
        final_top_k: int | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self.mode = settings.retrieval_mode if mode is None else mode
        if self.mode not in {"vector", "lexical", "hybrid"}:
            raise ValueError("Unsupported retrieval mode")
        self.candidate_top_k = (
            settings.retrieval_candidate_top_k if candidate_top_k is None else candidate_top_k
        )
        self.final_top_k = settings.retrieval_final_top_k if final_top_k is None else final_top_k
        self.rrf_k = rrf_k if rrf_k is not None else settings.retrieval_rrf_k
        if self.candidate_top_k <= 0 or self.final_top_k <= 0 or self.rrf_k < 0:
            raise ValueError("Retrieval limits must be positive and RRF k non-negative")
        if self.final_top_k > MAX_FUSED_CANDIDATES:
            raise ValueError(f"final_top_k must not exceed {MAX_FUSED_CANDIDATES}")
        self.vector_retriever = (
            vector_retriever if vector_retriever is not None
            else ChromaVectorRetriever() if self.mode in {"vector", "hybrid"} else None
        )
        self.lexical_retriever = (
            lexical_retriever if lexical_retriever is not None
            else SQLiteFTSIndex() if self.mode in {"lexical", "hybrid"} else None
        )
        self.exact_lookup = exact_lookup if exact_lookup is not None else _exact_jira_lookup

    def retrieve(
        self,
        query: str,
        collection_name: str = "default",
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        requested_top_k = min(top_k if top_k is not None else self.final_top_k, self.final_top_k)
        if requested_top_k < 0:
            raise ValueError("top_k must be non-negative")
        primary_names = _primary_names(self.mode)
        result_sets: list[list[RetrievalCandidate]] = []
        successful: list[str] = []
        empty: list[str] = []
        failures: list[str] = []

        for name in primary_names:
            retriever = self.vector_retriever if name == "vector" else self.lexical_retriever
            assert retriever is not None
            try:
                candidates = retriever.search(query, self.candidate_top_k, filters, collection_name)
            except Exception as error:
                failures.append(name)
                logger.warning(
                    "retrieval_path_unavailable",
                    retrieval_method=name,
                    collection_name=collection_name,
                    error_type=type(error).__name__,
                )
                continue
            successful.append(name)
            if candidates:
                result_sets.append(candidates)
            else:
                empty.append(name)

        exact_keys = extract_query_hints(query).jira_keys
        exact_candidates, exact_failure_count = self._exact_candidates(exact_keys, collection_name, filters)
        if exact_candidates:
            result_sets.append(exact_candidates)

        if len(failures) == len(primary_names):
            raise RetrievalUnavailableError("All configured retrieval backends are unavailable")

        fused = ReciprocalRankFusion(k=self.rrf_k).fuse(result_sets, requested_top_k)
        diagnostics: dict[str, Any] = {
            "configured_retrievers": primary_names,
            "successful_retrievers": successful,
            "empty_retrievers": empty,
            "failed_retrievers": failures,
            "exact_lookup": _exact_diagnostics(exact_keys, exact_candidates, exact_failure_count),
        }
        return RetrievalResult(
            candidates=fused,
            retrieval_mode=self.mode,
            failures=failures,
            diagnostics=diagnostics,
        )

    def _exact_candidates(
        self,
        jira_keys: list[str],
        collection_name: str,
        filters: dict[str, Any] | None,
    ) -> tuple[list[RetrievalCandidate], int]:
        candidates: list[RetrievalCandidate] = []
        seen: set[str] = set()
        failure_count = 0
        for jira_key in jira_keys:
            try:
                records = self.exact_lookup(jira_key, collection_name, filters)
            except Exception as error:
                failure_count += 1
                logger.warning(
                    "exact_jira_lookup_unavailable",
                    collection_name=collection_name,
                    error_type=type(error).__name__,
                )
                continue
            for record in records:
                candidate = _exact_candidate(record, collection_name)
                if candidate.chunk_id not in seen:
                    seen.add(candidate.chunk_id)
                    candidates.append(candidate)
        for rank, candidate in enumerate(candidates, start=1):
            candidate.rank_by_method = {"exact": rank}
        return candidates[: self.candidate_top_k], failure_count


def _primary_names(mode: RetrievalMode) -> list[str]:
    if mode == "hybrid":
        return ["vector", "lexical"]
    return [mode]


def _exact_diagnostics(
    jira_keys: list[str],
    candidates: list[RetrievalCandidate],
    failure_count: int,
) -> dict[str, int | str]:
    attempted_count = len(jira_keys)
    if not attempted_count:
        status = "not_requested"
    elif failure_count == attempted_count:
        status = "unavailable"
    elif failure_count:
        status = "partial_failure"
    elif candidates:
        status = "ok"
    else:
        status = "no_match"
    return {
        "status": status,
        "attempted_count": attempted_count,
        "failure_count": failure_count,
        "match_count": len(candidates),
    }


def _exact_jira_lookup(
    jira_key: str,
    collection_name: str,
    filters: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    return get_jira_key_context(jira_key, collection_name, filters)


def _exact_candidate(record: RetrievalCandidate | dict[str, Any], collection_name: str) -> RetrievalCandidate:
    if isinstance(record, RetrievalCandidate):
        candidate = record.model_copy(deep=True)
    else:
        metadata = dict(record.get("metadata") or {})
        candidate = RetrievalCandidate(
            content=str(record.get("content", "")),
            chunk_id=str(record.get("chunk_id") or metadata.get("chunk_id") or metadata.get("id") or ""),
            document_id=str(record.get("document_id") or metadata.get("document_id") or ""),
            source_type=str(record.get("source_type") or metadata.get("source_type") or metadata.get("type") or ""),
            source_url=str(record.get("source_url") or metadata.get("source_url") or ""),
            title=str(record.get("title") or metadata.get("title") or ""),
            metadata=metadata,
            score=float(record.get("score", 0.0)),
            collection_name=collection_name,
        )
    candidate.collection_name = collection_name
    candidate.exact_match = True
    candidate.retrieval_methods = ["exact"]
    candidate.method_scores = {"exact": candidate.score}
    return candidate
