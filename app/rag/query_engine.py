"""Retrieval, evidence selection, grounded generation, and citation orchestration."""

from threading import Lock
from typing import Any

import httpx
import structlog
from openai import OpenAI

from app.config import settings
from app.grounding.citations import CitationValidator
from app.grounding.evidence import EvidenceSelector
from app.grounding.generator import GroundedAnswerGenerator
from app.grounding.models import GeneratedAnswer, REFUSAL_ANSWER
from app.retrieval.pipeline import HybridRetrievalPipeline, RetrievalResult
from app.retrieval.vector import ChromaVectorRetriever

logger = structlog.get_logger()

_default_query_pipeline: "QueryPipeline | None" = None
_default_query_pipeline_lock = Lock()
_RETRIEVAL_FAILURES = frozenset({"vector", "lexical"})
_RERANKER_PROVIDERS = frozenset({"none", "openai", "qwen_local", "custom"})
_RERANKER_STATUSES = frozenset({"ok", "fallback", "disabled"})
_EXACT_LOOKUP_STATUSES = frozenset({
    "not_requested",
    "unavailable",
    "partial_failure",
    "ok",
    "no_match",
})
_SAFE_ERROR_TYPES = frozenset({
    "APIConnectionError",
    "APIError",
    "APITimeoutError",
    "AttributeError",
    "AuthenticationError",
    "BadRequestError",
    "CircuitOpenError",
    "ConflictError",
    "Exception",
    "ImportError",
    "IndexError",
    "InferenceBusyError",
    "InternalServerError",
    "JSONDecodeError",
    "NotFoundError",
    "PermissionDeniedError",
    "RateLimitError",
    "RuntimeError",
    "TimeoutError",
    "TypeError",
    "UnprocessableEntityError",
    "ValueError",
})


class QueryPipeline:
    """Injectable server-owned orchestration around the retrieval and model boundaries."""

    def __init__(
        self,
        retrieval_pipeline: Any | None = None,
        evidence_selector: Any | None = None,
        generator: Any | None = None,
        validator: Any | None = None,
        *,
        model: str | None = None,
        evidence_top_k: int | None = None,
        openai_client: Any | None = None,
    ) -> None:
        shared_client = openai_client
        self._owned_openai_client: Any | None = None
        self._closed = False
        needs_shared_client = generator is None or (
            retrieval_pipeline is None
            and (
                settings.retrieval_mode in {"vector", "hybrid"}
                or settings.reranker_provider == "openai"
            )
        )
        if shared_client is None and needs_shared_client:
            shared_client = _create_owned_openai_client()
            self._owned_openai_client = shared_client
        try:
            if retrieval_pipeline is None:
                vector_retriever = None
                if settings.retrieval_mode in {"vector", "hybrid"}:
                    assert shared_client is not None
                    vector_retriever = ChromaVectorRetriever(openai_client=shared_client)
                if settings.reranker_provider == "openai":
                    assert shared_client is not None
                    retrieval_pipeline = HybridRetrievalPipeline(
                        vector_retriever=vector_retriever,
                        reranker_provider="openai",
                        reranker_dependencies={"client": shared_client},
                    )
                else:
                    retrieval_pipeline = HybridRetrievalPipeline(
                        vector_retriever=vector_retriever
                    )
            if generator is None:
                generator = GroundedAnswerGenerator(
                    model=model,
                    client=shared_client,
                )

            self.retrieval_pipeline = retrieval_pipeline
            self.evidence_selector = (
                evidence_selector if evidence_selector is not None else EvidenceSelector()
            )
            self.generator = generator
            self.validator = validator if validator is not None else CitationValidator()
            self.model = model or getattr(generator, "model", settings.answer_openai_model)
            self.evidence_top_k = (
                settings.grounding_evidence_top_k
                if evidence_top_k is None
                else evidence_top_k
            )
            if not 5 <= self.evidence_top_k <= 10:
                raise ValueError("evidence_top_k must be between 5 and 10")
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Close resources owned by this pipeline exactly once."""
        if self._closed:
            return
        self._closed = True
        if self._owned_openai_client is None:
            return
        try:
            self._owned_openai_client.close()
        except Exception:
            logger.warning("query_pipeline_close_failed", resource="openai")

    def query(
        self,
        question: str,
        collection_name: str = "default",
        top_k: int | None = None,
        document_type: str | None = None,
    ) -> dict[str, Any]:
        requested_top_k = settings.top_k if top_k is None else top_k
        filters = {"document_type": document_type} if document_type else None
        retrieval = self.retrieval_pipeline.retrieve(
            question,
            collection_name=collection_name,
            top_k=requested_top_k,
            filters=filters,
        )
        evidence = self.evidence_selector.select(
            question,
            retrieval.candidates,
            top_k=self.evidence_top_k,
        )

        if evidence:
            generated = self.generator.generate(question, evidence)
        else:
            generated = GeneratedAnswer(answer=REFUSAL_ANSWER, citation_ids=[])
        grounded = self.validator.validate(generated, evidence)

        logger.info(
            "grounded_query_completed",
            grounding_status=grounded.status,
            retrieval_mode=retrieval.retrieval_mode,
            candidate_count=len(retrieval.candidates),
            evidence_count=len(evidence),
            citation_count=len(grounded.citations),
        )
        return {
            "answer": grounded.answer,
            "sources": [_source(item) for item in evidence],
            "model": self.model,
            "citations": [citation.model_dump() for citation in grounded.citations],
            "grounding": {"status": grounded.status},
            "retrieval_metadata": _retrieval_metadata(retrieval),
        }


def query(
    question: str,
    collection_name: str = "default",
    top_k: int | None = None,
    document_type: str | None = None,
) -> dict:
    return get_default_query_pipeline().query(
        question,
        collection_name=collection_name,
        top_k=top_k,
        document_type=document_type,
    )


def get_default_query_pipeline() -> QueryPipeline:
    global _default_query_pipeline
    if _default_query_pipeline is None:
        with _default_query_pipeline_lock:
            if _default_query_pipeline is None:
                _default_query_pipeline = QueryPipeline()
    return _default_query_pipeline


def close_default_query_pipeline() -> None:
    global _default_query_pipeline
    with _default_query_pipeline_lock:
        pipeline = _default_query_pipeline
        _default_query_pipeline = None
    if pipeline is not None:
        pipeline.close()


def _create_owned_openai_client() -> Any:
    http_client = httpx.Client()
    try:
        return OpenAI(
            api_key=settings.openai_api_key,
            http_client=http_client,
        )
    except Exception:
        http_client.close()
        raise


def _source(evidence: Any) -> dict[str, Any]:
    candidate = evidence.candidate
    return {
        "document_id": candidate.document_id,
        "source_type": candidate.source_type,
        "title": candidate.title,
        "document_type": str(candidate.metadata.get("document_type", "")),
        "excerpt": candidate.content[: settings.grounding_excerpt_max_chars],
        "score": candidate.score,
    }


def _retrieval_metadata(retrieval: RetrievalResult) -> dict[str, Any]:
    result: dict[str, Any] = {
        "mode": retrieval.retrieval_mode,
        "failures": _safe_failures(retrieval.failures),
        "reranker": _safe_reranker_metadata(retrieval.diagnostics.get("reranker")),
    }
    exact_lookup = _safe_exact_lookup_metadata(retrieval.diagnostics.get("exact_lookup"))
    if exact_lookup is not None:
        result["exact_lookup"] = exact_lookup
    return result


def _safe_failures(failures: list[str]) -> list[str]:
    selected: list[str] = []
    for failure in failures:
        if failure in _RETRIEVAL_FAILURES and failure not in selected:
            selected.append(failure)
    return selected


def _safe_reranker_metadata(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    provider = value.get("provider")
    status = value.get("status")
    if (
        not isinstance(provider, str)
        or provider not in _RERANKER_PROVIDERS
        or not isinstance(status, str)
        or status not in _RERANKER_STATUSES
    ):
        return {}
    result = {"provider": provider, "status": status}
    error_type = value.get("error_type")
    if (
        status == "fallback"
        and isinstance(error_type, str)
        and error_type in _SAFE_ERROR_TYPES
    ):
        result["error_type"] = error_type
    return result


def _safe_exact_lookup_metadata(value: Any) -> dict[str, str | int] | None:
    if not isinstance(value, dict):
        return None
    status = value.get("status")
    if not isinstance(status, str) or status not in _EXACT_LOOKUP_STATUSES:
        return None
    result: dict[str, str | int] = {"status": status}
    for key in ("attempted_count", "failure_count", "match_count"):
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return None
        result[key] = count
    return result
