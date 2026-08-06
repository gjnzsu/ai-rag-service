"""Retrieval, evidence selection, grounded generation, and citation orchestration."""

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
        if retrieval_pipeline is None:
            vector_retriever = None
            if settings.retrieval_mode in {"vector", "hybrid"}:
                shared_client = shared_client if shared_client is not None else _openai_client()
                vector_retriever = ChromaVectorRetriever(openai_client=shared_client)
            retrieval_pipeline = HybridRetrievalPipeline(vector_retriever=vector_retriever)
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
    return QueryPipeline().query(
        question,
        collection_name=collection_name,
        top_k=top_k,
        document_type=document_type,
    )


def _openai_client() -> Any:
    return OpenAI(
        api_key=settings.openai_api_key,
        http_client=httpx.Client(),
    )


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
    reranker = retrieval.diagnostics.get("reranker", {})
    return {
        "mode": retrieval.retrieval_mode,
        "failures": list(retrieval.failures),
        "reranker": dict(reranker) if isinstance(reranker, dict) else {},
    }
