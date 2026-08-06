"""Optional post-fusion reranking backends."""

from typing import Any

from app.config import Settings, settings as default_settings
from app.reranking.base import Reranker
from app.reranking.noop import NoOpReranker


def build_reranker(
    provider: str | None = None,
    settings: Settings = default_settings,
    **injectable_dependencies: Any,
) -> Reranker:
    """Construct only the configured reranking backend."""
    selected = settings.reranker_provider if provider is None else provider
    if selected == "none":
        return NoOpReranker()
    if selected == "openai":
        from app.reranking.openai import GPT5Reranker

        return GPT5Reranker(
            model=settings.reranker_openai_model,
            timeout_seconds=settings.reranker_openai_timeout_seconds,
            **injectable_dependencies,
        )
    if selected == "qwen_local":
        from app.reranking.qwen import Qwen3LocalReranker

        return Qwen3LocalReranker(
            model_name=settings.reranker_qwen_model,
            revision=settings.reranker_qwen_revision,
            max_candidates=settings.reranker_qwen_max_candidates,
            max_length=settings.reranker_qwen_max_length,
            batch_size=settings.reranker_qwen_batch_size,
            timeout_seconds=settings.reranker_qwen_timeout_seconds,
            circuit_breaker_seconds=settings.reranker_qwen_circuit_breaker_seconds,
            **injectable_dependencies,
        )
    raise ValueError("Unsupported reranker provider")


__all__ = ["NoOpReranker", "Reranker", "build_reranker"]
