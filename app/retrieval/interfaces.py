"""Small interfaces that allow retrieval backends to be substituted in tests."""

from typing import Any, Protocol

from app.retrieval.models import RetrievalCandidate


class VectorRetriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        collection_name: str,
    ) -> list[RetrievalCandidate]: ...


class LexicalRetriever(Protocol):
    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None,
        collection_name: str,
    ) -> list[RetrievalCandidate]: ...
