from types import SimpleNamespace

import pytest

from app.grounding.citations import CitationValidator
from app.grounding.models import (
    Evidence,
    GeneratedAnswer,
    GroundedAnswerResult,
    REFUSAL_ANSWER,
    TrustedCitation,
)
from app.rag import query_engine
from app.rag.query_engine import QueryPipeline, query
from app.retrieval.models import RetrievalCandidate
from app.retrieval.pipeline import RetrievalResult, RetrievalUnavailableError


def _candidate(chunk_id: str, *, document_id: str | None = None) -> RetrievalCandidate:
    document_id = document_id or f"document-{chunk_id}"
    return RetrievalCandidate(
        content=f"Canonical content for {chunk_id}",
        document_id=document_id,
        chunk_id=chunk_id,
        source_type="pdf",
        source_url=f"https://trusted.example/{chunk_id}",
        title=f"Title {chunk_id}",
        metadata={"document_type": "guide", "private": "metadata"},
        score=0.75,
        method_scores={"vector": 0.75},
        rrf_score=0.01,
        retrieval_methods=["vector"],
        fused_rank=1,
    )


def _retrieval(candidates, *, reranker_status="ok"):
    return RetrievalResult(
        candidates=candidates,
        retrieval_mode="hybrid",
        failures=["lexical"],
        diagnostics={
            "reranker": {"provider": "none", "status": reranker_status},
            "private": "must not escape",
        },
    )


class _RetrievalPipeline:
    reranker_provider = "none"

    def __init__(self, result=None, error=None, events=None):
        self.result = result
        self.error = error
        self.calls = []
        self.events = events if events is not None else []

    def retrieve(self, query, collection_name="default", top_k=None, filters=None):
        self.events.append("retrieve")
        self.calls.append((query, collection_name, top_k, filters))
        if self.error:
            raise self.error
        return self.result


class _Selector:
    def __init__(self, evidence, events=None):
        self.evidence = evidence
        self.calls = []
        self.events = events if events is not None else []

    def select(self, query, candidates, top_k):
        self.events.append("select")
        self.calls.append((query, candidates, top_k))
        return self.evidence


class _Generator:
    def __init__(self, generated, events=None):
        self.generated = generated
        self.calls = []
        self.events = events if events is not None else []

    def generate(self, query, evidence):
        self.events.append("generate")
        self.calls.append((query, evidence))
        return self.generated


class _Validator:
    def __init__(self, validated, events=None):
        self.validated = validated
        self.calls = []
        self.events = events if events is not None else []

    def validate(self, generated, evidence):
        self.events.append("validate")
        self.calls.append((generated, evidence))
        return self.validated


def _evidence(candidate, citation_id="E1"):
    return Evidence(citation_id=citation_id, candidate=candidate, prompt_content=candidate.content)


def test_pipeline_orchestrates_in_order_and_preserves_legacy_and_additive_fields():
    events = []
    selected_candidate = _candidate("selected")
    unselected_candidate = _candidate("unselected")
    selected = [_evidence(selected_candidate)]
    generated = GeneratedAnswer(answer="answer [E1]", citation_ids=["E1"])
    validated = GroundedAnswerResult(
        answer="answer [E1]",
        citations=[TrustedCitation(
            citation_id="E1",
            document_id=selected_candidate.document_id,
            chunk_id=selected_candidate.chunk_id,
            source_url=selected_candidate.source_url,
            excerpt=selected_candidate.content,
        )],
        status="supported",
    )
    retrieval = _RetrievalPipeline(_retrieval([selected_candidate, unselected_candidate]), events=events)
    selector = _Selector(selected, events=events)
    generator = _Generator(generated, events=events)
    validator = _Validator(validated, events=events)
    pipeline = QueryPipeline(
        retrieval_pipeline=retrieval,
        evidence_selector=selector,
        generator=generator,
        validator=validator,
        model="gpt-5-2025-08-07",
        evidence_top_k=5,
    )

    result = pipeline.query(
        "question", collection_name="alpha", top_k=7, document_type="guide"
    )

    assert events == ["retrieve", "select", "generate", "validate"]
    assert retrieval.calls == [("question", "alpha", 7, {"document_type": "guide"})]
    assert selector.calls == [("question", [selected_candidate, unselected_candidate], 5)]
    assert generator.calls == [("question", selected)]
    assert validator.calls == [(generated, selected)]
    assert result["answer"] == "answer [E1]"
    assert result["model"] == "gpt-5-2025-08-07"
    assert result["sources"] == [{
        "document_id": "document-selected",
        "source_type": "pdf",
        "title": "Title selected",
        "document_type": "guide",
        "excerpt": "Canonical content for selected",
        "score": 0.75,
    }]
    assert result["citations"] == [validated.citations[0].model_dump()]
    assert result["grounding"] == {"status": "supported"}
    assert result["retrieval_metadata"] == {
        "mode": "hybrid",
        "failures": ["lexical"],
        "reranker": {"provider": "none", "status": "ok"},
    }


def test_pipeline_empty_retrieval_refuses_without_generation_or_citations():
    events = []
    retrieval = _RetrievalPipeline(_retrieval([]), events=events)
    selector = _Selector([], events=events)
    generator = _Generator(GeneratedAnswer(answer="must not run", citation_ids=[]), events=events)
    validator = CitationValidator()
    pipeline = QueryPipeline(
        retrieval_pipeline=retrieval,
        evidence_selector=selector,
        generator=generator,
        validator=validator,
    )

    result = pipeline.query("question")

    assert events == ["retrieve", "select"]
    assert generator.calls == []
    assert result["answer"] == REFUSAL_ANSWER
    assert result["sources"] == []
    assert result["citations"] == []
    assert result["grounding"] == {"status": "insufficient_evidence"}


@pytest.mark.parametrize(
    "error",
    [RetrievalUnavailableError("both unavailable"), RuntimeError("unexpected retrieval failure")],
)
def test_pipeline_propagates_retrieval_failures_without_generation(error):
    retrieval = _RetrievalPipeline(error=error)
    selector = _Selector([])
    generator = _Generator(GeneratedAnswer(answer="bad", citation_ids=[]))
    pipeline = QueryPipeline(
        retrieval_pipeline=retrieval,
        evidence_selector=selector,
        generator=generator,
    )

    with pytest.raises(type(error), match=str(error)):
        pipeline.query("question")

    assert selector.calls == []
    assert generator.calls == []


def test_pipeline_reranker_fallback_candidates_still_generate():
    candidate = _candidate("fallback")
    evidence = [_evidence(candidate)]
    generator = _Generator(GeneratedAnswer(answer="answer [E1]", citation_ids=["E1"]))
    pipeline = QueryPipeline(
        retrieval_pipeline=_RetrievalPipeline(_retrieval([candidate], reranker_status="fallback")),
        evidence_selector=_Selector(evidence),
        generator=generator,
        validator=CitationValidator(),
    )

    result = pipeline.query("question")

    assert result["answer"] == "answer [E1]"
    assert result["grounding"] == {"status": "supported"}
    assert result["retrieval_metadata"]["reranker"]["status"] == "fallback"
    assert len(generator.calls) == 1


def test_pipeline_invalid_model_citation_cannot_create_mapping():
    candidate = _candidate("known")
    evidence = [_evidence(candidate)]
    pipeline = QueryPipeline(
        retrieval_pipeline=_RetrievalPipeline(_retrieval([candidate])),
        evidence_selector=_Selector(evidence),
        generator=_Generator(GeneratedAnswer(
            answer="answer from invented source",
            citation_ids=["E999", "https://evil.example"],
        )),
        validator=CitationValidator(),
    )

    result = pipeline.query("question")

    assert result["answer"] == REFUSAL_ANSWER
    assert result["citations"] == []
    assert result["grounding"] == {"status": "validation_failed"}
    assert "evil.example" not in repr(result)


def test_retrieval_metadata_whitelists_nested_diagnostics_and_safe_tokens_only():
    candidate = _candidate("known")
    evidence = [_evidence(candidate)]
    retrieval = RetrievalResult(
        candidates=[candidate],
        retrieval_mode="hybrid",
        failures=["lexical", "https://private.example/query"],
        diagnostics={
            "reranker": {
                "provider": "openai",
                "status": "fallback",
                "error_type": "TimeoutError",
                "query": "private customer question",
                "content": "private passage",
                "source_url": "https://private.example/source",
                "error_message": "secret API failure detail",
                "private": {"nested": "must not escape"},
            },
            "query": "top-level private question",
            "private": {"nested": "must not escape"},
        },
    )
    pipeline = QueryPipeline(
        retrieval_pipeline=_RetrievalPipeline(retrieval),
        evidence_selector=_Selector(evidence),
        generator=_Generator(GeneratedAnswer(answer="answer [E1]", citation_ids=["E1"])),
        validator=CitationValidator(),
    )

    result = pipeline.query("private customer question")

    assert result["retrieval_metadata"] == {
        "mode": "hybrid",
        "failures": ["lexical"],
        "reranker": {
            "provider": "openai",
            "status": "fallback",
            "error_type": "TimeoutError",
        },
    }
    rendered = repr(result["retrieval_metadata"])
    assert "private" not in rendered
    assert "source" not in rendered
    assert "secret" not in rendered


def test_retrieval_metadata_drops_invalid_reranker_types_and_tokens():
    retrieval = RetrievalResult(
        candidates=[],
        retrieval_mode="vector",
        diagnostics={
            "reranker": {
                "provider": ["openai"],
                "status": "fallback\nprivate",
                "error_type": "PrivateCustomerError",
            },
        },
    )
    pipeline = QueryPipeline(
        retrieval_pipeline=_RetrievalPipeline(retrieval),
        evidence_selector=_Selector([]),
        generator=_Generator(GeneratedAnswer(answer="must not run", citation_ids=[])),
        validator=CitationValidator(),
    )

    result = pipeline.query("question")

    assert result["retrieval_metadata"] == {
        "mode": "vector",
        "failures": [],
        "reranker": {},
    }


def test_retrieval_metadata_whitelists_exact_lookup_status_and_nonnegative_counts():
    retrieval = RetrievalResult(
        candidates=[],
        retrieval_mode="lexical",
        diagnostics={
            "exact_lookup": {
                "status": "partial_failure",
                "attempted_count": 3,
                "failure_count": 1,
                "match_count": 2,
                "query": "private question",
                "source_url": "https://private.example",
                "error_message": "secret failure",
                "nested": {"private": True},
            },
        },
    )
    pipeline = QueryPipeline(
        retrieval_pipeline=_RetrievalPipeline(retrieval),
        evidence_selector=_Selector([]),
        generator=_Generator(GeneratedAnswer(answer="must not run", citation_ids=[])),
        validator=CitationValidator(),
    )

    result = pipeline.query("question")

    assert result["retrieval_metadata"] == {
        "mode": "lexical",
        "failures": [],
        "reranker": {},
        "exact_lookup": {
            "status": "partial_failure",
            "attempted_count": 3,
            "failure_count": 1,
            "match_count": 2,
        },
    }


def test_default_pipeline_reuses_one_openai_client_for_vector_retrieval_and_generation(monkeypatch):
    shared_client = SimpleNamespace()
    openai_calls = []
    vector_clients = []

    def openai(**kwargs):
        openai_calls.append(kwargs)
        return shared_client

    class Vector:
        def __init__(self, openai_client):
            vector_clients.append(openai_client)

    monkeypatch.setattr("app.rag.query_engine.OpenAI", openai)
    monkeypatch.setattr("app.rag.query_engine.ChromaVectorRetriever", Vector)
    monkeypatch.setattr("app.rag.query_engine.httpx.Client", lambda: "compatible-http-client")

    pipeline = QueryPipeline()

    assert len(openai_calls) == 1
    assert openai_calls[0]["api_key"] == "test-key"
    assert openai_calls[0]["http_client"] == "compatible-http-client"
    assert vector_clients == [shared_client]
    assert pipeline.generator._client is shared_client


def test_public_query_lazily_reuses_one_application_owned_pipeline(monkeypatch):
    constructions = []
    calls = []

    class Pipeline:
        def __init__(self):
            constructions.append(True)

        def query(self, question, collection_name="default", top_k=None, document_type=None):
            calls.append((question, collection_name, top_k, document_type))
            return {"answer": question, "sources": [], "model": "model"}

    monkeypatch.setattr(query_engine, "QueryPipeline", Pipeline)
    monkeypatch.setattr(query_engine, "_default_query_pipeline", None, raising=False)

    first = query("first")
    second = query("second")

    assert first["answer"] == "first"
    assert second["answer"] == "second"
    assert constructions == [True]
    assert len(calls) == 2


def test_owned_client_is_shared_with_vector_generation_and_openai_reranker_then_closed_once(
    monkeypatch,
):
    class HttpClient:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    class OpenAIClient:
        def __init__(self, *, api_key, http_client):
            self.api_key = api_key
            self.http_client = http_client
            self.close_count = 0

        def close(self):
            self.close_count += 1
            self.http_client.close()

    http_client = HttpClient()
    openai_clients = []
    vector_clients = []
    hybrid_calls = []

    def openai(**kwargs):
        client = OpenAIClient(**kwargs)
        openai_clients.append(client)
        return client

    class Vector:
        def __init__(self, openai_client):
            vector_clients.append(openai_client)

    class Hybrid:
        def __init__(self, **kwargs):
            hybrid_calls.append(kwargs)

    monkeypatch.setattr(query_engine.settings, "retrieval_mode", "hybrid")
    monkeypatch.setattr(query_engine.settings, "reranker_provider", "openai")
    monkeypatch.setattr(query_engine, "OpenAI", openai)
    monkeypatch.setattr(query_engine.httpx, "Client", lambda: http_client)
    monkeypatch.setattr(query_engine, "ChromaVectorRetriever", Vector)
    monkeypatch.setattr(query_engine, "HybridRetrievalPipeline", Hybrid)

    pipeline = QueryPipeline()
    pipeline.close()
    pipeline.close()

    assert len(openai_clients) == 1
    shared_client = openai_clients[0]
    assert vector_clients == [shared_client]
    assert pipeline.generator._client is shared_client
    assert hybrid_calls == [{
        "vector_retriever": hybrid_calls[0]["vector_retriever"],
        "reranker_provider": "openai",
        "reranker_dependencies": {"client": shared_client},
    }]
    assert shared_client.close_count == 1
    assert http_client.close_count == 1


def test_lexical_only_pipeline_owns_shared_generation_client_without_vector_construction(monkeypatch):
    clients = []

    class Client:
        def __init__(self, **kwargs):
            clients.append(self)
            self.close_count = 0

        def close(self):
            self.close_count += 1

    monkeypatch.setattr(query_engine.settings, "retrieval_mode", "lexical")
    monkeypatch.setattr(query_engine.settings, "reranker_provider", "none")
    monkeypatch.setattr(query_engine, "OpenAI", Client)
    monkeypatch.setattr(query_engine.httpx, "Client", lambda: SimpleNamespace())
    monkeypatch.setattr(
        query_engine,
        "ChromaVectorRetriever",
        lambda **kwargs: pytest.fail("constructed vector retriever"),
    )
    monkeypatch.setattr(query_engine, "HybridRetrievalPipeline", lambda **kwargs: SimpleNamespace())

    pipeline = QueryPipeline()
    pipeline.close()
    pipeline.close()

    assert len(clients) == 1
    assert pipeline.generator._client is clients[0]
    assert clients[0].close_count == 1


def test_pipeline_closes_newly_owned_client_when_downstream_construction_fails(monkeypatch):
    class HttpClient:
        def __init__(self):
            self.close_count = 0

        def close(self):
            self.close_count += 1

    class Client:
        def __init__(self, *, api_key, http_client):
            self.http_client = http_client
            self.close_count = 0

        def close(self):
            self.close_count += 1
            self.http_client.close()

    http_client = HttpClient()
    clients = []

    def openai(**kwargs):
        client = Client(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(query_engine.settings, "retrieval_mode", "lexical")
    monkeypatch.setattr(query_engine.settings, "reranker_provider", "none")
    monkeypatch.setattr(query_engine, "OpenAI", openai)
    monkeypatch.setattr(query_engine.httpx, "Client", lambda: http_client)
    monkeypatch.setattr(
        query_engine,
        "HybridRetrievalPipeline",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("construction failed")),
    )

    with pytest.raises(RuntimeError, match="construction failed"):
        QueryPipeline()

    assert clients[0].close_count == 1
    assert http_client.close_count == 1


def test_default_pipeline_close_hook_is_idempotent_and_allows_lazy_recreation(monkeypatch):
    instances = []

    class Pipeline:
        def __init__(self):
            self.close_count = 0
            instances.append(self)

        def query(self, *args, **kwargs):
            return {"answer": "ok", "sources": [], "model": "model"}

        def close(self):
            self.close_count += 1

    monkeypatch.setattr(query_engine, "QueryPipeline", Pipeline)
    monkeypatch.setattr(query_engine, "_default_query_pipeline", None, raising=False)

    query("first")
    query_engine.close_default_query_pipeline()
    query_engine.close_default_query_pipeline()
    query("second")

    assert len(instances) == 2
    assert instances[0].close_count == 1
    assert instances[1].close_count == 0


def test_public_query_parameters_are_unchanged_and_delegate_to_default_pipeline(monkeypatch):
    calls = []

    class Pipeline:
        def query(self, question, collection_name="default", top_k=None, document_type=None):
            calls.append((question, collection_name, top_k, document_type))
            return {"answer": "ok", "sources": [], "model": "model"}

    monkeypatch.setattr("app.rag.query_engine.QueryPipeline", Pipeline)
    monkeypatch.setattr(query_engine, "_default_query_pipeline", None, raising=False)

    result = query("question", "alpha", 6, "guide")

    assert result["answer"] == "ok"
    assert calls == [("question", "alpha", 6, "guide")]


def test_pipeline_preserves_falsey_injected_components():
    class FalseySelector(_Selector):
        def __bool__(self):
            return False

    class FalseyValidator(_Validator):
        def __bool__(self):
            return False

    candidate = _candidate("one")
    evidence = [_evidence(candidate)]
    selector = FalseySelector(evidence)
    validator = FalseyValidator(GroundedAnswerResult(
        answer="answer [E1]",
        citations=[],
        status="supported",
    ))
    pipeline = QueryPipeline(
        retrieval_pipeline=_RetrievalPipeline(_retrieval([candidate])),
        evidence_selector=selector,
        generator=_Generator(GeneratedAnswer(answer="answer [E1]", citation_ids=["E1"])),
        validator=validator,
    )

    pipeline.query("question")

    assert pipeline.evidence_selector is selector
    assert pipeline.validator is validator
    assert len(selector.calls) == 1
    assert len(validator.calls) == 1
