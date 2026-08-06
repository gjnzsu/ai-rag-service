import json
import math
from types import SimpleNamespace

import pytest

from app.evaluation.models import EvaluationCase, QueryType, RankedEvaluationResult
from app.evaluation.runner import (
    EvaluationRunner,
    _production_hybrid_adapter,
    _production_reranker_adapter,
    main,
)
from app.retrieval.models import RetrievalCandidate


def _candidate(document_id: str, chunk_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(content=chunk_id, document_id=document_id, chunk_id=chunk_id)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "question": "What is the approved rate?",
            "query_type": "exact_fact",
            "relevant_document_ids": ["doc-1"],
            "expected_facts": ["Synthetic fact"],
            "should_abstain": False,
        },
        {
            "question": "Compare the synthetic sources.",
            "query_type": "cross_document",
            "relevant_document_ids": ["doc-1", "doc-2"],
            "relevant_chunk_ids": ["doc-1#1", "doc-2#1"],
            "expected_facts": [],
            "should_abstain": False,
        },
        {
            "question": "What is unavailable?",
            "query_type": "unanswerable",
            "should_abstain": True,
        },
    ],
)
def test_evaluation_case_accepts_document_only_and_optional_chunk_labels(payload):
    case = EvaluationCase.model_validate(payload)
    assert case.question == payload["question"]
    assert isinstance(case.query_type, QueryType)


@pytest.mark.parametrize(
    "payload",
    [
        {"question": "", "query_type": "exact_fact"},
        {"question": "q", "query_type": "unknown"},
        {"question": "q", "query_type": "exact_fact", "relevant_document_ids": [""]},
        {"question": "q", "query_type": "exact_fact", "relevant_chunk_ids": [" "]},
    ],
)
def test_evaluation_case_rejects_invalid_questions_types_and_ids(payload):
    with pytest.raises(ValueError):
        EvaluationCase.model_validate(payload)


def test_ranked_result_never_fabricates_citation_correctness():
    result = RankedEvaluationResult(ranked_document_ids=("doc-1",))
    assert result.human_citation_correctness is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latency_ms", math.nan),
        ("answer_latency_ms", math.inf),
        ("local_memory_mb", -math.inf),
    ],
)
def test_ranked_result_rejects_nonfinite_observations(field, value):
    with pytest.raises(ValueError):
        RankedEvaluationResult(**{field: value})


def test_runner_caches_hybrid_candidates_and_records_failures_and_recommendation():
    calls = {"vector": 0, "hybrid": 0}
    c1_inputs: list[list[RetrievalCandidate]] = []
    c2_inputs: list[list[RetrievalCandidate]] = []

    def vector(case):
        calls["vector"] += 1
        if case.question == "fails safely":
            raise RuntimeError("unavailable")
        return [_candidate("doc-v", "v1")]

    def hybrid(case):
        calls["hybrid"] += 1
        return [_candidate("doc-2", "h1"), _candidate("doc-1", "h2")]

    def c1(case, candidates):
        c1_inputs.append(candidates)
        candidates[0].document_id = "mutated"
        return list(reversed(candidates))

    def c2(case, candidates):
        c2_inputs.append(candidates)
        return candidates

    def evaluate(case, candidates):
        return RankedEvaluationResult(
            selected_chunk_ids=(candidates[0].chunk_id,),
            cited_chunk_ids=(candidates[0].chunk_id,),
            latency_ms=100.0,
            answer_latency_ms=6_000.0,
            input_tokens=10,
            output_tokens=5,
        )

    runner = EvaluationRunner(vector, hybrid, c1, c2, grounded_answer_evaluator=evaluate)
    report = runner.run(
        [
            EvaluationCase(
                question="works",
                query_type="exact_fact",
                relevant_document_ids=("doc-1",),
                relevant_chunk_ids=("h2",),
            ),
            EvaluationCase(question="fails safely", query_type="hard_negative", should_abstain=True),
        ]
    )

    assert calls == {"vector": 2, "hybrid": 2}
    assert [item.chunk_id for item in c1_inputs[0]] == ["h1", "h2"]
    assert [item.chunk_id for item in c2_inputs[0]] == ["h1", "h2"]
    assert c1_inputs[0] is not c2_inputs[0]
    assert report["configurations"]["A"]["questions"]
    assert report["configurations"]["C1"]["aggregate"]["mrr_at_10"] > report["configurations"]["B"]["aggregate"]["mrr_at_10"]
    assert report["configurations"]["C1"]["questions"][0]["changes_from_b"]["mrr_at_10"] > 0
    assert report["configurations"]["C1"]["questions"][0]["result"]["latency_ms"] >= 6_000.0
    assert report["failures"]
    assert report["recommendation"]["configuration"] == "C1"

    def slow_evaluate(case, candidates):
        return RankedEvaluationResult(answer_latency_ms=11_000.0)

    slow_report = EvaluationRunner(
        lambda case: [_candidate("doc-v", "v1")],
        lambda case: [_candidate("doc-2", "h1"), _candidate("doc-1", "h2")],
        lambda case, candidates: list(reversed(candidates)),
        lambda case, candidates: candidates,
        grounded_answer_evaluator=slow_evaluate,
    ).run([EvaluationCase(question="q", query_type="exact_fact", relevant_document_ids=("doc-1",))])
    assert slow_report["recommendation"]["configuration"] == "B"

    no_regression_report = EvaluationRunner(
        lambda case: [_candidate("doc-v", "v1")],
        lambda case: [
            _candidate("doc-2", "h1"),
            _candidate("doc-1", "h2"),
            _candidate("doc-3", "h3"),
        ],
        lambda case, candidates: [candidates[1]],
        lambda case, candidates: candidates,
        grounded_answer_evaluator=evaluate,
    ).run([
        EvaluationCase(
            question="q",
            query_type="cross_document",
            relevant_document_ids=("doc-1", "doc-3"),
        )
    ])
    assert no_regression_report["recommendation"]["configuration"] == "B"

    unmeasured_report = EvaluationRunner(
        lambda case: [_candidate("doc-v", "v1")],
        lambda case: [_candidate("doc-1", "h1")],
        lambda case, candidates: candidates,
        lambda case, candidates: candidates,
    ).run([EvaluationCase(question="q", query_type="unanswerable", should_abstain=True)])
    unmeasured = unmeasured_report["configurations"]["B"]
    assert unmeasured["questions"][0]["metrics"]["citation_validity"] is None
    assert unmeasured["questions"][0]["metrics"]["abstention_accuracy"] is None
    assert unmeasured["questions"][0]["result"]["answer_latency_ms"] is None
    assert unmeasured["aggregate"]["token_usage"]["input"] is None

    incomplete_c1 = EvaluationRunner(
        lambda case: [_candidate("doc-v", "v1")],
        lambda case: [_candidate("doc-2", "h1"), _candidate("doc-1", "h2")],
        lambda case, candidates: (_ for _ in ()).throw(RuntimeError()) if case.question == "two" else list(reversed(candidates)),
        lambda case, candidates: candidates,
        grounded_answer_evaluator=evaluate,
    ).run([
        EvaluationCase(question="one", query_type="exact_fact", relevant_document_ids=("doc-1",)),
        EvaluationCase(question="two", query_type="exact_fact", relevant_document_ids=("doc-1",)),
    ])
    assert incomplete_c1["recommendation"]["configuration"] == "B"

    bad = _candidate("doc-1", "bad")
    bad.score = math.nan
    bad_score_report = EvaluationRunner(
        lambda case: [_candidate("doc-v", "v1")],
        lambda case: [bad],
        lambda case, candidates: candidates,
        lambda case, candidates: candidates,
    ).run([EvaluationCase(question="q", query_type="exact_fact", relevant_document_ids=("doc-1",))])
    assert bad_score_report["configurations"]["B"]["questions"][0]["error"] == "ValueError"


def test_cli_reads_jsonl_and_writes_json_without_subprocess(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    output_path = tmp_path / "report.json"
    cases_path.write_text(
        json.dumps({"question": "q", "query_type": "unanswerable", "should_abstain": True}) + "\n",
        encoding="utf-8",
    )

    class FakeRunner:
        def run(self, cases):
            assert len(cases) == 1
            return {"configurations": {}, "failures": [], "recommendation": {"configuration": "B"}}

    main(["--cases", str(cases_path), "--output", str(output_path)], runner_factory=lambda: FakeRunner())

    assert json.loads(output_path.read_text(encoding="utf-8"))["recommendation"]["configuration"] == "B"


def test_live_adapters_use_production_reranker_signature_and_reject_degradation():
    calls = []

    class ProductionReranker:
        def rerank(self, query, candidates, top_k):
            calls.append((query, [candidate.chunk_id for candidate in candidates], top_k))
            return candidates

    case = EvaluationCase(question="production question", query_type="exact_fact")
    adapter = _production_reranker_adapter(ProductionReranker(), top_k=7)
    assert adapter(case, [_candidate("doc-1", "chunk-1")])[0].chunk_id == "chunk-1"
    assert calls == [("production question", ["chunk-1"], 7)]

    degraded = SimpleNamespace(
        retrieve=lambda *args, **kwargs: SimpleNamespace(
            candidates=[_candidate("doc-1", "chunk-1")],
            failures=["vector"],
            diagnostics={"configured_retrievers": ["vector", "lexical"], "successful_retrievers": ["lexical"]},
        )
    )
    with pytest.raises(RuntimeError):
        _production_hybrid_adapter(degraded, top_k=7)(case)
