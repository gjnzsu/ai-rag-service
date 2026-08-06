"""Injected evaluation runner and JSONL command-line boundary."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from app.evaluation.metrics import (
    abstention_accuracy,
    citation_correctness,
    citation_validity,
    context_precision,
    hit_at_k,
    latency_percentiles,
    mrr_at_k,
    recall_at_k,
)
from app.evaluation.models import EvaluationCase, RankedEvaluationResult
from app.retrieval.models import RetrievalCandidate

CandidateRetriever = Callable[[EvaluationCase], Sequence[RetrievalCandidate]]
CandidateReranker = Callable[[EvaluationCase, list[RetrievalCandidate]], Sequence[RetrievalCandidate]]
GroundedAnswerEvaluator = Callable[
    [EvaluationCase, Sequence[RetrievalCandidate]], RankedEvaluationResult
]

_CONFIGURATIONS = ("A", "B", "C1", "C2")
_EVALUATION_TOP_K = 20
_SAFE_ERROR_TYPES = frozenset({
    "AttributeError",
    "ImportError",
    "IndexError",
    "KeyError",
    "RuntimeError",
    "TimeoutError",
    "TypeError",
    "ValueError",
})


class EvaluationRunner:
    """Evaluate exactly A/B/C1/C2 using injected retrieval and reranking boundaries."""

    def __init__(
        self,
        vector_retrieval: CandidateRetriever,
        hybrid_retrieval: CandidateRetriever,
        gpt5_reranker: CandidateReranker,
        qwen_reranker: CandidateReranker,
        *,
        grounded_answer_evaluator: GroundedAnswerEvaluator | None = None,
    ) -> None:
        self.vector_retrieval = vector_retrieval
        self.hybrid_retrieval = hybrid_retrieval
        self.gpt5_reranker = gpt5_reranker
        self.qwen_reranker = qwen_reranker
        self.grounded_answer_evaluator = grounded_answer_evaluator

    def run(self, cases: Sequence[EvaluationCase]) -> dict[str, Any]:
        """Run all four configurations without exposing experiment controls to API callers."""
        records: dict[str, list[dict[str, Any]]] = {name: [] for name in _CONFIGURATIONS}
        failures: list[dict[str, Any]] = []
        for case_index, case in enumerate(cases):
            self._run_retrieval("A", case_index, case, self.vector_retrieval, records, failures)
            hybrid = self._run_hybrid(case_index, case, records, failures)
            if hybrid is None:
                self._record_failure("C1", case_index, case, RuntimeError(), records, failures)
                self._record_failure("C2", case_index, case, RuntimeError(), records, failures)
                continue
            cached_candidates, hybrid_elapsed_ms = hybrid
            self._run_reranker(
                "C1", case_index, case, cached_candidates, hybrid_elapsed_ms,
                self.gpt5_reranker, records, failures,
            )
            self._run_reranker(
                "C2", case_index, case, cached_candidates, hybrid_elapsed_ms,
                self.qwen_reranker, records, failures,
            )

        _attach_changes_from_baseline(records)
        configurations = {
            name: {"questions": records[name], "aggregate": _aggregate(records[name])}
            for name in _CONFIGURATIONS
        }
        return {
            "configurations": configurations,
            "failures": failures,
            "recommendation": _recommend(configurations, frozenset(range(len(cases)))),
        }

    def _run_retrieval(
        self,
        configuration: str,
        case_index: int,
        case: EvaluationCase,
        retrieve: CandidateRetriever,
        records: dict[str, list[dict[str, Any]]],
        failures: list[dict[str, Any]],
    ) -> None:
        try:
            started = time.perf_counter()
            candidates = list(retrieve(case))
            elapsed_ms = _elapsed_ms(started)
            _validate_candidates(candidates)
            records[configuration].append(self._record(case_index, case, candidates, elapsed_ms))
        except Exception as error:
            self._record_failure(configuration, case_index, case, error, records, failures)

    def _run_hybrid(
        self,
        case_index: int,
        case: EvaluationCase,
        records: dict[str, list[dict[str, Any]]],
        failures: list[dict[str, Any]],
    ) -> tuple[list[RetrievalCandidate], float] | None:
        try:
            started = time.perf_counter()
            candidates = [candidate.model_copy(deep=True) for candidate in self.hybrid_retrieval(case)]
            elapsed_ms = _elapsed_ms(started)
            _validate_candidates(candidates)
            records["B"].append(self._record(
                case_index,
                case,
                [candidate.model_copy(deep=True) for candidate in candidates],
                elapsed_ms,
            ))
            return candidates, elapsed_ms
        except Exception as error:
            self._record_failure("B", case_index, case, error, records, failures)
            return None

    def _run_reranker(
        self,
        configuration: str,
        case_index: int,
        case: EvaluationCase,
        cached_hybrid_candidates: Sequence[RetrievalCandidate],
        hybrid_elapsed_ms: float,
        reranker: CandidateReranker,
        records: dict[str, list[dict[str, Any]]],
        failures: list[dict[str, Any]],
    ) -> None:
        try:
            started = time.perf_counter()
            candidates = list(reranker(
                case,
                [candidate.model_copy(deep=True) for candidate in cached_hybrid_candidates],
            ))
            elapsed_ms = hybrid_elapsed_ms + _elapsed_ms(started)
            _validate_candidates(candidates)
            records[configuration].append(self._record(case_index, case, candidates, elapsed_ms))
        except Exception as error:
            self._record_failure(configuration, case_index, case, error, records, failures)

    def _record(
        self,
        case_index: int,
        case: EvaluationCase,
        candidates: Sequence[RetrievalCandidate],
        base_latency_ms: float,
    ) -> dict[str, Any]:
        ranked_document_ids = tuple(candidate.document_id for candidate in candidates)
        ranked_chunk_ids = tuple(candidate.chunk_id for candidate in candidates)
        answer_evaluated = self.grounded_answer_evaluator is not None
        if answer_evaluated:
            started = time.perf_counter()
            observed = self.grounded_answer_evaluator(case, candidates)
            answer_elapsed_ms = _elapsed_ms(started)
            answer_latency_ms = (
                observed.answer_latency_ms
                if observed.answer_latency_ms is not None
                else answer_elapsed_ms
            )
            result = observed.model_copy(update={
                "ranked_document_ids": ranked_document_ids,
                "ranked_chunk_ids": ranked_chunk_ids,
                "latency_ms": base_latency_ms + answer_latency_ms,
                "answer_latency_ms": answer_latency_ms,
            })
        else:
            result = RankedEvaluationResult(
                ranked_document_ids=ranked_document_ids,
                ranked_chunk_ids=ranked_chunk_ids,
                latency_ms=base_latency_ms,
            )
        return {
            "case_index": case_index,
            "query_type": case.query_type.value,
            "metrics": _case_metrics(case, result, answer_evaluated=answer_evaluated),
            "result": result.model_dump(mode="json"),
            "error": None,
        }

    @staticmethod
    def _record_failure(
        configuration: str,
        case_index: int,
        case: EvaluationCase,
        error: Exception,
        records: dict[str, list[dict[str, Any]]],
        failures: list[dict[str, Any]],
    ) -> None:
        error_type = _safe_error_type(error)
        item = {
            "case_index": case_index,
            "query_type": case.query_type.value,
            "metrics": None,
            "result": None,
            "error": error_type,
        }
        records[configuration].append(item)
        failures.append({"configuration": configuration, "case_index": case_index, "error": error_type})


def _case_metrics(
    case: EvaluationCase,
    result: RankedEvaluationResult,
    *,
    answer_evaluated: bool,
) -> dict[str, float | None]:
    ranked_ids, relevant_ids = _retrieval_label_level(case, result)
    metrics: dict[str, float | None] = {
        "recall_at_20": recall_at_k(ranked_ids, relevant_ids, k=20),
        "hit_at_5": hit_at_k(ranked_ids, relevant_ids, k=5),
        "mrr_at_10": mrr_at_k(ranked_ids, relevant_ids, k=10),
        "context_precision": context_precision(ranked_ids, relevant_ids, k=20),
        "citation_validity": None,
        "citation_correctness": None,
        "abstention_accuracy": None,
    }
    if not answer_evaluated:
        return metrics
    cited_ids, selected_ids = _citation_label_level(result)
    metrics.update({
        "citation_validity": citation_validity(cited_ids, selected_ids),
        "citation_correctness": citation_correctness(
            None
            if result.human_citation_correctness is None
            else (result.human_citation_correctness,)
        ),
        "abstention_accuracy": (
            abstention_accuracy(case.should_abstain, result.refused)
            if result.refused is not None
            else None
        ),
    })
    return metrics


def _retrieval_label_level(
    case: EvaluationCase, result: RankedEvaluationResult
) -> tuple[Sequence[str], Sequence[str]]:
    if case.relevant_chunk_ids:
        return result.ranked_chunk_ids, case.relevant_chunk_ids
    return result.ranked_document_ids, case.relevant_document_ids


def _citation_label_level(result: RankedEvaluationResult) -> tuple[Sequence[str], Sequence[str]]:
    if result.cited_chunk_ids and result.selected_chunk_ids:
        return result.cited_chunk_ids, result.selected_chunk_ids
    if result.cited_document_ids:
        return result.cited_document_ids, result.selected_document_ids
    if result.cited_chunk_ids:
        return result.cited_chunk_ids, result.selected_chunk_ids
    if result.selected_chunk_ids:
        return result.cited_chunk_ids, result.selected_chunk_ids
    return result.cited_document_ids, result.selected_document_ids


def _aggregate(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record["metrics"] is not None]
    metric_names = (
        "recall_at_20", "hit_at_5", "mrr_at_10", "context_precision",
        "citation_validity", "citation_correctness", "abstention_accuracy",
    )
    aggregate: dict[str, Any] = {
        name: _mean([record["metrics"][name] for record in successful])
        for name in metric_names
    }
    results = [record["result"] for record in successful]
    p50, p95 = latency_percentiles([result["latency_ms"] for result in results])
    answer_p50, answer_p95 = latency_percentiles([
        result["answer_latency_ms"] for result in results if result["answer_latency_ms"] is not None
    ])
    aggregate.update({
        "latency_ms": {"p50": p50, "p95": p95},
        "answer_latency_ms": {"p50": answer_p50, "p95": answer_p95},
        "token_usage": {
            "input": _sum_optional(results, "input_tokens"),
            "output": _sum_optional(results, "output_tokens"),
        },
        "local_resources": {
            "cpu_percent_mean": _mean([result["local_cpu_percent"] for result in results]),
            "memory_mb_mean": _mean([result["local_memory_mb"] for result in results]),
        },
        "successful_case_count": len(successful),
        "failed_case_count": len(records) - len(successful),
    })
    return aggregate


def _attach_changes_from_baseline(records: dict[str, list[dict[str, Any]]]) -> None:
    """Add per-question metric deltas for rerankers relative to cached hybrid B."""
    baseline = {record["case_index"]: record for record in records["B"]}
    for configuration in ("C1", "C2"):
        for record in records[configuration]:
            base = baseline.get(record["case_index"])
            if record["metrics"] is None or base is None or base["metrics"] is None:
                record["changes_from_b"] = None
                continue
            record["changes_from_b"] = {
                name: _delta(record["metrics"][name], base["metrics"][name])
                for name in record["metrics"]
            }


def _delta(value: float | None, baseline: float | None) -> float | None:
    return value - baseline if value is not None and baseline is not None else None


def _mean(values: Sequence[float | None]) -> float | None:
    measured = [value for value in values if value is not None]
    return sum(measured) / len(measured) if measured else None


def _sum_optional(results: Sequence[dict[str, Any]], key: str) -> int | None:
    values = [result[key] for result in results if result[key] is not None]
    return sum(values) if values else None


def _safe_error_type(error: Exception) -> str:
    """Keep failure records actionable without serializing error content or secrets."""
    name = type(error).__name__
    return name if name in _SAFE_ERROR_TYPES else "Exception"


def _recommend(
    configurations: dict[str, dict[str, Any]], expected_case_indexes: frozenset[int]
) -> dict[str, str]:
    baseline = configurations["B"]
    if _successful_case_indexes(baseline) != expected_case_indexes:
        return _rrf_recommendation()
    baseline_metrics = baseline["aggregate"]
    candidates: list[tuple[float, float, str]] = []
    for name in ("C1", "C2"):
        configuration = configurations[name]
        candidate = configuration["aggregate"]
        mrr = candidate["mrr_at_10"]
        context = candidate["context_precision"]
        recall = candidate["recall_at_20"]
        latency = candidate["answer_latency_ms"]["p95"]
        if (
            _successful_case_indexes(configuration) == expected_case_indexes
            and mrr is not None
            and context is not None
            and recall is not None
            and latency is not None
            and baseline_metrics["mrr_at_10"] is not None
            and baseline_metrics["context_precision"] is not None
            and baseline_metrics["recall_at_20"] is not None
            and recall >= baseline_metrics["recall_at_20"]
            and latency <= 10_000
            and (mrr > baseline_metrics["mrr_at_10"] or context > baseline_metrics["context_precision"])
        ):
            candidates.append((mrr - baseline_metrics["mrr_at_10"], context - baseline_metrics["context_precision"], name))
    if not candidates:
        return _rrf_recommendation()
    _, _, name = max(candidates)
    return {
        "configuration": name,
        "reason": "Measured gain with no Recall@20 regression and P95 generated-answer latency at or below 10 seconds.",
    }


def _successful_case_indexes(configuration: dict[str, Any]) -> frozenset[int]:
    return frozenset(
        record["case_index"] for record in configuration["questions"] if record["metrics"] is not None
    )


def _rrf_recommendation() -> dict[str, str]:
    return {
        "configuration": "B",
        "reason": "No reranker met the complete-case, repeatable-gain, no-recall-regression, and 5–10 second target.",
    }


def _validate_candidates(candidates: Sequence[RetrievalCandidate]) -> None:
    for candidate in candidates:
        for field in ("score", "rrf_score", "rerank_score"):
            _validate_finite_observation(getattr(candidate, field, None), field)
        for score in getattr(candidate, "method_scores", {}).values():
            _validate_finite_observation(score, "method score")


def _validate_finite_observation(value: Any, name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite")
    try:
        normalized = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite") from error
    if not math.isfinite(normalized):
        raise ValueError(f"{name} must be finite")


def _elapsed_ms(started: float) -> float:
    elapsed = (time.perf_counter() - started) * 1_000
    _validate_finite_observation(elapsed, "latency")
    return elapsed


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load one validated evaluation case per non-empty JSONL line."""
    cases: list[EvaluationCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cases.append(EvaluationCase.model_validate_json(line))
        except ValueError as error:
            raise ValueError(f"Invalid evaluation case at line {line_number}") from error
    return cases


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Atomically write a JSON report without leaving partial output files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as temporary:
            temporary_name = temporary.name
            json.dump(report, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
        os.replace(temporary_name, path)
    except Exception:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None, *, runner_factory: Callable[[], EvaluationRunner] | None = None) -> int:
    """Run the evaluation harness through a small, testable CLI boundary."""
    parser = argparse.ArgumentParser(description="Evaluate A/B/C1/C2 retrieval configurations.")
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    runner = (runner_factory or _default_runner)()
    write_report(arguments.output, runner.run(load_cases(arguments.cases)))
    return 0


def _production_reranker_adapter(reranker: Any, *, top_k: int) -> CandidateReranker:
    """Adapt production rerankers to the injected evaluation callable contract."""
    def evaluate(case: EvaluationCase, candidates: list[RetrievalCandidate]) -> Sequence[RetrievalCandidate]:
        ranked = reranker.rerank(case.question, candidates, top_k)
        if getattr(reranker, "last_status", "ok") != "ok":
            raise RuntimeError("Evaluation reranker fell back")
        return ranked

    return evaluate


def _production_hybrid_adapter(pipeline: Any, *, top_k: int) -> CandidateRetriever:
    """Require a healthy vector+lexical hybrid result for evaluation configuration B."""
    def evaluate(case: EvaluationCase) -> Sequence[RetrievalCandidate]:
        result = pipeline.retrieve(case.question, collection_name="default", top_k=top_k)
        diagnostics = result.diagnostics if isinstance(getattr(result, "diagnostics", None), dict) else {}
        configured = set(diagnostics.get("configured_retrievers", ()))
        successful = set(diagnostics.get("successful_retrievers", ()))
        if (
            getattr(result, "retrieval_mode", "hybrid") != "hybrid"
            or getattr(result, "failures", ())
            or configured != {"vector", "lexical"}
            or successful != configured
        ):
            raise RuntimeError("Hybrid evaluation retrieval degraded")
        return result.candidates

    return evaluate


def _default_runner() -> EvaluationRunner:
    """Construct live integrations only when the operator explicitly invokes the CLI."""
    from app.config import settings
    from app.grounding.citations import CitationValidator
    from app.grounding.evidence import EvidenceSelector
    from app.grounding.generator import GroundedAnswerGenerator
    from app.grounding.models import GeneratedAnswer, REFUSAL_ANSWER
    from app.reranking import build_reranker
    from app.reranking.noop import NoOpReranker
    from app.retrieval.pipeline import HybridRetrievalPipeline
    from app.retrieval.vector import ChromaVectorRetriever

    vector = ChromaVectorRetriever()
    hybrid = HybridRetrievalPipeline(
        mode="hybrid",
        reranker=NoOpReranker(),
        exact_lookup=lambda _key, _collection, _filters: [],
        final_top_k=_EVALUATION_TOP_K,
    )
    gpt5 = build_reranker("openai")
    qwen = build_reranker("qwen_local")
    selector = EvidenceSelector()
    generator = GroundedAnswerGenerator()
    validator = CitationValidator()

    def vector_retrieval(case: EvaluationCase) -> Sequence[RetrievalCandidate]:
        return vector.search(case.question, settings.retrieval_candidate_top_k, None, "default")

    def evaluate(case: EvaluationCase, candidates: Sequence[RetrievalCandidate]) -> RankedEvaluationResult:
        started = time.perf_counter()
        evidence = selector.select(case.question, list(candidates), top_k=settings.grounding_evidence_top_k)
        generated = generator.generate(case.question, evidence) if evidence else GeneratedAnswer(
            answer=REFUSAL_ANSWER, citation_ids=[]
        )
        grounded = validator.validate(generated, evidence)
        selected_chunks = tuple(item.candidate.chunk_id for item in evidence)
        selected_documents = tuple(item.candidate.document_id for item in evidence)
        return RankedEvaluationResult(
            selected_document_ids=selected_documents,
            selected_chunk_ids=selected_chunks,
            cited_document_ids=tuple(item.document_id for item in grounded.citations),
            cited_chunk_ids=tuple(item.chunk_id for item in grounded.citations),
            refused=grounded.answer == REFUSAL_ANSWER,
            answer_latency_ms=_elapsed_ms(started),
        )

    return EvaluationRunner(
        vector_retrieval,
        _production_hybrid_adapter(hybrid, top_k=_EVALUATION_TOP_K),
        _production_reranker_adapter(gpt5, top_k=_EVALUATION_TOP_K),
        _production_reranker_adapter(qwen, top_k=_EVALUATION_TOP_K),
        grounded_answer_evaluator=evaluate,
    )


if __name__ == "__main__":  # pragma: no cover - exercised through main(argv)
    raise SystemExit(main())
