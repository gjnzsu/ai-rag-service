from concurrent.futures import Future

import pytest

from app.reranking.qwen import Qwen3LocalReranker
from app.retrieval.models import RetrievalCandidate


def _candidate(chunk_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        content=f"passage {chunk_id}",
        document_id=f"doc:{chunk_id}",
        chunk_id=chunk_id,
        metadata={"nested": {"id": chunk_id}},
        rrf_score=1.0,
    )


class _InlineExecutor:
    def __init__(self):
        self.submissions = 0

    def submit(self, function, *args, **kwargs):
        self.submissions += 1
        future = Future()
        try:
            future.set_result(function(*args, **kwargs))
        except BaseException as error:
            future.set_exception(error)
        return future


def _reranker(*, revision="test-revision", scorer=None, tokenizer_loader=None, model_loader=None, **kwargs):
    return Qwen3LocalReranker(
        model_name="Qwen/Qwen3-Reranker-0.6B",
        revision=revision,
        max_candidates=kwargs.pop("max_candidates", 20),
        max_length=kwargs.pop("max_length", 128),
        batch_size=kwargs.pop("batch_size", 3),
        timeout_seconds=kwargs.pop("timeout_seconds", 1.0),
        circuit_breaker_seconds=kwargs.pop("circuit_breaker_seconds", 10.0),
        tokenizer_loader=tokenizer_loader or (lambda *args, **options: object()),
        model_loader=model_loader or (lambda *args, **options: _FakeModel()),
        scorer=scorer or (lambda model, tokenizer, pairs, max_length: [float(index) for index, _ in enumerate(pairs)]),
        executor=kwargs.pop("executor", _InlineExecutor()),
        **kwargs,
    )


class _FakeModel:
    def __init__(self):
        self.eval_calls = 0

    def eval(self):
        self.eval_calls += 1
        return self


def test_qwen_process_cache_loads_model_once_per_model_and_revision():
    load_calls = []
    tokenizer = object()
    model = _FakeModel()

    def tokenizer_loader(model_name, *, revision):
        load_calls.append(("tokenizer", model_name, revision))
        return tokenizer

    def model_loader(model_name, *, revision):
        load_calls.append(("model", model_name, revision))
        return model

    first = _reranker(
        revision="cache-once-revision", tokenizer_loader=tokenizer_loader, model_loader=model_loader
    )
    second = _reranker(
        revision="cache-once-revision", tokenizer_loader=tokenizer_loader, model_loader=model_loader
    )

    first.rerank("query", [_candidate("a")], top_k=1)
    second.rerank("query", [_candidate("b")], top_k=1)

    assert load_calls == [
        ("tokenizer", "Qwen/Qwen3-Reranker-0.6B", "cache-once-revision"),
        ("model", "Qwen/Qwen3-Reranker-0.6B", "cache-once-revision"),
    ]
    assert model.eval_calls == 1


def test_qwen_pins_revision_caps_candidates_bounds_batches_and_formats_pairs():
    load_calls = []
    score_calls = []

    def tokenizer_loader(model_name, *, revision):
        load_calls.append(("tokenizer", model_name, revision))
        return object()

    def model_loader(model_name, *, revision):
        load_calls.append(("model", model_name, revision))
        return _FakeModel()

    def scorer(model, tokenizer, pairs, max_length):
        score_calls.append((pairs, max_length))
        return [1.0] * len(pairs)

    candidates = [_candidate(f"c-{index}") for index in range(25)]
    reranker = _reranker(
        revision="pinned-sha", tokenizer_loader=tokenizer_loader, model_loader=model_loader,
        scorer=scorer, batch_size=3, max_length=64,
    )

    result = reranker.rerank("customer query", candidates, top_k=20)

    assert load_calls == [
        ("tokenizer", "Qwen/Qwen3-Reranker-0.6B", "pinned-sha"),
        ("model", "Qwen/Qwen3-Reranker-0.6B", "pinned-sha"),
    ]
    assert len(score_calls) == 7
    assert all(len(pairs) <= 3 and max_length == 64 for pairs, max_length in score_calls)
    flattened = [pair for pairs, _ in score_calls for pair in pairs]
    assert len(flattened) == 20
    assert all("<Query>: customer query" in pair for pair in flattened)
    assert all("<Document>: passage c-" in pair for pair in flattened)
    assert all("untrusted" in pair.lower() for pair in flattened)
    assert all("<|im_start|>assistant\n<think>\n\n</think>\n\n" in pair for pair in flattened)
    assert [item.chunk_id for item in result] == [f"c-{index}" for index in range(20)]


def test_qwen_orders_descending_with_stable_ties_and_does_not_mutate_inputs():
    candidates = [_candidate(chunk_id) for chunk_id in ["a", "b", "c", "d"]]
    before = [candidate.model_copy(deep=True) for candidate in candidates]

    reranker = _reranker(
        revision="ranking-revision",
        scorer=lambda model, tokenizer, pairs, max_length: [0.1, 0.9, 0.9, -0.2],
        batch_size=4,
    )
    result = reranker.rerank("query", candidates, top_k=3)

    assert [item.chunk_id for item in result] == ["b", "c", "a"]
    assert [item.rerank_score for item in result] == [0.9, 0.9, 0.1]
    assert candidates == before
    assert all(result_item is not source for result_item, source in zip(result, [candidates[1], candidates[2], candidates[0]], strict=True))
    result[0].metadata["nested"]["id"] = "changed"
    assert candidates[1].metadata["nested"]["id"] == "b"


@pytest.mark.parametrize("failure_stage", ["optional_import", "model_load", "inference"])
def test_qwen_import_load_or_inference_failure_returns_exact_rrf_order(failure_stage):
    def fail(*args, **kwargs):
        error = ImportError("torch missing") if failure_stage == "optional_import" else RuntimeError("secret")
        raise error

    kwargs = {"revision": f"failure-{failure_stage}"}
    if failure_stage == "optional_import":
        kwargs["tokenizer_loader"] = fail
    elif failure_stage == "model_load":
        kwargs["model_loader"] = fail
    else:
        kwargs["scorer"] = fail
    candidates = [_candidate("a"), _candidate("b"), _candidate("c")]
    reranker = _reranker(**kwargs)

    result = reranker.rerank("query", candidates, top_k=2)

    assert result == candidates[:2]
    assert all(item is not source for item, source in zip(result, candidates[:2], strict=True))
    assert reranker.last_status == "fallback"
    assert reranker.last_error_type in {"ImportError", "RuntimeError"}


def test_qwen_timeout_opens_circuit_and_prevents_more_work_until_cooldown():
    completed = [False]

    class _TimeoutFuture:
        def result(self, timeout):
            assert timeout == 0.25
            raise TimeoutError("slow inference")

        def cancel(self):
            return False

        def done(self):
            return completed[0]

    class _TimeoutExecutor:
        def __init__(self):
            self.submissions = 0

        def submit(self, function, *args):
            self.submissions += 1
            return _TimeoutFuture()

    now = [100.0]
    executor = _TimeoutExecutor()
    candidates = [_candidate("a"), _candidate("b")]
    reranker = _reranker(
        revision="timeout-revision", executor=executor, clock=lambda: now[0],
        timeout_seconds=0.25, circuit_breaker_seconds=10.0,
    )

    first = reranker.rerank("query", candidates, top_k=2)
    second = reranker.rerank("query", candidates, top_k=2)
    now[0] = 111.0
    third = reranker.rerank("query", candidates, top_k=2)
    assert executor.submissions == 1
    completed[0] = True
    now[0] = 122.0
    fourth = reranker.rerank("query", candidates, top_k=2)

    assert first == second == third == fourth == candidates
    assert executor.submissions == 2


def test_qwen_rejects_invalid_score_count_or_non_finite_scores():
    candidates = [_candidate("a"), _candidate("b")]
    wrong_count = _reranker(revision="wrong-count", scorer=lambda *args: [1.0])
    non_finite = _reranker(revision="non-finite", scorer=lambda *args: [1.0, float("nan")])

    assert wrong_count.rerank("query", candidates, top_k=2) == candidates
    assert non_finite.rerank("query", candidates, top_k=2) == candidates


def test_qwen_empty_input_zero_top_k_and_negative_top_k():
    executor = _InlineExecutor()
    reranker = _reranker(revision="empty-revision", executor=executor)

    assert reranker.rerank("query", [], top_k=2) == []
    assert reranker.rerank("query", [_candidate("a")], top_k=0) == []
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("query", [_candidate("a")], top_k=-1)
    assert executor.submissions == 0
