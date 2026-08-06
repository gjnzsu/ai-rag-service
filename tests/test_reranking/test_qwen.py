from concurrent.futures import Future
import hashlib
import math
import sys
from types import SimpleNamespace

import pytest

from app.reranking.qwen import Qwen3LocalReranker, _load_tokenizer, _score_batch
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
    revision = _commit(revision) if len(revision) != 40 else revision
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


def _commit(label: str) -> str:
    return hashlib.sha1(label.encode(), usedforsecurity=False).hexdigest()


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
        ("tokenizer", "Qwen/Qwen3-Reranker-0.6B", _commit("cache-once-revision")),
        ("model", "Qwen/Qwen3-Reranker-0.6B", _commit("cache-once-revision")),
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
        ("tokenizer", "Qwen/Qwen3-Reranker-0.6B", _commit("pinned-sha")),
        ("model", "Qwen/Qwen3-Reranker-0.6B", _commit("pinned-sha")),
    ]
    assert len(score_calls) == 7
    assert all(len(pairs) <= 3 and max_length == 64 for pairs, max_length in score_calls)
    flattened = [pair for pairs, _ in score_calls for pair in pairs]
    assert len(flattened) == 20
    assert all(pair.startswith("<Instruct>: Given a web search query") for pair in flattened)
    assert all("<Query>: customer query" in pair for pair in flattened)
    assert all("<Document>: passage c-" in pair for pair in flattened)
    assert all("<|im_start|>assistant" not in pair for pair in flattened)
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


def test_qwen_failure_fallback_uses_full_input_beyond_scoring_cap():
    candidates = [_candidate(chunk_id) for chunk_id in ["a", "b", "c", "d"]]
    reranker = _reranker(
        revision="full-fallback",
        max_candidates=2,
        scorer=lambda *args: (_ for _ in ()).throw(RuntimeError("inference failed")),
    )

    result = reranker.rerank("query", candidates, top_k=4)

    assert result == candidates
    assert all(item is not source for item, source in zip(result, candidates, strict=True))


def test_qwen_success_appends_unscored_candidates_after_bounded_ranked_prefix():
    candidates = [_candidate(chunk_id) for chunk_id in ["a", "b", "c", "d"]]
    reranker = _reranker(
        revision="bounded-success",
        max_candidates=2,
        batch_size=2,
        scorer=lambda *args: [0.1, 0.9],
    )

    result = reranker.rerank("query", candidates, top_k=4)

    assert [item.chunk_id for item in result] == ["b", "a", "c", "d"]
    assert [item.rerank_score for item in result] == [0.9, 0.1, None, None]
    assert all(result_item is not source for result_item, source in zip(result, [candidates[1], candidates[0], candidates[2], candidates[3]], strict=True))


def test_qwen_timeout_and_inflight_slot_are_shared_across_adapters():
    class _PendingFuture:
        def result(self, timeout):
            raise TimeoutError("still running")

        def cancel(self):
            return False

        def done(self):
            return False

    class _Executor:
        def __init__(self):
            self.submissions = 0

        def submit(self, function, *args):
            self.submissions += 1
            return _PendingFuture()

    first_executor = _Executor()
    second_executor = _InlineExecutor()
    revision = "shared-timeout-state"
    first = _reranker(revision=revision, executor=first_executor)
    second = _reranker(revision=revision, executor=second_executor)
    candidates = [_candidate("a"), _candidate("b")]

    first_result = first.rerank("query", candidates, top_k=2)
    second_result = second.rerank("query", candidates, top_k=2)

    assert first_result == second_result == candidates
    assert first_executor.submissions == 1
    assert second_executor.submissions == 0
    assert second.last_error_type in {"CircuitOpenError", "InferenceBusyError"}


def test_qwen_multiple_adapters_create_only_one_process_executor(monkeypatch):
    created = []

    def executor_factory(*args, **kwargs):
        executor = _InlineExecutor()
        created.append(executor)
        return executor

    monkeypatch.setattr("app.reranking.qwen.ThreadPoolExecutor", executor_factory)
    revision = _commit("one-process-executor")
    kwargs = {
        "revision": revision,
        "tokenizer_loader": lambda *args, **options: object(),
        "model_loader": lambda *args, **options: _FakeModel(),
        "scorer": lambda model, tokenizer, pairs, max_length: [1.0] * len(pairs),
    }
    first = Qwen3LocalReranker(**kwargs)
    second = Qwen3LocalReranker(**kwargs)

    first.rerank("query", [_candidate("a")], top_k=1)
    second.rerank("query", [_candidate("b")], top_k=1)

    assert len(created) == 1


def test_qwen_empty_input_zero_top_k_and_negative_top_k():
    executor = _InlineExecutor()
    reranker = _reranker(revision="empty-revision", executor=executor)

    assert reranker.rerank("query", [], top_k=2) == []
    assert reranker.rerank("query", [_candidate("a")], top_k=0) == []
    with pytest.raises(ValueError, match="top_k"):
        reranker.rerank("query", [_candidate("a")], top_k=-1)
    assert executor.submissions == 0


@pytest.mark.parametrize(
    ("model_name", "revision"),
    [
        ("Qwen/Qwen3-Reranker-4B", "e61197ed45024b0ed8a2d74b80b4d909f1255473"),
        ("Qwen/Qwen3-Reranker-0.6B", "main"),
        ("Qwen/Qwen3-Reranker-0.6B", "e61197e"),
        ("Qwen/Qwen3-Reranker-0.6B", "z" * 40),
    ],
)
def test_qwen_rejects_wrong_model_or_non_commit_revision(model_name, revision):
    with pytest.raises(ValueError, match="model|revision"):
        Qwen3LocalReranker(model_name=model_name, revision=revision)


def test_qwen_default_tokenizer_loader_requires_left_padding(monkeypatch):
    calls = []

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return object()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=_AutoTokenizer))

    _load_tokenizer("Qwen/Qwen3-Reranker-0.6B", revision="a" * 40)

    assert calls == [(
        "Qwen/Qwen3-Reranker-0.6B",
        {"revision": "a" * 40, "padding_side": "left"},
    )]


def test_qwen_official_token_recipe_reserves_suffix_and_scores_final_yes_no_logits():
    prefix_ids = [90, 91]
    suffix_ids = [98, 99]

    class _FakeTensor:
        def __init__(self, values):
            self.values = values

        def to(self, device):
            return self

    class _FinalLogits:
        def __init__(self, rows):
            self.rows = rows

        def detach(self):
            return self

        def float(self):
            return self

        def cpu(self):
            return self

        def tolist(self):
            return self.rows

    class _Logits:
        def __init__(self, rows):
            self.rows = rows
            self.indexes = []

        def __getitem__(self, indexes):
            self.indexes.append(indexes)
            return _FinalLogits(self.rows)

    class _Tokenizer:
        def __init__(self):
            self.call = None
            self.pad_call = None
            self.token_lookups = []

        def encode(self, text, *, add_special_tokens):
            assert add_special_tokens is False
            if text.startswith("<|im_start|>system"):
                return prefix_ids
            if text.startswith("<|im_end|>"):
                return suffix_ids
            raise AssertionError(f"unexpected encoded text: {text}")

        def __call__(self, pairs, **kwargs):
            self.call = (pairs, kwargs)
            body_limit = kwargs["max_length"]
            return {"input_ids": [
                list(range(1, body_limit + 1)),
                list(range(11, 11 + body_limit - 2)),
            ]}

        def pad(self, inputs, **kwargs):
            self.pad_call = ({"input_ids": [list(ids) for ids in inputs["input_ids"]]}, kwargs)
            width = max(len(ids) for ids in inputs["input_ids"])
            padded = [[0] * (width - len(ids)) + ids for ids in inputs["input_ids"]]
            masks = [[0] * (width - len(ids)) + [1] * len(ids) for ids in inputs["input_ids"]]
            return {"input_ids": _FakeTensor(padded), "attention_mask": _FakeTensor(masks)}

        def convert_tokens_to_ids(self, token):
            self.token_lookups.append(token)
            return {"no": 3, "yes": 7}[token]

    rows = [[0.0] * 10 for _ in range(2)]
    rows[0][3], rows[0][7] = 0.0, 2.0
    rows[1][3], rows[1][7] = 2.0, 0.0
    logits = _Logits(rows)

    class _Model:
        device = "cpu"

        def __init__(self):
            self.inputs = None

        def __call__(self, **inputs):
            self.inputs = inputs
            return SimpleNamespace(logits=logits)

    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *args):
            return False

    tokenizer = _Tokenizer()
    model = _Model()
    fake_torch = SimpleNamespace(no_grad=lambda: _NoGrad())

    scores = _score_batch(
        model,
        tokenizer,
        ["first body", "second body"],
        max_length=10,
        torch_module=fake_torch,
    )

    _, tokenize_kwargs = tokenizer.call
    assert tokenize_kwargs == {
        "padding": False,
        "truncation": "longest_first",
        "return_attention_mask": False,
        "max_length": 6,
    }
    padded_inputs, pad_kwargs = tokenizer.pad_call
    assert pad_kwargs == {"padding": True, "return_tensors": "pt", "max_length": 10}
    assert all(ids[:2] == prefix_ids and ids[-2:] == suffix_ids for ids in padded_inputs["input_ids"])
    assert all(len(ids) <= 10 for ids in padded_inputs["input_ids"])
    assert model.inputs["input_ids"].values[1][0:2] == [0, 0]
    assert model.inputs["input_ids"].values[1][-2:] == suffix_ids
    assert tokenizer.token_lookups == ["yes", "no"]
    assert logits.indexes == [(slice(None), -1, slice(None))]
    assert scores == pytest.approx([
        1 / (1 + math.exp(-2.0)),
        1 / (1 + math.exp(2.0)),
    ])
