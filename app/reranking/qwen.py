"""Lazy, fail-open local Qwen reranking adapter."""

from concurrent.futures import ThreadPoolExecutor
import math
import re
from threading import Lock
import time
from typing import Any, Callable

from app.reranking.base import validate_top_k
from app.retrieval.models import RetrievalCandidate

QWEN_MODEL_NAME = "Qwen/Qwen3-Reranker-0.6B"
MAX_CANDIDATES = 20
_DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the Instruct "
    'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = Lock()
_COMMIT_SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_EXECUTION_STATES: dict[tuple[str, str], "_ExecutionState"] = {}
_EXECUTION_STATES_LOCK = Lock()


class _ExecutionState:
    def __init__(self, executor: Any) -> None:
        self.executor = executor
        self.lock = Lock()
        self.inflight_future: Any | None = None
        self.circuit_open_until = 0.0


class Qwen3LocalReranker:
    """Score query-passage pairs locally within bounded operational limits."""

    provider = "qwen_local"

    def __init__(
        self,
        *,
        model_name: str = QWEN_MODEL_NAME,
        revision: str,
        max_candidates: int = MAX_CANDIDATES,
        max_length: int = 512,
        batch_size: int = 4,
        timeout_seconds: float = 5.0,
        circuit_breaker_seconds: float = 30.0,
        tokenizer_loader: Callable[..., Any] | None = None,
        model_loader: Callable[..., Any] | None = None,
        scorer: Callable[[Any, Any, list[str], int], Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
        executor: Any | None = None,
    ) -> None:
        if model_name != QWEN_MODEL_NAME:
            raise ValueError(f"model_name must be {QWEN_MODEL_NAME}")
        if not _COMMIT_SHA_PATTERN.fullmatch(revision):
            raise ValueError("revision must be a full commit SHA")
        if max_candidates <= 0 or max_length <= 0 or batch_size <= 0:
            raise ValueError("Qwen bounds must be positive")
        if timeout_seconds <= 0 or circuit_breaker_seconds < 0:
            raise ValueError("Qwen time limits are invalid")
        self.model_name = model_name
        self.revision = revision
        self.max_candidates = min(max_candidates, MAX_CANDIDATES)
        self.max_length = max_length
        self.batch_size = batch_size
        self.timeout_seconds = timeout_seconds
        self.circuit_breaker_seconds = circuit_breaker_seconds
        self.tokenizer_loader = tokenizer_loader or _load_tokenizer
        self.model_loader = model_loader or _load_model
        self.scorer = scorer or _score_batch
        self.clock = clock
        self._executor_override = executor
        self.last_status = "ok"
        self.last_error_type: str | None = None

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalCandidate],
        top_k: int,
    ) -> list[RetrievalCandidate]:
        validate_top_k(top_k)
        self.last_status = "ok"
        self.last_error_type = None
        bounded = candidates[: self.max_candidates]
        fallback = _copies(candidates[:top_k])
        if not bounded or top_k == 0:
            return fallback
        execution_state = self._execution_state()
        if self._circuit_is_open(execution_state):
            self.last_status = "fallback"
            self.last_error_type = "CircuitOpenError"
            return fallback

        pairs = [_format_pair(query, candidate.content) for candidate in bounded]
        future = None
        try:
            future = self._submit_if_idle(execution_state, pairs)
            if future is None:
                self._open_circuit(execution_state)
                self.last_status = "fallback"
                self.last_error_type = "InferenceBusyError"
                return fallback
            scores = future.result(timeout=self.timeout_seconds)
            validated = _validate_scores(scores, len(bounded))
        except Exception as error:
            if future is not None:
                future.cancel()
                if future.done():
                    self._clear_inflight(execution_state, future)
            self._open_circuit(execution_state)
            self.last_status = "fallback"
            self.last_error_type = type(error).__name__
            return fallback
        self._clear_inflight(execution_state, future)

        ordered = sorted(
            enumerate(bounded),
            key=lambda indexed: (-validated[indexed[0]], indexed[0]),
        )
        ordered_sources = [candidate for _, candidate in ordered]
        ordered_sources.extend(candidates[len(bounded) :])
        outputs = _copies(ordered_sources[:top_k])
        for output, (source_index, _) in zip(outputs, ordered, strict=False):
            output.rerank_score = validated[source_index]
        return outputs

    def _score_pairs(self, pairs: list[str]) -> list[float]:
        tokenizer, model = self._load_cached()
        scores: list[float] = []
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start : start + self.batch_size]
            batch_scores = self.scorer(model, tokenizer, batch, self.max_length)
            scores.extend(float(score) for score in batch_scores)
        return scores

    def _load_cached(self) -> tuple[Any, Any]:
        cache_key = (self.model_name, self.revision)
        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                return cached
            tokenizer = self.tokenizer_loader(self.model_name, revision=self.revision)
            model = self.model_loader(self.model_name, revision=self.revision)
            model.eval()
            cached = (tokenizer, model)
            _MODEL_CACHE[cache_key] = cached
            return cached

    def _execution_state(self) -> _ExecutionState:
        cache_key = (self.model_name, self.revision)
        with _EXECUTION_STATES_LOCK:
            state = _EXECUTION_STATES.get(cache_key)
            if state is None:
                executor = self._executor_override or ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="qwen-reranker",
                )
                state = _ExecutionState(executor)
                _EXECUTION_STATES[cache_key] = state
            return state

    def _submit_if_idle(self, state: _ExecutionState, pairs: list[str]) -> Any | None:
        with state.lock:
            if state.inflight_future is not None:
                if not state.inflight_future.done():
                    return None
                state.inflight_future = None
            future = state.executor.submit(self._score_pairs, pairs)
            state.inflight_future = future
            return future

    def _clear_inflight(self, state: _ExecutionState, future: Any) -> None:
        with state.lock:
            if state.inflight_future is future:
                state.inflight_future = None

    def _circuit_is_open(self, state: _ExecutionState) -> bool:
        with state.lock:
            return self.clock() < state.circuit_open_until

    def _open_circuit(self, state: _ExecutionState) -> None:
        with state.lock:
            state.circuit_open_until = self.clock() + self.circuit_breaker_seconds


def _format_pair(query: str, passage: str) -> str:
    return (
        f"<Instruct>: {_DEFAULT_INSTRUCTION}\n"
        f"<Query>: {query}\n"
        f"<Document>: {passage}"
    )


def _load_tokenizer(model_name: str, *, revision: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, revision=revision, padding_side="left")


def _load_model(model_name: str, *, revision: str) -> Any:
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(model_name, revision=revision)


def _score_batch(
    model: Any,
    tokenizer: Any,
    pairs: list[str],
    max_length: int,
    *,
    torch_module: Any | None = None,
) -> list[float]:
    if torch_module is None:
        import torch as torch_module

    prefix_tokens = tokenizer.encode(_PREFIX, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(_SUFFIX, add_special_tokens=False)
    body_budget = max_length - len(prefix_tokens) - len(suffix_tokens)
    if body_budget <= 0:
        raise ValueError("max_length cannot fit the Qwen prefix and suffix")
    encoded = tokenizer(
        pairs,
        padding=False,
        truncation="longest_first",
        return_attention_mask=False,
        max_length=body_budget,
    )
    for index, input_ids in enumerate(encoded["input_ids"]):
        encoded["input_ids"][index] = prefix_tokens + input_ids + suffix_tokens
    encoded = tokenizer.pad(
        encoded,
        padding=True,
        return_tensors="pt",
        max_length=max_length,
    )
    try:
        device = model.device
    except AttributeError:
        device = next(model.parameters()).device
    try:
        encoded = {name: value.to(device) for name, value in encoded.items()}
    except AttributeError:
        pass
    with torch_module.no_grad():
        final_logits = model(**encoded).logits[:, -1, :]
    rows = final_logits.detach().float().cpu().tolist()
    yes_id = tokenizer.convert_tokens_to_ids("yes")
    no_id = tokenizer.convert_tokens_to_ids("no")
    # This model-local relevance signal is not a calibrated correctness probability.
    return [_yes_relevance_signal(row[yes_id], row[no_id]) for row in rows]


def _yes_relevance_signal(yes_logit: float, no_logit: float) -> float:
    difference = float(yes_logit) - float(no_logit)
    if difference >= 0:
        return 1 / (1 + math.exp(-difference))
    exponential = math.exp(difference)
    return exponential / (1 + exponential)


def _validate_scores(scores: Any, expected_count: int) -> list[float]:
    values = list(scores)
    if len(values) != expected_count:
        raise ValueError("Qwen returned the wrong score count")
    validated = [float(value) for value in values]
    if not all(math.isfinite(value) for value in validated):
        raise ValueError("Qwen returned a non-finite score")
    return validated


def _copies(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    return [candidate.model_copy(deep=True) for candidate in candidates]
