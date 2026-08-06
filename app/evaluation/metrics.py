"""Pure, deterministic formulas used by the offline evaluation harness."""

from collections.abc import Sequence
import math


def recall_at_k(
    ranked_ids: Sequence[str], relevant_ids: Sequence[str], *, k: int = 20
) -> float:
    """Return unique relevant IDs recovered in the first ``k`` ranks.

    Empty relevant or ranked sets return ``0.0``. Duplicate ranked IDs count once.
    """
    relevant = set(relevant_ids)
    if not relevant or k <= 0:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant) / len(relevant)


def hit_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], *, k: int = 5) -> float:
    """Return ``1.0`` if a relevant ID appears in the first ``k`` ranks, else ``0.0``.

    Empty relevant or ranked sets return ``0.0``.
    """
    if k <= 0 or not relevant_ids:
        return 0.0
    return float(bool(set(ranked_ids[:k]) & set(relevant_ids)))


def mrr_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], *, k: int = 10) -> float:
    """Return reciprocal rank of the first relevant item in the first ``k`` ranks.

    Empty relevant or ranked sets, or no match, return ``0.0``. Duplicate IDs retain
    their first observed rank.
    """
    relevant = set(relevant_ids)
    if k <= 0 or not relevant:
        return 0.0
    for index, item_id in enumerate(ranked_ids[:k], start=1):
        if item_id in relevant:
            return 1.0 / index
    return 0.0


def context_precision(
    ranked_ids: Sequence[str], relevant_ids: Sequence[str], *, k: int = 20
) -> float:
    """Return precision over unique ranked contexts in the first ``k`` positions.

    Empty relevant or ranked sets return ``0.0``. Repeated retrieved IDs are counted
    once, preventing duplicate chunks from inflating or deflating the score.
    """
    unique_ranked = tuple(dict.fromkeys(ranked_ids[: max(k, 0)]))
    if not unique_ranked or not relevant_ids:
        return 0.0
    return len(set(unique_ranked) & set(relevant_ids)) / len(unique_ranked)


def citation_validity(cited_ids: Sequence[str], selected_ids: Sequence[str]) -> float:
    """Return the share of unique cited IDs that were selected as evidence.

    No citations returns ``1.0`` because there are no invalid references. Citations
    with an empty selected set return ``0.0``. Duplicate citations count once.
    """
    cited = set(cited_ids)
    if not cited:
        return 1.0
    return len(cited & set(selected_ids)) / len(cited)


def citation_correctness(human_labels: Sequence[bool] | None) -> float | None:
    """Return the mean human correctness label, or ``None`` when none were supplied.

    Missing or empty human labels are deliberately unmeasured rather than assumed
    correct; this function never fabricates a correctness score.
    """
    if not human_labels:
        return None
    return sum(human_labels) / len(human_labels)


def abstention_accuracy(should_abstain: bool, refused: bool) -> float:
    """Return ``1.0`` when the refusal matches the labelled abstention expectation."""
    return float(should_abstain is refused)


def latency_percentiles(samples_ms: Sequence[float]) -> tuple[float | None, float | None]:
    """Return linearly interpolated P50 and P95 latency in milliseconds.

    An empty sample returns ``(None, None)``. Zero-valued samples are valid and
    preserved, which is useful for deterministic unit-test fakes.
    """
    if not samples_ms:
        return None, None
    ordered = sorted(float(sample) for sample in samples_ms)
    if any(sample < 0 or not math.isfinite(sample) for sample in ordered):
        raise ValueError("latency samples must be finite and non-negative")
    return _percentile(ordered, 0.50), _percentile(ordered, 0.95)


def _percentile(ordered: Sequence[float], percentile: float) -> float:
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
