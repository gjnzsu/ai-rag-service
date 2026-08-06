import pytest

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


@pytest.mark.parametrize(
    ("ranked", "relevant", "expected_recall", "expected_hit", "expected_mrr"),
    [
        (("d1", "d2", "d3"), ("d2", "d4"), 0.5, 1.0, 0.5),
        (("d1", "d1", "d2"), ("d1", "d2"), 1.0, 1.0, 1.0),
        (("d1", "d2"), (), 0.0, 0.0, 0.0),
        ((), ("d1",), 0.0, 0.0, 0.0),
    ],
)
def test_rank_metrics_handle_exact_matches_duplicates_and_empty_sets(
    ranked,
    relevant,
    expected_recall,
    expected_hit,
    expected_mrr,
):
    assert recall_at_k(ranked, relevant, k=20) == expected_recall
    assert hit_at_k(ranked, relevant, k=5) == expected_hit
    assert mrr_at_k(ranked, relevant, k=10) == expected_mrr


@pytest.mark.parametrize(
    ("ranked", "relevant", "expected"),
    [
        (("d1", "d2", "d3"), ("d1", "d3"), 2 / 3),
        (("d1", "d1", "d2"), ("d1",), 0.5),
        ((), (), 0.0),
    ],
)
def test_context_precision_has_explicit_empty_and_duplicate_conventions(ranked, relevant, expected):
    assert context_precision(ranked, relevant, k=20) == expected


@pytest.mark.parametrize(
    ("cited", "selected", "labels", "expected_validity", "expected_correctness"),
    [
        (("c1", "c2"), ("c1", "c3"), (True, False), 0.5, 0.5),
        ((), ("c1",), (), 1.0, None),
    ],
)
def test_citation_metrics_handle_invalid_duplicate_and_unlabeled_citations(
    cited,
    selected,
    labels,
    expected_validity,
    expected_correctness,
):
    assert citation_validity(cited, selected) == expected_validity
    assert citation_correctness(labels) == expected_correctness


@pytest.mark.parametrize(
    ("should_abstain", "refused", "expected"),
    [(True, True, 1.0), (False, False, 1.0), (True, False, 0.0)],
)
def test_abstention_accuracy_is_binary_per_case(should_abstain, refused, expected):
    assert abstention_accuracy(should_abstain, refused) == expected


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ((0.0,), (0.0, 0.0)),
        ((10.0, 20.0, 30.0, 40.0), (25.0, 38.5)),
        ((), (None, None)),
    ],
)
def test_latency_percentiles_include_zero_and_empty_samples(samples, expected):
    assert latency_percentiles(samples) == expected
