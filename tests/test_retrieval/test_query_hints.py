import re

import pytest

from app.retrieval.query_hints import extract_query_hints


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Check PROJ-7", ["PROJ-7"]),
        ("Check proj-7", ["PROJ-7"]),
        ("Compare OPS-2 then APP9-10", ["OPS-2", "APP9-10"]),
        ("PROJ-7, proj-7; and PROJ-8.", ["PROJ-7", "PROJ-8"]),
        ("(PROJ-7)!", ["PROJ-7"]),
        ("No ticket here: project-abc, A-1, -7, PROJ7", []),
    ],
)
def test_extract_query_hints_normalizes_jira_keys_in_encounter_order(query, expected):
    assert extract_query_hints(query).jira_keys == expected


def test_extract_query_hints_accepts_a_configurable_pattern():
    assert extract_query_hints("work item #42", pattern=r"#\d+").jira_keys == ["#42"]


@pytest.mark.parametrize("pattern", ["", "["])
def test_extract_query_hints_rejects_empty_or_invalid_explicit_patterns(pattern):
    with pytest.raises((ValueError, re.error)):
        extract_query_hints("PROJ-7", pattern=pattern)
