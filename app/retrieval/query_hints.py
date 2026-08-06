"""Deterministic query hints used by retrieval orchestration."""

import re
from dataclasses import dataclass

from app.config import settings


@dataclass(frozen=True)
class QueryHints:
    """Normalized query hints, kept in their original encounter order."""

    jira_keys: list[str]


def extract_query_hints(query: str, pattern: str | None = None) -> QueryHints:
    """Find Jira issue keys without using a model or query planner."""
    selected_pattern = settings.jira_key_pattern if pattern is None else pattern
    if not selected_pattern:
        raise ValueError("Jira-key pattern must not be empty")
    compiled = re.compile(selected_pattern, flags=re.IGNORECASE)
    seen: set[str] = set()
    jira_keys: list[str] = []
    for match in compiled.finditer(query):
        key = match.group(0).upper()
        if key not in seen:
            seen.add(key)
            jira_keys.append(key)
    return QueryHints(jira_keys=jira_keys)
