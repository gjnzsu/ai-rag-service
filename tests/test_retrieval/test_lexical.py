from app.retrieval.lexical import SQLiteFTSIndex


def _chunk(
    chunk_id: str,
    content: str,
    *,
    document_id: str = "jira:PROJ-7",
    issue_key: str = "PROJ-7",
    title: str = "Login audit",
    source_type: str = "jira",
    source_url: str = "https://jira.example/browse/PROJ-7",
    **metadata,
) -> dict:
    return {
        "id": chunk_id,
        "chunk_id": chunk_id,
        "chunk_index": 0,
        "content": content,
        "document_id": document_id,
        "source_type": source_type,
        "source_url": source_url,
        "title": title,
        "issue_key": issue_key,
        "metadata": {"issue_key": issue_key, **metadata},
        **metadata,
    }


def test_fts_upsert_replaces_all_chunks_for_document(tmp_path):
    index = SQLiteFTSIndex(tmp_path / "lexical.db")
    old = [_chunk("jira:PROJ-7:chunk:0", "old term")]
    new = [_chunk("jira:PROJ-7:chunk:0", "new term")]

    index.upsert_document(old, "alpha")
    index.upsert_document(new, "alpha")

    assert [item.chunk_id for item in index.search("new", 10, None, "alpha")] == [
        "jira:PROJ-7:chunk:0"
    ]
    assert index.search("old", 10, None, "alpha") == []


def test_fts_is_collection_filter_and_weight_aware(tmp_path):
    index = SQLiteFTSIndex(tmp_path / "lexical.db")
    identifier = _chunk(
        "jira:PROJ-7:chunk:0",
        "ordinary body",
        project_key="PROJ",
    )
    body = _chunk(
        "jira:PROJ-8:chunk:0",
        "PROJ-7 is mentioned only in body",
        document_id="jira:PROJ-8",
        issue_key="PROJ-8",
        title="Ordinary title",
        project_key="OTHER",
    )
    other_collection = _chunk(
        "jira:PROJ-9:chunk:0",
        "login audit",
        document_id="jira:PROJ-9",
        issue_key="PROJ-9",
    )

    index.upsert_document([identifier], "alpha")
    index.upsert_document([body], "alpha")
    index.upsert_document([other_collection], "beta")

    results = index.search("PROJ-7", 10, {"project_key": {"in": ["PROJ"]}}, "alpha")

    assert [item.chunk_id for item in results] == ["jira:PROJ-7:chunk:0"]
    assert results[0].collection_name == "alpha"
    assert results[0].source_type == "jira"
    assert index.search("ordinary", 10, {"source_type": "jira"}, "beta") == []


def test_fts_delete_handles_empty_and_unsafe_queries_and_maps_trusted_fields(tmp_path):
    index = SQLiteFTSIndex(tmp_path / "lexical.db")
    chunk = _chunk("jira:PROJ-7:chunk:0", "login audit evidence", status="Open")
    index.upsert_document([chunk], "alpha")

    result = index.search("login!!! \" *", 10, {"status": "Open"}, "alpha")[0]

    assert result.document_id == "jira:PROJ-7"
    assert result.source_url == "https://jira.example/browse/PROJ-7"
    assert result.metadata == {"issue_key": "PROJ-7", "status": "Open"}
    assert index.search("", 10, None, "alpha") == []
    assert index.search("!!! \" OR *", 10, None, "alpha") == []
    assert index.delete_document("jira:PROJ-7", "alpha") is True
    assert index.search("login", 10, None, "alpha") == []


def test_fts_weights_rank_identifier_then_title_then_body(tmp_path, monkeypatch):
    monkeypatch.setattr("app.retrieval.lexical.settings.lexical_issue_key_weight", 100.0)
    monkeypatch.setattr("app.retrieval.lexical.settings.lexical_title_weight", 10.0)
    monkeypatch.setattr("app.retrieval.lexical.settings.lexical_content_weight", 1.0)
    index = SQLiteFTSIndex(tmp_path / "lexical.db")
    index.upsert_document([_chunk("jira:ID-3:chunk:0", "RANKTOKEN " * 20, document_id="jira:ID-3", issue_key="ID-3", title="ordinary")], "alpha")
    index.upsert_document([_chunk("jira:ID-2:chunk:0", "ordinary", document_id="jira:ID-2", issue_key="ID-2", title="RANKTOKEN")], "alpha")
    index.upsert_document([_chunk("jira:ID-1:chunk:0", "ordinary", document_id="jira:ID-1", issue_key="RANKTOKEN", title="ordinary")], "alpha")

    assert [item.document_id for item in index.search("RANKTOKEN", 10, None, "alpha")] == [
        "jira:ID-1", "jira:ID-2", "jira:ID-3",
    ]
