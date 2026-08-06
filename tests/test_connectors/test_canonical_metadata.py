from pathlib import Path

import pymupdf

from app.config import settings
from app.connectors.confluence import ConfluenceConnector
from app.connectors.jira import JiraConnector
from app.connectors.pdf import PDFConnector
from app.pipeline.chunker import chunk_documents


class _JiraClient:
    def jql(self, _jql, limit):
        assert limit == 1
        return {
            "issues": [
                {
                    "key": "PROJ-7",
                    "fields": {
                        "summary": "Login audit",
                        "description": "Record failed attempts",
                    },
                }
            ]
        }


class _ConfluenceClient:
    def get_all_pages_from_space(self, _space_key, **_kwargs):
        return [
            {
                "id": "12345",
                "title": "Audit design",
                "body": {"storage": {"value": "<p>Design notes</p>"}},
            }
        ]


def test_connectors_emit_canonical_ids_and_trusted_urls(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(settings, "jira_url", "https://jira.example/")
    monkeypatch.setattr(settings, "confluence_url", "https://wiki.example/")

    jira = JiraConnector.__new__(JiraConnector)
    jira.client = _JiraClient()
    jira_document = jira.fetch("PROJ", max_results=1)[0]

    confluence = ConfluenceConnector.__new__(ConfluenceConnector)
    confluence.client = _ConfluenceClient()
    confluence_document = confluence.fetch("TEAM")[0]

    pdf_path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    document.new_page().insert_text((50, 50), "PDF content")
    document.save(pdf_path)
    document.close()
    pdf_document = PDFConnector().fetch(str(pdf_path))[0]

    assert jira_document.id == "jira:PROJ-7"
    assert jira_document.metadata["issue_key"] == "PROJ-7"
    assert jira_document.metadata["key"] == "PROJ-7"
    assert jira_document.metadata["source_url"] == "https://jira.example/browse/PROJ-7"
    assert confluence_document.id == "confluence:12345"
    assert confluence_document.metadata["source_url"] == "https://wiki.example/pages/12345"
    assert pdf_document.id.startswith("pdf:")


def test_chunk_documents_uses_canonical_identity_and_resets_index_per_document():
    jira = JiraConnector.__new__(JiraConnector)
    jira.client = _JiraClient()
    document = jira.fetch("PROJ", max_results=1)[0]
    second = document.model_copy(update={"id": "jira:PROJ-8", "content": "Another issue"})

    chunks = chunk_documents([document, second])

    assert chunks[0]["document_id"] == "jira:PROJ-7"
    assert chunks[0]["id"] == chunks[0]["chunk_id"] == "jira:PROJ-7:chunk:0"
    assert chunks[0]["chunk_index"] == 0
    assert chunks[1]["id"] == chunks[1]["chunk_id"] == "jira:PROJ-8:chunk:0"
    assert chunks[1]["source_url"].endswith("/browse/PROJ-7")
