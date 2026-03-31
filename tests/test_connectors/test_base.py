from app.connectors.base import BaseConnector, Document


def test_document_model():
    doc = Document(
        id="abc123",
        content="Hello world",
        source_type="pdf",
        title="Test Doc",
        metadata={"filename": "test.pdf"},
    )
    assert doc.id == "abc123"
    assert doc.source_type == "pdf"
    assert doc.metadata["filename"] == "test.pdf"


def test_make_id_is_deterministic():
    id1 = BaseConnector.make_id("pdf", "file.pdf")
    id2 = BaseConnector.make_id("pdf", "file.pdf")
    assert id1 == id2
    assert len(id1) == 16


def test_make_id_differs_by_source():
    id_pdf = BaseConnector.make_id("pdf", "doc")
    id_jira = BaseConnector.make_id("jira", "doc")
    assert id_pdf != id_jira


def test_base_connector_fetch_raises():
    import pytest
    connector = BaseConnector()
    with pytest.raises(NotImplementedError):
        connector.fetch()
