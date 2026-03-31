from pathlib import Path

import pymupdf
import pytest

from app.connectors.pdf import PDFConnector


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Hello from PDF page 1")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_pdf_connector_returns_one_document(sample_pdf):
    connector = PDFConnector()
    docs = connector.fetch(file_path=str(sample_pdf))
    assert len(docs) == 1


def test_pdf_connector_document_fields(sample_pdf):
    connector = PDFConnector()
    doc = connector.fetch(file_path=str(sample_pdf))[0]
    assert doc.source_type == "pdf"
    assert "Hello from PDF page 1" in doc.content
    assert doc.title == sample_pdf.name
    assert doc.metadata["page_count"] == 1
    assert "filename" in doc.metadata
    assert "file_size_bytes" in doc.metadata


def test_pdf_connector_id_is_deterministic(sample_pdf):
    connector = PDFConnector()
    id1 = connector.fetch(file_path=str(sample_pdf))[0].id
    id2 = connector.fetch(file_path=str(sample_pdf))[0].id
    assert id1 == id2
