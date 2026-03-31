import pymupdf
import pytest


@pytest.fixture(scope="session")
def sample_pdf_path(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fixtures")
    pdf_path = tmp / "integration_sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "The USD to CNY exchange rate is 7.25.")
    page.insert_text((50, 80), "This document is about foreign exchange rates.")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path
