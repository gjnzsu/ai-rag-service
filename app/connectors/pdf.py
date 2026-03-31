from pathlib import Path

import pymupdf

from app.connectors.base import BaseConnector, Document


class PDFConnector(BaseConnector):
    def fetch(self, file_path: str, original_filename: str | None = None, **kwargs) -> list[Document]:
        path = Path(file_path)
        display_name = original_filename or path.name
        doc = pymupdf.open(file_path)
        pages_text = [page.get_text() for page in doc]
        content = "\n\n".join(pages_text)
        doc.close()
        return [
            Document(
                id=self.make_id("pdf", display_name),
                content=content,
                source_type="pdf",
                title=display_name,
                metadata={
                    "filename": display_name,
                    "page_count": len(pages_text),
                    "file_size_bytes": path.stat().st_size,
                },
            )
        ]
