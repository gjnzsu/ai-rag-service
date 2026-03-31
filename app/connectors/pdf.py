from pathlib import Path

import pymupdf

from app.connectors.base import BaseConnector, Document


class PDFConnector(BaseConnector):
    def fetch(self, file_path: str, **kwargs) -> list[Document]:
        path = Path(file_path)
        doc = pymupdf.open(file_path)
        pages_text = [page.get_text() for page in doc]
        content = "\n\n".join(pages_text)
        doc.close()
        return [
            Document(
                id=self.make_id("pdf", file_path),
                content=content,
                source_type="pdf",
                title=path.name,
                metadata={
                    "filename": path.name,
                    "page_count": len(pages_text),
                    "file_size_bytes": path.stat().st_size,
                },
            )
        ]
