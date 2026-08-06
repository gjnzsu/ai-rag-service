import hashlib
from pathlib import Path

import pymupdf

from app.connectors.base import BaseConnector, Document


class PDFConnector(BaseConnector):
    def fetch(self, file_path: str, original_filename: str | None = None, document_type: str | None = None, **kwargs) -> list[Document]:
        path = Path(file_path)
        display_name = original_filename or path.name
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        doc = pymupdf.open(file_path)
        pages_text = [page.get_text() for page in doc]
        content = "\n\n".join(pages_text)
        doc.close()

        # Auto-detect document type from filename if not provided
        if not document_type:
            filename_lower = display_name.lower()
            if "research" in filename_lower or "report" in filename_lower:
                document_type = "research_report"
            elif "rulebook" in filename_lower or "rule" in filename_lower:
                document_type = "rulebook"
            else:
                document_type = "general"

        return [
            Document(
                id=f"pdf:{file_digest}",
                content=content,
                source_type="pdf",
                title=display_name,
                metadata={
                    "filename": display_name,
                    "page_count": len(pages_text),
                    "file_size_bytes": path.stat().st_size,
                    "document_type": document_type,
                    "source_url": "",
                },
            )
        ]
