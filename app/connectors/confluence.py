import re

from atlassian import Confluence

from app.config import settings
from app.connectors.base import BaseConnector, Document


class ConfluenceConnector(BaseConnector):
    def __init__(self):
        self.client = Confluence(
            url=settings.confluence_url,
            username=settings.jira_email,
            password=settings.jira_api_token,
            cloud=True,
        )

    def fetch(self, space_key: str, max_pages: int = 50, **kwargs) -> list[Document]:
        pages = self.client.get_all_pages_from_space(
            space_key, start=0, limit=max_pages,
            expand="body.storage,version",
        )
        documents = []
        for page in pages:
            html = page.get("body", {}).get("storage", {}).get("value", "")
            plain = re.sub(r"<[^>]+>", " ", html)
            plain = re.sub(r"\s+", " ", plain).strip()
            version = page.get("version", {})
            by = version.get("by") or {}
            documents.append(
                Document(
                    id=self.make_id("confluence", page["id"]),
                    content=plain,
                    source_type="confluence",
                    title=page.get("title", ""),
                    metadata={
                        "page_id": page["id"],
                        "space_key": space_key,
                        "author": by.get("displayName", ""),
                        "last_modified": version.get("when", ""),
                    },
                )
            )
        return documents
