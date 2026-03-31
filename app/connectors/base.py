import hashlib

from pydantic import BaseModel, Field


class Document(BaseModel):
    id: str
    content: str
    source_type: str  # "pdf" | "jira" | "confluence" | "fx"
    title: str
    metadata: dict = Field(default_factory=dict)


class BaseConnector:
    def fetch(self, **kwargs) -> list[Document]:
        raise NotImplementedError

    @staticmethod
    def make_id(source_type: str, unique_str: str) -> str:
        raw = f"{source_type}:{unique_str}".encode()
        return hashlib.sha256(raw).hexdigest()[:16]
