# RAG Service POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python/FastAPI RAG service that ingests PDFs, Jira/Confluence content, and FX rates into ChromaDB and answers natural-language questions via GPT-4o.

**Architecture:** Three-layer design — connectors (per-source) → ingestion pipeline (chunk/embed/store) → REST API. All connectors produce a shared `Document` model; the pipeline and query layer are source-agnostic.

**Tech Stack:** Python 3.11+, FastAPI, Uvicorn, OpenAI (`text-embedding-3-small` + `gpt-4o`), ChromaDB, LangChain (text splitter), pymupdf, atlassian-python-api, pydantic-settings, structlog, pytest.

---

## File Map

```
ai-rag-service/
├── app/
│   ├── __init__.py
│   ├── main.py                  # FastAPI app factory + /health
│   ├── config.py                # pydantic-settings Settings
│   ├── api/
│   │   ├── __init__.py
│   │   ├── ingest.py            # POST /ingest/{pdf,jira,confluence,fx}
│   │   └── query.py             # POST /query
│   ├── connectors/
│   │   ├── __init__.py
│   │   ├── base.py              # Document model + BaseConnector
│   │   ├── pdf.py
│   │   ├── jira.py
│   │   ├── confluence.py
│   │   └── fx.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── chunker.py           # RecursiveCharacterTextSplitter wrapper
│   │   ├── embedder.py          # OpenAI embedding batches
│   │   └── store.py             # ChromaDB upsert/query
│   └── rag/
│       ├── __init__.py
│       └── query_engine.py      # embed → retrieve → generate
├── tests/
│   ├── conftest.py
│   ├── test_connectors/
│   │   ├── __init__.py
│   │   ├── test_pdf.py
│   │   ├── test_jira.py
│   │   ├── test_confluence.py
│   │   └── test_fx.py
│   ├── test_pipeline/
│   │   ├── __init__.py
│   │   ├── test_chunker.py
│   │   ├── test_embedder.py
│   │   └── test_store.py
│   ├── test_rag/
│   │   ├── __init__.py
│   │   └── test_query_engine.py
│   ├── test_api/
│   │   ├── __init__.py
│   │   ├── test_ingest.py
│   │   └── test_query.py
│   └── test_integration.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `app/__init__.py`, `app/api/__init__.py`, `app/connectors/__init__.py`, `app/pipeline/__init__.py`, `app/rag/__init__.py`
- Create: `tests/__init__.py`, `tests/test_connectors/__init__.py`, `tests/test_pipeline/__init__.py`, `tests/test_rag/__init__.py`, `tests/test_api/__init__.py`
- Create: `app/config.py`
- Create: `app/main.py`

- [ ] **Step 1: Create requirements.txt**

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
openai>=1.30.0
chromadb>=0.5.0
langchain-text-splitters>=0.2.0
pymupdf>=1.24.0
atlassian-python-api>=3.41.0
pydantic-settings>=2.3.0
structlog>=24.1.0
python-multipart>=0.0.9
httpx>=0.27.0
pytest>=8.2.0
pytest-asyncio>=0.23.0
```

- [ ] **Step 2: Create .env.example**

```
OPENAI_API_KEY=sk-...
JIRA_URL=https://yourorg.atlassian.net
JIRA_EMAIL=you@example.com
JIRA_API_TOKEN=
CONFLUENCE_URL=https://yourorg.atlassian.net
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
LOG_LEVEL=INFO
```

- [ ] **Step 3: Create app/config.py**

```python
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str
    jira_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    confluence_url: str = ""
    chroma_persist_dir: str = "./chroma_db"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    log_level: str = "INFO"


settings = Settings()
```

- [ ] **Step 4: Create app/main.py**

```python
import structlog
from fastapi import FastAPI

from app.api.ingest import router as ingest_router
from app.api.query import router as query_router

logger = structlog.get_logger()


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Service", version="0.1.0")
    app.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
    app.include_router(query_router, tags=["query"])

    @app.get("/health")
    def health():
        return {"status": "ok", "version": "0.1.0"}

    return app


app = create_app()
```

- [ ] **Step 5: Create all empty `__init__.py` files**

```bash
touch app/__init__.py app/api/__init__.py app/connectors/__init__.py \
      app/pipeline/__init__.py app/rag/__init__.py \
      tests/__init__.py tests/test_connectors/__init__.py \
      tests/test_pipeline/__init__.py tests/test_rag/__init__.py \
      tests/test_api/__init__.py
```

- [ ] **Step 6: Create pytest.ini**

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 7: Install dependencies**

```bash
pip install -r requirements.txt
```

- [ ] **Step 8: Verify app loads**

```bash
OPENAI_API_KEY=test python -c "from app.main import app; print('OK')"
```
Expected: `OK`

- [ ] **Step 9: Commit**

```bash
git add .
git commit -m "chore: project scaffold — FastAPI app, config, requirements"
```

---

## Task 2: Base Connector & Document Model

**Files:**
- Create: `app/connectors/base.py`
- Create: `tests/test_connectors/test_base.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_connectors/test_base.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_base.py -v
```
Expected: FAIL (ImportError — module does not exist yet)

- [ ] **Step 3: Create app/connectors/base.py**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_base.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/connectors/base.py tests/test_connectors/test_base.py
git commit -m "feat: add Document model and BaseConnector"
```

---

## Task 3: PDF Connector

**Files:**
- Create: `app/connectors/pdf.py`
- Create: `tests/test_connectors/test_pdf.py`
- Create: `tests/fixtures/sample.pdf` (generated in test setup)

- [ ] **Step 1: Write the failing test**

Create `tests/test_connectors/test_pdf.py`:

```python
import tempfile
from pathlib import Path

import pymupdf
import pytest

from app.connectors.pdf import PDFConnector


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a minimal PDF with known text content."""
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_pdf.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create app/connectors/pdf.py**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_pdf.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/connectors/pdf.py tests/test_connectors/test_pdf.py
git commit -m "feat: add PDFConnector"
```

---

## Task 4: Ingestion Pipeline — Chunker

**Files:**
- Create: `app/pipeline/chunker.py`
- Create: `tests/test_pipeline/test_chunker.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline/test_chunker.py`:

```python
from app.connectors.base import Document
from app.pipeline.chunker import chunk_documents


def make_doc(content: str, source_type: str = "pdf") -> Document:
    return Document(
        id="doc1",
        content=content,
        source_type=source_type,
        title="Test Doc",
        metadata={"filename": "test.pdf"},
    )


def test_short_doc_produces_one_chunk():
    doc = make_doc("Short content.")
    chunks = chunk_documents([doc], chunk_size=512, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "Short content."


def test_long_doc_produces_multiple_chunks():
    long_text = "word " * 300  # ~1500 chars
    doc = make_doc(long_text)
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1


def test_chunk_has_required_fields():
    doc = make_doc("Some content here.")
    chunk = chunk_documents([doc])[0]
    assert "id" in chunk
    assert "content" in chunk
    assert "document_id" in chunk
    assert chunk["document_id"] == "doc1"
    assert chunk["source_type"] == "pdf"
    assert chunk["title"] == "Test Doc"


def test_chunk_ids_are_unique():
    long_text = "word " * 300
    doc = make_doc(long_text)
    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=10)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_pipeline/test_chunker.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create app/pipeline/chunker.py**

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.connectors.base import Document
from app.config import settings


def chunk_documents(
    documents: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for doc in documents:
        texts = splitter.split_text(doc.content)
        for i, text in enumerate(texts):
            flat_meta = {k: str(v) for k, v in doc.metadata.items()}
            chunks.append({
                "id": f"{doc.id}_chunk_{i}",
                "content": text,
                "document_id": doc.id,
                "source_type": doc.source_type,
                "title": doc.title,
                **flat_meta,
            })
    return chunks
```

- [ ] **Step 4: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_pipeline/test_chunker.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/chunker.py tests/test_pipeline/test_chunker.py
git commit -m "feat: add document chunker pipeline step"
```

---

## Task 5: Ingestion Pipeline — Embedder

**Files:**
- Create: `app/pipeline/embedder.py`
- Create: `tests/test_pipeline/test_embedder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline/test_embedder.py`:

```python
from unittest.mock import MagicMock, patch

from app.pipeline.embedder import embed_chunks


def make_chunks(n: int) -> list[dict]:
    return [{"id": f"chunk_{i}", "content": f"text {i}"} for i in range(n)]


def mock_embedding_response(texts):
    mock_resp = MagicMock()
    mock_resp.data = [MagicMock(embedding=[0.1] * 1536) for _ in texts]
    return mock_resp


@patch("app.pipeline.embedder.OpenAI")
def test_embed_chunks_returns_one_vector_per_chunk(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.side_effect = lambda **kwargs: mock_embedding_response(kwargs["input"])

    chunks = make_chunks(3)
    embeddings = embed_chunks(chunks)
    assert len(embeddings) == 3
    assert len(embeddings[0]) == 1536


@patch("app.pipeline.embedder.OpenAI")
def test_embed_chunks_batches_correctly(mock_openai_cls):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.side_effect = lambda **kwargs: mock_embedding_response(kwargs["input"])

    chunks = make_chunks(250)  # exceeds batch size of 100
    embeddings = embed_chunks(chunks)
    assert len(embeddings) == 250
    # 250 chunks / 100 batch size = 3 API calls
    assert mock_client.embeddings.create.call_count == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_pipeline/test_embedder.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create app/pipeline/embedder.py**

```python
from openai import OpenAI

from app.config import settings

BATCH_SIZE = 100


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    client = OpenAI(api_key=settings.openai_api_key)
    texts = [c["content"] for c in chunks]
    embeddings: list[list[float]] = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        embeddings.extend([e.embedding for e in response.data])
    return embeddings
```

- [ ] **Step 4: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_pipeline/test_embedder.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/embedder.py tests/test_pipeline/test_embedder.py
git commit -m "feat: add OpenAI embedder pipeline step"
```

---

## Task 6: Ingestion Pipeline — ChromaDB Store

**Files:**
- Create: `app/pipeline/store.py`
- Create: `tests/test_pipeline/test_store.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pipeline/test_store.py`:

```python
import tempfile

import pytest

from app.pipeline.store import upsert_chunks, query_collection


@pytest.fixture
def tmp_chroma(tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.store.settings.chroma_persist_dir", str(tmp_path))
    return str(tmp_path)


def test_upsert_and_query(tmp_chroma):
    chunks = [
        {"id": "c1", "content": "the cat sat on the mat",
         "document_id": "doc1", "source_type": "pdf", "title": "Test"},
        {"id": "c2", "content": "dogs love to run in parks",
         "document_id": "doc1", "source_type": "pdf", "title": "Test"},
    ]
    embeddings = [[0.1] * 1536, [0.9] * 1536]
    upsert_chunks(chunks, embeddings, collection_name="test_col")

    results = query_collection(
        query_embedding=[0.1] * 1536,
        collection_name="test_col",
        top_k=1,
    )
    assert len(results["documents"][0]) == 1
    assert results["documents"][0][0] == "the cat sat on the mat"


def test_upsert_is_idempotent(tmp_chroma):
    chunks = [{"id": "c1", "content": "hello",
               "document_id": "doc1", "source_type": "pdf", "title": "T"}]
    embeddings = [[0.1] * 1536]
    upsert_chunks(chunks, embeddings, collection_name="test_col")
    upsert_chunks(chunks, embeddings, collection_name="test_col")  # should not raise

    results = query_collection([0.1] * 1536, collection_name="test_col", top_k=5)
    assert len(results["documents"][0]) == 1  # still one, not two
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_pipeline/test_store.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create app/pipeline/store.py**

```python
import chromadb

from app.config import settings


def _get_collection(collection_name: str):
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(
    chunks: list[dict],
    embeddings: list[list[float]],
    collection_name: str = "default",
) -> None:
    collection = _get_collection(collection_name)
    metadatas = [
        {k: str(v) for k, v in chunk.items() if k != "content"}
        for chunk in chunks
    ]
    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["content"] for c in chunks],
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_collection(
    query_embedding: list[float],
    collection_name: str = "default",
    top_k: int = 5,
) -> dict:
    collection = _get_collection(collection_name)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_pipeline/test_store.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/store.py tests/test_pipeline/test_store.py
git commit -m "feat: add ChromaDB store pipeline step"
```

---

## Task 7: PDF Ingest API Route (end-to-end)

**Files:**
- Create: `app/api/ingest.py`
- Create: `tests/test_api/test_ingest.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api/test_ingest.py`:

```python
import io
from pathlib import Path
from unittest.mock import patch, MagicMock

import pymupdf
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def pdf_bytes(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "RAG test content")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path.read_bytes()


@patch("app.api.ingest.embed_chunks")
@patch("app.api.ingest.upsert_chunks")
def test_ingest_pdf_success(mock_upsert, mock_embed, pdf_bytes):
    mock_embed.return_value = [[0.1] * 1536]
    mock_upsert.return_value = None

    response = client.post(
        "/ingest/pdf",
        files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
    )
    assert response.status_code == 200
    data = response.json()
    assert "ingested_chunks" in data
    assert "document_id" in data
    assert data["ingested_chunks"] >= 1


def test_ingest_pdf_wrong_content_type():
    response = client.post(
        "/ingest/pdf",
        files={"file": ("test.txt", b"not a pdf", "text/plain")},
    )
    assert response.status_code == 422


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_api/test_ingest.py -v
```
Expected: FAIL (ImportError — ingest router not defined)

- [ ] **Step 3: Create app/api/ingest.py**

```python
import tempfile
from pathlib import Path

import structlog
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.connectors.pdf import PDFConnector
from app.pipeline.chunker import chunk_documents
from app.pipeline.embedder import embed_chunks
from app.pipeline.store import upsert_chunks

logger = structlog.get_logger()
router = APIRouter()


class IngestResponse(BaseModel):
    ingested_chunks: int
    document_id: str


@router.post("/pdf", response_model=IngestResponse)
def ingest_pdf(
    file: UploadFile = File(...),
    collection: str = "default",
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=422, detail="Only PDF files are accepted")
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        connector = PDFConnector()
        documents = connector.fetch(file_path=tmp_path)
        # restore original filename in metadata
        for doc in documents:
            doc.metadata["filename"] = file.filename
        chunks = chunk_documents(documents)
        embeddings = embed_chunks(chunks)
        upsert_chunks(chunks, embeddings, collection_name=collection)
        logger.info("pdf_ingested", filename=file.filename, chunks=len(chunks))
        return IngestResponse(
            ingested_chunks=len(chunks),
            document_id=documents[0].id,
        )
    except Exception as e:
        logger.error("pdf_ingest_error", error=str(e))
        raise HTTPException(status_code=422, detail=f"PDF parsing failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)
```

- [ ] **Step 4: Create stub app/api/query.py** (needed for app to load)

```python
from fastapi import APIRouter

router = APIRouter()
```

- [ ] **Step 5: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_api/test_ingest.py -v
```
Expected: 3 PASSED

- [ ] **Step 6: Commit**

```bash
git add app/api/ingest.py app/api/query.py tests/test_api/test_ingest.py
git commit -m "feat: add PDF ingest API route end-to-end"
```

---

## Task 8: Jira Connector

**Files:**
- Create: `app/connectors/jira.py`
- Create: `tests/test_connectors/test_jira.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_connectors/test_jira.py`:

```python
from unittest.mock import MagicMock, patch

from app.connectors.jira import JiraConnector


def make_mock_issue(key="PROJ-1", summary="Fix bug", description="Details here"):
    return {
        "key": key,
        "fields": {
            "summary": summary,
            "description": description,
            "status": {"name": "Open"},
            "priority": {"name": "High"},
            "reporter": {"displayName": "Alice"},
            "created": "2024-01-01T00:00:00Z",
            "comment": {"comments": []},
        },
    }


@patch("app.connectors.jira.Jira")
def test_jira_connector_returns_documents(mock_jira_cls):
    mock_client = MagicMock()
    mock_jira_cls.return_value = mock_client
    mock_client.jql.return_value = {"issues": [make_mock_issue()]}

    connector = JiraConnector()
    docs = connector.fetch(project_key="PROJ")
    assert len(docs) == 1
    assert docs[0].source_type == "jira"
    assert "Fix bug" in docs[0].content
    assert docs[0].metadata["issue_key"] == "PROJ-1"


@patch("app.connectors.jira.Jira")
def test_jira_connector_includes_comments(mock_jira_cls):
    mock_client = MagicMock()
    mock_jira_cls.return_value = mock_client
    issue = make_mock_issue()
    issue["fields"]["comment"]["comments"] = [
        {"author": {"displayName": "Bob"}, "body": "Looks good"}
    ]
    mock_client.jql.return_value = {"issues": [issue]}

    connector = JiraConnector()
    docs = connector.fetch(project_key="PROJ")
    assert "Looks good" in docs[0].content


@patch("app.connectors.jira.Jira")
def test_jira_connector_respects_max_results(mock_jira_cls):
    mock_client = MagicMock()
    mock_jira_cls.return_value = mock_client
    mock_client.jql.return_value = {"issues": []}

    connector = JiraConnector()
    connector.fetch(project_key="PROJ", max_results=10)
    call_kwargs = mock_client.jql.call_args
    assert call_kwargs[1]["limit"] == 10 or call_kwargs[0][1] == 10
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_jira.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create app/connectors/jira.py**

```python
from atlassian import Jira

from app.config import settings
from app.connectors.base import BaseConnector, Document


class JiraConnector(BaseConnector):
    def __init__(self):
        self.client = Jira(
            url=settings.jira_url,
            username=settings.jira_email,
            password=settings.jira_api_token,
            cloud=True,
        )

    def fetch(self, project_key: str, max_results: int = 100, **kwargs) -> list[Document]:
        jql = f"project = {project_key} ORDER BY created DESC"
        issues = self.client.jql(jql, limit=max_results)["issues"]
        documents = []
        for issue in issues:
            fields = issue["fields"]
            summary = fields.get("summary", "")
            description = fields.get("description") or ""
            content = f"Summary: {summary}\n\nDescription: {description}"
            comments = fields.get("comment", {}).get("comments", [])
            if comments:
                comment_lines = [
                    f"Comment by {c['author']['displayName']}: {c['body']}"
                    for c in comments
                ]
                content += "\n\nComments:\n" + "\n".join(comment_lines)
            documents.append(
                Document(
                    id=self.make_id("jira", issue["key"]),
                    content=content,
                    source_type="jira",
                    title=f"[{issue['key']}] {summary}",
                    metadata={
                        "issue_key": issue["key"],
                        "status": fields.get("status", {}).get("name", ""),
                        "priority": fields.get("priority", {}).get("name", "") if fields.get("priority") else "",
                        "reporter": fields.get("reporter", {}).get("displayName", "") if fields.get("reporter") else "",
                        "created_at": fields.get("created", ""),
                    },
                )
            )
        return documents
```

- [ ] **Step 4: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_jira.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/connectors/jira.py tests/test_connectors/test_jira.py
git commit -m "feat: add JiraConnector"
```

---

## Task 9: Confluence Connector

**Files:**
- Create: `app/connectors/confluence.py`
- Create: `tests/test_connectors/test_confluence.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_connectors/test_confluence.py`:

```python
from unittest.mock import MagicMock, patch

from app.connectors.confluence import ConfluenceConnector


def make_mock_page(page_id="123", title="My Page", html="<p>Hello <b>world</b></p>"):
    return {
        "id": page_id,
        "title": title,
        "body": {"storage": {"value": html}},
        "version": {
            "by": {"displayName": "Alice"},
            "when": "2024-01-01T00:00:00Z",
        },
    }


@patch("app.connectors.confluence.Confluence")
def test_confluence_connector_returns_documents(mock_conf_cls):
    mock_client = MagicMock()
    mock_conf_cls.return_value = mock_client
    mock_client.get_all_pages_from_space.return_value = [make_mock_page()]

    connector = ConfluenceConnector()
    docs = connector.fetch(space_key="TEAM")
    assert len(docs) == 1
    assert docs[0].source_type == "confluence"
    assert docs[0].title == "My Page"
    assert docs[0].metadata["space_key"] == "TEAM"


@patch("app.connectors.confluence.Confluence")
def test_confluence_html_is_stripped(mock_conf_cls):
    mock_client = MagicMock()
    mock_conf_cls.return_value = mock_client
    mock_client.get_all_pages_from_space.return_value = [
        make_mock_page(html="<h1>Title</h1><p>Body text here</p>")
    ]

    connector = ConfluenceConnector()
    doc = connector.fetch(space_key="TEAM")[0]
    assert "<" not in doc.content
    assert "Body text here" in doc.content


@patch("app.connectors.confluence.Confluence")
def test_confluence_connector_metadata(mock_conf_cls):
    mock_client = MagicMock()
    mock_conf_cls.return_value = mock_client
    mock_client.get_all_pages_from_space.return_value = [make_mock_page(page_id="42")]

    connector = ConfluenceConnector()
    doc = connector.fetch(space_key="TEAM")[0]
    assert doc.metadata["page_id"] == "42"
    assert doc.metadata["author"] == "Alice"
    assert doc.metadata["last_modified"] == "2024-01-01T00:00:00Z"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_confluence.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create app/connectors/confluence.py**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_confluence.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/connectors/confluence.py tests/test_connectors/test_confluence.py
git commit -m "feat: add ConfluenceConnector"
```

---

## Task 10: Jira + Confluence Ingest Routes

**Files:**
- Modify: `app/api/ingest.py`
- Modify: `tests/test_api/test_ingest.py`

- [ ] **Step 1: Add tests for Jira and Confluence routes**

Append to `tests/test_api/test_ingest.py`:

```python
@patch("app.api.ingest.JiraConnector")
@patch("app.api.ingest.embed_chunks")
@patch("app.api.ingest.upsert_chunks")
def test_ingest_jira_success(mock_upsert, mock_embed, mock_jira_cls):
    from app.connectors.base import Document
    mock_instance = MagicMock()
    mock_jira_cls.return_value = mock_instance
    mock_instance.fetch.return_value = [
        Document(id="j1", content="Issue content", source_type="jira",
                 title="[PROJ-1] Bug", metadata={"issue_key": "PROJ-1"})
    ]
    mock_embed.return_value = [[0.1] * 1536]
    mock_upsert.return_value = None

    response = client.post("/ingest/jira", json={"project_key": "PROJ"})
    assert response.status_code == 200
    data = response.json()
    assert data["ingested_chunks"] >= 1
    assert data["issues_fetched"] == 1


@patch("app.api.ingest.ConfluenceConnector")
@patch("app.api.ingest.embed_chunks")
@patch("app.api.ingest.upsert_chunks")
def test_ingest_confluence_success(mock_upsert, mock_embed, mock_conf_cls):
    from app.connectors.base import Document
    mock_instance = MagicMock()
    mock_conf_cls.return_value = mock_instance
    mock_instance.fetch.return_value = [
        Document(id="c1", content="Page content", source_type="confluence",
                 title="My Page", metadata={"page_id": "123", "space_key": "TEAM"})
    ]
    mock_embed.return_value = [[0.1] * 1536]
    mock_upsert.return_value = None

    response = client.post("/ingest/confluence", json={"space_key": "TEAM"})
    assert response.status_code == 200
    data = response.json()
    assert data["ingested_chunks"] >= 1
    assert data["pages_fetched"] == 1
```

- [ ] **Step 2: Run new tests to verify they fail**

```bash
OPENAI_API_KEY=test pytest tests/test_api/test_ingest.py::test_ingest_jira_success tests/test_api/test_ingest.py::test_ingest_confluence_success -v
```
Expected: FAIL

- [ ] **Step 3: Add Jira + Confluence routes to app/api/ingest.py**

Append to `app/api/ingest.py` (after existing imports and PDF route):

```python
from app.connectors.jira import JiraConnector
from app.connectors.confluence import ConfluenceConnector


class JiraIngestRequest(BaseModel):
    project_key: str
    max_results: int = 100
    collection: str = "default"


class JiraIngestResponse(BaseModel):
    ingested_chunks: int
    issues_fetched: int


class ConfluenceIngestRequest(BaseModel):
    space_key: str
    max_pages: int = 50
    collection: str = "default"


class ConfluenceIngestResponse(BaseModel):
    ingested_chunks: int
    pages_fetched: int


@router.post("/jira", response_model=JiraIngestResponse)
def ingest_jira(request: JiraIngestRequest):
    try:
        connector = JiraConnector()
        documents = connector.fetch(
            project_key=request.project_key,
            max_results=request.max_results,
        )
        chunks = chunk_documents(documents)
        embeddings = embed_chunks(chunks)
        upsert_chunks(chunks, embeddings, collection_name=request.collection)
        logger.info("jira_ingested", project=request.project_key, issues=len(documents))
        return JiraIngestResponse(ingested_chunks=len(chunks), issues_fetched=len(documents))
    except Exception as e:
        logger.error("jira_ingest_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/confluence", response_model=ConfluenceIngestResponse)
def ingest_confluence(request: ConfluenceIngestRequest):
    try:
        connector = ConfluenceConnector()
        documents = connector.fetch(
            space_key=request.space_key,
            max_pages=request.max_pages,
        )
        chunks = chunk_documents(documents)
        embeddings = embed_chunks(chunks)
        upsert_chunks(chunks, embeddings, collection_name=request.collection)
        logger.info("confluence_ingested", space=request.space_key, pages=len(documents))
        return ConfluenceIngestResponse(ingested_chunks=len(chunks), pages_fetched=len(documents))
    except Exception as e:
        logger.error("confluence_ingest_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
```

- [ ] **Step 4: Run all ingest tests to verify they pass**

```bash
OPENAI_API_KEY=test pytest tests/test_api/test_ingest.py -v
```
Expected: 5 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/api/ingest.py tests/test_api/test_ingest.py
git commit -m "feat: add Jira and Confluence ingest API routes"
```

---

## Task 11: FX Connector + Ingest Route

**Files:**
- Create: `app/connectors/fx.py`
- Create: `tests/test_connectors/test_fx.py`
- Modify: `app/api/ingest.py`
- Modify: `tests/test_api/test_ingest.py`

- [ ] **Step 1: Write the failing connector test**

Create `tests/test_connectors/test_fx.py`:

```python
from app.connectors.fx import FXConnector


def test_fx_connector_returns_one_document():
    connector = FXConnector()
    docs = connector.fetch(base_currency="USD", date_str="2026-03-31")
    assert len(docs) == 1


def test_fx_document_fields():
    connector = FXConnector()
    doc = connector.fetch(base_currency="USD", date_str="2026-03-31")[0]
    assert doc.source_type == "fx"
    assert "USD" in doc.title
    assert "2026-03-31" in doc.content
    assert doc.metadata["base_currency"] == "USD"
    assert doc.metadata["source"] == "mock"


def test_fx_content_contains_rates():
    connector = FXConnector()
    doc = connector.fetch(base_currency="USD", date_str="2026-03-31")[0]
    assert "CNY" in doc.content
    assert "EUR" in doc.content
    assert "1 USD =" in doc.content


def test_fx_rates_count_in_metadata():
    connector = FXConnector()
    doc = connector.fetch(base_currency="USD")[0]
    assert int(doc.metadata["rates_count"]) > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_fx.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create app/connectors/fx.py**

```python
from datetime import date

from app.connectors.base import BaseConnector, Document

MOCK_RATES_USD: dict[str, float] = {
    "EUR": 0.92,
    "GBP": 0.79,
    "JPY": 151.23,
    "CNY": 7.25,
    "HKD": 7.82,
    "SGD": 1.35,
    "AUD": 1.53,
    "CAD": 1.37,
    "CHF": 0.90,
    "INR": 83.50,
}


class FXConnector(BaseConnector):
    def fetch(
        self,
        base_currency: str = "USD",
        date_str: str | None = None,
        **kwargs,
    ) -> list[Document]:
        if date_str is None:
            date_str = date.today().isoformat()
        lines = [
            f"1 {base_currency} = {rate} {target} as of {date_str}"
            for target, rate in MOCK_RATES_USD.items()
        ]
        content = "\n".join(lines)
        return [
            Document(
                id=self.make_id("fx", f"{base_currency}_{date_str}"),
                content=content,
                source_type="fx",
                title=f"FX Rates ({base_currency} base, {date_str})",
                metadata={
                    "base_currency": base_currency,
                    "date": date_str,
                    "source": "mock",
                    "rates_count": len(MOCK_RATES_USD),
                },
            )
        ]
```

- [ ] **Step 4: Run connector test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_fx.py -v
```
Expected: 4 PASSED

- [ ] **Step 5: Add FX route test** — append to `tests/test_api/test_ingest.py`:

```python
@patch("app.api.ingest.embed_chunks")
@patch("app.api.ingest.upsert_chunks")
def test_ingest_fx_success(mock_upsert, mock_embed):
    mock_embed.return_value = [[0.1] * 1536]
    mock_upsert.return_value = None

    response = client.post("/ingest/fx", json={"base_currency": "USD"})
    assert response.status_code == 200
    data = response.json()
    assert data["ingested_chunks"] >= 1
    assert data["rates_count"] > 0
```

- [ ] **Step 6: Add FX route to app/api/ingest.py** — append after Confluence route:

```python
from app.connectors.fx import FXConnector


class FXIngestRequest(BaseModel):
    base_currency: str = "USD"
    date_str: str | None = None
    collection: str = "default"


class FXIngestResponse(BaseModel):
    ingested_chunks: int
    rates_count: int


@router.post("/fx", response_model=FXIngestResponse)
def ingest_fx(request: FXIngestRequest):
    try:
        connector = FXConnector()
        documents = connector.fetch(
            base_currency=request.base_currency,
            date_str=request.date_str,
        )
        chunks = chunk_documents(documents)
        embeddings = embed_chunks(chunks)
        upsert_chunks(chunks, embeddings, collection_name=request.collection)
        rates_count = int(documents[0].metadata.get("rates_count", 0))
        logger.info("fx_ingested", currency=request.base_currency, chunks=len(chunks))
        return FXIngestResponse(ingested_chunks=len(chunks), rates_count=rates_count)
    except Exception as e:
        logger.error("fx_ingest_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
```

- [ ] **Step 7: Run all ingest + FX connector tests**

```bash
OPENAI_API_KEY=test pytest tests/test_connectors/test_fx.py tests/test_api/test_ingest.py -v
```
Expected: 10 PASSED

- [ ] **Step 8: Commit**

```bash
git add app/connectors/fx.py tests/test_connectors/test_fx.py app/api/ingest.py tests/test_api/test_ingest.py
git commit -m "feat: add FXConnector and FX ingest API route"
```

---

## Task 12: Query Engine

**Files:**
- Create: `app/rag/query_engine.py`
- Create: `tests/test_rag/test_query_engine.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_rag/test_query_engine.py`:

```python
from unittest.mock import MagicMock, patch

from app.rag.query_engine import query


def mock_chroma_results(docs, metas, distances):
    return {
        "documents": [docs],
        "metadatas": [metas],
        "distances": [distances],
    }


@patch("app.rag.query_engine.chromadb")
@patch("app.rag.query_engine.OpenAI")
def test_query_returns_answer_and_sources(mock_openai_cls, mock_chromadb):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="The rate is 7.25."))]
    )
    mock_chroma_client = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_chroma_client
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    mock_collection.query.return_value = mock_chroma_results(
        ["1 USD = 7.25 CNY"],
        [{"document_id": "d1", "source_type": "fx", "title": "FX Rates"}],
        [0.05],
    )

    result = query("What is USD to CNY?")
    assert result["answer"] == "The rate is 7.25."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["source_type"] == "fx"
    assert result["model"] == "gpt-4o"


@patch("app.rag.query_engine.chromadb")
@patch("app.rag.query_engine.OpenAI")
def test_query_empty_collection_returns_no_info(mock_openai_cls, mock_chromadb):
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1] * 1536)]
    )
    mock_chroma_client = MagicMock()
    mock_chromadb.PersistentClient.return_value = mock_chroma_client
    mock_collection = MagicMock()
    mock_chroma_client.get_collection.return_value = mock_collection
    mock_collection.query.return_value = mock_chroma_results([], [], [])

    result = query("What is the answer?")
    assert "don't have enough information" in result["answer"]
    assert result["sources"] == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_rag/test_query_engine.py -v
```
Expected: FAIL (ImportError)

- [ ] **Step 3: Create app/rag/query_engine.py**

```python
import chromadb
from openai import OpenAI

from app.config import settings


def query(
    question: str,
    collection_name: str = "default",
    top_k: int | None = None,
) -> dict:
    top_k = top_k or settings.top_k
    openai_client = OpenAI(api_key=settings.openai_api_key)

    # 1. Embed the question
    q_embedding = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=[question],
    ).data[0].embedding

    # 2. Retrieve from ChromaDB
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_collection(name=collection_name)
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    if not chunks:
        return {
            "answer": "I don't have enough information to answer this question.",
            "sources": [],
            "model": "gpt-4o",
        }

    # 3. Build grounded prompt
    context_parts = [
        f"[{m.get('title', 'Unknown')} ({m.get('source_type', 'unknown')})]
{chunk}"
        for chunk, m in zip(chunks, metadatas)
    ]
    context = "\n\n".join(context_parts)
    prompt = (
        "You are a helpful assistant. Answer the question using only the provided context.\n"
        "If the answer is not in the context, say "
        "\"I don't have enough information to answer this question.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    # 4. Generate answer
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    answer = response.choices[0].message.content

    # 5. Build sources
    sources = [
        {
            "document_id": m.get("document_id", ""),
            "source_type": m.get("source_type", ""),
            "title": m.get("title", ""),
            "excerpt": chunk[:200],
            "score": round(1 - dist, 4),
        }
        for chunk, m, dist in zip(chunks, metadatas, distances)
    ]
    return {"answer": answer, "sources": sources, "model": "gpt-4o"}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_rag/test_query_engine.py -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add app/rag/query_engine.py tests/test_rag/test_query_engine.py
git commit -m "feat: add RAG query engine (retrieve + generate)"
```

---

## Task 13: Query API Route

**Files:**
- Modify: `app/api/query.py`
- Create: `tests/test_api/test_query.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_api/test_query.py`:

```python
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


@patch("app.api.query.query_engine.query")
def test_query_success(mock_query):
    mock_query.return_value = {
        "answer": "The USD to CNY rate is 7.25.",
        "sources": [
            {
                "document_id": "d1",
                "source_type": "fx",
                "title": "FX Rates",
                "excerpt": "1 USD = 7.25 CNY",
                "score": 0.95,
            }
        ],
        "model": "gpt-4o",
    }

    response = client.post("/query", json={"question": "What is USD to CNY?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "The USD to CNY rate is 7.25."
    assert len(data["sources"]) == 1
    assert data["model"] == "gpt-4o"


@patch("app.api.query.query_engine.query")
def test_query_passes_top_k(mock_query):
    mock_query.return_value = {"answer": "ok", "sources": [], "model": "gpt-4o"}
    client.post("/query", json={"question": "test", "top_k": 3})
    call_kwargs = mock_query.call_args[1]
    assert call_kwargs.get("top_k") == 3


def test_query_missing_question():
    response = client.post("/query", json={})
    assert response.status_code == 422


@patch("app.api.query.query_engine.query")
def test_query_engine_error_returns_502(mock_query):
    mock_query.side_effect = Exception("ChromaDB unavailable")
    response = client.post("/query", json={"question": "anything"})
    assert response.status_code == 502
```

- [ ] **Step 2: Run test to verify it fails**

```bash
OPENAI_API_KEY=test pytest tests/test_api/test_query.py -v
```
Expected: FAIL

- [ ] **Step 3: Replace app/api/query.py with full implementation**

```python
import structlog
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.rag import query_engine

logger = structlog.get_logger()
router = APIRouter()


class QueryRequest(BaseModel):
    question: str
    collection: str = "default"
    top_k: int = 5


class SourceItem(BaseModel):
    document_id: str
    source_type: str
    title: str
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    model: str


@router.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    try:
        result = query_engine.query(
            question=request.question,
            collection_name=request.collection,
            top_k=request.top_k,
        )
        logger.info("query_served", question=request.question[:80])
        return QueryResponse(**result)
    except Exception as e:
        logger.error("query_error", error=str(e))
        raise HTTPException(status_code=502, detail=str(e))
```

- [ ] **Step 4: Add `app/rag/__init__.py`**

```bash
touch app/rag/__init__.py
```

- [ ] **Step 5: Run test to verify it passes**

```bash
OPENAI_API_KEY=test pytest tests/test_api/test_query.py -v
```
Expected: 4 PASSED

- [ ] **Step 6: Commit**

```bash
git add app/api/query.py app/rag/__init__.py tests/test_api/test_query.py
git commit -m "feat: add query API route"
```

---

## Task 14: Integration Test

**Files:**
- Create: `tests/test_integration.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create tests/conftest.py**

```python
import pytest
import pymupdf


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
```

- [ ] **Step 2: Write the failing integration test**

Create `tests/test_integration.py`:

```python
"""Integration test: PDF ingest → ChromaDB → query pipeline (mocks OpenAI only)."""
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def patch_openai():
    """Replace all OpenAI calls with deterministic mocks."""
    fake_embedding = [0.1] * 1536

    def fake_embed(**kwargs):
        n = len(kwargs["input"])
        return MagicMock(data=[MagicMock(embedding=fake_embedding) for _ in range(n)])

    def fake_chat(**kwargs):
        return MagicMock(
            choices=[MagicMock(message=MagicMock(
                content="The USD to CNY rate is 7.25 according to the document."
            ))]
        )

    with patch("app.pipeline.embedder.OpenAI") as mock_emb_cls, \
         patch("app.rag.query_engine.OpenAI") as mock_q_cls:
        mock_emb = MagicMock()
        mock_emb.embeddings.create.side_effect = fake_embed
        mock_emb_cls.return_value = mock_emb

        mock_q = MagicMock()
        mock_q.embeddings.create.side_effect = fake_embed
        mock_q.chat.completions.create.side_effect = fake_chat
        mock_q_cls.return_value = mock_q
        yield


def test_ingest_pdf_then_query(sample_pdf_path, tmp_path, monkeypatch):
    monkeypatch.setattr("app.pipeline.store.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr("app.rag.query_engine.settings.chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr("app.rag.query_engine.settings.top_k", 5)

    # Ingest
    with open(sample_pdf_path, "rb") as f:
        resp = client.post(
            "/ingest/pdf",
            files={"file": ("sample.pdf", f, "application/pdf")},
            params={"collection": "integration_test"},
        )
    assert resp.status_code == 200
    assert resp.json()["ingested_chunks"] >= 1

    # Query
    resp = client.post(
        "/query",
        json={"question": "What is the USD to CNY rate?", "collection": "integration_test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert len(data["sources"]) >= 1
    assert data["sources"][0]["source_type"] == "pdf"
```

- [ ] **Step 3: Run integration test**

```bash
OPENAI_API_KEY=test pytest tests/test_integration.py -v
```
Expected: 1 PASSED

- [ ] **Step 4: Run the full test suite**

```bash
OPENAI_API_KEY=test pytest tests/ -v
```
Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_integration.py
git commit -m "test: add end-to-end integration test for ingest→query pipeline"
```

---

## Task 15: README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Create README.md**

```markdown
# AI RAG Service

A Retrieval-Augmented Generation (RAG) knowledge base service. Ingest PDFs,
Jira issues, Confluence pages, and FX rate data; query them in natural language.

## Stack

Python 3.11+, FastAPI, OpenAI (text-embedding-3-small + gpt-4o), ChromaDB,
LangChain text splitter, pymupdf, atlassian-python-api.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Fill in OPENAI_API_KEY and Atlassian credentials in .env
```

## Run

```bash
uvicorn app.main:app --reload
```

Service available at http://localhost:8000. Docs at http://localhost:8000/docs.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest/pdf` | Upload a PDF file |
| POST | `/ingest/jira` | Ingest Jira project issues |
| POST | `/ingest/confluence` | Ingest Confluence space pages |
| POST | `/ingest/fx` | Ingest FX rate data (mock) |
| POST | `/query` | Ask a question against the knowledge base |
| GET | `/health` | Health check |

## Examples

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/ingest/pdf -F "file=@report.pdf"

# Ingest Jira project
curl -X POST http://localhost:8000/ingest/jira \
  -H 'Content-Type: application/json' \
  -d '{"project_key": "MYPROJ"}'

# Ingest Confluence space
curl -X POST http://localhost:8000/ingest/confluence \
  -H 'Content-Type: application/json' \
  -d '{"space_key": "TEAM"}'

# Ingest FX rates (mock)
curl -X POST http://localhost:8000/ingest/fx \
  -H 'Content-Type: application/json' \
  -d '{"base_currency": "USD"}'

# Query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the USD to CNY exchange rate?"}'
```

## Tests

```bash
OPENAI_API_KEY=test pytest tests/ -v
```

## Adding a New Data Source

1. Create `app/connectors/your_source.py` inheriting `BaseConnector`
2. Implement `fetch(**kwargs) -> list[Document]`
3. Add a route in `app/api/ingest.py`
4. Add unit tests in `tests/test_connectors/`
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add README with setup, API reference, and usage examples"
```

---

## Verification Checklist

```bash
# 1. Full test suite passes
OPENAI_API_KEY=test pytest tests/ -v

# 2. Service starts cleanly
uvicorn app.main:app --reload

# 3. Interactive API docs reachable
open http://localhost:8000/docs

# 4. Ingest a PDF
curl -X POST http://localhost:8000/ingest/pdf -F "file=@any.pdf"

# 5. Ingest mock FX data
curl -X POST http://localhost:8000/ingest/fx \
  -H 'Content-Type: application/json' -d '{"base_currency":"USD"}'

# 6. Query
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the USD to CNY rate?"}'

# 7. Health check
curl http://localhost:8000/health
```






