# AI RAG Service

A Retrieval-Augmented Generation (RAG) knowledge base service. Ingest PDFs,
Jira issues, Confluence pages, and FX rate data; query them in natural language.

## Stack

Python 3.11+, FastAPI, OpenAI (`text-embedding-3-small` + `gpt-4o`), ChromaDB,
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

Service available at http://localhost:8000. Interactive docs at http://localhost:8000/docs.

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/ingest/pdf` | Upload a PDF file (`multipart/form-data`, field: `file`) |
| POST | `/ingest/jira` | Ingest Jira project issues (`{project_key, max_results?}`) |
| POST | `/ingest/confluence` | Ingest Confluence space pages (`{space_key, max_pages?}`) |
| POST | `/ingest/fx` | Ingest FX rate data — mock (`{base_currency?, date_str?}`) |
| POST | `/query` | Ask a question (`{question, collection?, top_k?}`) |
| GET | `/health` | Health check |

All ingest endpoints accept an optional `collection` field to namespace documents.
The `/query` endpoint accepts `collection` to query a specific namespace.

## Examples

```bash
# Ingest a PDF
curl -X POST http://localhost:8000/ingest/pdf \
  -F "file=@report.pdf"

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

## Configuration

All settings are loaded from `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `JIRA_URL` | `""` | Atlassian Cloud base URL |
| `JIRA_EMAIL` | `""` | Atlassian account email |
| `JIRA_API_TOKEN` | `""` | Atlassian API token |
| `CONFLUENCE_URL` | `""` | Atlassian Cloud base URL (usually same as JIRA_URL) |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage directory |
| `CHUNK_SIZE` | `512` | Token chunk size for splitting |
| `CHUNK_OVERLAP` | `50` | Token overlap between chunks |
| `TOP_K` | `5` | Number of chunks retrieved per query |
| `LOG_LEVEL` | `INFO` | Log level |

## Tests

```bash
OPENAI_API_KEY=test pytest tests/ -v
```

## Adding a New Data Source

1. Create `app/connectors/your_source.py` inheriting `BaseConnector`
2. Implement `fetch(**kwargs) -> list[Document]`
3. Add request/response models and a route in `app/api/ingest.py`
4. Add unit tests in `tests/test_connectors/`

## Project Structure

```
app/
  connectors/   # One module per data source
  pipeline/     # Chunker, embedder, ChromaDB store
  rag/          # Query engine (retrieve + generate)
  api/          # FastAPI routers
tests/          # Unit + integration tests
```
