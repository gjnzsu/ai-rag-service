# AI RAG Service

A RAG (Retrieval-Augmented Generation) service for ingesting data from Jira, Confluence, Market Data (FX), and PDF files, using OpenAI and ChromaDB.

## 🚀 GKE Deployment

The service is deployed to Google Kubernetes Engine (GKE).

- **Base URL:** `http://34.10.130.210`
- **Swagger UI:** `http://34.10.130.210/docs`
- **Redoc UI:** `http://34.10.130.210/redoc`
- **Health Check:** `http://34.10.130.210/health`

## 🛠 Tech Stack

- **Framework:** FastAPI
- **LLM:** OpenAI (GPT-4o, text-embedding-3-small)
- **Vector DB:** ChromaDB (Persistent storage on GKE PVC)
- **Cloud:** Google Cloud Platform (GKE, Artifact Registry, Cloud Build)
- **Parsing:** PyMuPDF, Atlassian Python API

## 📥 Ingestion Endpoints

### 📄 PDF Ingestion
```bash
curl -X POST -F "file=@your_file.pdf" http://34.10.130.210/ingest/pdf
```

### 🎫 Jira Ingestion
```bash
curl -X POST -H "Content-Type: application/json" -d '{"project_key": "SCRUM"}' http://34.10.130.210/ingest/jira
```

### 📝 Confluence Ingestion
```bash
curl -X POST -H "Content-Type: application/json" -d '{"space_key": "SCRUM"}' http://34.10.130.210/ingest/confluence
```

### 📈 Market Data (FX) Ingestion
```bash
curl -X POST -H "Content-Type: application/json" -d '{"base_currency": "USD"}' http://34.10.130.210/ingest/fx
```

## 🔍 Query Endpoint

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"question": "What is the summary of SCRUM-1?"}' \
  http://34.10.130.210/query
```

## Platform Lifecycle API

The lifecycle API is the platform-aligned ingestion and retrieval contract for
callers that already have extracted text plus business metadata. Existing
compatibility endpoints such as `/ingest/pdf` and `/query` remain supported.

### Document Upsert

Use this endpoint for Jira issues, Confluence pages, or any future source that
can provide plain text content and metadata.

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "document_id": "jira_issue:PROJ-123",
    "content": "Summary: Add login auditing\n\nDescription: Capture login audit events.",
    "metadata": {
      "type": "jira_issue",
      "key": "PROJ-123",
      "project_key": "PROJ",
      "title": "Add login auditing",
      "url": "https://jira.example/browse/PROJ-123",
      "status": "To Do",
      "priority": "High"
    }
  }' \
  http://34.10.130.210/documents/upsert
```

Confluence pages should preserve their Jira relationship in metadata:

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "content": "Design notes for PROJ-123 login auditing.",
    "metadata": {
      "type": "confluence_page",
      "title": "PROJ-123 Login Auditing Design",
      "url": "https://wiki.example/pages/123",
      "space_key": "TEAM",
      "related_jira": "PROJ-123"
    }
  }' \
  http://34.10.130.210/documents/upsert
```

### Metadata-Filtered Retrieval

```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{
    "query": "login audit acceptance criteria",
    "top_k": 5,
    "filters": {
      "type": "jira_issue",
      "project_key": {"in": ["PROJ", "AUTH"]}
    }
  }' \
  http://34.10.130.210/retrieve
```

Supported filter forms are equality and `in`.

### Jira-Key Context Lookup

Use exact Jira-key context lookup when a caller needs the Jira issue and related
Confluence pages regardless of semantic similarity.

```bash
curl http://34.10.130.210/context/jira/PROJ-123
```

### Document Lookup And Delete

```bash
curl http://34.10.130.210/documents/jira_issue:PROJ-123

curl -X DELETE http://34.10.130.210/documents/jira_issue:PROJ-123
```

### PDF Compatibility And Future Direction

`/ingest/pdf` is still the file-upload endpoint for PDF parsing and indexing,
and `/query` is still the answer-generating query endpoint used by existing
consumers such as AI Market Studio. Internally, PDF ingestion now shares the
same indexing path as lifecycle upsert. A future migration can add a
lifecycle-style file endpoint or move callers to `/documents/upsert` after they
extract PDF text themselves.

## 🏗 Local Development

### Prerequisites
- Python 3.12
- Docker (for GKE build verification)
- Google Cloud SDK

### Setup
1. Clone the repository
2. Create a `.env` file with your API keys:
   ```env
   OPENAI_API_KEY=...
   JIRA_URL=...
   JIRA_EMAIL=...
   JIRA_API_TOKEN=...
   CONFLUENCE_URL=...
   ```
3. Install dependencies: `pip install -r requirements.txt`
4. Run the app: `python -m uvicorn app.main:app --reload`

## 🚢 Deployment Automation

To redeploy to GKE after changes:
```bash
bash deploy.sh
```
This script automates building the Docker image with Cloud Build and applying Kubernetes manifests to GKE.
