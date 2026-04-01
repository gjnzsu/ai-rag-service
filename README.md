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
