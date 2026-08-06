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
- **LLM:** OpenAI GPT-5 pinned snapshot (`gpt-5-2025-08-07`)
- **Embedding Model:** OpenAI text-embedding-3-small
- **Vector DB:** ChromaDB (Persistent storage on GKE PVC)
- **Cloud:** Google Cloud Platform (GKE, Artifact Registry, Cloud Build)
- **Parsing:** PyMuPDF, Atlassian Python API

### Model Responsibilities and Integration Modes

- `text-embedding-3-small` converts document chunks and user queries into
  vectors for similarity search in ChromaDB. It does not generate answers.
- `POST /query` provides an end-to-end RAG flow: this service retrieves the
  relevant context and uses its configured pinned GPT-5 model to generate the final
  answer.
- `POST /retrieve` provides retrieval only: it returns the relevant chunks and
  metadata without calling the answer model. AI applications that
  already have their own configured LLM should normally use this endpoint and
  pass the retrieved context to their own LLM.

For example, if `ai-market-studio` already defines its own LLM, the recommended
integration is to call `/retrieve` and let that LLM generate the final answer.
Use `/query` when the caller wants this RAG service to own both retrieval and
answer generation with its pinned GPT-5 model.

## Hybrid retrieval, reranking, and grounding

The default `RETRIEVAL_MODE=hybrid` combines SQLite FTS/BM25-style lexical
retrieval with Chroma vector retrieval using reciprocal-rank fusion (RRF). The
candidate and final limits default to 30 and 10 respectively, with
`RETRIEVAL_RRF_K=60`; lexical field weights default to 10 (Jira key), 5
(title), and 1 (content). `JIRA_KEY_PATTERN` defaults to
`\b[A-Z][A-Z0-9]+-\d+\b`; `RETRIEVAL_SCORE_THRESHOLD` is empty/disabled by
default. `vector` and `lexical` remain supported safe
retrieval modes; if one hybrid backend is unavailable, the pipeline uses the
available backend and records a bounded diagnostic. If all configured primary
retrieval backends fail, retrieval fails rather than inventing an answer.

Existing Chroma-only data must be reindexed to build `lexical.db` before
lexical or hybrid retrieval can find it. Re-run the normal document ingestion
for the corpus after enabling lexical retrieval; do not treat the pre-existing
Chroma collection as a lexical index.

`RERANKER_PROVIDER=none` is the safe default. `openai` selects the pinned
`RERANKER_OPENAI_MODEL=gpt-5-2025-08-07` with a five-second timeout;
`qwen_local` selects the pinned
`Qwen/Qwen3-Reranker-0.6B` revision. Both rerankers are bounded to a small
candidate list (Qwen: 20 candidates, 512 tokens, batch size 4, five-second
timeout, 30-second circuit-breaker) and safely fall back to the RRF order on
failure. The local Qwen path is optional: install `requirements-qwen.txt` only
on the operator host that will run it. Grounded answers select 5–10 evidence
items (default 5), cap prompt content at 4,000 characters and excerpts at 200,
and will refuse when evidence is insufficient; retrieval alone cannot guarantee
that a generated answer is correct. Answer generation uses the pinned
`ANSWER_OPENAI_MODEL=gpt-5-2025-08-07` with a 15-second timeout.

## Repeatable evaluation

The evaluation harness compares exactly these configurations: A vector only;
B BM25 + vector + RRF; C1 B's cached candidates with the pinned GPT-5
reranker; and C2 those same cached B candidates with the pinned Qwen reranker.
For each question, C1 and C2 receive independent deep copies of the identical
B ordering, so neither reranker can alter the other's input. Evaluation B
disables supplementary exact Jira lookup and marks a case failed if either
primary backend degrades, rather than misreporting a single-backend result as
hybrid. A, B, C1, and C2 retain/rank a comparable top-20 pool, so Recall@20 is
evaluated at the same depth for every configuration.

`evaluation/cases.example.jsonl` contains only synthetic placeholder cases
(exact fact, cross-document, hard negative, and unanswerable). It is a schema
example, not a measured corpus or a source of real Jira, Confluence, or PDF
labels. Start with roughly 30 manually labelled cases. Document-level labels
are sufficient initially; add chunk IDs only when refinement is useful.

Run an evaluation from an environment configured for the service:

```powershell
python -m app.evaluation.runner --cases evaluation/cases.example.jsonl --output evaluation/report.json --corpus-revision synthetic-example-v1 --index-revision local-index-v1
```

The JSON report keeps per-question (case-indexed, never question-text) results
and aggregates: Recall@20, Hit@5, MRR@10, Context Precision, citation validity
and human-labelled correctness when available, abstention accuracy, P50/P95
latency, token usage when available, and optional local CPU/memory observations.
Unmeasured fields remain `null`; the harness never invents results. Each run
also records the service commit, normalized case checksum, effective retrieval
settings, pinned model revisions, corpus revision, and index revision. A reranker
is recommended only when it has a measured MRR@10 or Context Precision gain,
does not regress Recall@20, and keeps P95 generated-answer latency at or below
the 5–10 second operational target. Otherwise the recommendation remains B
(RRF/no reranker).

For the standard local quality gate, use the currently active Python
environment (it deliberately does not select a merely present `.venv`):

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\quality-check.ps1
```

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
and `/query` remains available for consumers that want `ai-rag-service` to own
answer generation. Applications that already have their own LLM, including AI
Market Studio, use `/retrieve` and generate the final answer themselves.
Internally, PDF ingestion now shares the same indexing path as lifecycle
upsert. A future migration can add a lifecycle-style file endpoint or move
callers to `/documents/upsert` after they extract PDF text themselves.

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
