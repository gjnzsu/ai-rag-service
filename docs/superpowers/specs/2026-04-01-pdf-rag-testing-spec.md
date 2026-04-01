# Spec: Testing RAG with AI Engineering Rulebook.pdf

## Overview
This spec outlines the testing strategy for the `AI Engineering Rulebook.pdf` document in the RAG service. The goal is to verify that the PDF is correctly ingested, chunked, stored in ChromaDB, and retrievable via a question-answering query.

## Test Setup
- **Target File:** `AI Engieering Rulebook.pdf` (local root directory)
- **Test Type:** Integration test using `FastAPI`'s `TestClient`.
- **Storage:** Temporary ChromaDB instance to avoid data persistence.
- **Model Mocks:** Mocking OpenAI Embeddings and ChatCompletions.

## Success Criteria
- Ingestion returns 200 OK and a `document_id`.
- The document is chunked correctly (based on `ingested_chunks` count).
- Retrieval for a specific question returns relevant source chunks from the file.
- Final answer is generated using the retrieved context.

## Test Execution Plan
1. Use `TestClient` to POST the PDF to `/ingest/pdf`.
2. Assert successful ingestion and capture `document_id`.
3. POST a relevant query to `/query`.
4. Assert that `sources` in the response contains the correctly identified document.
5. Verify that metadata (filename, etc.) is preserved.