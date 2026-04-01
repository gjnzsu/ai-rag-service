import sys
import os
# Add current directory to path
sys.path.append(os.getcwd())

from app.connectors.pdf import PDFConnector
from app.pipeline.chunker import chunk_documents
from app.pipeline.embedder import embed_chunks
from app.pipeline.store import upsert_chunks
from app.rag import query_engine
from app.config import settings

def test_new_pdf_rag():
    print("--- Testing RAG with Attention Is All You Need.pdf ---")
    pdf_path = "tests/test_docs/Attention Is All You Need.pdf"
    collection = "test_pdf_rag"

    print("1. Ingesting PDF...")
    connector = PDFConnector()
    documents = connector.fetch(file_path=pdf_path)
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    # To avoid crash/timeout in sandbox, we'll only embed and store the first 40 chunks
    # to capture more of the main content.
    sample_chunks = chunks[:40]
    print(f"   Embedding and storing first {len(sample_chunks)} chunks...")
    embeddings = embed_chunks(sample_chunks)
    upsert_chunks(sample_chunks, embeddings, collection_name=collection)
    print("   Ingestion complete.")

    print("\n2. Querying: 'What is the Transformer architecture?'")
    result = query_engine.query(
        question="What is the Transformer architecture?",
        collection_name=collection,
        top_k=3
    )

    # Use utf-8 encoding for output since we have special characters (e.g. Polish characters in author names)
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print(f"\nAnswer: {result['answer']}")
    print("\nSources:")
    for s in result['sources']:
        print(f"- [{s['title']}] (Score: {s['score']}): {s['excerpt']}...")

if __name__ == "__main__":
    try:
        test_new_pdf_rag()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
