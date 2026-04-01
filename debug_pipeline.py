import sys
import os
# Add current directory to path
sys.path.append(os.getcwd())

from app.connectors.pdf import PDFConnector
from app.pipeline.chunker import chunk_documents
from app.pipeline.embedder import embed_chunks
from app.pipeline.store import upsert_chunks
from app.config import settings

def test_pdf_pipeline():
    print("Starting PDF pipeline test...")
    pdf_path = "tests/test_docs/Attention Is All You Need.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        return

    print("1. Fetching PDF documents...")
    connector = PDFConnector()
    documents = connector.fetch(file_path=pdf_path)
    print(f"   Fetched {len(documents)} documents.")

    print("2. Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"   Created {len(chunks)} chunks.")

    print("3. Embedding chunks (calling OpenAI)...")
    # Only embed first 2 chunks to save time/quota
    test_chunks = chunks[:2]
    embeddings = embed_chunks(test_chunks)
    print(f"   Generated {len(embeddings)} embeddings.")

    print("4. Upserting to ChromaDB...")
    upsert_chunks(test_chunks, embeddings, collection_name="test_collection")
    print("   Upsert successful.")

if __name__ == "__main__":
    try:
        test_pdf_pipeline()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
