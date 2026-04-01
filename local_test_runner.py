import os
import sys
from fastapi.testclient import TestClient

# Add current directory to path
sys.path.append(os.getcwd())

from app.main import app

client = TestClient(app)

def test_local_features():
    print("--- Starting Local Feature Tests ---")

    # 1. Health
    print("\n1. Testing /health...")
    response = client.get("/health")
    print(f"   Status: {response.status_code}, Body: {response.json()}")

    # 2. Market Data (FX)
    print("\n2. Testing /ingest/fx...")
    response = client.post("/ingest/fx", json={"base_currency": "USD"})
    print(f"   Status: {response.status_code}, Body: {response.json()}")

    # 3. PDF Ingestion
    pdf_path = "tests/test_docs/Attention Is All You Need.pdf"
    print(f"\n3. Testing /ingest/pdf with {pdf_path}...")
    if os.path.exists(pdf_path):
        with open(pdf_path, "rb") as f:
            response = client.post(
                "/ingest/pdf",
                files={"file": ("Attention.pdf", f, "application/pdf")}
            )
        print(f"   Status: {response.status_code}, Body: {response.json()}")
    else:
        print("   Error: Test PDF not found.")

    # 4. Query
    print("\n4. Testing /query...")
    response = client.post("/query", json={"question": "What is the main topic of the document?"})
    print(f"   Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Answer: {data.get('answer')[:100]}...")
        print(f"   Sources found: {len(data.get('sources', []))}")
    else:
        print(f"   Error Body: {response.text}")

if __name__ == "__main__":
    try:
        test_local_features()
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
