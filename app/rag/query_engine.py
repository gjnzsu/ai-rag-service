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
        f"[{m.get('title', 'Unknown')} ({m.get('source_type', 'unknown')})]\n{chunk}"
        for chunk, m in zip(chunks, metadatas)
    ]
    context = "\n\n".join(context_parts)
    prompt = (
        "You are a helpful assistant. Answer the question using only the provided context.\n"
        'If the answer is not in the context, say '
        '"I don\'t have enough information to answer this question."\n\n'
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
