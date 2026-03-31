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
