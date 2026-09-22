from openai import OpenAI

from app.config import settings
from app.model_access import gateway_options, gateway_http_client

BATCH_SIZE = 100


def embed_chunks(chunks: list[dict]) -> list[list[float]]:
    client = OpenAI(**gateway_options(settings), http_client=gateway_http_client())
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
