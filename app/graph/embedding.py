"""Bounded embedding batches with source ordering and measured usage."""


class OpenAIChunkEmbedder:
    def __init__(self, client):
        self.client = client
        self.total_tokens = 0

    def __call__(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for start in range(0, len(texts), 100):
            batch = texts[start:start + 100]
            response = self.client.embeddings.create(model="text-embedding-3-small", input=batch)
            entries = sorted(response.data, key=lambda item: item.index)
            if [entry.index for entry in entries] != list(range(len(batch))):
                raise ValueError("embedding response indexes do not match the input")
            vectors.extend(entry.embedding for entry in entries)
            if response.usage is not None:
                self.total_tokens += response.usage.total_tokens
        return vectors
