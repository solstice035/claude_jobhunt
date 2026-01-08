from openai import AsyncOpenAI
from typing import List
import numpy as np
from app.config import get_settings

settings = get_settings()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


def get_openai_client() -> AsyncOpenAI:
    return AsyncOpenAI(api_key=settings.openai_api_key)


async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text"""
    text = text.replace("\n", " ").strip()
    if not text:
        return [0.0] * EMBEDDING_DIMENSIONS

    client = get_openai_client()
    response = await client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts in a single API call"""
    cleaned_texts = [t.replace("\n", " ").strip() for t in texts]

    # Filter out empty texts but track their indices
    non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]
    non_empty_texts = [cleaned_texts[i] for i in non_empty_indices]

    if not non_empty_texts:
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]

    client = get_openai_client()

    # Batch in groups of 100 (OpenAI limit is 2048)
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(non_empty_texts), batch_size):
        batch = non_empty_texts[i:i + batch_size]
        response = await client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
        )
        all_embeddings.extend([d.embedding for d in response.data])

    # Reconstruct full list with zero vectors for empty texts
    result = [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]
    for idx, emb in zip(non_empty_indices, all_embeddings):
        result[idx] = emb

    return result


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(vec1)
    b = np.array(vec2)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
