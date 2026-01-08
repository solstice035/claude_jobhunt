"""
OpenAI Embeddings Service - Text Vectorization for Semantic Search

This module provides async functions to convert text into dense vector
representations using OpenAI's text-embedding-3-small model.

Key Functions:
    - get_embedding(): Single text → 1536-dim vector
    - get_embeddings_batch(): Multiple texts → batch processing (100/request)
    - cosine_similarity(): Compare two vectors (-1 to 1)

Model Details:
    - Model: text-embedding-3-small
    - Dimensions: 1536
    - Max tokens: 8191
    - Cost: $0.02 / 1M tokens (as of 2024)

Performance:
    - Batch processing reduces API calls by 100x
    - Empty texts return zero vectors (no API call)
"""

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
    """
    Get embeddings for multiple texts with automatic batching.

    Handles up to 2048 texts per API call, but batches internally at 100
    for optimal throughput. Empty texts receive zero vectors without API calls.

    Args:
        texts: List of text strings to embed

    Returns:
        List of 1536-dim embedding vectors, same order as input

    Complexity: O(n/100) API calls where n = number of non-empty texts
    """
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
    """
    Calculate cosine similarity between two embedding vectors.

    Formula: cos(θ) = (a · b) / (||a|| × ||b||)

    Args:
        vec1: First embedding vector
        vec2: Second embedding vector

    Returns:
        Similarity score from -1 (opposite) to 1 (identical).
        Returns 0.0 if either vector has zero magnitude.
    """
    a = np.array(vec1)
    b = np.array(vec2)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
