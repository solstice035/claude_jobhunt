"""
Embedding Providers - Abstract Interface for Multiple Embedding Models

This module provides a unified interface for text embeddings, supporting
multiple providers including OpenAI and local models (sentence-transformers).

Provider Comparison:
    | Model                      | MTEB Score | Dimensions | Cost       |
    |----------------------------|------------|------------|------------|
    | OpenAI text-embedding-3-s  | 62.3       | 1536       | $0.02/1M   |
    | OpenAI text-embedding-3-l  | 64.6       | 3072       | $0.13/1M   |
    | nomic-embed-text-v1.5      | 69.1       | 768        | Free       |
    | BGE-large-en-v1.5          | 64.2       | 1024       | Free       |

Key Classes:
    - EmbeddingProvider: Abstract base class (Protocol)
    - OpenAIEmbeddings: OpenAI API provider
    - LocalEmbeddings: Local sentence-transformers models
    - MockEmbeddingProvider: Deterministic mock for testing

A/B Testing:
    - EmbeddingExperiment: Configuration for experiments
    - route_to_experiment(): Consistent routing by ID hash
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# Available models by provider
AVAILABLE_MODELS: Dict[str, List[str]] = {
    "openai": [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ],
    "local": [
        "nomic-ai/nomic-embed-text-v1.5",
        "BAAI/bge-large-en-v1.5",
        "intfloat/e5-large-v2",
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/all-mpnet-base-v2",
    ],
}

# Model dimension mappings
MODEL_DIMENSIONS: Dict[str, int] = {
    # OpenAI models
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    # Local models
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-large-v2": 1024,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
}


@runtime_checkable
class EmbeddingProvider(Protocol):
    """
    Protocol defining the embedding provider interface.

    All embedding providers must implement:
    - embed(): Single text to embedding
    - embed_batch(): Multiple texts to embeddings
    - dimensions: Embedding vector size
    """

    @property
    def dimensions(self) -> int:
        """Return the embedding vector dimensions."""
        ...

    async def embed(self, text: str) -> List[float]:
        """Embed a single text string."""
        ...

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple text strings efficiently."""
        ...


class OpenAIEmbeddings:
    """
    OpenAI API embeddings provider.

    Uses OpenAI's text-embedding-3 models for high-quality embeddings.
    Supports batching for efficient bulk processing.

    Attributes:
        model: OpenAI embedding model name
        api_key: OpenAI API key

    Example:
        >>> provider = OpenAIEmbeddings(api_key="sk-...")
        >>> embedding = await provider.embed("Python developer")
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small"
    ) -> None:
        """
        Initialize OpenAI embeddings provider.

        Args:
            api_key: OpenAI API key
            model: Model name (default: text-embedding-3-small)
        """
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Get or create async OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: pip install openai"
                )
        return self._client

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions for current model."""
        return MODEL_DIMENSIONS.get(self.model, 1536)

    async def embed(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        text = text.replace("\n", " ").strip()
        if not text:
            return [0.0] * self.dimensions

        client = self._get_client()
        response = await client.embeddings.create(
            input=[text],
            model=self.model,
        )
        return response.data[0].embedding

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Embed multiple texts with automatic batching.

        Args:
            texts: List of texts to embed
            batch_size: Maximum texts per API call (default 100)

        Returns:
            List of embedding vectors in same order as input
        """
        cleaned_texts = [t.replace("\n", " ").strip() for t in texts]

        # Track empty text indices
        non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]
        non_empty_texts = [cleaned_texts[i] for i in non_empty_indices]

        if not non_empty_texts:
            return [[0.0] * self.dimensions for _ in texts]

        client = self._get_client()

        # Process in batches
        all_embeddings = []
        for i in range(0, len(non_empty_texts), batch_size):
            batch = non_empty_texts[i:i + batch_size]
            response = await client.embeddings.create(
                input=batch,
                model=self.model,
            )
            all_embeddings.extend([d.embedding for d in response.data])

        # Reconstruct with zero vectors for empty texts
        result = [[0.0] * self.dimensions for _ in texts]
        for idx, emb in zip(non_empty_indices, all_embeddings):
            result[idx] = emb

        return result


class LocalEmbeddings:
    """
    Local embeddings using sentence-transformers.

    Runs embedding models locally without API calls. Useful for:
    - Cost savings (no per-token fees)
    - Privacy (data stays local)
    - Offline operation

    Default model: nomic-embed-text-v1.5 (69.1 MTEB, 768 dims)

    Attributes:
        model_name: HuggingFace model name/path
        dimensions: Embedding vector size

    Example:
        >>> provider = LocalEmbeddings()
        >>> embedding = await provider.embed("Python developer")
    """

    def __init__(
        self,
        model_name: str = "nomic-ai/nomic-embed-text-v1.5",
        lazy_load: bool = True
    ) -> None:
        """
        Initialize local embeddings provider.

        Args:
            model_name: HuggingFace model name or path
            lazy_load: If True, defer model loading until first use
        """
        self.model_name = model_name
        self._model = None
        self._lazy_load = lazy_load

        if not lazy_load:
            self._load_model()

    def _load_model(self) -> None:
        """Load the embedding model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, trust_remote_code=True)
            logger.info("Embedding model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self._model = None

    @property
    def model(self):
        """Get model, loading lazily if needed."""
        if self._model is None and self._lazy_load:
            self._load_model()
        return self._model

    @property
    def dimensions(self) -> int:
        """Return embedding dimensions for current model."""
        return MODEL_DIMENSIONS.get(self.model_name, 768)

    async def embed(self, text: str) -> List[float]:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector
        """
        text = text.strip()
        if not text:
            return [0.0] * self.dimensions

        if self.model is None:
            logger.warning("Model not loaded, returning zero vector")
            return [0.0] * self.dimensions

        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self.model.encode(text).tolist()
        )
        return embedding

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors in same order as input
        """
        cleaned_texts = [t.strip() for t in texts]

        # Handle empty inputs
        if all(not t for t in cleaned_texts):
            return [[0.0] * self.dimensions for _ in texts]

        if self.model is None:
            logger.warning("Model not loaded, returning zero vectors")
            return [[0.0] * self.dimensions for _ in texts]

        # Run in executor
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self.model.encode(cleaned_texts).tolist()
        )

        # Replace empty text embeddings with zeros
        for i, text in enumerate(cleaned_texts):
            if not text:
                embeddings[i] = [0.0] * self.dimensions

        return embeddings


class MockEmbeddingProvider:
    """
    Mock embedding provider for testing.

    Generates deterministic embeddings based on text hash.
    Useful for unit tests without loading real models.

    Attributes:
        dimensions: Configurable embedding dimensions
    """

    def __init__(self, dimensions: int = 768) -> None:
        """
        Initialize mock provider.

        Args:
            dimensions: Embedding vector size
        """
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        """Return configured dimensions."""
        return self._dimensions

    def _text_to_embedding(self, text: str) -> List[float]:
        """Generate deterministic embedding from text hash."""
        if not text:
            return [0.0] * self._dimensions

        # Use hash to generate deterministic values
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # Generate embedding from hash
        embedding = []
        for i in range(self._dimensions):
            # Use different parts of hash for different dimensions
            idx = (i * 2) % len(text_hash)
            char_val = int(text_hash[idx:idx+2], 16)
            # Normalize to [-1, 1]
            embedding.append((char_val / 127.5) - 1)

        return embedding

    async def embed(self, text: str) -> List[float]:
        """Embed text using deterministic hash-based method."""
        return self._text_to_embedding(text)

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts."""
        return [self._text_to_embedding(t) for t in texts]


# A/B Testing Support

@dataclass
class EmbeddingExperiment:
    """
    Configuration for an embedding A/B experiment.

    Attributes:
        name: Experiment identifier
        provider_name: Provider to use ("openai", "local")
        sample_percentage: Fraction of traffic (0.0 to 1.0)
        model_name: Optional specific model override
    """
    name: str
    provider_name: str
    sample_percentage: float
    model_name: Optional[str] = None


def route_to_experiment(
    entity_id: str,
    experiments: List[Dict[str, Any]]
) -> str:
    """
    Route an entity to an experiment based on ID hash.

    Uses consistent hashing so same ID always routes to same experiment.
    This ensures a job/profile sees consistent embeddings.

    Args:
        entity_id: Unique identifier (e.g., job ID, profile ID)
        experiments: List of {"name": str, "percentage": float}

    Returns:
        Name of selected experiment

    Example:
        >>> experiments = [
        ...     {"name": "control", "percentage": 0.8},
        ...     {"name": "treatment", "percentage": 0.2},
        ... ]
        >>> route_to_experiment("job-123", experiments)
        'control'
    """
    if not experiments:
        return "default"

    # Hash the entity ID to get consistent routing
    hash_val = int(hashlib.md5(entity_id.encode()).hexdigest(), 16)
    # Normalize to 0-1
    normalized = (hash_val % 10000) / 10000

    # Route based on cumulative percentages
    cumulative = 0.0
    for exp in experiments:
        cumulative += exp.get("percentage", 0)
        if normalized < cumulative:
            return exp["name"]

    # Fallback to last experiment
    return experiments[-1]["name"]


def get_embedding_provider(
    provider_name: str = "openai",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    lazy_load: bool = True,
    **kwargs: Any
) -> EmbeddingProvider:
    """
    Factory function to create embedding provider instances.

    Args:
        provider_name: Provider type - "openai", "local", or "mock"
        api_key: API key for cloud providers (required for OpenAI)
        model_name: Optional model name override
        lazy_load: For local models, defer loading until first use
        **kwargs: Additional provider-specific arguments

    Returns:
        EmbeddingProvider instance

    Raises:
        ValueError: If provider is unknown or required args missing

    Example:
        >>> provider = get_embedding_provider("openai", api_key="sk-...")
        >>> provider = get_embedding_provider("local")
    """
    provider_name = provider_name.lower()

    if provider_name == "openai":
        if not api_key:
            raise ValueError("OpenAI embeddings require api_key")
        return OpenAIEmbeddings(
            api_key=api_key,
            model=model_name or "text-embedding-3-small"
        )

    elif provider_name == "local":
        return LocalEmbeddings(
            model_name=model_name or "nomic-ai/nomic-embed-text-v1.5",
            lazy_load=lazy_load
        )

    elif provider_name == "mock":
        dimensions = kwargs.get("dimensions", 768)
        return MockEmbeddingProvider(dimensions=dimensions)

    else:
        raise ValueError(
            f"Unknown embedding provider: {provider_name}. "
            f"Supported: openai, local, mock"
        )


async def get_embedding_for_experiment(
    text: str,
    entity_id: str,
    experiments: List[EmbeddingExperiment],
    api_key: Optional[str] = None
) -> tuple[str, List[float]]:
    """
    Get embedding using A/B experiment routing.

    Routes entity to appropriate provider based on experiment config,
    then returns embedding along with experiment name for tracking.

    Args:
        text: Text to embed
        entity_id: ID for consistent routing (e.g., job ID)
        experiments: List of EmbeddingExperiment configs
        api_key: OpenAI API key if needed

    Returns:
        Tuple of (experiment_name, embedding_vector)
    """
    # Convert to dict format for routing
    exp_configs = [
        {"name": e.name, "percentage": e.sample_percentage}
        for e in experiments
    ]

    selected_name = route_to_experiment(entity_id, exp_configs)

    # Find selected experiment config
    selected = next(
        (e for e in experiments if e.name == selected_name),
        experiments[0]
    )

    # Get provider for experiment
    provider = get_embedding_provider(
        provider_name=selected.provider_name,
        api_key=api_key,
        model_name=selected.model_name,
        lazy_load=True
    )

    embedding = await provider.embed(text)

    return selected.name, embedding
