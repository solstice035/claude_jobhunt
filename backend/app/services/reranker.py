"""
Cross-Encoder Re-ranking Service - Precision Stage for Two-Stage Retrieval

This module provides cross-encoder re-ranking to improve match precision
after initial retrieval. Cross-encoders see both query and document together,
enabling deeper semantic comparison than bi-encoders.

Performance Comparison:
    | Stage      | Model Type  | Speed   | Accuracy |
    |------------|-------------|---------|----------|
    | Retrieval  | Bi-encoder  | ~10ms   | ~87%     |
    | Re-ranking | Cross-enc   | ~100ms  | ~92%     |

Provider Options:
    - Local: Uses sentence-transformers CrossEncoder (free, requires GPU)
    - Cohere: Uses Cohere Rerank API (paid, high quality)

Architecture:
    Candidates (100+) → Cross-Encoder → Top-K (20) Reranked Results

Key Classes:
    - RerankerBase: Abstract interface for all rerankers
    - LocalCrossEncoderReranker: Local sentence-transformers model
    - CohereReranker: Cohere API integration
    - get_reranker(): Factory function for creating rerankers
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Default local model - good balance of speed and quality
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Alternative models:
# - "cross-encoder/ms-marco-MiniLM-L-6-v2" - Faster, slightly lower quality
# - "cross-encoder/ms-marco-TinyBERT-L-2-v2" - Fastest, lower quality
# - "cross-encoder/stsb-roberta-large" - Highest quality, slower


class RerankerBase(ABC):
    """
    Abstract base class for re-ranking providers.

    All rerankers must implement the rerank method which takes
    a query and list of documents, returning documents sorted
    by relevance score.
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents by relevance to query.

        Args:
            query: The search query (e.g., CV summary or job preferences)
            documents: List of dicts with 'id' and 'text' keys
            top_k: Maximum number of results to return

        Returns:
            List of (document_id, relevance_score) tuples sorted by score desc
        """
        pass

    async def rerank_async(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Async version of rerank. Default implementation runs sync in executor.

        Args:
            query: The search query
            documents: Documents to rerank
            top_k: Maximum results

        Returns:
            List of (document_id, score) tuples
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.rerank, query, documents, top_k
        )


class LocalCrossEncoderReranker(RerankerBase):
    """
    Local cross-encoder using sentence-transformers.

    Uses a pre-trained cross-encoder model to compute relevance scores
    between query and documents. The model sees both texts together,
    enabling richer interaction modeling than bi-encoders.

    Model loading is lazy by default to avoid startup delays.

    Attributes:
        model_name: Name of the cross-encoder model
        model: Loaded CrossEncoder instance (lazy)

    Example:
        >>> reranker = LocalCrossEncoderReranker()
        >>> results = reranker.rerank("Python dev", [{"id": "1", "text": "..."}])
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER_MODEL,
        lazy_load: bool = True
    ) -> None:
        """
        Initialize local cross-encoder reranker.

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
        """Load the cross-encoder model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            logger.info("Cross-encoder model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self._model = None
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            self._model = None

    @property
    def model(self):
        """Get model, loading lazily if needed."""
        if self._model is None and self._lazy_load:
            self._load_model()
        return self._model

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using local cross-encoder.

        Creates query-document pairs and scores them using the
        cross-encoder model. Returns documents sorted by score.

        Args:
            query: Search query text
            documents: List of dicts with 'id' and 'text' keys
            top_k: Maximum results to return

        Returns:
            List of (document_id, relevance_score) sorted by score desc
        """
        if not query or not query.strip():
            return []

        if not documents:
            return []

        # Check if model is available
        if self.model is None:
            # Fallback: return documents in original order with default scores
            logger.warning("Cross-encoder model not available, returning original order")
            return [(doc["id"], 0.5) for doc in documents[:top_k]]

        # Create query-document pairs
        pairs = [(query, doc.get("text", "")) for doc in documents]

        try:
            # Get cross-encoder scores
            scores = self.model.predict(pairs)

            # Combine with document IDs
            scored_docs = list(zip(
                [doc["id"] for doc in documents],
                [float(s) for s in scores]
            ))

            # Sort by score descending
            scored_docs.sort(key=lambda x: -x[1])

            return scored_docs[:top_k]

        except Exception as e:
            logger.error(f"Cross-encoder scoring failed: {e}")
            # Return original order as fallback
            return [(doc["id"], 0.5) for doc in documents[:top_k]]


class CohereReranker(RerankerBase):
    """
    Cohere API-based re-ranker.

    Uses Cohere's rerank endpoint which provides high-quality
    cross-encoder re-ranking as a service.

    Pricing: ~$1 per 1K queries (as of 2024)
    Model: rerank-english-v3.0

    Attributes:
        api_key: Cohere API key
        model: Cohere rerank model name

    Example:
        >>> reranker = CohereReranker(api_key="your-key")
        >>> results = reranker.rerank("Python dev", documents)
    """

    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0"
    ) -> None:
        """
        Initialize Cohere reranker.

        Args:
            api_key: Cohere API key
            model: Cohere rerank model name
        """
        self.api_key = api_key
        self.model = model
        self._client = None

    def _get_client(self):
        """Get or create Cohere client."""
        if self._client is None:
            try:
                import cohere
                self._client = cohere.Client(api_key=self.api_key)
            except ImportError:
                logger.error(
                    "cohere package not installed. "
                    "Install with: pip install cohere"
                )
                raise
        return self._client

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Re-rank documents using Cohere API.

        Args:
            query: Search query text
            documents: List of dicts with 'id' and 'text' keys
            top_k: Maximum results to return

        Returns:
            List of (document_id, relevance_score) sorted by score desc
        """
        if not query or not documents:
            return []

        try:
            client = self._get_client()

            # Extract document texts
            doc_texts = [doc.get("text", "") for doc in documents]

            # Call Cohere rerank API
            response = client.rerank(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=min(top_k, len(documents))
            )

            # Map results back to document IDs
            results = []
            for result in response.results:
                doc_id = documents[result.index]["id"]
                score = result.relevance_score
                results.append((doc_id, score))

            return results

        except Exception as e:
            logger.error(f"Cohere rerank failed: {e}")
            # Return original order as fallback
            return [(doc["id"], 0.5) for doc in documents[:top_k]]

    async def rerank_async(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """
        Async re-ranking using Cohere.

        Note: Currently wraps sync API in executor.
        Future: Use cohere AsyncClient when available.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self.rerank, query, documents, top_k
        )


class MockReranker(RerankerBase):
    """
    Mock reranker for testing without loading models.

    Returns documents sorted by text length (longer = higher score)
    as a simple deterministic behavior for tests.
    """

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, str]],
        top_k: int = 20
    ) -> List[Tuple[str, float]]:
        """Mock rerank using text length as proxy for relevance."""
        if not query or not documents:
            return []

        # Score by text length (simple, deterministic)
        scored = [
            (doc["id"], len(doc.get("text", "")))
            for doc in documents
        ]

        # Normalize scores to 0-1
        max_len = max(s[1] for s in scored) if scored else 1
        normalized = [
            (doc_id, score / max_len)
            for doc_id, score in scored
        ]

        normalized.sort(key=lambda x: -x[1])
        return normalized[:top_k]


def get_reranker(
    provider: str = "local",
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any
) -> RerankerBase:
    """
    Factory function to create reranker instances.

    Args:
        provider: Reranker provider - "local", "cohere", or "mock"
        api_key: API key for cloud providers (required for cohere)
        model_name: Optional model name override
        **kwargs: Additional provider-specific arguments

    Returns:
        RerankerBase instance

    Raises:
        ValueError: If provider is unknown

    Example:
        >>> reranker = get_reranker("local")
        >>> reranker = get_reranker("cohere", api_key="...")
    """
    provider = provider.lower()

    if provider == "local":
        return LocalCrossEncoderReranker(
            model_name=model_name or DEFAULT_CROSS_ENCODER_MODEL,
            **kwargs
        )

    elif provider == "cohere":
        if not api_key:
            raise ValueError("Cohere reranker requires api_key")
        return CohereReranker(
            api_key=api_key,
            model=model_name or "rerank-english-v3.0"
        )

    elif provider == "mock":
        return MockReranker()

    else:
        raise ValueError(
            f"Unknown reranker provider: {provider}. "
            f"Supported: local, cohere, mock"
        )


async def rerank_jobs(
    query: str,
    jobs: List[dict],
    top_k: int = 20,
    provider: str = "local",
    api_key: Optional[str] = None
) -> List[Tuple[dict, float]]:
    """
    Convenience function to rerank job documents.

    Takes full job objects and returns them with relevance scores.
    Useful for the two-stage retrieval pipeline.

    Args:
        query: Search query (CV text or summary)
        jobs: List of job dicts with 'id', 'title', 'description'
        top_k: Maximum results to return
        provider: Reranker provider
        api_key: API key if needed

    Returns:
        List of (job_dict, relevance_score) tuples sorted by score desc

    Example:
        >>> jobs = [{"id": "1", "title": "...", "description": "..."}]
        >>> results = await rerank_jobs("Python dev", jobs)
        >>> for job, score in results:
        ...     print(f"{job['title']}: {score:.2f}")
    """
    # Convert jobs to reranker format
    documents = []
    job_lookup = {}

    for job in jobs:
        job_id = job.get("id", str(len(documents)))
        title = job.get("title", "")
        description = job.get("description", "")
        company = job.get("company", "")

        # Combine relevant text fields
        text = f"{title} at {company}\n{description[:2000]}"

        documents.append({"id": job_id, "text": text})
        job_lookup[job_id] = job

    # Get reranker and process
    reranker = get_reranker(provider=provider, api_key=api_key, lazy_load=True)
    reranked = await reranker.rerank_async(query, documents, top_k)

    # Map back to full job objects
    results = []
    for job_id, score in reranked:
        if job_id in job_lookup:
            results.append((job_lookup[job_id], score))

    return results
