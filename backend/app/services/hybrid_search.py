"""
Hybrid Search Service - BM25 + Semantic Search with RRF Fusion

This module implements a hybrid search approach that combines:
1. BM25 keyword search for exact term matching
2. Semantic search using embeddings for meaning-based matching
3. Reciprocal Rank Fusion (RRF) for combining rankings

Research shows hybrid search achieves 5-10% better recall than either approach alone.

Architecture:
    Query → [BM25 Search] → Rankings
         → [Semantic Search] → Rankings
                    ↓
            RRF Fusion
                    ↓
            Top-K Candidates

Key Classes:
    - JobBM25Index: BM25 keyword search index
    - HybridSearchService: Combined search with configurable weights

Complexity:
    - BM25 Index Build: O(n * m) where n=docs, m=avg tokens
    - BM25 Search: O(n * q) where q=query tokens
    - RRF Fusion: O(k * r) where k=top_k, r=ranking sources
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None

try:
    import nltk
    from nltk.tokenize import word_tokenize
    HAS_NLTK = True
except ImportError:
    HAS_NLTK = False


def _simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenizer fallback when NLTK is not available.

    Splits on whitespace and punctuation, lowercases, filters short tokens.

    Args:
        text: Input text to tokenize

    Returns:
        List of lowercase tokens
    """
    if not text:
        return []

    # Lowercase and split on non-alphanumeric characters
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())

    # Filter very short tokens
    return [t for t in tokens if len(t) > 1]


def _tokenize(text: str) -> List[str]:
    """
    Tokenize text using NLTK word_tokenize or fallback.

    Args:
        text: Input text to tokenize

    Returns:
        List of lowercase tokens
    """
    if not text:
        return []

    if HAS_NLTK:
        try:
            tokens = word_tokenize(text.lower())
            # Filter non-alphanumeric and very short tokens
            return [t for t in tokens if len(t) > 1 and t.isalnum()]
        except LookupError:
            # NLTK data not downloaded, fall back to simple tokenizer
            pass

    return _simple_tokenize(text)


def cosine_similarity_simple(vec1: List[float], vec2: List[float]) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Similarity score from -1 to 1
    """
    if not vec1 or not vec2:
        return 0.0

    a = np.array(vec1)
    b = np.array(vec2)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


class JobBM25Index:
    """
    BM25 keyword search index for job documents.

    Uses Okapi BM25 algorithm for term frequency-based ranking.
    Stores job IDs separately from tokenized content for efficient retrieval.

    Attributes:
        job_ids: List of job document IDs
        bm25: BM25Okapi index instance

    Example:
        >>> documents = [{"id": "1", "text": "Python developer"}]
        >>> index = JobBM25Index(documents)
        >>> results = index.search("Python", top_k=5)
        >>> print(results)  # ["1"]
    """

    def __init__(self, documents: List[Dict[str, str]]) -> None:
        """
        Initialize BM25 index from documents.

        Args:
            documents: List of dicts with 'id' and 'text' keys.
                      'text' is tokenized and indexed.
        """
        self.job_ids: List[str] = []
        self.tokenized_corpus: List[List[str]] = []

        if not documents:
            self.bm25 = None
            return

        for doc in documents:
            doc_id = doc.get("id", "")
            text = doc.get("text", "")

            self.job_ids.append(doc_id)
            self.tokenized_corpus.append(_tokenize(text))

        if HAS_BM25 and self.tokenized_corpus:
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            self.bm25 = None

    def search(self, query: str, top_k: int = 100) -> List[str]:
        """
        Search for documents matching query.

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of document IDs ranked by relevance
        """
        results = self.search_with_scores(query, top_k)
        return [doc_id for doc_id, _ in results]

    def search_with_scores(
        self, query: str, top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Search for documents with BM25 scores.

        Args:
            query: Search query string
            top_k: Maximum number of results to return

        Returns:
            List of (document_id, score) tuples sorted by score descending
        """
        if not query or not query.strip():
            return []

        if not self.bm25 or not self.job_ids:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)

        # Get top-k indices sorted by score
        top_indices = np.argsort(scores)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.job_ids[idx], float(scores[idx])))

        return results


def reciprocal_rank_fusion(
    rankings: List[List[str]],
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combine multiple rankings using Reciprocal Rank Fusion (RRF).

    RRF is a robust rank aggregation method that:
    - Handles rankings of different lengths
    - Is less sensitive to outliers than score-based fusion
    - Works well when source scores are not comparable

    Formula: score(d) = sum(1 / (k + rank_i(d))) for each ranking i

    Args:
        rankings: List of ranking lists, each containing document IDs
                 ordered from most to least relevant
        k: Smoothing constant (default 60, standard in literature)
           Higher k = smaller differences between ranks

    Returns:
        List of (document_id, fused_score) tuples sorted by score descending

    Example:
        >>> rankings = [["a", "b", "c"], ["b", "a", "d"]]
        >>> results = reciprocal_rank_fusion(rankings, k=60)
        >>> # 'a' and 'b' will have highest scores (appear in both)
    """
    if not rankings:
        return []

    scores: Dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (k + rank)

    # Sort by fused score descending
    sorted_results = sorted(scores.items(), key=lambda x: -x[1])

    return sorted_results


class HybridSearchService:
    """
    Hybrid search combining BM25 keyword and semantic embedding search.

    Uses Reciprocal Rank Fusion to combine rankings from:
    1. BM25 term-frequency based search
    2. Cosine similarity on embeddings

    Configurable weights allow tuning the balance between exact
    keyword matching and semantic understanding.

    Attributes:
        bm25_index: JobBM25Index for keyword search
        job_embeddings: Dict mapping job ID to embedding vector
        job_data: Dict mapping job ID to full document

    Example:
        >>> service = HybridSearchService()
        >>> service.build_index(jobs)
        >>> results = service.search(
        ...     query_text="Python developer",
        ...     query_embedding=[...],
        ...     top_k=10
        ... )
    """

    def __init__(self) -> None:
        """Initialize empty hybrid search service."""
        self.bm25_index: Optional[JobBM25Index] = None
        self.job_embeddings: Dict[str, List[float]] = {}
        self.job_data: Dict[str, dict] = {}

    def build_index(self, jobs: List[dict]) -> None:
        """
        Build search indices from job documents.

        Args:
            jobs: List of job documents with keys:
                  - id: Unique identifier
                  - title: Job title
                  - description: Job description text
                  - embedding: Pre-computed embedding vector (optional)
        """
        # Build BM25 index from combined title + description
        bm25_docs = []
        for job in jobs:
            job_id = job.get("id", "")
            title = job.get("title", "")
            description = job.get("description", "")

            combined_text = f"{title} {description}"
            bm25_docs.append({"id": job_id, "text": combined_text})

            # Store embedding if available
            if "embedding" in job and job["embedding"]:
                self.job_embeddings[job_id] = job["embedding"]

            # Store full document for later retrieval
            self.job_data[job_id] = job

        self.bm25_index = JobBM25Index(bm25_docs)

    def _semantic_search(
        self,
        query_embedding: List[float],
        top_k: int = 100
    ) -> List[Tuple[str, float]]:
        """
        Perform semantic search using embedding similarity.

        Args:
            query_embedding: Query embedding vector
            top_k: Maximum results to return

        Returns:
            List of (job_id, similarity_score) sorted by score descending
        """
        if not query_embedding or not self.job_embeddings:
            return []

        scores = []
        for job_id, job_embedding in self.job_embeddings.items():
            sim = cosine_similarity_simple(query_embedding, job_embedding)
            scores.append((job_id, sim))

        # Sort by similarity descending
        scores.sort(key=lambda x: -x[1])

        return scores[:top_k]

    def search(
        self,
        query_text: str,
        query_embedding: Optional[List[float]],
        top_k: int = 50,
        bm25_weight: float = 0.5,
        semantic_weight: float = 0.5,
        use_rrf: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Perform hybrid search combining BM25 and semantic search.

        Args:
            query_text: Text query for BM25 search
            query_embedding: Embedding vector for semantic search (optional)
            top_k: Maximum number of results to return
            bm25_weight: Weight for BM25 ranking (0-1)
            semantic_weight: Weight for semantic ranking (0-1)
            use_rrf: If True, use RRF fusion. If False, use weighted scores.

        Returns:
            List of (job_id, combined_score) sorted by score descending
        """
        rankings: List[List[str]] = []

        # BM25 search
        if query_text and query_text.strip() and self.bm25_index:
            bm25_results = self.bm25_index.search(query_text, top_k=top_k * 2)
            if bm25_results:
                # Add multiple copies based on weight for RRF
                weight_copies = max(1, int(bm25_weight * 10))
                for _ in range(weight_copies):
                    rankings.append(bm25_results)

        # Semantic search
        if query_embedding is not None:
            semantic_results = self._semantic_search(query_embedding, top_k=top_k * 2)
            if semantic_results:
                semantic_ids = [r[0] for r in semantic_results]
                # Add multiple copies based on weight for RRF
                weight_copies = max(1, int(semantic_weight * 10))
                for _ in range(weight_copies):
                    rankings.append(semantic_ids)

        if not rankings:
            return []

        if use_rrf:
            # Use RRF fusion
            fused = reciprocal_rank_fusion(rankings, k=60)
            return fused[:top_k]
        else:
            # Simple score-based fusion (not RRF)
            return self._weighted_score_fusion(
                query_text, query_embedding, top_k, bm25_weight, semantic_weight
            )

    def _weighted_score_fusion(
        self,
        query_text: str,
        query_embedding: Optional[List[float]],
        top_k: int,
        bm25_weight: float,
        semantic_weight: float
    ) -> List[Tuple[str, float]]:
        """
        Combine search results using weighted score fusion.

        Alternative to RRF that directly combines normalized scores.
        Less robust but allows finer weight control.

        Args:
            query_text: Text query for BM25
            query_embedding: Embedding for semantic search
            top_k: Max results
            bm25_weight: BM25 score weight
            semantic_weight: Semantic score weight

        Returns:
            Combined scored results
        """
        combined_scores: Dict[str, float] = defaultdict(float)

        # BM25 scores
        if query_text and self.bm25_index:
            bm25_results = self.bm25_index.search_with_scores(query_text, top_k * 2)
            if bm25_results:
                # Normalize BM25 scores
                max_score = max(s for _, s in bm25_results) if bm25_results else 1
                for job_id, score in bm25_results:
                    normalized = score / max_score if max_score > 0 else 0
                    combined_scores[job_id] += normalized * bm25_weight

        # Semantic scores
        if query_embedding:
            semantic_results = self._semantic_search(query_embedding, top_k * 2)
            for job_id, score in semantic_results:
                # Cosine similarity is already normalized (-1 to 1), shift to 0-1
                normalized = (score + 1) / 2
                combined_scores[job_id] += normalized * semantic_weight

        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: -x[1])

        return sorted_results[:top_k]

    def search_with_skill_filter(
        self,
        query_text: str,
        query_embedding: Optional[List[float]],
        required_skills: List[str],
        top_k: int = 50
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search with post-filtering by required skills.

        Performs hybrid search then filters results to only include
        jobs that mention required skills.

        Args:
            query_text: Text query
            query_embedding: Embedding vector
            required_skills: List of skill names that must appear
            top_k: Maximum results after filtering

        Returns:
            Filtered list of (job_id, score) tuples
        """
        # Get more candidates than needed since we'll filter
        candidates = self.search(
            query_text=query_text,
            query_embedding=query_embedding,
            top_k=top_k * 3
        )

        if not required_skills:
            return candidates[:top_k]

        # Filter by skills
        filtered = []
        for job_id, score in candidates:
            job = self.job_data.get(job_id, {})
            description = job.get("description", "").lower()
            title = job.get("title", "").lower()
            combined = f"{title} {description}"

            # Check if all required skills are present
            has_all_skills = all(
                skill.lower() in combined
                for skill in required_skills
            )

            if has_all_skills:
                filtered.append((job_id, score))

            if len(filtered) >= top_k:
                break

        return filtered
