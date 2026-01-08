"""
Multi-Layer Redis Caching Service

Implements a 3-tier caching strategy for job matching:
- L1 Response Cache (5min TTL): Full API responses
- L2 Match Score Cache (1hr TTL): Job-profile match calculations
- L3 Embedding Cache (24hr TTL): OpenAI embeddings

Cache Key Patterns:
    - resp:{endpoint}:{params_hash} - API responses
    - match:{job_id}:{profile_hash} - Match scores
    - emb:{content_hash} - Embedding vectors

Usage:
    cache = await get_cache()

    # Response caching
    cached = await cache.get_response("/api/jobs", {"status": "new"})
    if not cached:
        response = await fetch_jobs()
        await cache.set_response("/api/jobs", {"status": "new"}, response)

    # Match score caching
    result = await cache.get_match_score(job_id, profile_hash)
    if result:
        score, reasons = result
"""

import json
import hashlib
import logging
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import redis.asyncio as redis
from redis.asyncio.connection import ConnectionPool

from app.config import get_settings

logger = logging.getLogger(__name__)


class CacheLayer(Enum):
    """Cache layers with TTL values in seconds."""

    RESPONSE = ("response", 300)       # 5 minutes
    MATCH_SCORE = ("match_score", 3600)  # 1 hour
    EMBEDDING = ("embedding", 86400)    # 24 hours

    def __init__(self, layer_name: str, ttl: int):
        self.layer_name = layer_name
        self._ttl = ttl

    @property
    def ttl(self) -> int:
        return self._ttl


def hash_content(*args: Any) -> str:
    """
    Generate a 16-character hex hash from content.

    Handles strings, dicts, lists, and other JSON-serializable types.
    Dict keys are sorted for consistent hashing.

    Args:
        *args: Content to hash (will be JSON serialized)

    Returns:
        16-character hex string
    """
    content = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class MatchCache:
    """
    Multi-layer Redis cache for job matching operations.

    Provides graceful degradation when Redis is unavailable,
    returning None instead of raising exceptions.

    Attributes:
        redis: Async Redis client
        stats: Dict tracking hits/misses per layer
    """

    def __init__(self, redis_url: str):
        """
        Initialize cache with Redis URL.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379)
        """
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.stats: Dict[str, Dict[str, int]] = {
            "hits": {"response": 0, "match_score": 0, "embedding": 0},
            "misses": {"response": 0, "match_score": 0, "embedding": 0},
        }

    async def _ensure_connected(self) -> Optional[redis.Redis]:
        """Ensure Redis connection is established."""
        if self.redis is None:
            try:
                self.redis = redis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True
                )
            except Exception as e:
                logger.warning(f"Failed to connect to Redis: {e}")
                return None
        return self.redis

    # ==================== L1: Response Cache ====================

    async def get_response(
        self,
        endpoint: str,
        params: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached API response.

        Args:
            endpoint: API endpoint path (e.g., "/api/jobs")
            params: Request parameters for cache key

        Returns:
            Cached response dict or None on miss/error
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return None

            key = f"resp:{endpoint}:{hash_content(params)}"
            cached = await client.get(key)

            if cached:
                self.stats["hits"]["response"] += 1
                return json.loads(cached)

            self.stats["misses"]["response"] += 1
            return None

        except Exception as e:
            logger.warning(f"Redis get error (response cache): {e}")
            self.stats["misses"]["response"] += 1
            return None

    async def set_response(
        self,
        endpoint: str,
        params: Dict[str, Any],
        response: Dict[str, Any]
    ) -> bool:
        """
        Cache API response.

        Args:
            endpoint: API endpoint path
            params: Request parameters
            response: Response data to cache

        Returns:
            True if cached successfully, False otherwise
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return False

            key = f"resp:{endpoint}:{hash_content(params)}"
            await client.setex(key, CacheLayer.RESPONSE.ttl, json.dumps(response))
            return True

        except Exception as e:
            logger.warning(f"Redis set error (response cache): {e}")
            return False

    # ==================== L2: Match Score Cache ====================

    async def get_match_score(
        self,
        job_id: str,
        profile_hash: str
    ) -> Optional[Tuple[float, List[str]]]:
        """
        Get cached match score for job-profile pair.

        Args:
            job_id: Job UUID
            profile_hash: Hash of profile content

        Returns:
            Tuple of (score, reasons) or None on miss
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return None

            key = f"match:{job_id}:{profile_hash}"
            cached = await client.get(key)

            if cached:
                self.stats["hits"]["match_score"] += 1
                data = json.loads(cached)
                return data["score"], data["reasons"]

            self.stats["misses"]["match_score"] += 1
            return None

        except Exception as e:
            logger.warning(f"Redis get error (match score cache): {e}")
            self.stats["misses"]["match_score"] += 1
            return None

    async def set_match_score(
        self,
        job_id: str,
        profile_hash: str,
        score: float,
        reasons: List[str]
    ) -> bool:
        """
        Cache match score for job-profile pair.

        Args:
            job_id: Job UUID
            profile_hash: Hash of profile content
            score: Calculated match score (0-100)
            reasons: List of match reason strings

        Returns:
            True if cached successfully
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return False

            key = f"match:{job_id}:{profile_hash}"
            data = {"score": score, "reasons": reasons}
            await client.setex(key, CacheLayer.MATCH_SCORE.ttl, json.dumps(data))
            return True

        except Exception as e:
            logger.warning(f"Redis set error (match score cache): {e}")
            return False

    # ==================== L3: Embedding Cache ====================

    async def get_embedding(self, content_hash: str) -> Optional[List[float]]:
        """
        Get cached embedding vector.

        Args:
            content_hash: Hash of text content

        Returns:
            1536-dim embedding vector or None on miss
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return None

            key = f"emb:{content_hash}"
            cached = await client.get(key)

            if cached:
                self.stats["hits"]["embedding"] += 1
                return json.loads(cached)

            self.stats["misses"]["embedding"] += 1
            return None

        except Exception as e:
            logger.warning(f"Redis get error (embedding cache): {e}")
            self.stats["misses"]["embedding"] += 1
            return None

    async def set_embedding(
        self,
        content_hash: str,
        embedding: List[float]
    ) -> bool:
        """
        Cache embedding vector.

        Args:
            content_hash: Hash of original text content
            embedding: 1536-dim embedding vector

        Returns:
            True if cached successfully
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return False

            key = f"emb:{content_hash}"
            await client.setex(key, CacheLayer.EMBEDDING.ttl, json.dumps(embedding))
            return True

        except Exception as e:
            logger.warning(f"Redis set error (embedding cache): {e}")
            return False

    # ==================== Cache Invalidation ====================

    async def invalidate_response_cache(self, endpoint: str) -> int:
        """
        Invalidate all cached responses for an endpoint.

        Args:
            endpoint: API endpoint path to invalidate

        Returns:
            Number of keys deleted
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return 0

            pattern = f"resp:{endpoint}:*"
            keys = await client.keys(pattern)

            if keys:
                return await client.delete(*keys)
            return 0

        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return 0

    async def invalidate_profile_matches(self, profile_hash: str) -> int:
        """
        Invalidate all match scores for a profile.

        Args:
            profile_hash: Profile content hash

        Returns:
            Number of keys deleted
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return 0

            pattern = f"match:*:{profile_hash}"
            keys = await client.keys(pattern)

            if keys:
                return await client.delete(*keys)
            return 0

        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return 0

    async def invalidate_embedding(self, content_hash: str) -> bool:
        """
        Invalidate a specific embedding cache entry.

        Args:
            content_hash: Hash of content to invalidate

        Returns:
            True if key was deleted
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return False

            key = f"emb:{content_hash}"
            result = await client.delete(key)
            return result > 0

        except Exception as e:
            logger.warning(f"Redis delete error: {e}")
            return False

    async def invalidate_all_for_profile(
        self,
        profile_id: str,
        profile_hash: str,
        cv_hash: str
    ) -> Dict[str, int]:
        """
        Invalidate all caches when profile changes.

        Args:
            profile_id: Profile ID
            profile_hash: New profile content hash
            cv_hash: Hash of CV text for embedding lookup

        Returns:
            Dict with counts of invalidated keys per layer
        """
        results = {
            "match_scores": 0,
            "embeddings": 0,
            "responses": 0,
        }

        try:
            # Invalidate match scores
            results["match_scores"] = await self.invalidate_profile_matches(profile_hash)

            # Invalidate CV embedding
            if await self.invalidate_embedding(cv_hash):
                results["embeddings"] = 1

            # Invalidate job-related responses
            results["responses"] = await self.invalidate_response_cache("/api/jobs")

        except Exception as e:
            logger.warning(f"Error during profile cache invalidation: {e}")

        return results

    # ==================== Health & Stats ====================

    async def health_check(self) -> bool:
        """
        Check Redis connection health.

        Returns:
            True if Redis is responsive
        """
        try:
            client = await self._ensure_connected()
            if not client:
                return False

            await client.ping()
            return True

        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get cache statistics including hit rates.

        Returns:
            Dict with stats per cache layer
        """
        stats = {}

        for layer in ["response", "match_score", "embedding"]:
            hits = self.stats["hits"][layer]
            misses = self.stats["misses"][layer]
            total = hits + misses

            stats[layer] = {
                "hits": hits,
                "misses": misses,
                "total": total,
                "hit_rate": hits / total if total > 0 else 0.0,
            }

        return stats

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis:
            await self.redis.close()
            self.redis = None


# ==================== Factory Function ====================

_cache_instance: Optional[MatchCache] = None


async def get_cache(redis_url: Optional[str] = None) -> MatchCache:
    """
    Get or create cache singleton.

    Args:
        redis_url: Optional Redis URL (uses settings if not provided)

    Returns:
        MatchCache instance
    """
    global _cache_instance

    if _cache_instance is None:
        url = redis_url or get_settings().redis_url
        _cache_instance = MatchCache(redis_url=url)

    return _cache_instance
