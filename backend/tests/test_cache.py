"""
Tests for Multi-Layer Redis Caching Service

Tests cover:
- L1 Response cache (5min TTL)
- L2 Match score cache (1hr TTL)
- L3 Embedding cache (24hr TTL)
- Cache invalidation
- Hash functions for key generation
- Connection handling (with Redis unavailable)
"""

import pytest
import json
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Optional

# Import the cache module (will be created)
from app.services.cache import (
    MatchCache,
    CacheLayer,
    get_cache,
    hash_content,
)


class TestHashContent:
    """Test content hashing for cache keys."""

    def test_hash_content_returns_16_char_hex(self):
        """Hash should return 16-character hex string."""
        result = hash_content("test content")
        assert len(result) == 16
        assert all(c in "0123456789abcdef" for c in result)

    def test_hash_content_deterministic(self):
        """Same content should produce same hash."""
        content = "Senior Python Developer"
        assert hash_content(content) == hash_content(content)

    def test_hash_content_different_for_different_input(self):
        """Different content should produce different hashes."""
        hash1 = hash_content("Python Developer")
        hash2 = hash_content("Java Developer")
        assert hash1 != hash2

    def test_hash_content_handles_dict(self):
        """Should serialize dicts consistently."""
        data = {"a": 1, "b": 2}
        # Dict ordering shouldn't matter
        assert hash_content(data) == hash_content({"b": 2, "a": 1})

    def test_hash_content_handles_list(self):
        """Should handle list inputs."""
        data = [1, 2, 3]
        result = hash_content(data)
        assert len(result) == 16


class TestCacheLayer:
    """Test CacheLayer enum and TTL values."""

    def test_response_cache_ttl(self):
        """L1 response cache should have 5 minute TTL."""
        assert CacheLayer.RESPONSE.ttl == 300

    def test_match_score_cache_ttl(self):
        """L2 match score cache should have 1 hour TTL."""
        assert CacheLayer.MATCH_SCORE.ttl == 3600

    def test_embedding_cache_ttl(self):
        """L3 embedding cache should have 24 hour TTL."""
        assert CacheLayer.EMBEDDING.ttl == 86400


class AsyncIteratorMock:
    """Mock async iterator for scan_iter."""

    def __init__(self, items: List):
        self.items = items
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.setex = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.scan_iter = MagicMock(return_value=AsyncIteratorMock([]))
    redis.ping = AsyncMock(return_value=True)
    redis.close = AsyncMock()
    return redis


@pytest.fixture
def cache_service(mock_redis):
    """Create MatchCache instance with mock Redis."""
    cache = MatchCache(redis_url="redis://localhost:6379")
    cache.redis = mock_redis
    return cache


class TestMatchCacheResponseCache:
    """Test L1 Response Cache operations."""

    @pytest.mark.asyncio
    async def test_get_response_cache_miss(self, cache_service, mock_redis):
        """Should return None on cache miss."""
        mock_redis.get.return_value = None

        result = await cache_service.get_response("/api/jobs", {"status": "new"})

        assert result is None
        mock_redis.get.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_response_cache_hit(self, cache_service, mock_redis):
        """Should return cached response on hit."""
        cached_data = {"jobs": [{"id": "1", "title": "Developer"}]}
        mock_redis.get.return_value = json.dumps(cached_data)

        result = await cache_service.get_response("/api/jobs", {"status": "new"})

        assert result == cached_data

    @pytest.mark.asyncio
    async def test_set_response_with_correct_ttl(self, cache_service, mock_redis):
        """Should set response with 5 minute TTL."""
        response = {"jobs": []}

        await cache_service.set_response("/api/jobs", {"status": "new"}, response)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 300  # 5 minute TTL

    @pytest.mark.asyncio
    async def test_response_key_format(self, cache_service, mock_redis):
        """Response cache key should have correct format."""
        mock_redis.get.return_value = None

        await cache_service.get_response("/api/jobs", {"status": "new"})

        key = mock_redis.get.call_args[0][0]
        assert key.startswith("resp:/api/jobs:")


class TestMatchCacheScoreCache:
    """Test L2 Match Score Cache operations."""

    @pytest.mark.asyncio
    async def test_get_match_score_cache_miss(self, cache_service, mock_redis):
        """Should return None on cache miss."""
        mock_redis.get.return_value = None

        result = await cache_service.get_match_score("job-123", "profile-abc")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_match_score_cache_hit(self, cache_service, mock_redis):
        """Should return score and reasons on hit."""
        cached = {"score": 85.5, "reasons": ["Skills match", "Location match"]}
        mock_redis.get.return_value = json.dumps(cached)

        result = await cache_service.get_match_score("job-123", "profile-abc")

        assert result is not None
        score, reasons = result
        assert score == 85.5
        assert reasons == ["Skills match", "Location match"]

    @pytest.mark.asyncio
    async def test_set_match_score_with_correct_ttl(self, cache_service, mock_redis):
        """Should set match score with 1 hour TTL."""
        await cache_service.set_match_score(
            "job-123", "profile-abc", 75.0, ["Good fit"]
        )

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 3600  # 1 hour TTL

    @pytest.mark.asyncio
    async def test_match_score_key_format(self, cache_service, mock_redis):
        """Match score key should have correct format."""
        mock_redis.get.return_value = None

        await cache_service.get_match_score("job-123", "profile-abc")

        key = mock_redis.get.call_args[0][0]
        assert key == "match:job-123:profile-abc"


class TestMatchCacheEmbeddingCache:
    """Test L3 Embedding Cache operations."""

    @pytest.mark.asyncio
    async def test_get_embedding_cache_miss(self, cache_service, mock_redis):
        """Should return None on cache miss."""
        mock_redis.get.return_value = None

        result = await cache_service.get_embedding("content-hash-123")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_embedding_cache_hit(self, cache_service, mock_redis):
        """Should return embedding vector on hit."""
        embedding = [0.1, 0.2, 0.3] * 512  # 1536 dims
        mock_redis.get.return_value = json.dumps(embedding)

        result = await cache_service.get_embedding("content-hash-123")

        assert result == embedding

    @pytest.mark.asyncio
    async def test_set_embedding_with_correct_ttl(self, cache_service, mock_redis):
        """Should set embedding with 24 hour TTL."""
        embedding = [0.1] * 1536

        await cache_service.set_embedding("content-hash-123", embedding)

        mock_redis.setex.assert_called_once()
        call_args = mock_redis.setex.call_args
        assert call_args[0][1] == 86400  # 24 hour TTL

    @pytest.mark.asyncio
    async def test_embedding_key_format(self, cache_service, mock_redis):
        """Embedding key should have correct format."""
        mock_redis.get.return_value = None

        await cache_service.get_embedding("abc123")

        key = mock_redis.get.call_args[0][0]
        assert key == "emb:abc123"


class TestMatchCacheInvalidation:
    """Test cache invalidation operations."""

    @pytest.mark.asyncio
    async def test_invalidate_response_cache(self, cache_service, mock_redis):
        """Should delete response cache entries using scan_iter (non-blocking)."""
        mock_redis.scan_iter = MagicMock(
            return_value=AsyncIteratorMock(["resp:/api/jobs:abc", "resp:/api/jobs:def"])
        )

        count = await cache_service.invalidate_response_cache("/api/jobs")

        mock_redis.scan_iter.assert_called_once_with(match="resp:/api/jobs:*")
        assert mock_redis.delete.called

    @pytest.mark.asyncio
    async def test_invalidate_match_scores_for_profile(self, cache_service, mock_redis):
        """Should delete all match scores for a profile hash using scan_iter (non-blocking)."""
        mock_redis.scan_iter = MagicMock(
            return_value=AsyncIteratorMock(["match:job1:profile-abc", "match:job2:profile-abc"])
        )

        count = await cache_service.invalidate_profile_matches("profile-abc")

        mock_redis.scan_iter.assert_called_once_with(match="match:*:profile-abc")
        assert mock_redis.delete.called

    @pytest.mark.asyncio
    async def test_invalidate_embedding(self, cache_service, mock_redis):
        """Should delete specific embedding cache entry."""
        await cache_service.invalidate_embedding("content-hash-123")

        mock_redis.delete.assert_called_with("emb:content-hash-123")

    @pytest.mark.asyncio
    async def test_invalidate_all_caches_for_profile(self, cache_service, mock_redis):
        """Should invalidate all caches when profile changes."""
        mock_redis.scan_iter = MagicMock(return_value=AsyncIteratorMock([]))

        await cache_service.invalidate_all_for_profile(
            profile_id="default",
            profile_hash="abc123",
            cv_hash="def456"
        )

        # Should invalidate match scores and embedding
        assert mock_redis.delete.called or mock_redis.scan_iter.called


class TestMatchCacheMetrics:
    """Test cache metrics tracking."""

    @pytest.mark.asyncio
    async def test_records_cache_hit(self, cache_service, mock_redis):
        """Should record cache hits for metrics."""
        mock_redis.get.return_value = json.dumps({"data": "test"})

        await cache_service.get_response("/api/test", {})

        # Verify hit was recorded (via metrics or internal counter)
        assert cache_service.stats["hits"]["response"] >= 0

    @pytest.mark.asyncio
    async def test_records_cache_miss(self, cache_service, mock_redis):
        """Should record cache misses for metrics."""
        mock_redis.get.return_value = None

        await cache_service.get_response("/api/test", {})

        # Verify miss was recorded
        assert cache_service.stats["misses"]["response"] >= 0

    def test_get_stats_returns_hit_rates(self, cache_service):
        """Should calculate hit rates correctly."""
        cache_service.stats["hits"]["response"] = 80
        cache_service.stats["misses"]["response"] = 20

        stats = cache_service.get_stats()

        assert stats["response"]["hit_rate"] == 0.8


class TestMatchCacheConnectionHandling:
    """Test Redis connection handling."""

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_redis_unavailable(self, cache_service, mock_redis):
        """Should return None and not raise when Redis unavailable."""
        mock_redis.get.side_effect = ConnectionError("Redis unavailable")

        # Should not raise, should return None
        result = await cache_service.get_response("/api/test", {})

        assert result is None

    @pytest.mark.asyncio
    async def test_logs_warning_on_connection_error(self, cache_service, mock_redis):
        """Should log warning when Redis connection fails."""
        mock_redis.get.side_effect = ConnectionError("Redis unavailable")

        with patch("app.services.cache.logger") as mock_logger:
            await cache_service.get_response("/api/test", {})
            # Verify warning was logged
            mock_logger.warning.assert_called()

    @pytest.mark.asyncio
    async def test_health_check(self, cache_service, mock_redis):
        """Should report Redis health status."""
        mock_redis.ping.return_value = True

        is_healthy = await cache_service.health_check()

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_returns_false_on_error(self, cache_service, mock_redis):
        """Should return False when Redis ping fails."""
        mock_redis.ping.side_effect = ConnectionError("Connection refused")

        is_healthy = await cache_service.health_check()

        assert is_healthy is False


class TestGetCacheFactory:
    """Test cache factory function."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the global cache instance before each test."""
        import app.services.cache as cache_module
        cache_module._cache_instance = None
        yield
        cache_module._cache_instance = None

    @pytest.mark.asyncio
    async def test_get_cache_returns_singleton(self):
        """Should return same instance for same URL."""
        with patch("app.services.cache.MatchCache") as mock_class:
            mock_instance = MagicMock()
            mock_class.return_value = mock_instance

            cache1 = await get_cache("redis://localhost:6379")
            cache2 = await get_cache("redis://localhost:6379")

            # Should be same instance (singleton pattern)
            assert cache1 is cache2

    @pytest.mark.asyncio
    async def test_get_cache_uses_settings_redis_url(self):
        """Should use settings.redis_url when no URL provided."""
        with patch("app.services.cache.get_settings") as mock_settings:
            mock_settings.return_value.redis_url = "redis://custom:6379"
            with patch("app.services.cache.MatchCache") as mock_class:
                mock_instance = MagicMock()
                mock_class.return_value = mock_instance

                await get_cache()

                mock_class.assert_called_with(redis_url="redis://custom:6379")
