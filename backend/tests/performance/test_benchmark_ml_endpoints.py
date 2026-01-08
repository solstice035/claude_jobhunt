"""
Pytest Benchmark Tests for ML Endpoints

This module provides pytest-based benchmarks for measuring performance of ML endpoints.
Can be integrated with CI/CD pipelines to detect performance regressions.

Usage:
    # Run benchmarks
    pytest tests/performance/test_benchmark_ml_endpoints.py -v

    # Run with timing output
    pytest tests/performance/test_benchmark_ml_endpoints.py -v --durations=0

    # Run with pytest-benchmark (if installed)
    pytest tests/performance/test_benchmark_ml_endpoints.py --benchmark-only

Requirements:
    - pytest
    - pytest-asyncio
    - httpx
    - pytest-benchmark (optional, for detailed benchmarks)
"""

import asyncio
import statistics
import time
from typing import List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Try to import actual services for integration tests
try:
    from app.services.hybrid_search import HybridSearchService
    from app.services.skill_extractor import SkillExtractor
    from app.services.reranker import LocalReranker

    SERVICES_AVAILABLE = True
except ImportError:
    SERVICES_AVAILABLE = False


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def sample_job_documents() -> List[dict]:
    """Sample job documents for testing."""
    return [
        {
            "id": f"job-{i}",
            "title": f"Software Engineer {i}",
            "description": f"Looking for a Python developer with experience in AWS, Docker, "
                          f"and microservices. Job number {i}.",
            "embedding": [0.1 * (i % 10)] * 384,  # Mock embedding
        }
        for i in range(100)
    ]


@pytest.fixture
def sample_query_text() -> str:
    """Sample query text for testing."""
    return "Python developer with AWS and Docker experience for microservices architecture"


@pytest.fixture
def sample_cv_text() -> str:
    """Sample CV text for skill extraction."""
    return """
    Experienced Software Engineer with 8+ years in Python development.

    Technical Skills:
    - Languages: Python, JavaScript, TypeScript, Go
    - Cloud: AWS (EC2, S3, Lambda, ECS), Azure, GCP
    - Containers: Docker, Kubernetes, Helm
    - Databases: PostgreSQL, MongoDB, Redis, Elasticsearch
    - Frameworks: FastAPI, Django, React, Node.js

    Experience:
    - Led development of microservices architecture serving 1M+ users
    - Implemented CI/CD pipelines using GitHub Actions and ArgoCD
    - Designed and deployed ML models using TensorFlow and PyTorch
    """


# ==============================================================================
# Performance Measurement Utilities
# ==============================================================================

async def measure_async_execution(
    func,
    iterations: int = 10,
    warmup: int = 2
) -> Tuple[float, float, float, float]:
    """
    Measure async function execution time.

    Returns:
        Tuple of (avg_ms, p50_ms, p95_ms, p99_ms)
    """
    times = []

    # Warmup runs
    for _ in range(warmup):
        await func()

    # Measured runs
    for _ in range(iterations):
        start = time.perf_counter()
        await func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    import numpy as np
    return (
        statistics.mean(times),
        float(np.percentile(times, 50)),
        float(np.percentile(times, 95)),
        float(np.percentile(times, 99)),
    )


def measure_sync_execution(
    func,
    iterations: int = 10,
    warmup: int = 2
) -> Tuple[float, float, float, float]:
    """
    Measure sync function execution time.

    Returns:
        Tuple of (avg_ms, p50_ms, p95_ms, p99_ms)
    """
    times = []

    # Warmup runs
    for _ in range(warmup):
        func()

    # Measured runs
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    import numpy as np
    return (
        statistics.mean(times),
        float(np.percentile(times, 50)),
        float(np.percentile(times, 95)),
        float(np.percentile(times, 99)),
    )


# ==============================================================================
# Hybrid Search Benchmarks
# ==============================================================================

@pytest.mark.skipif(not SERVICES_AVAILABLE, reason="Services not available")
class TestHybridSearchPerformance:
    """Performance tests for HybridSearchService."""

    def test_bm25_index_build_performance(self, sample_job_documents):
        """Benchmark BM25 index building."""
        service = HybridSearchService()

        avg, p50, p95, p99 = measure_sync_execution(
            lambda: service.build_index(sample_job_documents),
            iterations=5,
            warmup=1
        )

        print(f"\nBM25 Index Build (100 docs):")
        print(f"  Avg: {avg:.2f}ms, p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")

        # Performance assertions
        assert avg < 1000, f"Index build too slow: {avg:.2f}ms (expected < 1000ms)"
        assert p95 < 2000, f"p95 too high: {p95:.2f}ms (expected < 2000ms)"

    def test_bm25_search_performance(self, sample_job_documents, sample_query_text):
        """Benchmark BM25 search."""
        service = HybridSearchService()
        service.build_index(sample_job_documents)

        avg, p50, p95, p99 = measure_sync_execution(
            lambda: service._bm25_search(sample_query_text, top_k=20),
            iterations=20,
            warmup=3
        )

        print(f"\nBM25 Search (100 docs, top_k=20):")
        print(f"  Avg: {avg:.2f}ms, p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")

        # BM25 should be very fast
        assert avg < 50, f"BM25 search too slow: {avg:.2f}ms (expected < 50ms)"
        assert p95 < 100, f"p95 too high: {p95:.2f}ms (expected < 100ms)"

    def test_semantic_search_performance(self, sample_job_documents, sample_query_text):
        """Benchmark semantic (embedding) search."""
        service = HybridSearchService()
        service.build_index(sample_job_documents)

        # Create a mock query embedding
        query_embedding = [0.1] * 384

        avg, p50, p95, p99 = measure_sync_execution(
            lambda: service._semantic_search(query_embedding, top_k=20),
            iterations=20,
            warmup=3
        )

        print(f"\nSemantic Search (100 docs, top_k=20):")
        print(f"  Avg: {avg:.2f}ms, p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")

        # Semantic search should be reasonably fast
        assert avg < 100, f"Semantic search too slow: {avg:.2f}ms (expected < 100ms)"
        assert p95 < 200, f"p95 too high: {p95:.2f}ms (expected < 200ms)"

    def test_hybrid_search_performance(self, sample_job_documents, sample_query_text):
        """Benchmark combined hybrid search."""
        service = HybridSearchService()
        service.build_index(sample_job_documents)

        query_embedding = [0.1] * 384

        avg, p50, p95, p99 = measure_sync_execution(
            lambda: service.search(
                query_text=sample_query_text,
                query_embedding=query_embedding,
                top_k=20,
                use_rrf=True
            ),
            iterations=20,
            warmup=3
        )

        print(f"\nHybrid Search with RRF (100 docs, top_k=20):")
        print(f"  Avg: {avg:.2f}ms, p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")

        # Hybrid search should complete within reasonable time
        assert avg < 200, f"Hybrid search too slow: {avg:.2f}ms (expected < 200ms)"
        assert p95 < 400, f"p95 too high: {p95:.2f}ms (expected < 400ms)"


# ==============================================================================
# Reranker Benchmarks
# ==============================================================================

@pytest.mark.skipif(not SERVICES_AVAILABLE, reason="Services not available")
class TestRerankerPerformance:
    """Performance tests for reranker models."""

    @pytest.fixture
    def sample_documents_for_rerank(self) -> List[dict]:
        """Sample documents for reranking."""
        return [
            {
                "id": f"doc-{i}",
                "text": f"Python developer with experience in AWS, Docker, and microservices. "
                       f"Looking for senior engineers to build scalable systems. Document {i}."
            }
            for i in range(20)
        ]

    def test_local_reranker_load_time(self):
        """Benchmark reranker model loading."""
        start = time.perf_counter()
        reranker = LocalReranker(lazy_load=False)
        load_time = (time.perf_counter() - start) * 1000

        print(f"\nLocal Reranker Load Time: {load_time:.2f}ms")

        # Model loading can be slow, but should complete
        assert load_time < 60000, f"Model loading too slow: {load_time:.2f}ms (expected < 60s)"

    @pytest.mark.asyncio
    async def test_local_reranker_inference_performance(self, sample_documents_for_rerank):
        """Benchmark reranker inference."""
        reranker = LocalReranker(lazy_load=False)
        query = "Senior Python developer for AWS microservices"

        avg, p50, p95, p99 = await measure_async_execution(
            lambda: reranker.rerank_async(query, sample_documents_for_rerank, top_k=10),
            iterations=10,
            warmup=2
        )

        print(f"\nLocal Reranker Inference (20 docs, top_k=10):")
        print(f"  Avg: {avg:.2f}ms, p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")

        # Reranking should complete in reasonable time
        # Note: First inference may be slower due to model warmup
        assert avg < 5000, f"Reranking too slow: {avg:.2f}ms (expected < 5000ms)"


# ==============================================================================
# Skill Extraction Benchmarks
# ==============================================================================

@pytest.mark.skipif(not SERVICES_AVAILABLE, reason="Services not available")
class TestSkillExtractionPerformance:
    """Performance tests for skill extraction."""

    @pytest.mark.asyncio
    async def test_skill_extraction_short_text(self):
        """Benchmark skill extraction from short text."""
        extractor = SkillExtractor()
        short_text = "Python developer with AWS and Docker experience"

        # Mock the LLM call for benchmark
        with patch.object(extractor, '_extract_with_llm') as mock_llm:
            mock_llm.return_value = [
                {"name": "Python", "category": "technical", "required": True, "confidence": "high"},
                {"name": "AWS", "category": "technical", "required": True, "confidence": "high"},
                {"name": "Docker", "category": "technical", "required": True, "confidence": "high"},
            ]

            avg, p50, p95, p99 = await measure_async_execution(
                lambda: extractor.extract_skills(short_text),
                iterations=20,
                warmup=3
            )

        print(f"\nSkill Extraction (short text, mocked LLM):")
        print(f"  Avg: {avg:.2f}ms, p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")

    @pytest.mark.asyncio
    async def test_skill_extraction_long_text(self, sample_cv_text):
        """Benchmark skill extraction from long CV text."""
        extractor = SkillExtractor()

        # Mock the LLM call
        with patch.object(extractor, '_extract_with_llm') as mock_llm:
            mock_llm.return_value = [
                {"name": "Python", "category": "technical", "required": True, "confidence": "high"},
                {"name": "AWS", "category": "technical", "required": True, "confidence": "high"},
                {"name": "Docker", "category": "technical", "required": True, "confidence": "high"},
                {"name": "Kubernetes", "category": "technical", "required": True, "confidence": "high"},
            ]

            avg, p50, p95, p99 = await measure_async_execution(
                lambda: extractor.extract_skills(sample_cv_text),
                iterations=10,
                warmup=2
            )

        print(f"\nSkill Extraction (long CV text, mocked LLM):")
        print(f"  Avg: {avg:.2f}ms, p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")


# ==============================================================================
# Mock-Based API Endpoint Benchmarks
# ==============================================================================

class TestMockedEndpointPerformance:
    """
    Performance tests using mocked services.

    These tests measure the overhead of the API layer without
    actual ML model inference.
    """

    @pytest.fixture
    def mock_hybrid_search_service(self):
        """Create a mocked hybrid search service."""
        service = MagicMock()
        service.search.return_value = [
            ("job-1", 0.95),
            ("job-2", 0.85),
            ("job-3", 0.75),
        ]
        return service

    def test_api_serialization_overhead(self):
        """Benchmark API response serialization."""
        from pydantic import BaseModel
        from typing import List, Optional

        class MockJobResult(BaseModel):
            job_id: str
            title: str
            company: str
            location: str
            match_score: float
            hybrid_score: float
            rerank_score: Optional[float] = None
            description_preview: str

        # Create sample data
        results = [
            MockJobResult(
                job_id=f"job-{i}",
                title=f"Software Engineer {i}",
                company=f"Company {i}",
                location="London",
                match_score=0.9 - (i * 0.05),
                hybrid_score=0.85 - (i * 0.05),
                rerank_score=0.9 - (i * 0.03),
                description_preview="A" * 300  # 300 char description
            )
            for i in range(50)
        ]

        avg, p50, p95, p99 = measure_sync_execution(
            lambda: [r.model_dump() for r in results],
            iterations=100,
            warmup=10
        )

        print(f"\nAPI Serialization (50 results):")
        print(f"  Avg: {avg:.2f}ms, p50: {p50:.2f}ms, p95: {p95:.2f}ms, p99: {p99:.2f}ms")

        # Serialization should be very fast
        assert avg < 10, f"Serialization too slow: {avg:.2f}ms (expected < 10ms)"


# ==============================================================================
# Throughput Tests
# ==============================================================================

class TestThroughput:
    """Throughput measurement tests."""

    def test_concurrent_requests_simulation(self):
        """Simulate concurrent request handling."""
        import asyncio

        async def simulate_request():
            # Simulate request processing time
            await asyncio.sleep(0.01)  # 10ms
            return True

        async def run_concurrent(num_requests: int, concurrency: int):
            """Run requests with limited concurrency."""
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded_request():
                async with semaphore:
                    return await simulate_request()

            start = time.perf_counter()
            tasks = [bounded_request() for _ in range(num_requests)]
            await asyncio.gather(*tasks)
            duration = time.perf_counter() - start

            throughput = num_requests / duration
            return throughput

        # Test different concurrency levels
        results = {}
        for concurrency in [1, 5, 10, 20]:
            throughput = asyncio.run(run_concurrent(100, concurrency))
            results[concurrency] = throughput

        print("\nThroughput by Concurrency Level:")
        for conc, tput in results.items():
            print(f"  Concurrency {conc}: {tput:.1f} req/s")


# ==============================================================================
# Performance Regression Detection
# ==============================================================================

class TestPerformanceBaselines:
    """
    Tests to detect performance regressions against baselines.

    Update these baselines when performance improvements are made.
    """

    # Performance baselines (in milliseconds)
    BASELINES = {
        "bm25_search": {"p95": 100, "p99": 200},
        "semantic_search": {"p95": 200, "p99": 400},
        "hybrid_search": {"p95": 400, "p99": 800},
        "api_serialization": {"p95": 5, "p99": 10},
    }

    def test_baselines_defined(self):
        """Verify all baselines are defined."""
        required_baselines = [
            "bm25_search",
            "semantic_search",
            "hybrid_search",
            "api_serialization"
        ]

        for baseline in required_baselines:
            assert baseline in self.BASELINES, f"Missing baseline: {baseline}"
            assert "p95" in self.BASELINES[baseline], f"Missing p95 for {baseline}"
            assert "p99" in self.BASELINES[baseline], f"Missing p99 for {baseline}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
