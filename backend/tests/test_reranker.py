"""
Tests for cross-encoder re-ranking service.

Tests cover:
- Local cross-encoder re-ranking (sentence-transformers)
- Cohere API re-ranking (mocked)
- Re-ranker provider abstraction

Run with: cd backend && pytest tests/test_reranker.py -v
"""
import pytest
from typing import List, Tuple
from unittest.mock import Mock, AsyncMock, patch


class TestLocalCrossEncoder:
    """Tests for local cross-encoder re-ranking."""

    def test_local_reranker_initialization(self):
        """Should initialize with default model."""
        from app.services.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker()
        assert reranker is not None

    def test_local_reranker_custom_model(self):
        """Should accept custom model name."""
        from app.services.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker(
            model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        assert reranker.model_name == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_local_reranker_reranks_documents(self):
        """Should rerank documents by relevance to query."""
        from app.services.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker(lazy_load=True)

        query = "Python developer with Django experience"
        documents = [
            {"id": "1", "text": "Java developer with Spring Boot"},
            {"id": "2", "text": "Python engineer specializing in Django and Flask"},
            {"id": "3", "text": "Frontend React developer"},
        ]

        results = reranker.rerank(query, documents, top_k=2)

        assert len(results) <= 2
        # Document 2 (Python + Django) should rank highest
        assert results[0][0] == "2"

    def test_local_reranker_returns_scores(self):
        """Should return relevance scores with documents."""
        from app.services.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker(lazy_load=True)

        query = "Machine learning engineer"
        documents = [
            {"id": "1", "text": "ML engineer with PyTorch experience"},
            {"id": "2", "text": "Data analyst role"},
        ]

        results = reranker.rerank(query, documents, top_k=2)

        for doc_id, score in results:
            assert isinstance(doc_id, str)
            assert isinstance(score, float)

        # ML engineer should score higher
        assert results[0][1] > results[1][1]

    def test_local_reranker_handles_empty_documents(self):
        """Should handle empty document list."""
        from app.services.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker(lazy_load=True)

        results = reranker.rerank("Python", [], top_k=5)
        assert results == []

    def test_local_reranker_handles_empty_query(self):
        """Should handle empty query."""
        from app.services.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker(lazy_load=True)

        documents = [{"id": "1", "text": "Python developer"}]
        results = reranker.rerank("", documents, top_k=5)

        assert results == []

    def test_local_reranker_respects_top_k(self):
        """Should not return more than top_k results."""
        from app.services.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker(lazy_load=True)

        documents = [
            {"id": str(i), "text": f"Python developer {i}"}
            for i in range(20)
        ]

        results = reranker.rerank("Python", documents, top_k=5)
        assert len(results) <= 5


class TestCohereReranker:
    """Tests for Cohere API re-ranking."""

    @pytest.fixture
    def mock_cohere_client(self):
        """Create mock Cohere client."""
        mock_result = Mock()
        mock_result.results = [
            Mock(index=1, relevance_score=0.95),
            Mock(index=0, relevance_score=0.75),
        ]

        mock_client = Mock()
        mock_client.rerank = Mock(return_value=mock_result)
        return mock_client

    def test_cohere_reranker_initialization(self):
        """Should initialize with API key."""
        from app.services.reranker import CohereReranker

        reranker = CohereReranker(api_key="test-key")
        assert reranker is not None

    def test_cohere_reranker_uses_api(self, mock_cohere_client):
        """Should call Cohere API for reranking."""
        from app.services.reranker import CohereReranker

        reranker = CohereReranker(api_key="test-key")
        reranker._client = mock_cohere_client

        documents = [
            {"id": "1", "text": "Java developer"},
            {"id": "2", "text": "Python developer"},
        ]

        results = reranker.rerank("Python", documents, top_k=2)

        mock_cohere_client.rerank.assert_called_once()
        assert len(results) == 2

    def test_cohere_reranker_handles_api_error(self, mock_cohere_client):
        """Should handle API errors gracefully."""
        from app.services.reranker import CohereReranker

        mock_cohere_client.rerank.side_effect = Exception("API Error")

        reranker = CohereReranker(api_key="test-key")
        reranker._client = mock_cohere_client

        documents = [{"id": "1", "text": "Python"}]

        # Should not raise, returns empty or original order
        results = reranker.rerank("Python", documents, top_k=1)
        assert isinstance(results, list)

    def test_cohere_reranker_maps_results_correctly(self, mock_cohere_client):
        """Should map API results back to original documents."""
        from app.services.reranker import CohereReranker

        reranker = CohereReranker(api_key="test-key")
        reranker._client = mock_cohere_client

        documents = [
            {"id": "doc_a", "text": "Java developer"},
            {"id": "doc_b", "text": "Python developer"},
        ]

        results = reranker.rerank("Python", documents, top_k=2)

        # Mock returns index 1 (doc_b) first with higher score
        assert results[0][0] == "doc_b"
        assert results[0][1] == 0.95


class TestRerankerFactory:
    """Tests for reranker provider factory."""

    def test_factory_creates_local_reranker(self):
        """Should create local cross-encoder by default."""
        from app.services.reranker import get_reranker

        reranker = get_reranker(provider="local")
        assert reranker is not None

    def test_factory_creates_cohere_reranker(self):
        """Should create Cohere reranker when specified."""
        from app.services.reranker import get_reranker

        reranker = get_reranker(provider="cohere", api_key="test-key")
        assert reranker is not None

    def test_factory_invalid_provider(self):
        """Should raise error for unknown provider."""
        from app.services.reranker import get_reranker

        with pytest.raises(ValueError):
            get_reranker(provider="unknown")


class TestRerankerInterface:
    """Tests for abstract reranker interface."""

    def test_all_rerankers_implement_interface(self):
        """All rerankers should implement the base interface."""
        from app.services.reranker import (
            RerankerBase,
            LocalCrossEncoderReranker,
            CohereReranker,
        )

        # Both should be subclasses of RerankerBase
        assert issubclass(LocalCrossEncoderReranker, RerankerBase)
        assert issubclass(CohereReranker, RerankerBase)

    def test_interface_has_required_methods(self):
        """Interface should define rerank method."""
        from app.services.reranker import RerankerBase

        assert hasattr(RerankerBase, "rerank")


class TestAsyncReranker:
    """Tests for async reranking operations."""

    @pytest.mark.asyncio
    async def test_async_rerank(self):
        """Should support async reranking."""
        from app.services.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker(lazy_load=True)

        documents = [
            {"id": "1", "text": "Python developer"},
            {"id": "2", "text": "Java developer"},
        ]

        results = await reranker.rerank_async("Python", documents, top_k=2)

        assert len(results) <= 2
        assert results[0][0] == "1"  # Python match


class TestRerankerBatching:
    """Tests for batch reranking optimization."""

    def test_rerank_preserves_order_with_identical_scores(self):
        """Should maintain stable ordering for equal scores."""
        from app.services.reranker import LocalCrossEncoderReranker

        reranker = LocalCrossEncoderReranker(lazy_load=True)

        # Very similar documents
        documents = [
            {"id": "a", "text": "Python developer role"},
            {"id": "b", "text": "Python developer position"},
            {"id": "c", "text": "Python developer job"},
        ]

        results1 = reranker.rerank("Python developer", documents, top_k=3)
        results2 = reranker.rerank("Python developer", documents, top_k=3)

        # Order should be consistent across calls
        ids1 = [r[0] for r in results1]
        ids2 = [r[0] for r in results2]
        assert ids1 == ids2


class TestTwoStageRetrieval:
    """Integration tests for two-stage retrieval pipeline."""

    def test_hybrid_search_then_rerank(self):
        """Should work with hybrid search results as input."""
        from app.services.hybrid_search import HybridSearchService
        from app.services.reranker import LocalCrossEncoderReranker

        # Stage 1: Hybrid search
        jobs = [
            {
                "id": "1",
                "title": "Senior Python Developer",
                "description": "Django and FastAPI experience required",
                "embedding": [0.1] * 10,
            },
            {
                "id": "2",
                "title": "Java Engineer",
                "description": "Spring Boot microservices",
                "embedding": [0.2] * 10,
            },
            {
                "id": "3",
                "title": "Python ML Engineer",
                "description": "TensorFlow and PyTorch for ML systems",
                "embedding": [0.3] * 10,
            },
        ]

        search_service = HybridSearchService()
        search_service.build_index(jobs)

        candidates = search_service.search(
            query_text="Python developer",
            query_embedding=[0.1] * 10,
            top_k=3
        )

        # Stage 2: Rerank candidates
        reranker = LocalCrossEncoderReranker(lazy_load=True)

        # Convert candidates to reranker format
        candidate_docs = []
        for job_id, _ in candidates:
            job = search_service.job_data[job_id]
            candidate_docs.append({
                "id": job_id,
                "text": f"{job['title']} {job['description']}"
            })

        reranked = reranker.rerank(
            query="Python developer with Django experience",
            documents=candidate_docs,
            top_k=2
        )

        assert len(reranked) <= 2
        # Senior Python Developer (Django) should be top after reranking
        assert reranked[0][0] == "1"
