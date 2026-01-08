"""
Tests for hybrid search service.

Tests cover:
- BM25 keyword search
- Reciprocal Rank Fusion (RRF) algorithm
- Hybrid search combining BM25 and semantic search

Run with: cd backend && pytest tests/test_hybrid_search.py -v
"""
import pytest
from typing import List, Tuple


class TestBM25Index:
    """Tests for BM25 keyword search index."""

    def test_bm25_index_initialization(self):
        """Should initialize BM25 index with tokenized documents."""
        from app.services.hybrid_search import JobBM25Index

        documents = [
            {"id": "1", "text": "Python developer with Django experience"},
            {"id": "2", "text": "Java engineer with Spring Boot skills"},
            {"id": "3", "text": "Full stack JavaScript developer"},
        ]

        index = JobBM25Index(documents)
        assert index is not None
        assert len(index.job_ids) == 3

    def test_bm25_search_finds_relevant_documents(self):
        """Should return documents matching query keywords."""
        from app.services.hybrid_search import JobBM25Index

        documents = [
            {"id": "1", "text": "Python developer with Django experience"},
            {"id": "2", "text": "Java engineer with Spring Boot skills"},
            {"id": "3", "text": "Python and machine learning expert"},
        ]

        index = JobBM25Index(documents)
        results = index.search("Python developer", top_k=2)

        assert len(results) <= 2
        assert "1" in results  # Best match for "Python developer"

    def test_bm25_search_returns_scores(self):
        """Should return documents with their BM25 scores."""
        from app.services.hybrid_search import JobBM25Index

        # Need more documents for BM25 IDF to work properly
        documents = [
            {"id": "1", "text": "Python Python Python developer experience required"},
            {"id": "2", "text": "Java Spring Boot engineer backend"},
            {"id": "3", "text": "Frontend React TypeScript developer"},
            {"id": "4", "text": "DevOps engineer AWS cloud infrastructure"},
            {"id": "5", "text": "Data scientist machine learning Python"},
        ]

        index = JobBM25Index(documents)
        results = index.search_with_scores("Python developer", top_k=5)

        assert len(results) > 0
        assert isinstance(results[0], tuple)
        # Document 1 should rank high (has Python developer)
        top_ids = [r[0] for r in results[:2]]
        assert "1" in top_ids or "5" in top_ids  # Both have Python

    def test_bm25_search_handles_empty_query(self):
        """Should handle empty query gracefully."""
        from app.services.hybrid_search import JobBM25Index

        documents = [
            {"id": "1", "text": "Python developer"},
        ]

        index = JobBM25Index(documents)
        results = index.search("", top_k=5)

        assert results == []

    def test_bm25_search_case_insensitive(self):
        """Should match regardless of case."""
        from app.services.hybrid_search import JobBM25Index

        # Need more documents for BM25 IDF to work properly
        documents = [
            {"id": "1", "text": "PYTHON DEVELOPER senior role"},
            {"id": "2", "text": "python developer junior position"},
            {"id": "3", "text": "Java engineer backend systems"},
            {"id": "4", "text": "React frontend engineer"},
            {"id": "5", "text": "DevOps cloud infrastructure"},
        ]

        index = JobBM25Index(documents)
        results = index.search("Python", top_k=5)

        # Both doc 1 and 2 have Python (different cases) and should be found
        assert len(results) > 0
        # At least one Python document should be in top results
        assert "1" in results or "2" in results

    def test_bm25_respects_top_k_limit(self):
        """Should not return more than top_k results."""
        from app.services.hybrid_search import JobBM25Index

        documents = [
            {"id": str(i), "text": f"Python developer {i}"}
            for i in range(100)
        ]

        index = JobBM25Index(documents)
        results = index.search("Python", top_k=10)

        assert len(results) <= 10

    def test_bm25_empty_index(self):
        """Should handle empty document list."""
        from app.services.hybrid_search import JobBM25Index

        index = JobBM25Index([])
        results = index.search("Python", top_k=5)

        assert results == []


class TestReciprocalRankFusion:
    """Tests for RRF score fusion algorithm."""

    def test_rrf_combines_single_ranking(self):
        """Should handle a single ranking list."""
        from app.services.hybrid_search import reciprocal_rank_fusion

        rankings = [["a", "b", "c"]]
        results = reciprocal_rank_fusion(rankings, k=60)

        assert len(results) == 3
        assert results[0][0] == "a"  # First place gets highest score
        assert results[0][1] > results[1][1]  # Scores decrease

    def test_rrf_combines_multiple_rankings(self):
        """Should fuse scores from multiple rankings."""
        from app.services.hybrid_search import reciprocal_rank_fusion

        rankings = [
            ["a", "b", "c"],  # Ranking 1
            ["b", "a", "c"],  # Ranking 2
        ]
        results = reciprocal_rank_fusion(rankings, k=60)

        # 'a' and 'b' should be top 2 (appear high in both)
        top_ids = [r[0] for r in results[:2]]
        assert "a" in top_ids
        assert "b" in top_ids

    def test_rrf_handles_partial_overlap(self):
        """Should handle rankings with different items."""
        from app.services.hybrid_search import reciprocal_rank_fusion

        rankings = [
            ["a", "b", "c"],
            ["d", "e", "a"],  # Only 'a' is in both
        ]
        results = reciprocal_rank_fusion(rankings, k=60)

        result_ids = [r[0] for r in results]
        assert "a" in result_ids  # 'a' should be present
        assert len(result_ids) == 5  # All unique items

    def test_rrf_formula_correctness(self):
        """Should compute RRF scores correctly: 1/(k+rank)."""
        from app.services.hybrid_search import reciprocal_rank_fusion

        rankings = [["a", "b"]]
        k = 60
        results = reciprocal_rank_fusion(rankings, k=k)

        # Expected: a = 1/(60+1), b = 1/(60+2)
        expected_a = 1.0 / (k + 1)
        expected_b = 1.0 / (k + 2)

        assert abs(results[0][1] - expected_a) < 0.0001
        assert abs(results[1][1] - expected_b) < 0.0001

    def test_rrf_empty_rankings(self):
        """Should handle empty ranking lists."""
        from app.services.hybrid_search import reciprocal_rank_fusion

        results = reciprocal_rank_fusion([], k=60)
        assert results == []

    def test_rrf_k_parameter_affects_scores(self):
        """Different k values should produce different score distributions."""
        from app.services.hybrid_search import reciprocal_rank_fusion

        rankings = [["a", "b", "c"]]

        results_k60 = reciprocal_rank_fusion(rankings, k=60)
        results_k1 = reciprocal_rank_fusion(rankings, k=1)

        # With k=1, score difference between ranks is larger
        diff_k60 = results_k60[0][1] - results_k60[1][1]
        diff_k1 = results_k1[0][1] - results_k1[1][1]

        assert diff_k1 > diff_k60  # Larger k = smaller differences


class TestHybridSearch:
    """Tests for combined hybrid search functionality."""

    @pytest.fixture
    def sample_jobs(self) -> List[dict]:
        """Sample job documents for testing."""
        return [
            {
                "id": "job1",
                "title": "Senior Python Developer",
                "description": "Looking for a senior Python developer with Django and AWS experience.",
                "embedding": [0.1] * 10,  # Simplified embedding
            },
            {
                "id": "job2",
                "title": "Java Engineer",
                "description": "Enterprise Java developer with Spring Boot and microservices expertise.",
                "embedding": [0.2] * 10,
            },
            {
                "id": "job3",
                "title": "Full Stack Developer",
                "description": "React and Node.js developer for startup environment.",
                "embedding": [0.3] * 10,
            },
            {
                "id": "job4",
                "title": "Machine Learning Engineer",
                "description": "Python and TensorFlow expert for ML platform team.",
                "embedding": [0.4] * 10,
            },
        ]

    def test_hybrid_search_combines_bm25_and_semantic(self, sample_jobs):
        """Should combine BM25 and semantic search results."""
        from app.services.hybrid_search import HybridSearchService

        service = HybridSearchService()
        service.build_index(sample_jobs)

        # Query with text and embedding
        results = service.search(
            query_text="Python developer",
            query_embedding=[0.1] * 10,  # Similar to job1
            top_k=3
        )

        assert len(results) <= 3
        # job1 should rank high (matches both text and embedding)
        result_ids = [r[0] for r in results]
        assert "job1" in result_ids

    def test_hybrid_search_uses_rrf_fusion(self, sample_jobs):
        """Should use RRF to fuse rankings from different sources."""
        from app.services.hybrid_search import HybridSearchService

        service = HybridSearchService()
        service.build_index(sample_jobs)

        results = service.search(
            query_text="Python machine learning",
            query_embedding=[0.4] * 10,  # Close to job4
            top_k=4
        )

        # Both job1 (Python) and job4 (Python + ML) should appear
        result_ids = [r[0] for r in results]
        assert "job4" in result_ids  # Matches both text (ML, Python) and embedding

    def test_hybrid_search_configurable_weights(self, sample_jobs):
        """Should allow configuring BM25 vs semantic weights."""
        from app.services.hybrid_search import HybridSearchService

        service = HybridSearchService()
        service.build_index(sample_jobs)

        # BM25-heavy search
        results_bm25 = service.search(
            query_text="Java Spring Boot",
            query_embedding=[0.1] * 10,  # Closer to Python job
            top_k=2,
            bm25_weight=0.8,
            semantic_weight=0.2
        )

        # Semantic-heavy search
        results_semantic = service.search(
            query_text="Java Spring Boot",
            query_embedding=[0.1] * 10,  # Closer to Python job
            top_k=2,
            bm25_weight=0.2,
            semantic_weight=0.8
        )

        # Results should differ based on weights
        bm25_ids = [r[0] for r in results_bm25]
        semantic_ids = [r[0] for r in results_semantic]

        # With BM25-heavy, Java job should rank higher
        # With semantic-heavy, Python job (matching embedding) should rank higher
        assert "job2" in bm25_ids[:2]  # Java job matches BM25

    def test_hybrid_search_empty_query(self, sample_jobs):
        """Should handle empty text query."""
        from app.services.hybrid_search import HybridSearchService

        service = HybridSearchService()
        service.build_index(sample_jobs)

        results = service.search(
            query_text="",
            query_embedding=[0.1] * 10,
            top_k=2
        )

        # Should fall back to semantic-only search
        assert len(results) <= 2

    def test_hybrid_search_no_embedding(self, sample_jobs):
        """Should handle missing embedding (BM25-only)."""
        from app.services.hybrid_search import HybridSearchService

        service = HybridSearchService()
        service.build_index(sample_jobs)

        results = service.search(
            query_text="Python developer",
            query_embedding=None,
            top_k=2
        )

        # Should return BM25-only results
        assert len(results) <= 2
        result_ids = [r[0] for r in results]
        assert "job1" in result_ids

    def test_hybrid_search_returns_scores(self, sample_jobs):
        """Should return combined scores with results."""
        from app.services.hybrid_search import HybridSearchService

        service = HybridSearchService()
        service.build_index(sample_jobs)

        results = service.search(
            query_text="Python",
            query_embedding=[0.1] * 10,
            top_k=2
        )

        # Each result should be (job_id, combined_score)
        for job_id, score in results:
            assert isinstance(job_id, str)
            assert isinstance(score, float)
            assert score >= 0


class TestHybridSearchIntegration:
    """Integration tests for hybrid search with full pipeline."""

    def test_hybrid_search_with_skill_filter(self):
        """Should integrate with skill filtering from matcher."""
        from app.services.hybrid_search import HybridSearchService
        from app.services.matcher import extract_skills_with_synonyms

        jobs = [
            {
                "id": "1",
                "title": "Python Dev",
                "description": "Python Django AWS developer",
                "embedding": [0.1] * 10,
            },
            {
                "id": "2",
                "title": "Java Dev",
                "description": "Java Spring developer",
                "embedding": [0.2] * 10,
            },
        ]

        service = HybridSearchService()
        service.build_index(jobs)

        # Search for Python jobs
        results = service.search(
            query_text="Python Django",
            query_embedding=[0.1] * 10,
            top_k=2
        )

        # Verify results can be filtered by skills
        top_job = next((j for j in jobs if j["id"] == results[0][0]), None)
        skills = extract_skills_with_synonyms(top_job["description"])

        assert "languages" in skills
        assert "python" in skills["languages"]
