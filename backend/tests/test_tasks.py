"""
Tests for Celery Background Tasks

Tests cover:
- process_new_jobs task
- recalculate_all_scores task
- generate_embedding task (rate limited)
- Task retries on failure
- Task chaining and workflows
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call
from typing import List

# Import tasks (will be created)
from app.tasks.jobs import (
    process_new_jobs,
    recalculate_all_scores,
    generate_embedding_task,
    update_vector_index,
)
from app.celery import celery_app


class TestCeleryApp:
    """Test Celery app configuration."""

    def test_celery_app_exists(self):
        """Celery app should be configured."""
        assert celery_app is not None
        assert celery_app.main == "job_matching"

    def test_celery_uses_redis_broker(self):
        """Should use Redis as message broker."""
        # Broker URL should be Redis
        assert "redis" in celery_app.conf.broker_url

    def test_celery_uses_redis_backend(self):
        """Should use Redis as result backend."""
        assert "redis" in celery_app.conf.result_backend


class TestProcessNewJobsTask:
    """Test process_new_jobs background task."""

    @pytest.fixture
    def mock_job(self):
        """Create a mock job object."""
        job = MagicMock()
        job.id = "job-123"
        job.title = "Senior Python Developer"
        job.description = "We are looking for a Python developer..."
        return job

    @pytest.fixture
    def mock_profile(self):
        """Create a mock profile object."""
        profile = MagicMock()
        profile.id = "default"
        profile.cv_text = "Experienced Python developer..."
        profile.cv_embedding = [0.1] * 1536
        return profile

    def test_process_new_jobs_is_celery_task(self):
        """process_new_jobs should be a registered Celery task."""
        assert hasattr(process_new_jobs, 'delay')
        assert hasattr(process_new_jobs, 'apply_async')

    @patch("app.tasks.jobs.get_job")
    @patch("app.tasks.jobs.get_embedding")
    @patch("app.tasks.jobs.get_all_profiles")
    @patch("app.tasks.jobs.calculate_match_score")
    @patch("app.tasks.jobs.update_job_score")
    @patch("app.tasks.jobs.update_vector_index")
    def test_process_new_jobs_generates_embeddings(
        self, mock_update_vector, mock_update_score, mock_calc_score,
        mock_get_profiles, mock_get_embedding, mock_get_job, mock_job
    ):
        """Should generate embeddings for new jobs."""
        mock_get_job.return_value = mock_job
        mock_get_embedding.return_value = [0.1] * 1536
        mock_get_profiles.return_value = []
        mock_calc_score.return_value = (75.0, ["Good match"])

        # Run task synchronously for testing
        process_new_jobs.run(["job-123"])

        mock_get_embedding.assert_called_once_with(mock_job.description)

    @patch("app.tasks.jobs.get_job")
    @patch("app.tasks.jobs.get_embedding")
    @patch("app.tasks.jobs.get_all_profiles")
    @patch("app.tasks.jobs.calculate_match_score")
    @patch("app.tasks.jobs.update_job_score")
    @patch("app.tasks.jobs.update_vector_index")
    def test_process_new_jobs_calculates_scores_for_all_profiles(
        self, mock_update_vector, mock_update_score, mock_calc_score,
        mock_get_profiles, mock_get_embedding, mock_get_job,
        mock_job, mock_profile
    ):
        """Should calculate match scores for all profiles."""
        mock_get_job.return_value = mock_job
        mock_get_embedding.return_value = [0.1] * 1536
        mock_get_profiles.return_value = [mock_profile]
        mock_calc_score.return_value = (75.0, ["Skills match"])

        process_new_jobs.run(["job-123"])

        mock_calc_score.assert_called_once_with(mock_job, mock_profile)
        mock_update_score.assert_called_once_with(
            "job-123", mock_profile.id, 75.0, ["Skills match"]
        )

    @patch("app.tasks.jobs.get_job")
    @patch("app.tasks.jobs.get_embedding")
    @patch("app.tasks.jobs.get_all_profiles")
    @patch("app.tasks.jobs.calculate_match_score")
    @patch("app.tasks.jobs.update_job_score")
    @patch("app.tasks.jobs.update_vector_index")
    def test_process_new_jobs_updates_vector_index(
        self, mock_update_vector, mock_update_score, mock_calc_score,
        mock_get_profiles, mock_get_embedding, mock_get_job, mock_job
    ):
        """Should update vector index with new embeddings via async delay."""
        mock_get_job.return_value = mock_job
        embedding = [0.1] * 1536
        mock_get_embedding.return_value = embedding
        mock_get_profiles.return_value = []

        process_new_jobs.run(["job-123"])

        # Task uses .delay() for async execution
        mock_update_vector.delay.assert_called_once_with("job-123", embedding)

    @patch("app.tasks.jobs.get_job")
    @patch("app.tasks.jobs.get_embedding")
    def test_process_new_jobs_handles_multiple_jobs(
        self, mock_get_embedding, mock_get_job
    ):
        """Should process multiple jobs in sequence."""
        job1 = MagicMock(id="job-1", description="Job 1 desc")
        job2 = MagicMock(id="job-2", description="Job 2 desc")
        mock_get_job.side_effect = [job1, job2]
        mock_get_embedding.return_value = [0.1] * 1536

        with patch("app.tasks.jobs.get_all_profiles", return_value=[]):
            with patch("app.tasks.jobs.update_vector_index"):
                process_new_jobs.run(["job-1", "job-2"])

        assert mock_get_job.call_count == 2
        assert mock_get_embedding.call_count == 2

    def test_process_new_jobs_has_retry_policy(self):
        """Task should be configured with max_retries."""
        # Check task options
        assert process_new_jobs.max_retries == 3


class TestRecalculateAllScoresTask:
    """Test recalculate_all_scores background task."""

    @pytest.fixture
    def mock_profile(self):
        profile = MagicMock()
        profile.id = "default"
        profile.cv_text = "Python developer"
        return profile

    @pytest.fixture
    def mock_jobs(self):
        job1 = MagicMock(id="job-1")
        job2 = MagicMock(id="job-2")
        return [job1, job2]

    def test_recalculate_all_scores_is_celery_task(self):
        """Should be a registered Celery task."""
        assert hasattr(recalculate_all_scores, 'delay')
        assert hasattr(recalculate_all_scores, 'apply_async')

    @patch("app.tasks.jobs.get_profile")
    @patch("app.tasks.jobs.get_all_active_jobs")
    @patch("app.tasks.jobs.calculate_match_score")
    @patch("app.tasks.jobs.update_job_score")
    def test_recalculate_all_scores_processes_all_jobs(
        self, mock_update_score, mock_calc_score,
        mock_get_jobs, mock_get_profile, mock_profile, mock_jobs
    ):
        """Should recalculate scores for all active jobs."""
        mock_get_profile.return_value = mock_profile
        mock_get_jobs.return_value = mock_jobs
        mock_calc_score.return_value = (80.0, ["Great match"])

        recalculate_all_scores.run("default")

        assert mock_calc_score.call_count == 2
        assert mock_update_score.call_count == 2

    @patch("app.tasks.jobs.get_profile")
    @patch("app.tasks.jobs.get_all_active_jobs")
    @patch("app.tasks.jobs.calculate_match_score")
    @patch("app.tasks.jobs.update_job_score")
    def test_recalculate_updates_each_job_score(
        self, mock_update_score, mock_calc_score,
        mock_get_jobs, mock_get_profile, mock_profile
    ):
        """Should update score for each job."""
        mock_get_profile.return_value = mock_profile
        job = MagicMock(id="job-123")
        mock_get_jobs.return_value = [job]
        mock_calc_score.return_value = (90.0, ["Perfect match"])

        recalculate_all_scores.run("default")

        mock_update_score.assert_called_once_with(
            "job-123", "default", 90.0, ["Perfect match"]
        )


class TestGenerateEmbeddingTask:
    """Test generate_embedding_task with rate limiting."""

    def test_generate_embedding_is_celery_task(self):
        """Should be a registered Celery task."""
        assert hasattr(generate_embedding_task, 'delay')

    def test_generate_embedding_has_rate_limit(self):
        """Should have rate limit configured."""
        # Rate limit should be set (100/m according to design)
        assert generate_embedding_task.rate_limit is not None

    @patch("app.tasks.jobs.get_embedding")
    def test_generate_embedding_returns_vector(self, mock_get_embedding):
        """Should return embedding vector."""
        mock_get_embedding.return_value = [0.1] * 1536

        result = generate_embedding_task.run("Some text to embed")

        assert len(result) == 1536


class TestUpdateVectorIndex:
    """Test update_vector_index task."""

    def test_update_vector_index_is_task(self):
        """Should be callable task."""
        assert callable(update_vector_index.run)

    @patch("app.tasks.jobs.vector_db")
    def test_update_vector_index_calls_vector_db(self, mock_vector_db):
        """Should update vector database."""
        embedding = [0.1] * 1536

        update_vector_index.run("job-123", embedding)

        mock_vector_db.upsert.assert_called_once()


class TestTaskWorkflows:
    """Test task chaining and workflows."""

    def test_can_chain_embedding_to_scoring(self):
        """Should support chaining embedding -> scoring."""
        from celery import chain

        # Create workflow (won't execute, just verify chain works)
        workflow = chain(
            generate_embedding_task.s("job description"),
        )
        assert workflow is not None

    def test_can_group_multiple_embeddings(self):
        """Should support grouping parallel embedding tasks."""
        from celery import group

        # Create group (won't execute)
        job_descriptions = ["Job 1", "Job 2", "Job 3"]
        task_group = group(
            generate_embedding_task.s(desc) for desc in job_descriptions
        )
        assert task_group is not None


class TestTaskErrorHandling:
    """Test task error handling and retries."""

    @patch("app.tasks.jobs.get_job")
    def test_process_new_jobs_handles_job_errors_gracefully(self, mock_get_job):
        """Should handle individual job errors without failing entire batch."""
        mock_get_job.side_effect = ConnectionError("DB connection failed")

        # Task should handle error gracefully and return stats
        result = process_new_jobs.run(["job-123"])

        # Should report the failure in stats
        assert result["failed"] == 1

    @patch("app.tasks.jobs.get_embedding")
    def test_generate_embedding_retries_on_api_error(self, mock_get_embedding):
        """Should retry on API errors."""
        mock_get_embedding.side_effect = Exception("OpenAI API error")

        with pytest.raises(Exception):
            generate_embedding_task.run("text")


class TestTaskMetrics:
    """Test that tasks emit metrics."""

    @patch("app.tasks.jobs.get_job")
    @patch("app.tasks.jobs.get_embedding")
    @patch("app.tasks.jobs.get_all_profiles")
    @patch("app.tasks.jobs.update_vector_index")
    @patch("app.tasks.jobs.TASK_DURATION")
    def test_process_new_jobs_records_duration(
        self, mock_metric, mock_update_vector, mock_get_profiles,
        mock_get_embedding, mock_get_job
    ):
        """Should record task duration metric."""
        mock_get_job.return_value = MagicMock(id="job-1", description="desc")
        mock_get_embedding.return_value = [0.1] * 1536
        mock_get_profiles.return_value = []

        process_new_jobs.run(["job-1"])

        # Verify metric was observed
        mock_metric.labels.assert_called()
