"""
Background Tasks for Job Processing

Celery tasks for:
- Processing newly fetched jobs (embedding + scoring)
- Recalculating match scores when profile changes
- Generating embeddings (rate-limited)
- Updating vector database index

All tasks support:
- Automatic retries on failure
- Rate limiting for API calls
- Prometheus metrics
- Graceful error handling
"""

import logging
import time
from typing import List, Optional, Tuple

from celery import shared_task
from prometheus_client import Histogram, Counter

from app.celery import celery_app

logger = logging.getLogger(__name__)

# ==================== Prometheus Metrics ====================

TASK_DURATION = Histogram(
    "celery_task_duration_seconds",
    "Time spent executing Celery tasks",
    ["task_name"]
)

TASK_FAILURES = Counter(
    "celery_task_failures_total",
    "Number of Celery task failures",
    ["task_name"]
)

EMBEDDINGS_GENERATED = Counter(
    "embeddings_generated_total",
    "Number of embeddings generated"
)

SCORES_CALCULATED = Counter(
    "match_scores_calculated_total",
    "Number of match scores calculated"
)


# ==================== Helper Functions ====================
# These are placeholders that will be replaced with actual implementations

def get_job(job_id: str):
    """Get job by ID from database."""
    from app.database import get_db_session
    from app.models import Job

    # Synchronous database access for Celery tasks
    # In production, use async session with appropriate handling
    session = get_db_session()
    try:
        return session.query(Job).filter(Job.id == job_id).first()
    finally:
        session.close()


def get_embedding(text: str) -> List[float]:
    """Generate embedding for text (synchronous wrapper)."""
    import asyncio
    from app.services.embeddings import get_embedding as async_get_embedding

    # Run async function synchronously for Celery
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(async_get_embedding(text))
    finally:
        loop.close()


def get_all_profiles():
    """Get all user profiles."""
    from app.database import get_db_session
    from app.models import Profile

    session = get_db_session()
    try:
        return session.query(Profile).all()
    finally:
        session.close()


def get_profile(profile_id: str):
    """Get profile by ID."""
    from app.database import get_db_session
    from app.models import Profile

    session = get_db_session()
    try:
        return session.query(Profile).filter(Profile.id == profile_id).first()
    finally:
        session.close()


def get_all_active_jobs():
    """Get all jobs that are not archived."""
    from app.database import get_db_session
    from app.models import Job

    session = get_db_session()
    try:
        return session.query(Job).filter(Job.status != "archived").all()
    finally:
        session.close()


def calculate_match_score(job, profile) -> Tuple[float, List[str]]:
    """Calculate match score between job and profile."""
    # Placeholder - actual implementation would use matcher service
    from app.services.matcher import JobMatcher

    matcher = JobMatcher()
    return matcher.calculate_score(job, profile)


def update_job_score(
    job_id: str,
    profile_id: str,
    score: float,
    reasons: List[str]
):
    """Update job's match score in database."""
    from app.database import get_db_session
    from app.models import Job

    session = get_db_session()
    try:
        job = session.query(Job).filter(Job.id == job_id).first()
        if job:
            job.match_score = score
            job.match_reasons = reasons
            session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# Global vector_db reference (lazy loaded)
vector_db = None


def get_vector_db():
    """Get or initialize vector database client."""
    global vector_db
    if vector_db is None:
        from app.services.vector_db import get_vector_db as init_vector_db
        vector_db = init_vector_db()
    return vector_db


# ==================== Celery Tasks ====================

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def process_new_jobs(self, job_ids: List[str]) -> dict:
    """
    Background task to process newly fetched jobs.

    For each job:
    1. Generate embedding from description
    2. Calculate match scores for all profiles
    3. Update vector index

    Args:
        job_ids: List of job UUIDs to process

    Returns:
        Dict with processing statistics
    """
    start_time = time.time()
    stats = {
        "processed": 0,
        "failed": 0,
        "embeddings_generated": 0,
        "scores_calculated": 0,
    }

    try:
        for job_id in job_ids:
            try:
                # Get job from database
                job = get_job(job_id)
                if not job:
                    logger.warning(f"Job not found: {job_id}")
                    stats["failed"] += 1
                    continue

                # Generate embedding
                embedding = get_embedding(job.description)
                stats["embeddings_generated"] += 1
                EMBEDDINGS_GENERATED.inc()

                # Calculate match scores for all profiles
                profiles = get_all_profiles()
                for profile in profiles:
                    score, reasons = calculate_match_score(job, profile)
                    update_job_score(job_id, profile.id, score, reasons)
                    stats["scores_calculated"] += 1
                    SCORES_CALCULATED.inc()

                # Update vector index
                update_vector_index.delay(job_id, embedding)

                stats["processed"] += 1
                logger.info(f"Processed job {job_id}: score={score:.1f}")

            except Exception as e:
                logger.error(f"Error processing job {job_id}: {e}")
                stats["failed"] += 1
                TASK_FAILURES.labels(task_name="process_new_jobs").inc()

    except Exception as exc:
        TASK_FAILURES.labels(task_name="process_new_jobs").inc()
        logger.error(f"Task failed: {exc}")
        raise self.retry(exc=exc, countdown=60)

    finally:
        duration = time.time() - start_time
        TASK_DURATION.labels(task_name="process_new_jobs").observe(duration)

    return stats


@celery_app.task(bind=True, max_retries=3)
def recalculate_all_scores(self, profile_id: str) -> dict:
    """
    Recalculate match scores when profile changes.

    Called when user updates CV or preferences to ensure
    all job scores reflect the new profile.

    Args:
        profile_id: Profile UUID that was updated

    Returns:
        Dict with recalculation statistics
    """
    start_time = time.time()
    stats = {
        "jobs_processed": 0,
        "failed": 0,
    }

    try:
        profile = get_profile(profile_id)
        if not profile:
            logger.error(f"Profile not found: {profile_id}")
            return {"error": "Profile not found"}

        jobs = get_all_active_jobs()
        logger.info(f"Recalculating scores for {len(jobs)} jobs")

        for job in jobs:
            try:
                score, reasons = calculate_match_score(job, profile)
                update_job_score(job.id, profile_id, score, reasons)
                stats["jobs_processed"] += 1
                SCORES_CALCULATED.inc()

            except Exception as e:
                logger.error(f"Error scoring job {job.id}: {e}")
                stats["failed"] += 1

    except Exception as exc:
        TASK_FAILURES.labels(task_name="recalculate_all_scores").inc()
        raise self.retry(exc=exc, countdown=60)

    finally:
        duration = time.time() - start_time
        TASK_DURATION.labels(task_name="recalculate_all_scores").observe(duration)

    return stats


@celery_app.task(bind=True, rate_limit="100/m", max_retries=3)
def generate_embedding_task(self, text: str) -> List[float]:
    """
    Generate embedding for text with rate limiting.

    Rate limited to 100 calls per minute to stay within
    OpenAI API limits and control costs.

    Args:
        text: Text to embed

    Returns:
        1536-dimensional embedding vector
    """
    start_time = time.time()

    try:
        embedding = get_embedding(text)
        EMBEDDINGS_GENERATED.inc()
        return embedding

    except Exception as exc:
        TASK_FAILURES.labels(task_name="generate_embedding_task").inc()
        raise self.retry(exc=exc, countdown=30)

    finally:
        duration = time.time() - start_time
        TASK_DURATION.labels(task_name="generate_embedding_task").observe(duration)


@celery_app.task(bind=True, max_retries=3)
def update_vector_index(self, job_id: str, embedding: List[float]) -> bool:
    """
    Update vector database with new embedding.

    Args:
        job_id: Job UUID
        embedding: 1536-dim embedding vector

    Returns:
        True if update successful
    """
    start_time = time.time()

    try:
        db = get_vector_db()
        db.upsert(
            ids=[job_id],
            embeddings=[embedding],
        )
        logger.info(f"Updated vector index for job {job_id}")
        return True

    except Exception as exc:
        TASK_FAILURES.labels(task_name="update_vector_index").inc()
        logger.error(f"Failed to update vector index: {exc}")
        raise self.retry(exc=exc, countdown=30)

    finally:
        duration = time.time() - start_time
        TASK_DURATION.labels(task_name="update_vector_index").observe(duration)
