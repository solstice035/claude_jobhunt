"""
Celery Application Configuration

Configures Celery for background job processing with:
- Redis as message broker and result backend
- Task autodiscovery from app.tasks module
- Retry policies for reliability
- Rate limiting for external API calls

Usage:
    # Start worker:
    celery -A app.celery worker --loglevel=info

    # Start beat scheduler (for periodic tasks):
    celery -A app.celery beat --loglevel=info

    # Enqueue a task:
    from app.tasks.jobs import process_new_jobs
    process_new_jobs.delay(["job-123", "job-456"])
"""

from celery import Celery
from app.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "job_matching",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

# Configure Celery
celery_app.conf.update(
    # Task settings
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,

    # Worker settings
    worker_prefetch_multiplier=1,  # Fair task distribution
    worker_concurrency=4,  # Number of concurrent workers

    # Result settings
    result_expires=3600,  # Results expire after 1 hour
    task_track_started=True,

    # Retry settings
    task_acks_late=True,  # Acknowledge after completion
    task_reject_on_worker_lost=True,

    # Rate limiting
    worker_disable_rate_limits=False,

    # Task routing (optional, for future scaling)
    task_routes={
        "app.tasks.jobs.generate_embedding_task": {"queue": "embeddings"},
        "app.tasks.jobs.process_new_jobs": {"queue": "processing"},
        "app.tasks.jobs.recalculate_all_scores": {"queue": "scoring"},
    },

    # Default queue
    task_default_queue="default",
)

# Autodiscover tasks
celery_app.autodiscover_tasks(["app.tasks"])
