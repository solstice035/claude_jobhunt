"""
Celery Task Modules

Background tasks for job processing:
- jobs.py: Job embedding generation, scoring, and indexing
"""

from app.tasks.jobs import (
    process_new_jobs,
    recalculate_all_scores,
    generate_embedding_task,
    update_vector_index,
)

__all__ = [
    "process_new_jobs",
    "recalculate_all_scores",
    "generate_embedding_task",
    "update_vector_index",
]
