"""
Background Job Scheduler - Periodic Job Fetching and Processing

This module manages automated job scraping using APScheduler.

Processing Pipeline:
    1. Fetch jobs from Adzuna for each search query
    2. Deduplicate against existing database entries (batched query)
    3. Generate OpenAI embeddings for new job descriptions
    4. Calculate match scores against user's CV profile
    5. Store jobs with scores in database

Default Schedule: Every 6 hours (configurable via SCRAPE_INTERVAL_HOURS)

Performance Optimizations:
    - Batch URL hash lookup (single IN query vs N+1)
    - Batch embedding generation (100 texts per API call)
    - Set-based deduplication for O(1) lookups
"""

import asyncio
import logging
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select
from app.database import async_session
from app.models import Job, Profile
from app.services.scrapers import AdzunaScraper
from app.services.scrapers.adzuna import generate_url_hash, generate_content_hash
from app.services.embeddings import get_embedding, get_embeddings_batch
from app.services.matcher import calculate_match_score
from app.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()
scheduler = AsyncIOScheduler()

# Search queries to run (customizable via profile in future)
DEFAULT_SEARCH_QUERIES = [
    "technology director",
    "head of technology",
    "principal consultant",
    "CTO",
    "engineering director",
]


async def fetch_and_process_jobs():
    """
    Main scheduled task that fetches, deduplicates, and scores jobs.

    Flow:
        1. For each search query, fetch up to 150 jobs from Adzuna
        2. Generate URL hashes and batch-check for duplicates
        3. Insert new jobs and generate embeddings
        4. Calculate match scores if profile exists
        5. Commit all changes

    Side Effects:
        - Creates Job records in database
        - Updates Profile.cv_embedding if missing
        - Prints progress to stdout for monitoring
    """
    logger.info("Starting job fetch...")

    scraper = AdzunaScraper()
    all_jobs = []

    # Fetch jobs for each search query
    for query in DEFAULT_SEARCH_QUERIES:
        try:
            jobs = await scraper.fetch_jobs(query, location="uk")
            all_jobs.extend(jobs)
            logger.info("Fetched %d jobs for query: %s", len(jobs), query)
        except Exception as e:
            logger.error("Error fetching jobs for %s: %s", query, e)

    if not all_jobs:
        logger.info("No jobs fetched")
        return

    async with async_session() as db:
        # Get profile for matching
        result = await db.execute(select(Profile).where(Profile.id == "default"))
        profile = result.scalar_one_or_none()

        if not profile or not profile.cv_text:
            logger.info("No profile/CV configured, skipping matching")
            cv_embedding = None
        else:
            # Get or generate CV embedding
            if not profile.cv_embedding:
                logger.info("Generating CV embedding...")
                profile.cv_embedding = await get_embedding(profile.cv_text)
                await db.commit()
            cv_embedding = profile.cv_embedding

        # Generate all URL hashes upfront
        job_hashes = {generate_url_hash(job.url): job for job in all_jobs}

        # Fetch all existing URL hashes in a single query (fixes N+1)
        existing_result = await db.execute(
            select(Job.url_hash).where(Job.url_hash.in_(job_hashes.keys()))
        )
        existing_url_hashes = set(row[0] for row in existing_result.fetchall())

        # Generate content hashes for new jobs only
        new_job_candidates = {
            url_hash: job_data
            for url_hash, job_data in job_hashes.items()
            if url_hash not in existing_url_hashes
        }

        # Build list of (content_hash, url_hash, job_data) tuples - preserves ALL jobs
        # including duplicates within the same batch (critical for proper deduplication)
        jobs_with_hashes = []
        unique_content_hashes = set()
        for url_hash, job_data in new_job_candidates.items():
            content_hash = generate_content_hash(job_data.title, job_data.description)
            jobs_with_hashes.append((content_hash, url_hash, job_data))
            unique_content_hashes.add(content_hash)

        # Query for existing jobs with matching content hashes
        existing_content_result = await db.execute(
            select(Job.id, Job.content_hash).where(
                Job.content_hash.in_(unique_content_hashes),
                Job.is_duplicate_of.is_(None),  # Only match against original jobs
            )
        )
        existing_content_map = {row[1]: row[0] for row in existing_content_result.fetchall()}

        # Deduplicate and insert new jobs
        new_jobs_count = 0
        duplicate_jobs_count = 0
        jobs_to_embed = []
        job_objects = []

        for content_hash, url_hash, job_data in jobs_with_hashes:
            # Check if this is a content duplicate (either from DB or from this batch)
            original_job_id = existing_content_map.get(content_hash)

            job = Job(
                title=job_data.title,
                company=job_data.company,
                location=job_data.location,
                salary_min=job_data.salary_min,
                salary_max=job_data.salary_max,
                description=job_data.description,
                url=job_data.url,
                url_hash=url_hash,
                content_hash=content_hash,
                is_duplicate_of=original_job_id,  # None if new, ID if duplicate
                source=job_data.source,
                posted_at=job_data.posted_at,
                status="new",
            )

            db.add(job)

            if original_job_id:
                duplicate_jobs_count += 1
            else:
                # Only generate embeddings for non-duplicate jobs
                jobs_to_embed.append(job_data.description)
                job_objects.append(job)
                new_jobs_count += 1
                # Track this content hash for future duplicates in this batch
                existing_content_map[content_hash] = job.id

        if job_objects:
            await db.commit()

            # Generate embeddings for new jobs
            logger.info("Generating embeddings for %d new jobs...", len(job_objects))
            embeddings = await get_embeddings_batch(jobs_to_embed)

            # Calculate match scores
            for job, embedding in zip(job_objects, embeddings):
                job.embedding = embedding

                if cv_embedding and profile:
                    score, reasons = calculate_match_score(
                        job_embedding=embedding,
                        job_description=job.description,
                        job_title=job.title,
                        job_location=job.location,
                        job_salary_min=job.salary_min,
                        job_salary_max=job.salary_max,
                        cv_embedding=cv_embedding,
                        cv_text=profile.cv_text,
                        target_roles=profile.target_roles or [],
                        preferred_locations=profile.locations or [],
                        exclude_keywords=profile.exclude_keywords or [],
                        salary_min=profile.salary_min,
                        salary_target=profile.salary_target,
                        score_weights=profile.score_weights or {},
                    )
                    job.match_score = score
                    job.match_reasons = reasons

            await db.commit()

        logger.info("Added %d new jobs, %d duplicates detected", new_jobs_count, duplicate_jobs_count)


async def trigger_manual_refresh():
    """Trigger an immediate job refresh"""
    await fetch_and_process_jobs()


def start_scheduler():
    """Start the background scheduler"""
    scheduler.add_job(
        fetch_and_process_jobs,
        trigger=IntervalTrigger(hours=settings.scrape_interval_hours),
        id="fetch_jobs",
        replace_existing=True,
        max_instances=1,
        misfire_grace_time=300,
    )
    scheduler.start()
    logger.info("Scheduler started: fetching jobs every %d hours", settings.scrape_interval_hours)


def stop_scheduler():
    """Stop the background scheduler"""
    scheduler.shutdown()
