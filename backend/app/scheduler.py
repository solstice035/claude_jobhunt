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
from datetime import datetime, timezone
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select
from app.database import async_session
from app.models import Job, Profile
from app.services.scrapers import AdzunaScraper
from app.services.scrapers.adzuna import generate_url_hash
from app.services.embeddings import get_embedding, get_embeddings_batch
from app.services.matcher import calculate_match_score
from app.config import get_settings

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
    print(f"[{datetime.now(timezone.utc).isoformat()}] Starting job fetch...")

    scraper = AdzunaScraper()
    all_jobs = []

    # Fetch jobs for each search query
    for query in DEFAULT_SEARCH_QUERIES:
        try:
            jobs = await scraper.fetch_jobs(query, location="uk")
            all_jobs.extend(jobs)
            print(f"  Fetched {len(jobs)} jobs for query: {query}")
        except Exception as e:
            print(f"  Error fetching jobs for {query}: {e}")

    if not all_jobs:
        print("  No jobs fetched")
        return

    async with async_session() as db:
        # Get profile for matching
        result = await db.execute(select(Profile).where(Profile.id == "default"))
        profile = result.scalar_one_or_none()

        if not profile or not profile.cv_text:
            print("  No profile/CV configured, skipping matching")
            cv_embedding = None
        else:
            # Get or generate CV embedding
            if not profile.cv_embedding:
                print("  Generating CV embedding...")
                profile.cv_embedding = await get_embedding(profile.cv_text)
                await db.commit()
            cv_embedding = profile.cv_embedding

        # Generate all URL hashes upfront
        job_hashes = {generate_url_hash(job.url): job for job in all_jobs}

        # Fetch all existing hashes in a single query (fixes N+1)
        existing_result = await db.execute(
            select(Job.url_hash).where(Job.url_hash.in_(job_hashes.keys()))
        )
        existing_hashes = set(row[0] for row in existing_result.fetchall())

        # Deduplicate and insert new jobs
        new_jobs_count = 0
        jobs_to_embed = []
        job_objects = []

        for url_hash, job_data in job_hashes.items():
            # Skip if job already exists (O(1) set lookup)
            if url_hash in existing_hashes:
                continue

            job = Job(
                title=job_data.title,
                company=job_data.company,
                location=job_data.location,
                salary_min=job_data.salary_min,
                salary_max=job_data.salary_max,
                description=job_data.description,
                url=job_data.url,
                url_hash=url_hash,
                source=job_data.source,
                posted_at=job_data.posted_at,
                status="new",
            )

            db.add(job)
            jobs_to_embed.append(job_data.description)
            job_objects.append(job)
            new_jobs_count += 1

        if job_objects:
            await db.commit()

            # Generate embeddings for new jobs
            print(f"  Generating embeddings for {len(job_objects)} new jobs...")
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
                        cv_embedding=cv_embedding,
                        cv_text=profile.cv_text,
                        target_roles=profile.target_roles or [],
                        preferred_locations=profile.locations or [],
                        score_weights=profile.score_weights or {},
                    )
                    job.match_score = score
                    job.match_reasons = reasons

            await db.commit()

        print(f"  Added {new_jobs_count} new jobs")


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
    )
    scheduler.start()
    print(f"Scheduler started: fetching jobs every {settings.scrape_interval_hours} hours")


def stop_scheduler():
    """Stop the background scheduler"""
    scheduler.shutdown()
