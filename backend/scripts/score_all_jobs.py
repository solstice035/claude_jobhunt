#!/usr/bin/env python3
"""
Score all existing jobs against the master CV.
Designed for one-time batch scoring after data migration.
"""
import asyncio
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from sqlalchemy import select
from sqlalchemy.orm import selectinload
from app.database import async_session
from app.models import Job, Profile
from app.services.embeddings import get_embedding, get_embeddings_batch
from app.services.matcher import calculate_match_score


async def score_all_jobs():
    """Score all jobs that currently have match_score = 0"""

    async with async_session() as db:
        # Get the default profile
        result = await db.execute(select(Profile).where(Profile.id == "default"))
        profile = result.scalar_one_or_none()

        if not profile or not profile.cv_text:
            print("❌ No profile found or CV is empty")
            return

        print(f"📄 Profile loaded: {len(profile.cv_text)} characters")
        print(f"   CV embedding found: {profile.cv_embedding is not None}")

        # Get or generate CV embedding
        if not profile.cv_embedding:
            print("🔄 Generating CV embedding...")
            profile.cv_embedding = await get_embedding(profile.cv_text)
            await db.commit()

        cv_embedding = profile.cv_embedding
        print("✅ CV embedding ready")

        # Get all jobs with score = 0 (excluding duplicates)        
        jobs_query = (
            select(Job)
            .where(Job.status == "new")
            .where(Job.is_duplicate_of.is_(None))
            .order_by(Job.created_at.desc())
        )
        print(f"📝 SQL Query for unscored jobs (all 'new' status):\n{jobs_query}")
        
        result = await db.execute(jobs_query)
        jobs = result.scalars().all()

        if not jobs:
            print("ℹ️  No jobs to score")
            return

        print(f"📊 Found {len(jobs)} jobs to score")

        # Batch generate embeddings for jobs that don't have them
        jobs_needing_embeddings = [j for j in jobs if not j.embedding]
        if jobs_needing_embeddings:
            print(f"🔄 Generating embeddings for {len(jobs_needing_embeddings)} jobs...")
            descriptions = [j.description for j in jobs_needing_embeddings]
            embeddings = await get_embeddings_batch(descriptions)

            for job, embedding in zip(jobs_needing_embeddings, embeddings):
                job.embedding = embedding

            await db.commit()
            print("✅ Embeddings generated")

        # Score all jobs
        print("🔄 Calculating match scores...")
        scored_count = 0

        for i, job in enumerate(jobs, 1):
            if not job.embedding:
                print(f"⚠️  Job {job.id} has no embedding, skipping")
                continue

            score, reasons = calculate_match_score(
                job_embedding=job.embedding,
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
            scored_count += 1

            if i % 100 == 0:
                print(f"   Progress: {i}/{len(jobs)} scored")
                await db.commit()

        # Final commit
        await db.commit()

        print(f"✅ Scored {scored_count} jobs successfully")

        # Show distribution
        result = await db.execute(
            select(Job.match_score)
            .where(Job.match_score > 0)
            .where(Job.is_duplicate_of.is_(None))
        )
        scores = [row[0] for row in result.fetchall()]

        if scores:
            print(f"\n📈 Score distribution:")
            print(f"   Min: {min(scores):.1f}")
            print(f"   Max: {max(scores):.1f}")
            print(f"   Average: {sum(scores)/len(scores):.1f}")
            print(f"   Top 70+: {len([s for s in scores if s >= 70])}")
            print(f"   Good 60-69: {len([s for s in scores if 60 <= s < 70])}")
            print(f"   Fair 50-59: {len([s for s in scores if 50 <= s < 60])}")


if __name__ == "__main__":
    asyncio.run(score_all_jobs())
