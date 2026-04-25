#!/usr/bin/env python3
"""
Re-match all jobs with updated profile

This script:
1. Reads the master CV from agent-data/master-cv.md
2. Updates the profile in the database
3. Resets all existing match scores to 0
4. Triggers background re-matching for all jobs

Usage:
    python scripts/rematch_all_jobs.py [--dry-run]
"""

import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from sqlalchemy import text
from app.database import get_db_session
from app.models import Profile, Job


async def read_master_cv(cv_path: Path) -> str:
    """Read master CV from markdown file."""
    if not cv_path.exists():
        raise FileNotFoundError(f"Master CV not found: {cv_path}")
    
    with open(cv_path, 'r', encoding='utf-8') as f:
        return f.read()


async def update_profile(cv_text: str, dry_run: bool = False) -> None:
    """Update the default profile with new CV text."""
    async for db in get_db_session():
        # Get existing profile
        result = await db.execute(
            text("SELECT id, LENGTH(cv_text) as old_length FROM jobhunt_profiles WHERE id = 'default'")
        )
        profile_info = result.first()
        
        if not profile_info:
            print("❌ No default profile found!")
            return
        
        old_length = profile_info.old_length
        new_length = len(cv_text)
        
        print(f"📄 Profile CV:")
        print(f"   Old: {old_length:,} characters")
        print(f"   New: {new_length:,} characters")
        print(f"   Δ: {new_length - old_length:+,} characters")
        
        if dry_run:
            print("   [DRY RUN] Would update profile")
            return
        
        # Update profile
        await db.execute(
            text("""
                UPDATE jobhunt_profiles 
                SET cv_text = :cv_text,
                    cv_embedding = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = 'default'
            """),
            {"cv_text": cv_text}
        )
        await db.commit()
        print("   ✅ Profile updated")


async def reset_match_scores(dry_run: bool = False) -> dict:
    """Reset all match scores to 0 and clear match reasons."""
    async for db in get_db_session():
        # Get current stats
        result = await db.execute(
            text("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(CASE WHEN match_score > 0 THEN 1 END) as scored,
                    MIN(match_score) as min_score,
                    MAX(match_score) as max_score,
                    AVG(match_score) as avg_score
                FROM jobhunt_jobs 
                WHERE status != 'duplicate'
            """)
        )
        stats = result.first()
        
        print(f"\n🔄 Match Scores:")
        print(f"   Total jobs: {stats.total:,}")
        print(f"   Currently scored: {stats.scored:,}")
        if stats.scored > 0:
            print(f"   Score range: {stats.min_score:.1f} - {stats.max_score:.1f}")
            print(f"   Average: {stats.avg_score:.2f}")
        
        if dry_run:
            print("   [DRY RUN] Would reset all scores to 0")
            return {"total": stats.total, "reset": 0}
        
        # Reset all scores
        result = await db.execute(
            text("""
                UPDATE jobhunt_jobs 
                SET match_score = 0,
                    match_reasons = '[]'::jsonb,
                    embedding = NULL,
                    updated_at = CURRENT_TIMESTAMP
                WHERE status != 'duplicate'
            """)
        )
        await db.commit()
        
        rows_updated = result.rowcount
        print(f"   ✅ Reset {rows_updated:,} job scores to 0")
        
        return {"total": stats.total, "reset": rows_updated}


async def trigger_rematching(dry_run: bool = False) -> None:
    """
    Trigger re-matching via the background task system.
    
    Note: This requires Celery to be running. If Celery is not running,
    scores will be recalculated on-demand when jobs are viewed.
    """
    if dry_run:
        print("\n🚀 Re-matching:")
        print("   [DRY RUN] Would trigger background re-matching")
        print("   (Requires Celery worker to be running)")
        return
    
    try:
        from app.tasks.jobs import recalculate_all_scores
        
        # Queue the task
        task = recalculate_all_scores.delay("default")
        print(f"\n🚀 Re-matching:")
        print(f"   ✅ Queued background task: {task.id}")
        print(f"   Monitor progress at: http://localhost:5555")
        print(f"   Or check: celery -A app.celery inspect active")
        
    except Exception as e:
        print(f"\n⚠️  Could not queue background task: {e}")
        print("   Scores will be recalculated on-demand when jobs are viewed")


async def main():
    parser = argparse.ArgumentParser(description="Re-match all jobs with updated profile")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()
    
    print("=" * 60)
    print("Re-match All Jobs")
    print("=" * 60)
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - No changes will be made\n")
    
    # Get paths
    repo_root = Path(__file__).parent.parent
    cv_path = repo_root / "agent-data" / "master-cv.md"
    
    try:
        # Read master CV
        print(f"📖 Reading master CV from: {cv_path}")
        cv_text = await read_master_cv(cv_path)
        print(f"   ✅ Loaded {len(cv_text):,} characters\n")
        
        # Update profile
        await update_profile(cv_text, dry_run=args.dry_run)
        
        # Reset match scores
        stats = await reset_match_scores(dry_run=args.dry_run)
        
        # Trigger re-matching
        if not args.dry_run:
            await trigger_rematching(dry_run=args.dry_run)
        
        print("\n" + "=" * 60)
        if args.dry_run:
            print("✅ Dry run complete - no changes made")
        else:
            print("✅ Profile updated and re-matching triggered")
            print(f"   {stats['reset']:,} jobs queued for re-scoring")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
