#!/usr/bin/env python3
"""
Migration: Add content_hash and is_duplicate_of columns to jobs table.

This migration:
1. Adds content_hash column (nullable initially)
2. Adds is_duplicate_of column (foreign key to jobs.id)
3. Generates content hashes for all existing jobs
4. Identifies duplicates (same content_hash) and marks newer ones as duplicates
5. Creates index on content_hash

Run from backend directory:
    python -m scripts.migrate_add_content_hash
"""

import hashlib
import sqlite3
from pathlib import Path


def generate_content_hash(title: str, description: str) -> str:
    """Generate normalized content hash from title and description."""
    norm_title = " ".join(title.lower().split())
    norm_desc = " ".join(description.lower().split())
    content = f"{norm_title}|{norm_desc}"
    return hashlib.sha256(content.encode()).hexdigest()


def migrate(db_path: str = "data/jobs.db"):
    """Run the migration."""
    db_path = Path(db_path)
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Check if columns already exist
        cursor.execute("PRAGMA table_info(jobs)")
        columns = {row[1] for row in cursor.fetchall()}

        # Step 1: Add content_hash column if not exists
        if "content_hash" not in columns:
            print("Adding content_hash column...")
            cursor.execute("ALTER TABLE jobs ADD COLUMN content_hash VARCHAR(64)")
            conn.commit()
            print("  Done.")
        else:
            print("content_hash column already exists.")

        # Step 2: Add is_duplicate_of column if not exists
        if "is_duplicate_of" not in columns:
            print("Adding is_duplicate_of column...")
            cursor.execute("ALTER TABLE jobs ADD COLUMN is_duplicate_of VARCHAR(36)")
            conn.commit()
            print("  Done.")
        else:
            print("is_duplicate_of column already exists.")

        # Step 3: Generate content hashes for existing jobs
        print("Generating content hashes for existing jobs...")
        cursor.execute("SELECT id, title, description FROM jobs WHERE content_hash IS NULL")
        jobs = cursor.fetchall()
        print(f"  Found {len(jobs)} jobs without content hash.")

        for job_id, title, description in jobs:
            content_hash = generate_content_hash(title, description)
            cursor.execute(
                "UPDATE jobs SET content_hash = ? WHERE id = ?",
                (content_hash, job_id)
            )

        conn.commit()
        print(f"  Updated {len(jobs)} jobs with content hashes.")

        # Step 4: Identify and mark duplicates
        print("Identifying duplicate jobs...")
        cursor.execute("""
            SELECT content_hash, COUNT(*) as cnt
            FROM jobs
            WHERE is_duplicate_of IS NULL
            GROUP BY content_hash
            HAVING cnt > 1
        """)
        duplicate_hashes = cursor.fetchall()
        print(f"  Found {len(duplicate_hashes)} content hashes with duplicates.")

        total_marked = 0
        for content_hash, count in duplicate_hashes:
            # Get all jobs with this content hash, ordered by created_at (oldest first)
            cursor.execute("""
                SELECT id FROM jobs
                WHERE content_hash = ? AND is_duplicate_of IS NULL
                ORDER BY created_at ASC
            """, (content_hash,))
            job_ids = [row[0] for row in cursor.fetchall()]

            # Keep the first (oldest) as original, mark others as duplicates
            original_id = job_ids[0]
            duplicates = job_ids[1:]

            for dup_id in duplicates:
                cursor.execute(
                    "UPDATE jobs SET is_duplicate_of = ? WHERE id = ?",
                    (original_id, dup_id)
                )
                total_marked += 1

        conn.commit()
        print(f"  Marked {total_marked} jobs as duplicates.")

        # Step 5: Create index on content_hash if not exists
        print("Creating index on content_hash...")
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS ix_jobs_content_hash ON jobs(content_hash)
        """)
        conn.commit()
        print("  Done.")

        # Summary
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE is_duplicate_of IS NULL")
        visible_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM jobs WHERE is_duplicate_of IS NOT NULL")
        hidden_count = cursor.fetchone()[0]

        print("\n=== Migration Complete ===")
        print(f"Visible jobs (original): {visible_count}")
        print(f"Hidden jobs (duplicates): {hidden_count}")

    except Exception as e:
        conn.rollback()
        print(f"Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    migrate()
