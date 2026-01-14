# Duplicate Jobs Deduplication Design

**Date:** 2026-01-14
**Status:** Approved
**Branch:** `bugfix/duplicate-jobs`

## Problem

Jobs are being duplicated in two ways:

1. **True duplicates:** Same job with different Adzuna redirect URLs (tracking params vary)
2. **Near duplicates:** Same job posted by different agencies (Hays vs Hays Technology) with different locations

Current deduplication only uses `url_hash`, which fails when URLs differ.

## Solution

Content-based deduplication using `title + description` hash.

### Hash Strategy

```
content_hash = SHA-256(normalize(title) + normalize(description))
```

**Normalization:**
- Lowercase
- Strip leading/trailing whitespace
- Collapse multiple spaces to single space

**Why title + description:**
- Same job = same description, even across agencies
- Salary/location can vary between agency listings
- Description is the most reliable identifier

### Database Changes

Add to `jobs` table:

| Column | Type | Constraints |
|--------|------|-------------|
| `content_hash` | VARCHAR(64) | NOT NULL, indexed |
| `is_duplicate_of` | VARCHAR(36) | FK to jobs.id, nullable |

Note: `content_hash` is NOT unique - we store duplicates but mark them.

### Duplicate Handling Flow

**On job fetch:**
1. Generate `content_hash` for incoming job
2. Query for existing job with same hash
3. If exists: set `is_duplicate_of` = original job's ID
4. Insert job (preserves data, marked as duplicate)

**On job listing:**
- Default: `WHERE is_duplicate_of IS NULL`
- Hides duplicates from UI
- Original (oldest) job remains visible

### Files to Modify

1. **`backend/app/models/job.py`**
   - Add `content_hash` column
   - Add `is_duplicate_of` column with FK

2. **`backend/app/scheduler.py`**
   - Add `generate_content_hash()` function
   - Update `fetch_and_process_jobs()` to detect duplicates

3. **`backend/app/api/jobs.py`**
   - Update list endpoint to filter `is_duplicate_of IS NULL`

### Migration Plan

1. Add columns (nullable)
2. Backfill `content_hash` for existing jobs
3. Identify duplicates (group by hash, mark newer as duplicates)
4. Alter column to NOT NULL

### Cleanup Script

```python
# Pseudocode
for content_hash in (SELECT content_hash, COUNT(*) FROM jobs GROUP BY content_hash HAVING COUNT(*) > 1):
    jobs = SELECT * FROM jobs WHERE content_hash = ? ORDER BY created_at ASC
    original = jobs[0]
    for duplicate in jobs[1:]:
        UPDATE jobs SET is_duplicate_of = original.id WHERE id = duplicate.id
```

### Verification

1. Check count: `SELECT COUNT(*) FROM jobs WHERE is_duplicate_of IS NULL`
2. UI shows no duplicate job cards
3. Trigger job refresh, verify new duplicates caught
4. Check a known duplicate is properly hidden

## Future Enhancements (Not in scope)

- "Show duplicates" toggle in UI
- "Also posted by" list on job detail
- Fuzzy matching for near-near duplicates
