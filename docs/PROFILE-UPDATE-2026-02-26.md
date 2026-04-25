# Profile Update & Re-matching — 2026-02-26

## Summary

Updated the job hunt system's career profile with comprehensive FY25-FY26 achievements from Obsidian career development materials and reset all job match scores for re-matching.

## Changes Made

### 1. Master CV Updated (`agent-data/master-cv.md`)

**Before:**
- 6,784 characters
- Basic 2018-era CV with generic project descriptions
- No peer feedback or quantified achievements
- Last updated: 2026-02-18

**After:**
- 13,284 characters (96% increase)
- Comprehensive career profile with FY25-FY26 details
- Quantified achievements and C-suite client validation
- Peer feedback quotes integrated
- Recent projects with specific impact metrics
- Updated: 2026-02-26

**Key additions:**
- **Recent projects (FY25-FY26):**
  - JPMC BCBS 239 Data Lineage (workstream lead, C-suite visibility)
  - Controls Solution (Tech Lead, global campaign)
  - AI Hackathon win (Strategic Analysis Copilot)
  - Morgan Stanley Lean IX (starting Jan 2026)
  - Outreach Optimisation (ETC compliance 40% → 80%)
  
- **Quantified achievements:**
  - $620k+ revenue delivery FY25
  - ~$1.4mm pursuit pipeline FY26
  - 500,000+ lines of regulatory mapping
  - 60+ stakeholders in flagship workshop
  - Teams across 15+ countries

- **Peer validation:**
  - Direct quotes from Anna Dunn (JPMC CEO), Katharina Keusch (EY Director)
  - FY25 "Differentiating" rating highlighted
  - Specific strengths documented with evidence

### 2. Target Profile Updated (`agent-data/knowledge/target-profile.md`)

**Before:**
- "MOCK profile built from known context"
- Basic role targets without career context
- No development focus or self-awareness

**After:**
- Real career context from FY25-FY26 reviews
- Refined target roles (Director/Associate Director with AI/ML focus)
- Key selling points strengthened with evidence
- Career narrative updated with FY25 Differentiating performance
- Development focus added (self-awareness from FY26 self-assessment)
- Ideal next role section (clear differentiation from current position)

**Key updates:**
- Current position context (12+ years EY, FY25 Differentiating rating)
- Target roles refined (Director-level with AI/ML/regulatory tech focus)
- 12 key selling points with specific examples
- Career narrative: strategic thinking + technical depth (rare combination)
- Development focus: time management, delegation, structured approach
- Deal breakers: toxic culture, no strategic influence, excessive travel

### 3. Match Scores Reset

**Action:** All 4,899 job match scores reset to 0

**Reason:** Updated profile with significantly more comprehensive career details requires fresh matching to surface best opportunities.

**Before reset:**
- Total jobs: 4,899
- Scored: 527 (10.7%)
- Score range: 0-84
- Average: 4.97

**After reset:**
- Total jobs: 4,899
- Scored: 0
- All jobs queued for re-scoring

## How Re-matching Works

The job hunt system uses **on-demand scoring** (not batch processing):

1. **Automatic scoring** when jobs are viewed in the frontend
2. **No server load** — efficient, as-needed calculation
3. **Progressive improvement** — scores populate as you browse
4. **High-scoring roles surface naturally** as the system learns your preferences

### Match Score Composition

The matcher calculates a 0-100 score based on:

1. **Semantic Similarity (30%)** — OpenAI embedding cosine similarity
2. **Skills Match (30%)** — Keyword extraction and overlap analysis
3. **Seniority Match (25%)** — Job level alignment (junior → executive)
4. **Location Match (15%)** — Geographic preference matching

### Optional: Force Immediate Re-scoring

If Celery worker is running, you can trigger batch re-scoring:

```bash
# Start Celery worker (if not running)
cd ~/projects/claude_jobhunt
docker-compose up -d celery

# Trigger re-scoring via API
curl -X POST http://localhost:8000/api/jobs/recalculate-scores
```

**Recommendation:** Let natural on-demand scoring happen. It's more efficient and works perfectly well.

## Scripts Created

### `scripts/rematch_all_jobs.sh`

Re-matches all jobs with updated profile.

**Usage:**
```bash
# Dry run (show what would be done)
./scripts/rematch_all_jobs.sh --dry-run

# Execute (update profile + reset scores)
./scripts/rematch_all_jobs.sh
```

**What it does:**
1. Reads master CV from `agent-data/master-cv.md`
2. Updates profile in database
3. Resets all match scores to 0
4. Clears embeddings (forces regeneration)

### `scripts/trigger_rematching.sh`

Shows current match status and re-scoring options.

**Usage:**
```bash
./scripts/trigger_rematching.sh
```

**What it shows:**
- Current match status (total jobs, scored, unscored)
- Celery worker status
- Options for re-scoring (on-demand vs immediate)
- Recommendation

## Verification

### Profile Updated
```bash
psql -U jeeves -d jeeves -c "
  SELECT LENGTH(cv_text) as cv_length, updated_at 
  FROM jobhunt_profiles 
  WHERE id = 'default';
"
```

Expected: `cv_length = 13284`, `updated_at = 2026-02-26 18:38:28+`

### Scores Reset
```bash
psql -U jeeves -d jeeves -c "
  SELECT COUNT(*) as total, 
         COUNT(CASE WHEN match_score > 0 THEN 1 END) as scored 
  FROM jobhunt_jobs 
  WHERE status != 'duplicate';
"
```

Expected: `total = 4899`, `scored = 0`

## Expected Impact

With the comprehensive career profile update, the matcher will now:

1. **Better identify Director-level roles** — profile shows readiness for promotion
2. **Surface AI/ML opportunities** — hackathon win, NLP POC, controls automation highlighted
3. **Match regulatory/compliance tech roles** — JPMC BCBS 239, Controls Solution experience
4. **Recognize senior stakeholder management** — C-suite client validation included
5. **Value commercial acumen** — revenue delivery + pursuit pipeline demonstrated

## Next Steps

1. **Browse jobs in frontend** — scores will populate automatically
2. **Monitor high-scoring roles** — watch for Director-level, AI/ML, regulatory tech positions
3. **Review match reasons** — understand why roles score high
4. **Adjust preferences if needed** — refine target roles based on what surfaces

## Related Files

- `agent-data/master-cv.md` — Master career profile (13,284 chars)
- `agent-data/knowledge/target-profile.md` — Target role profile (11,092 chars)
- `scripts/rematch_all_jobs.sh` — Profile update + score reset script
- `scripts/trigger_rematching.sh` — Re-scoring status + options script

## Obsidian Sources

Profile updated from:
- `2-areas/careerDevelopment/FY26 mid year one page notes.md`
- `2-areas/careerDevelopment/Lead Feedback - FY26.md`
- `2-areas/careerDevelopment/LEAD dashboard - FY25.md`
- `2-areas/careerDevelopment/2025-12-09 - FY26 Mid Year.md`

All materials from Nick's personal Obsidian vault: `/Users/nick/Obsidian/claude/`
