# Job Hunt System — Comprehensive Test Report

**Date:** 2026-02-15 16:53 GMT  
**Tester:** Jeeves (subagent)  
**Purpose:** Consolidation & validation before Phase 4

---

## Test Matrix Summary

| # | Test | Result | Severity |
|---|------|--------|----------|
| 1.1 | POST /auth/login | ✅ PASS | — |
| 1.2 | POST /auth/logout | ✅ PASS | — |
| 1.3 | GET /auth/check | ✅ PASS | — |
| 1.4 | GET /jobs (list) | ✅ PASS | — |
| 1.5 | GET /jobs/{id} | ✅ PASS | — |
| 1.6 | GET /jobs?status=new | ✅ PASS | — |
| 1.7 | GET /jobs?created_after= | ✅ PASS | — |
| 1.8 | GET /jobs?min_score=75 | ⚠️ BROKEN | HIGH |
| 1.9 | POST /jobs/refresh | ✅ PASS | — |
| 1.10 | PATCH /jobs/{id} | ✅ PASS | — |
| 1.11 | GET /stats | ✅ PASS | — |
| 1.12 | POST /skills/extract | ✅ PASS | — |
| 1.13 | GET /skills/gaps | ⚠️ BLOCKED | HIGH |
| 1.14 | GET /skills/gaps/summary | ⚠️ BLOCKED | HIGH |
| 1.15 | GET /skills/search | ✅ PASS (empty) | — |
| 1.16 | GET /skills/recommendations | Not tested | — |
| 1.17 | POST /skills/infer | Not tested | — |
| 1.18 | GET /profile | ✅ PASS (empty) | — |
| 1.19 | PUT /profile | Not tested | — |
| 1.20 | GET /health | ✅ PASS | — |
| 1.21 | POST /api/search/hybrid | Not tested (no index) | — |
| 1.22 | GET /api/search/status | ✅ PASS | — |
| 1.23 | POST /api/agents/log | ❌ NOT IMPLEMENTED | CRITICAL |
| 1.24 | GET /api/agents/log | ❌ NOT IMPLEMENTED | CRITICAL |
| 1.25 | GET /api/agents/dashboard | ❌ NOT IMPLEMENTED | CRITICAL |
| 1.26 | GET /api/agents/follow-ups-due | ❌ NOT IMPLEMENTED | CRITICAL |
| 1.27 | POST /api/agents/follow-ups | ❌ NOT IMPLEMENTED | CRITICAL |
| 1.28 | PATCH /api/agents/follow-ups/{id} | ❌ NOT IMPLEMENTED | CRITICAL |
| 1.29 | GET /api/agents/networking-contacts | ❌ NOT IMPLEMENTED | HIGH |
| 1.30 | POST /api/agents/networking-contacts | ❌ NOT IMPLEMENTED | HIGH |
| 2.1 | Coordinator workspace | ✅ PASS | — |
| 2.2 | Market-intel workspace | ✅ PASS | — |
| 2.3 | App-tracker workspace | ✅ PASS | — |
| 2.4 | CV-tailor workspace | ✅ PASS | — |
| 2.5 | Cover-letter workspace | ✅ PASS | — |
| 3.1 | Agent spawning | ⏭ SKIP | — |
| 4.1 | E2E: jobs → skills → Slack | ✅ PASS (partial) | — |
| 5.1 | Slack #briefing | ✅ PASS | — |
| 5.2 | Slack #daily | ✅ PASS | — |
| 5.3 | Slack #general | ✅ PASS | — |
| 5.4 | Slack formatting | ✅ PASS | — |
| 6.1 | DB schema | ✅ PASS | — |
| 6.2 | Job data quality | ⚠️ ISSUES | MEDIUM |
| 6.3 | agent-data dirs | ⚠️ MISMATCH | MEDIUM |
| 6.4 | Reset script | ✅ PASS | — |
| 7.1 | Config audit | ✅ PASS | — |
| 7.2 | Model assignments | ✅ PASS | — |
| 7.3 | Subagent permissions | ✅ PASS | — |

---

## 1. API Endpoint Coverage

### What Works
- **Auth**: Login/logout/check all work correctly. Cookie-based JWT.
- **Jobs CRUD**: List, get by ID, patch status — all working. Pagination via `skip`/`limit` works. `created_after` filter works.
- **Stats**: Returns correct counts. Currently 722 jobs (increased from 717 after refresh triggered during testing).
- **Skills extract**: Works well — tested with a JD, correctly identified Python, AWS, Docker, ML.
- **Health**: Returns healthy.
- **Search status**: Returns status showing hybrid search available but BM25 index empty (0 entries).

### What's Broken

**`min_score` filter is broken (HIGH)**: `GET /jobs?min_score=75` returns ALL 722 jobs despite every job having `match_score: 0.0`. The filter is either not implemented or not applied.

**Skills gaps requires profile CV text (HIGH)**: `GET /skills/gaps` and `/skills/gaps/summary` both return 400: "Profile CV text is required for gap analysis". The profile exists but `cv_text` is empty. This blocks the entire skills gap analysis pipeline.

**Note**: The API contract specifies `GET /skills/gaps/{job_id}` but the actual implementation is `GET /skills/gaps` (no job_id path param). The JOBHUNT-API.md skill file doesn't document this endpoint at all.

### What's NOT Implemented (all 404)

These endpoints are specified in `07-API-CONTRACT.md` but DO NOT EXIST in the backend:

| Endpoint | Purpose | Impact |
|----------|---------|--------|
| `POST /api/agents/log` | Agent audit trail | Blocks agent monitoring |
| `GET /api/agents/log` | Retrieve agent activity | Blocks coordinator oversight |
| `GET /api/agents/dashboard` | Aggregated monitoring | Blocks operational dashboard |
| `GET /api/agents/follow-ups-due` | Due follow-ups | Blocks app-tracker workflow |
| `POST /api/agents/follow-ups` | Schedule follow-ups | Blocks app-tracker workflow |
| `PATCH /api/agents/follow-ups/{id}` | Complete follow-ups | Blocks app-tracker workflow |
| `GET /api/agents/networking-contacts` | Contact list | Phase 4 — expected |
| `POST /api/agents/networking-contacts` | Add contacts | Phase 4 — expected |
| `PATCH /api/agents/networking-contacts/{id}` | Update contacts | Phase 4 — expected |

**No `agent_log`, `follow_ups`, or `networking_contacts` tables exist in the DB schema.**

### Endpoints That Exist But Aren't in the Contract
- `GET /skills/recommendations`
- `POST /skills/infer`
- `GET /skills/esco/{uri}`
- `POST /api/search/hybrid`
- `POST /api/search/rebuild-index`
- `POST /api/search/rerank`
- `GET /api/search/status`
- `GET /profile` / `PUT /profile`

The skill files reference a simpler API than what actually exists. The search infrastructure is more sophisticated than documented.

---

## 2. Agent Workspace Validation

### All 5 Agents: Common Findings

**✅ What's good across the board:**
- Every workspace has SOUL.md, AGENTS.md, USER.md, IDENTITY.md
- Every workspace has skills/JOBHUNT-API.md and skills/SLACK.md
- SOUL.md files are coherent, well-written, personality-appropriate
- AGENTS.md files have clear procedures with working curl examples
- Auth procedure is consistent (cookie-based JWT, same password)
- USER.md files all contain accurate Nick background info

**⚠️ Issues found:**

| Agent | Issue | Severity |
|-------|-------|----------|
| All | JOBHUNT-API.md skill only documents basic endpoints (stats, jobs, refresh, patch) — missing skills/extract, skills/gaps, search, profile endpoints | MEDIUM |
| All | AGENTS.md references `~/projects/claude_jobhunt/agent-data/cv/master-cv.md` but actual path is `~/projects/claude_jobhunt/agent-data/master-cv.md` (no `cv/` subdir) | HIGH |
| Coordinator | AGENTS.md references agent endpoints (dashboard, follow-ups) that don't exist | HIGH |
| App-tracker | Entire follow-up workflow depends on `/api/agents/follow-ups` which doesn't exist | CRITICAL |
| Market-intel | References `POST /jobs/refresh` correctly ✅ | — |
| CV-tailor | No TOOLS.md file (others have it or don't need it) | LOW |
| Cover-letter | Has web_search in spec but skill file doesn't document it | LOW |

### Cross-Reference vs 06-AGENT-SPECIFICATION-CARDS.md

| Spec Attribute | coordinator | market-intel | app-tracker | cv-tailor | cover-letter |
|---------------|-------------|--------------|-------------|-----------|--------------|
| Model match | ✅ Sonnet | ✅ Sonnet | ✅ Sonnet | ✅ Opus | ✅ Opus |
| Tools match | Partial* | ✅ | Partial* | ✅ | ✅ |
| Triggers defined | ❌ No crons | ❌ No crons | ❌ No crons | N/A (on-demand) | N/A (on-demand) |
| Role coherent | ✅ | ✅ | ✅ | ✅ | ✅ |

*Coordinator spec lists `sessions_spawn`, `sessions_send`, `sessions_list`, `cron`, `lobster` — none of these are standard OpenClaw tools. The actual mechanism is subagent spawning via `allowAgents`.

*App-tracker spec lists `message` tool — present via Slack skill ✅.

### Missing Agents from Spec
The spec defines 7 agents but only 5 are implemented:
- ❌ `interview-prep` — Not built, no workspace
- ❌ `networking` — Not built, no workspace

These are presumably Phase 4+.

---

## 3. Agent Spawning Tests

**SKIPPED** — I'm running as a subagent of main, and spawning would create additional sessions that could interfere. The config shows:

- **main** can spawn: matron, archivist, market-intel, coordinator, app-tracker, cv-tailor, cover-letter ✅
- **coordinator** can spawn: market-intel, app-tracker, cv-tailor, cover-letter ✅
- **matron** can spawn: archivist ✅
- All others: no subagent permissions (leaf agents) ✅

This permission structure is correct per the spec — coordinator is the hub.

---

## 4. End-to-End Pipeline Test

### Step 1: Get a real job ✅
`GET /jobs?status=new&limit=1` → Got KPMG Associate Director - Tech CoE (job ID: 40b252eb...)

### Step 2: Extract skills ✅
`POST /skills/extract` with KPMG job description → Returned 4 skills (Python, AWS, Docker, ML). Works but the truncated descriptions from Adzuna limit extraction quality.

### Step 3: Skills gaps ❌ BLOCKED
`GET /skills/gaps` returns 400 — profile CV text is empty. **This is the #1 blocker for the entire pipeline.** Without a populated profile, no match scoring or gap analysis works.

### Step 4: Master CV ✅
File exists at `~/projects/claude_jobhunt/agent-data/master-cv.md`. It's a mock CV, clearly marked as such. Contains accurate Nick background (EY, Irish Guards, AI/ML).

### Step 5: Slack posting ✅
Successfully posted a formatted job alert to #briefing. `slack_post.py` works reliably.

**Pipeline verdict: 3/5 steps work. Profile population is the critical blocker.**

---

## 5. Slack Integration Test

| Channel | Post | Format | Result |
|---------|------|--------|--------|
| #briefing | ✅ | *bold* _italic_ `code` links | ✅ All rendered |
| #daily | ✅ | Basic text | ✅ |
| #general | ✅ | Basic text | ✅ |
| #overnight-sprints-channel | Not tested | — | — |
| #reading-list | Not tested | — | — |

Slack `slack_post.py` helper works well. Channel name mapping (briefing → C0ACJQZMP6H etc.) is correct.

**Note:** No delete capability tested — the script only posts, doesn't return message IDs in a way that enables deletion (though ts is returned).

---

## 6. Data Integrity

### DB Schema
Three tables exist:
- `jobs` — 722 rows, proper schema with id, title, company, location, salary, description, url, match_score, status, etc.
- `profiles` — Exists but cv_text is empty
- `esco_skills` — Exists (ESCO taxonomy for skills matching)

**Missing tables** (per API contract):
- ❌ `agent_log` — needed for agent activity tracking
- ❌ `follow_ups` — needed for app-tracker workflow  
- ❌ `networking_contacts` — needed for Phase 4

### Job Data Quality (5 samples)

| Job | Title | Company | Salary | Description | URL | Issues |
|-----|-------|---------|--------|-------------|-----|--------|
| 1 | Associate Director - Tech CoE | KPMG UK | £55,180 | Truncated (~600 chars) | ✅ Valid Adzuna | salary_min = salary_max |
| 2 | Senior Business Analyst | Inspire People | £45,000 | Truncated | ✅ | Below Nick's £100k target |
| 3 | Enterprise Architect | Amazon | £102,556 | Truncated | ✅ | Good match potential |
| 4 | Senior Director, Audit Manager | BNY Mellon | £92,408 | Truncated | ✅ | salary_min = salary_max |
| 5 | Head of Business Development | Capita | £89,602 | Truncated | ✅ | salary_min = salary_max |

**Issues:**
- All descriptions truncated to ~600 chars (Adzuna API limit?) — agents can't do proper skills extraction on partial JDs
- Many jobs have `salary_min == salary_max` (single value, not a range)
- ALL `match_score = 0.0` — scoring not working because profile is empty
- ALL `match_reasons = []` — same reason
- No relevance filtering — 722 jobs include everything from £3/yr teaching assistants to £221k directors
- No duplicate detection in practice (all `is_duplicate_of` is null)

### agent-data Directory Structure

**Mismatch between reset script expectations and actual directories:**

| Expected (reset script) | Actual | Status |
|------------------------|--------|--------|
| pipeline/leads | ✅ exists | OK |
| pipeline/active | ❌ missing | Mismatch |
| pipeline/applied | ❌ missing | Mismatch |
| pipeline/rejected | ❌ missing | Mismatch |
| cv/tailored | ✅ exists | OK |
| cover-letters | ✅ exists | OK |
| research | ❌ missing | Mismatch |
| briefs | ❌ missing | Mismatch |
| interview-prep | ✅ exists | OK |
| networking/drafts | ❌ missing | Mismatch |
| networking/sent | ❌ missing | Mismatch |

**Extra dirs that exist but aren't in reset script:**
- `company-research/` (spec says this, not `research/`)
- `outreach/` (not in reset script)
- `pipeline/materials/` (not in reset script)
- `reports/`, `reports/weekly/` (not in reset script)
- `scripts/`, `templates/` (not in reset script)

The reset script references dirs that don't exist and misses dirs that do. Non-critical but sloppy.

### Reset Script
Runs cleanly ✅. Preserves master-cv.md and target-profile.md. Clears agent workspace memory dirs. Does not create missing directories.

---

## 7. Configuration Audit

### Agent Registry (openclaw.json)

| Agent | Model | Workspace | allowAgents | Status |
|-------|-------|-----------|-------------|--------|
| main | claude-opus-4-6 | workspace/ | matron, archivist, market-intel, coordinator, app-tracker, cv-tailor, cover-letter | ✅ |
| matron | claude-sonnet-4-5 | workspace-matron/ | archivist | ✅ |
| archivist | claude-sonnet-4-5 | workspace/ (shared with main!) | none | ⚠️ |
| market-intel | claude-sonnet-4-5 | workspace-market-intel/ | none | ✅ |
| coordinator | claude-sonnet-4-5 | workspace-coordinator/ | market-intel, app-tracker, cv-tailor, cover-letter | ✅ |
| app-tracker | claude-sonnet-4-5 | workspace-app-tracker/ | none | ✅ |
| cv-tailor | claude-opus-4-5 | workspace-cv-tailor/ | none | ✅ |
| cover-letter | claude-opus-4-5 | workspace-cover-letter/ | none | ✅ |

**Model assignments match spec ✅** — Opus for creative (cv-tailor, cover-letter), Sonnet for operational (coordinator, market-intel, app-tracker).

**Issue:** Archivist shares workspace with main. This could cause conflicts if both read/write the same files simultaneously.

**Missing from spec:**
- `interview-prep` agent not registered (Phase 4)
- `networking` agent not registered (Phase 4)

### Subagent Permissions
- Main → can reach all agents ✅ (includes matron + archivist which are non-jobhunt)
- Coordinator → can spawn the 4 specialist agents ✅
- Specialists → leaf nodes, can't spawn ✅
- **Gap:** Coordinator can't see interview-prep or networking (not registered yet) — expected

---

## 8. Issues & Gap Analysis

### CRITICAL

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| C1 | Agent API endpoints not implemented (log, dashboard, follow-ups) | App-tracker is non-functional; no agent monitoring possible | Build the `/api/agents/*` routes + DB tables |
| C2 | Profile CV text empty → match scoring broken → all scores 0.0 | No job relevance filtering, no skills gap analysis, pipeline is blind | `PUT /profile` with master CV text |
| C3 | `min_score` filter broken | Can't filter high-quality matches even if scoring worked | Fix the query filter in backend |

### HIGH

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| H1 | Job descriptions truncated (~600 chars) | Skills extraction and matching quality severely limited | Investigate Adzuna API — can we get full descriptions? Or scrape job URLs? |
| H2 | Master CV path wrong in agent AGENTS.md files | Agents will fail to find the CV when dispatched | Fix path: `agent-data/master-cv.md` not `agent-data/cv/master-cv.md` |
| H3 | JOBHUNT-API.md skill incomplete | Agents don't know about skills, search, profile endpoints | Update skill file across all 5 workspaces |
| H4 | No cron jobs configured for any agent | Coordinator daily planning, market-intel 3x daily scans not automated | Set up crons per spec |

### MEDIUM

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| M1 | agent-data directory structure mismatches reset script | Reset script silently skips missing dirs | Align dirs or update reset script |
| M2 | 722 unfiltered jobs (£3/yr to £221k) | Noise overwhelms signal | Need profile + scoring to filter |
| M3 | Search index empty (BM25 = 0) | Hybrid search non-functional | Run `/api/search/rebuild-index` |
| M4 | No duplicate detection active | Possible duplicate jobs in DB | Check content_hash implementation |
| M5 | Networking/interview-prep agents not built | Phase 4 blocked | Build when ready |

### LOW

| # | Issue | Impact | Fix |
|---|-------|--------|-----|
| L1 | Archivist shares workspace with main | Potential file conflicts | Give archivist own workspace |
| L2 | Slack message deletion not scriptable | Test messages linger | Add delete to slack_post.py |
| L3 | cv-tailor missing TOOLS.md | Inconsistency | Create file or confirm not needed |
| L4 | API contract documents Bearer token auth but system uses cookies | Potential confusion | Align docs or implement both |

---

## 9. Recommendations — Priority Order

### Before Phase 4 (do these NOW)

1. **Populate the profile** — `PUT /profile` with master CV text, target roles, sectors, locations, salary. This unblocks match scoring AND skills gap analysis. Probably 30 minutes of work.

2. **Fix `min_score` filter** — Backend bug. Without this, agents can't filter for relevant jobs.

3. **Build agent API endpoints** — At minimum: `POST/GET /api/agents/log` and `GET /api/agents/dashboard`. The follow-ups endpoints are needed for app-tracker but could wait slightly.

4. **Update JOBHUNT-API.md** across all 5 workspaces with the full endpoint list.

5. **Fix master CV path** in coordinator's AGENTS.md (and any other agents that reference it).

6. **Set up coordinator + market-intel cron jobs** — The spec defines schedules; implement them.

### Phase 4 Prerequisites
- Agent log + dashboard endpoints working
- Follow-up endpoints working  
- Profile populated and scoring functional
- At least one successful end-to-end pipeline run (job → skills → CV → cover letter → Slack approval)

---

## 10. What's Actually Working Well

- **Core job API** is solid — list, get, patch, refresh all work
- **Auth** is clean and consistent
- **Slack integration** is reliable and well-abstracted
- **Agent workspaces** are well-structured with coherent SOUL.md personalities
- **Skills extraction** works (even with truncated descriptions)
- **Config/permissions** are correctly set up
- **Reset script** works cleanly
- **Docker setup** is stable (backend + frontend running)
- **Model assignments** are sensible (Opus for creative, Sonnet for operational)

The foundations are solid. The main gap is between what's *documented* in the API contract and what's *implemented* in the backend. About 40% of the contracted API surface doesn't exist yet.
