# Changelog

## 2026-02-15 — OpenClaw Agent Integration (Phases 1-3)

### Added
- **5 OpenClaw agent workspaces** — coordinator, market-intel, app-tracker, cv-tailor, cover-letter
  - Each with SOUL.md, AGENTS.md, IDENTITY.md, USER.md
  - Skill files: JOBHUNT-API.md (full API docs), SLACK.md (posting helper)
- **Agent API endpoints** — new routes for agent activity tracking:
  - `POST/GET /api/agents/log` — agent audit trail
  - `GET /api/agents/dashboard` — aggregated monitoring
  - `GET/POST/PATCH /api/agents/follow-ups` — follow-up scheduling
  - `GET/POST/PATCH /api/agents/networking-contacts` — contact management
- **New DB models** — `agent_log`, `follow_ups`, `networking_contacts` tables
- **Slack integration** — bot token + app token configured, posting tested to 5 channels
- **Helper scripts:**
  - `scripts/slack_post.py` — Slack channel posting utility
  - `scripts/reset-test-data.sh` — clean sweep of agent-generated data (preserves Adzuna DB)
- **Mock master CV** at `agent-data/master-cv.md`
- **Target profile** at `agent-data/knowledge/target-profile.md`
- **agent-data directory structure** — pipeline/, cv/, cover-letters/, research/, briefs/, etc.
- **Test reports** — comprehensive QA + UAT reports

### Fixed
- **min_score filter** — parameter name mismatch (`score_min` vs `min_score`), now accepts both
- **docker-compose.yml** — DATABASE_URL corrected from PostgreSQL to SQLite
- **Profile population** — master CV uploaded via PUT /profile, unblocking match scoring

### Known Issues
- Job descriptions truncated (~600 chars) — Adzuna API limitation
- Match scoring returns 0.0 for most jobs — profile needs real CV for meaningful scores
- Skills/gaps endpoint can hang — needs investigation
- No Brave Search API key — web_search non-functional
- Interview-prep and networking agents not yet built (Phase 4)
- No cron jobs configured yet for automated scanning

### Agent Architecture
```
main (Opus 4) → coordinator (Sonnet 4.5) → market-intel (Sonnet 4.5)
                                          → app-tracker (Sonnet 4.5)
                                          → cv-tailor (Opus 4.5)
                                          → cover-letter (Opus 4.5)
```

### Documentation
- QA Test Report: TEST-REPORT-2026-02-15.md
- UAT Report: UAT-REPORT-2026-02-15.md
- Implementation Guide: docs/openclawintegration/IMPLEMENTATION_GUIDE.md
