---
tags: [job-hunt, openclaw, agents, phase-1]
created: 2026-02-15
---

# Job Hunt — Phase 1 Progress

## Overview
Building a seven-agent autonomous job search system on OpenClaw, layered on top of Nick's existing `claude_jobhunt` FastAPI/Next.js application.

## Repo
- **Source:** `~/projects/claude_jobhunt` (cloned from `solstice035/claude_jobhunt`)
- **Docs:** `docs/openclawintegration/` — comprehensive plan written by LLM, reviewed and adapted

## Phase 1: Foundation — Market Intelligence Agent

### Completed ✅
- [x] Repo cloned
- [x] `.env` configured with API keys (from Obsidian vault note)
- [x] Docker dev stack running (Colima) — backend :8000, frontend :3000
- [x] All API endpoints verified working (stats, jobs, skills/extract, search, refresh)
- [x] Correct API paths mapped (`/skills/*` not `/api/skills/*`, `/api/search/*`)
- [x] 717 jobs in DB from Adzuna (no match scores — needs CV profile)
- [x] Market Intelligence agent workspace created (`~/.openclaw/workspace-market-intel/`)
- [x] Agent registered in OpenClaw config
- [x] Test scan completed — agent can auth, query, and report
- [x] Full scan running with lead report output
- [x] Agent data directories created

### Pending
- [ ] **Slack setup** — BLOCKER: needs Nick to create Slack app (Socket Mode)
- [ ] **CV Profile** — BLOCKER: needs Nick's master CV to enable match scoring
- [ ] Add API key auth to FastAPI (currently using cookie auth — temporary workaround)
- [ ] Set up cron jobs for scheduled scanning (3x daily)
- [ ] Add `agent_activities` table to SQLite for audit trail
- [ ] Test Slack message delivery from agent

### Architecture Notes
- **Auth:** Cookie-based JWT (30-day expiry). Agents log in via POST /auth/login. Temporary workaround until API key middleware is added.
- **API Paths:** Skills at `/skills/*`, search at `/api/search/*`, jobs at `/jobs/*`
- **Docker:** `docker-compose` (not `docker compose`) on this machine
- **No Redis/ChromaDB** in dev compose — backend runs without them (graceful degradation)

## Blockers for Nick
1. **Create Slack workspace + app** — see setup instructions below
2. **Provide master CV** — comprehensive reference document for Phase 0
3. **Set profile/CV** in the app — enables match scoring

## Slack App Setup Instructions
1. Go to https://api.slack.com/apps → Create New App → From scratch
2. Name: `JobHunt Agent` | Workspace: your workspace
3. **Socket Mode:** Enable, generate App-Level Token (`connections:write` scope) → gives `xapp-*` token
4. **OAuth & Permissions** — Bot Token Scopes: `chat:write`, `channels:read`, `channels:history`, `im:read`, `im:write`, `im:history`, `app_mentions:read`, `reactions:read`
5. **Event Subscriptions:** Enable, subscribe to: `message.channels`, `message.im`, `app_mention`, `reaction_added`
6. Install to workspace → gives `xoxb-*` token
7. Create channels: `#job-search`, `#job-agents-ops`
8. Invite bot: `/invite @JobHunt Agent` in both channels
9. Give Jeeves both tokens

## Cost Estimate
- Market Intel (Sonnet, 3x daily): ~£12/month
- All 7 agents at full operation: ~£55-80/month
