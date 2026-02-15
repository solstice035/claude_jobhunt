# OpenClaw Multi-Agent Job Search System — Technical Implementation Guide

> **Purpose**: Step-by-step implementation guide for building a seven-agent autonomous job search system on OpenClaw, layered on top of the existing `claude_jobhunt` application.
>
> **Author**: Nick (EY Senior Manager, Banking & Capital Markets)
> **Platform**: OpenClaw on Mac Mini (macOS)
> **Existing System**: https://github.com/solstice035/claude_jobhunt
> **Target**: Fully autonomous job search pipeline with human-in-the-loop approvals

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Decision Log](#2-decision-log)
3. [Prerequisites & Environment Setup](#3-prerequisites--environment-setup)
4. [Phase 1: Foundation — Market Intelligence Agent](#4-phase-1-foundation--market-intelligence-agent)
5. [Phase 2: Coordination Layer](#5-phase-2-coordination-layer)
6. [Phase 3: Materials Pipeline](#6-phase-3-materials-pipeline)
7. [Phase 4: Interview & Networking](#7-phase-4-interview--networking)
8. [Phase 5: Polish & Monitoring](#8-phase-5-polish--monitoring)
9. [Agent Specifications (Complete)](#9-agent-specifications-complete)
10. [FastAPI Backend Extensions](#10-fastapi-backend-extensions)
11. [Database Schema Extensions](#11-database-schema-extensions)
12. [OpenClaw Configuration Reference](#12-openclaw-configuration-reference)
13. [Operational Runbooks](#13-operational-runbooks)
14. [Cost Model & Resource Planning](#14-cost-model--resource-planning)

---

## 1. Architecture Overview

### System Topology

```
┌──────────────────────────────────────────────────────────────────────┐
│                         MAC MINI (macOS)                             │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   OpenClaw Gateway                           │    │
│  │                 (Node.js, port 18789)                        │    │
│  │                                                               │    │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │    │
│  │  │Coordinator│  │Market    │  │CV Tailor │  │Cover     │   │    │
│  │  │  Agent    │  │Intel     │  │  Agent   │  │Letter    │   │    │
│  │  │          │──▶│  Agent   │  │          │  │  Agent   │   │    │
│  │  │          │──▶│          │  │          │  │          │   │    │
│  │  │          │──▶├──────────┤  ├──────────┤  ├──────────┤   │    │
│  │  │          │   │App       │  │Interview │  │Networking│   │    │
│  │  │          │──▶│Tracker   │  │  Prep    │  │  Agent   │   │    │
│  │  │          │──▶│  Agent   │  │  Agent   │  │          │   │    │
│  │  └──────────┘   └──────────┘  └──────────┘  └──────────┘   │    │
│  │       │              │                                        │    │
│  │       │    exec+curl │                                        │    │
│  └───────┼──────────────┼────────────────────────────────────────┘    │
│          │              │                                              │
│          ▼              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              Existing claude_jobhunt Stack                   │    │
│  │                                                               │    │
│  │  FastAPI (:8000)  │  Next.js (:3000)  │  SQLite + Redis     │    │
│  │  ChromaDB         │  Prometheus/Grafana │  APScheduler        │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  ┌──────────────┐                                                    │
│  │  Slack (API)  │  ◀── Socket Mode (no public URL needed)          │
│  └──────────────┘                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow: Job Discovery → Application Submission

```
[Cron: 07:00 daily]
       │
       ▼
┌──────────────┐    POST /jobs/refresh     ┌────────────────┐
│  Coordinator │───────────────────────────▶│  FastAPI        │
│              │                            │  (Adzuna fetch) │
│              │    GET /jobs?status=new     │                │
│              │◀───────────────────────────│                │
└──────┬───────┘                            └────────────────┘
       │
       │ sessions_spawn (for each high-score job)
       │
       ├──▶ [Market Intel] ── web_search company ──▶ writes company-research/
       │
       ├──▶ [CV Tailor] ── reads master-cv.md + job desc ──▶ writes tailored CV
       │
       ├──▶ [Cover Letter] ── reads company research + CV ──▶ writes cover letter
       │
       ▼
┌──────────────┐
│  Coordinator │──── Slack message: "Materials ready for Barclays Director role"
│              │     [Approve] [Reject] [Edit]
└──────┬───────┘
       │
       │ Nick clicks [Approve] in Slack
       │
       ▼
┌──────────────┐    PATCH /jobs/<id>/status    ┌────────────────┐
│  App Tracker │───────────────────────────────▶│  FastAPI        │
│              │    (status = "applied")         │  (updates DB)  │
└──────────────┘                                └────────────────┘
```

### Agent Communication Model

```
                    ┌─────────────────┐
                    │   Coordinator    │
                    │ (hub / spawner)  │
                    └────────┬────────┘
                             │
              sessions_spawn │ sessions_send
                             │
        ┌────────┬───────────┼───────────┬────────┬────────┐
        │        │           │           │        │        │
        ▼        ▼           ▼           ▼        ▼        ▼
    ┌───────┐┌───────┐ ┌─────────┐ ┌─────────┐┌──────┐┌──────┐
    │Market ││CV     │ │Cover    │ │App      ││Inter-││Net-  │
    │Intel  ││Tailor │ │Letter   │ │Tracker  ││view  ││work  │
    └───┬───┘└───┬───┘ └────┬────┘ └────┬────┘└──┬───┘└──┬───┘
        │        │          │           │        │       │
        └────────┴──────────┴───────────┘        │       │
                        │                         │       │
                   exec + curl                    │       │
                        │                         │       │
                        ▼                         │       │
              ┌──────────────────┐               │       │
              │  FastAPI Backend  │               │       │
              │  (localhost:8000) │               │       │
              └──────────────────┘               │       │
                                                  │       │
                                          web_search    web_search
                                          web_fetch     browser
```

**Key constraint**: Sub-agents CANNOT spawn sub-sub-agents. Only the Coordinator spawns. This is an OpenClaw platform limitation — design around it.

---

## 2. Decision Log

This section records architectural decisions with rationale. Reference these when implementation questions arise.

### D1: OpenClaw as sole orchestrator (no n8n)

**Decision**: Use OpenClaw's cron, heartbeat, Lobster workflows, and webhooks for all orchestration. Do NOT deploy n8n.

**Rationale**: OpenClaw provides native scheduling (cron), continuous monitoring (heartbeat), deterministic multi-step workflows with approval gates (Lobster), and event-driven triggers (webhooks). Adding n8n introduces a separate runtime, Docker containers, a separate UI, and another failure domain — all for capabilities OpenClaw already provides. Nick has n8n experience to fall back on if needed, but start without it.

**Reversibility**: High. If OpenClaw's scheduling proves insufficient, n8n can be added later alongside it. The `exec + curl` pattern works identically whether triggered by OpenClaw cron or n8n Execute Command node.

### D2: exec + curl for API integration (not MCP)

**Decision**: All agent-to-FastAPI communication uses `exec` tool running `curl` + `jq` commands, defined in a shared SKILL.md file.

**Rationale**: OpenClaw's native MCP client support is not yet merged into core (Feb 2026). Community plugins exist but add fragility. `exec + curl` is universally reliable, easy to debug (`curl -v`), and the output is parseable by any agent. The shared skill pattern means API calling conventions are defined once and used by all agents.

**Reversibility**: High. When MCP client lands in OpenClaw core, API calls can migrate to MCP tools with no architectural change — just swap the skill instructions.

### D3: Model allocation — Sonnet for routine, Opus for creative

**Decision**: Use Claude Sonnet 4.5 for Coordinator, Market Intel, App Tracker, and Networking agents. Use Claude Opus for CV Tailor, Cover Letter, and Interview Prep agents.

**Rationale**: Routine tasks (API calls, scheduling, tracking, web research) don't need Opus-level reasoning. Creative writing tasks (CV tailoring, cover letters, interview prep) benefit significantly from Opus's superior writing quality. This halves API costs for ~70% of agent invocations.

**Reversibility**: Trivial — change the `model` field in each agent's configuration.

### D4: Flat agent hierarchy (no delegation chains)

**Decision**: Coordinator is the only agent that spawns other agents. No agent-to-agent spawning beyond this.

**Rationale**: OpenClaw sub-agents cannot spawn sub-sub-agents (platform constraint). Even if they could, deep delegation chains are harder to debug and monitor. A flat hub-and-spoke model keeps the system predictable and auditable.

### D5: Slack as primary human interface (Socket Mode)

**Decision**: Use Slack with Socket Mode for all human-in-the-loop interactions. No public webhook URL needed.

**Rationale**: Mac Mini is behind NAT — no easy public endpoint. Socket Mode uses outbound WebSocket connections, bypassing NAT entirely. Slack is accessible from phone, desktop, and web — Nick can approve applications from anywhere. Interactive messages (approve/reject buttons) are not natively supported by OpenClaw, so approvals use text responses or emoji reactions on agent messages.

### D6: File-system as shared state (supplementing SQLite)

**Decision**: Agents share context through both the file system (`~/jobhunt/agent-data/`) and the existing SQLite database (via FastAPI API calls). Long-form content (CVs, cover letters, research briefs) lives on disk. Structured pipeline data lives in SQLite.

**Rationale**: OpenClaw agents natively read/write files. SQLite is accessed via API calls. This separation plays to each system's strengths: files for documents humans review, database for queryable pipeline state.

### D7: Isolated cron sessions for batch work

**Decision**: All scheduled agent runs use `--session isolated` to start fresh sessions rather than continuing long-running ones.

**Rationale**: Context window management. Agents processing many job descriptions in one session will hit compaction limits and lose important context. Fresh sessions start clean with only the agent's system prompt and current task — no accumulated cruft from previous runs.

---

## 3. Prerequisites & Environment Setup

### 3.1 Verify Existing System

Before touching OpenClaw, confirm the existing `claude_jobhunt` stack is healthy:

```bash
# Check FastAPI is running
curl -s http://localhost:8000/docs | head -5
# Expected: HTML of Swagger docs page

# Check jobs endpoint
curl -s http://localhost:8000/stats | jq '.'
# Expected: JSON with job counts

# Check Adzuna integration
curl -s -X POST http://localhost:8000/jobs/refresh | jq '.'
# Expected: JSON confirmation of fetch triggered

# Check frontend
curl -s http://localhost:3000 | head -5
# Expected: HTML of Next.js app

# Check Redis (if running)
redis-cli ping
# Expected: PONG

# Check Docker containers
docker ps
# Expected: List of running containers
```

**Document the results.** If anything is broken, fix it before proceeding. The agents depend on every endpoint working.

### 3.2 Install OpenClaw

```bash
# Ensure Node.js 22+ is installed
node --version
# If not: brew install node@22

# Install OpenClaw globally
npm install -g openclaw@latest

# Verify installation
openclaw --version

# Run onboarding — this creates config directories and installs the launchd daemon
openclaw onboard --install-daemon

# Verify daemon is running
launchctl list | grep openclaw
# Expected: PID and status for com.openclaw.gateway

# Check Gateway health
curl -s http://127.0.0.1:18789/health
# Expected: JSON health status
```

### 3.3 Configure Anthropic API Access

```bash
# Add Anthropic API key to OpenClaw
openclaw config set auth.anthropic.apiKey "sk-ant-..."

# Or set via environment variable in the launchd plist
# Edit ~/Library/LaunchAgents/com.openclaw.gateway.plist and add:
# <key>ANTHROPIC_API_KEY</key>
# <string>sk-ant-...</string>

# Verify model access
openclaw model test anthropic/claude-sonnet-4-5
# Expected: Successful test response
```

### 3.4 Create Directory Structure

```bash
# Agent workspaces (OpenClaw standard location)
mkdir -p ~/.openclaw/workspace-coordinator
mkdir -p ~/.openclaw/workspace-market-intel
mkdir -p ~/.openclaw/workspace-cv-tailor
mkdir -p ~/.openclaw/workspace-cover-letter
mkdir -p ~/.openclaw/workspace-app-tracker
mkdir -p ~/.openclaw/workspace-interview-prep
mkdir -p ~/.openclaw/workspace-networking

# Shared skills directory
mkdir -p ~/.openclaw/skills/jobhunt-api
mkdir -p ~/.openclaw/skills/company-research
mkdir -p ~/.openclaw/skills/email-monitor

# Shared agent data (within the existing repo)
cd ~/claude_jobhunt  # or wherever the repo lives
mkdir -p agent-data/knowledge
mkdir -p agent-data/pipeline/leads
mkdir -p agent-data/pipeline/materials
mkdir -p agent-data/cv/tailored
mkdir -p agent-data/cover-letters
mkdir -p agent-data/interview-prep
mkdir -p agent-data/outreach
mkdir -p agent-data/reports/weekly
mkdir -p agent-data/company-research
mkdir -p agent-data/templates
```

### 3.5 Phase 0: Master CV Foundation

**Goal**: Build a comprehensive, validated master CV before any agent work begins.
**Duration**: Week 0 (~5-10 hours)
**Effort**: Primarily Nick + one Opus session

The entire system's output quality depends on how good `master-cv.md` is. Every tailored CV and cover letter draws from this source. Treat this as foundational — skipping or rushing Phase 0 degrades everything downstream.

#### 3.5.1 Gather All Source Material

Collect everything that might contain relevant experience, skills, or achievements:

- Current CV (all versions)
- LinkedIn profile (export as PDF)
- EY performance reviews and project summaries
- Military service record highlights
- Project documentation for key deliverables
- Training certificates and qualifications
- Any portfolio or case study materials

#### 3.5.2 Structure the Master CV

Use a Claude Opus session to help structure a comprehensive `master-cv.md`:

```bash
# Create the master CV file
touch ~/claude_jobhunt/agent-data/knowledge/master-cv.md
```

The master CV should include **everything** — it's not a 2-page document but a comprehensive reference:

- **Every role** with full bullet points (not just the last 3 positions)
- **Every project** with quantified outcomes where possible
- **Every skill** including technical, leadership, domain, and soft skills
- **Every qualification** and certification
- **Key metrics**: team sizes managed, budgets owned, revenue influenced, efficiency gains delivered

#### 3.5.3 Validate with Nick

Review the structured master CV for:
- [ ] Accuracy — no inflated claims
- [ ] Completeness — no significant projects or skills missing
- [ ] Metrics — every major achievement has a number attached
- [ ] Narrative coherence — the career story makes sense (military → consulting → AI/ML)

#### 3.5.4 Store and Version

```bash
# Commit the validated master CV
cd ~/claude_jobhunt
git add agent-data/knowledge/master-cv.md
git commit -m "Add validated master CV for agent system"
```

**Phase 0 is complete when**: A comprehensive, validated `master-cv.md` exists that captures every project, metric, skill, and achievement Nick might want to reference. The CV Tailor agent will select and reframe from this master copy — its output can only be as good as this input.

### 3.6 Create Target Profile

```bash
cat > ~/claude_jobhunt/agent-data/knowledge/target-profile.md << 'EOF'
# Job Search Target Profile

## Target Roles
- Senior Manager / Director / Associate Director / Principal Consultant
- Technology Consulting, Digital Transformation, AI/ML Strategy
- Banking & Capital Markets, Financial Services, Insurance

## Domain Expertise to Emphasise
- Regulatory compliance: Basel III/IV, MiFID II, DORA, T+1 settlement
- AI/ML integration for financial services
- Controls solutions and risk management
- Digital transformation and process optimization

## Target Employers
### Tier 1 (Priority)
- Big 4: Deloitte, PwC, KPMG (already at EY — lateral move)
- MBB: McKinsey, BCG, Bain (technology practices)
- Tier-1 banks: JPMorgan, Goldman Sachs, Barclays, HSBC, Morgan Stanley

### Tier 2
- Boutique consultancies: Oliver Wyman, Capco, Capgemini Invent
- Fintech scale-ups: Thought Machine, 10x Banking, Revolut, Monzo
- Technology firms: Palantir, Microsoft, Google (financial services verticals)

## Location
- Within 60-minute commute of Guildford, UK
- London preferred
- Remote/hybrid roles accepted
- Will not relocate internationally

## Compensation
- Minimum: £100,000 base
- Target: £120,000-£150,000+ base
- Open to equity/bonus-heavy packages at startups

## Exclusion List
- Roles requiring >4 days/week in office outside London
- Pure software engineering roles (want consulting/strategy component)
- Contracts shorter than 6 months
- Companies: [add any specific exclusions]

## Key Differentiators
1. Military leadership (Irish Guards Captain) — unique among consulting hires
2. Bridges business and technology — can talk to the board AND write code
3. AI/ML specialist with production deployment experience
4. Deep regulatory compliance knowledge in banking
EOF
```

### 3.7 Create Exclusion List

```bash
cat > ~/claude_jobhunt/agent-data/knowledge/exclusion-list.md << 'EOF'
# Exclusion List

## Companies to Skip
- [Add companies you don't want to apply to]

## Roles to Skip
- Any role with "Junior" or "Associate" (non-Associate Director) in title
- Pure software engineering without consulting/advisory component
- Roles requiring security clearance you don't hold
- Commission-only or self-employed contractor positions

## Recruiters to Ignore
- [Add specific agencies or recruiters if needed]
EOF
```

---

## 4. Phase 1: Foundation — Market Intelligence Agent

**Goal**: One agent calling the existing FastAPI backend, running on a cron schedule, reporting to Slack.
**Duration**: Week 1–2
**Effort**: ~15 hours

### 4.1 Extend FastAPI Backend

Add the following to your existing FastAPI application:

#### 4.1.1 API Key Authentication Middleware

Create `backend/app/middleware/agent_auth.py`:

```python
"""
Simple API key authentication for agent access.
Agents authenticate with a bearer token in the Authorization header.
Human users continue using the existing session/JWT auth.
"""
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
import os
import secrets

AGENT_API_KEY = os.getenv("AGENT_API_KEY", "")

class AgentAuthMiddleware(BaseHTTPMiddleware):
    """
    Checks for agent API key on /api/agents/* endpoints.
    All other endpoints fall through to existing auth.
    """
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/api/agents"):
            auth_header = request.headers.get("Authorization", "")
            if not auth_header.startswith("Bearer "):
                raise HTTPException(status_code=401, detail="Missing bearer token")
            token = auth_header.replace("Bearer ", "")
            if not secrets.compare_digest(token, AGENT_API_KEY):
                raise HTTPException(status_code=403, detail="Invalid agent API key")
        return await call_next(request)
```

Generate and store the API key:

```bash
# Generate a secure API key
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to your .env file
echo 'AGENT_API_KEY=<generated-key>' >> ~/claude_jobhunt/.env
```

#### 4.1.2 Agent Activity Logging Endpoint

Create `backend/app/api/agents.py`:

```python
"""
Agent coordination endpoints.
Tracks agent activities, provides health checks, and exposes
pipeline data in agent-friendly formats.
"""
from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from datetime import datetime, timedelta
from typing import Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/agents", tags=["agents"])

class AgentActivityLog(BaseModel):
    agent_id: str
    action_type: str  # 'api_call', 'file_write', 'slack_msg', 'error'
    description: str
    job_id: Optional[int] = None
    status: str = "completed"  # 'completed', 'failed', 'pending'
    error_message: Optional[str] = None
    tokens_used: Optional[int] = None

@router.post("/log")
async def log_agent_activity(activity: AgentActivityLog, db: AsyncSession = Depends(get_db)):
    """Record an agent action for audit trail."""
    await db.execute(text("""
        INSERT INTO agent_activities 
        (agent_id, action_type, description, job_id, status, error_message, tokens_used)
        VALUES (:agent_id, :action_type, :description, :job_id, :status, :error_message, :tokens_used)
    """), activity.dict())
    await db.commit()
    return {"status": "logged"}

@router.get("/log")
async def get_agent_activities(
    agent_id: Optional[str] = None,
    since_hours: int = Query(default=24, le=168),
    limit: int = Query(default=50, le=200),
    db: AsyncSession = Depends(get_db)
):
    """Retrieve recent agent activity."""
    since = datetime.utcnow() - timedelta(hours=since_hours)
    query = "SELECT * FROM agent_activities WHERE created_at > :since"
    params = {"since": since}
    if agent_id:
        query += " AND agent_id = :agent_id"
        params["agent_id"] = agent_id
    query += " ORDER BY created_at DESC LIMIT :limit"
    params["limit"] = limit
    result = await db.execute(text(query), params)
    return [dict(row._mapping) for row in result]

@router.get("/dashboard")
async def agent_dashboard(db: AsyncSession = Depends(get_db)):
    """Overview for the agent monitoring dashboard."""
    # Last activity per agent
    result = await db.execute(text("""
        SELECT agent_id, 
               MAX(created_at) as last_active,
               COUNT(*) as total_actions,
               SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as error_count
        FROM agent_activities 
        WHERE created_at > datetime('now', '-24 hours')
        GROUP BY agent_id
    """))
    agents = [dict(row._mapping) for row in result]
    
    # Pipeline summary
    pipeline = await db.execute(text("""
        SELECT status, COUNT(*) as count 
        FROM jobs 
        GROUP BY status
    """))
    
    return {
        "agents": agents,
        "pipeline": [dict(row._mapping) for row in pipeline],
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/follow-ups-due")
async def get_due_follow_ups(db: AsyncSession = Depends(get_db)):
    """Get applications needing follow-up."""
    result = await db.execute(text("""
        SELECT f.*, j.title, j.company 
        FROM follow_ups f
        JOIN jobs j ON f.job_id = j.id
        WHERE f.completed_at IS NULL 
        AND f.due_date <= date('now')
        ORDER BY f.due_date ASC
    """))
    return [dict(row._mapping) for row in result]
```

Register the router in `backend/app/main.py`:

```python
from app.api.agents import router as agents_router
app.include_router(agents_router)
```

#### 4.1.3 Add created_after Filter to Jobs Endpoint

In your existing jobs endpoint (likely `backend/app/api/jobs.py`), add a `created_after` query parameter:

```python
from datetime import datetime
from typing import Optional

@router.get("/jobs")
async def list_jobs(
    status: Optional[str] = None,
    min_score: Optional[float] = None,
    created_after: Optional[str] = None,  # ISO format datetime
    # ... existing params
):
    query = "SELECT * FROM jobs WHERE 1=1"
    params = {}
    
    if created_after:
        query += " AND created_at > :created_after"
        params["created_after"] = created_after
    
    # ... rest of existing filtering logic
```

### 4.2 Create Database Tables

Add to your existing migration or run directly:

```sql
-- Agent activity audit trail
CREATE TABLE IF NOT EXISTS agent_activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    description TEXT,
    job_id INTEGER REFERENCES jobs(id),
    input_summary TEXT,
    output_summary TEXT,
    status TEXT DEFAULT 'completed',
    error_message TEXT,
    tokens_used INTEGER,
    cost_estimate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Follow-up tracking
CREATE TABLE IF NOT EXISTS follow_ups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    follow_up_type TEXT NOT NULL,
    due_date DATE NOT NULL,
    completed_at TIMESTAMP,
    agent_id TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Networking contacts (Phase 4)
CREATE TABLE IF NOT EXISTS networking_contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    company TEXT,
    role TEXT,
    linkedin_url TEXT,
    email TEXT,
    connection_type TEXT,
    shared_background TEXT,
    outreach_status TEXT DEFAULT 'identified',
    message_draft TEXT,
    approved_at TIMESTAMP,
    sent_at TIMESTAMP,
    response_received BOOLEAN DEFAULT FALSE,
    follow_up_due DATE,
    job_ids TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_agent_activities_agent ON agent_activities(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_activities_created ON agent_activities(created_at);
CREATE INDEX IF NOT EXISTS idx_follow_ups_due ON follow_ups(due_date) WHERE completed_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_networking_status ON networking_contacts(outreach_status);
```

Run this against your SQLite database:

```bash
sqlite3 ~/claude_jobhunt/data/jobs.db < schema_extensions.sql
```

### 4.3 Create the Shared JobHunt API Skill

This skill is referenced by every agent that calls your FastAPI backend.

```bash
cat > ~/.openclaw/skills/jobhunt-api/SKILL.md << 'SKILL_EOF'
---
name: jobhunt-api
description: >
  Interface with Nick's JobHunt FastAPI backend for job search, matching,
  pipeline management, skill analysis, and statistics. Use this skill
  whenever interacting with the job search database or triggering job fetches.
metadata:
  openclaw:
    requires:
      bins: ["curl", "jq"]
      env: ["JOBHUNT_API_URL", "AGENT_API_KEY"]
    always: true
---

# JobHunt API Skill

Base URL: $JOBHUNT_API_URL (default: http://localhost:8000)
Auth: Bearer token in AGENT_API_KEY (required for /api/agents/* endpoints)

## Core Job Operations

### Trigger Job Fetch from Adzuna
```bash
curl -s -X POST "$JOBHUNT_API_URL/jobs/refresh" | jq '.'
```
Returns confirmation. New jobs appear after ~30 seconds of ingestion.

### List Jobs with Filters
```bash
# All new jobs
curl -s "$JOBHUNT_API_URL/jobs?status=new" | jq '.'

# High-scoring new jobs from last 6 hours
curl -s "$JOBHUNT_API_URL/jobs?status=new&min_score=0.75&created_after=$(date -u -v-6H +%Y-%m-%dT%H:%M:%S)" | jq '.'

# Jobs by specific status
curl -s "$JOBHUNT_API_URL/jobs?status=saved" | jq '.'
curl -s "$JOBHUNT_API_URL/jobs?status=applied" | jq '.'
```

### Get Single Job Details
```bash
curl -s "$JOBHUNT_API_URL/jobs/<JOB_ID>" | jq '.'
```

### Update Job Status
```bash
curl -s -X PATCH "$JOBHUNT_API_URL/jobs/<JOB_ID>" \
  -H "Content-Type: application/json" \
  -d '{"status": "applied", "notes": "Applied via company portal"}' | jq '.'
```

Valid statuses: new, saved, applied, interviewing, offered, rejected, withdrawn

### Get Dashboard Statistics
```bash
curl -s "$JOBHUNT_API_URL/stats" | jq '.'
```

## Search & Matching

### Hybrid Search (BM25 + Semantic)
```bash
curl -s -X POST "$JOBHUNT_API_URL/api/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "senior technology consulting banking AI", "limit": 20}' | jq '.'
```

### Re-rank Results
```bash
curl -s -X POST "$JOBHUNT_API_URL/api/search/rerank" \
  -H "Content-Type: application/json" \
  -d '{"query": "regulatory compliance director", "job_ids": [1, 5, 12, 23]}' | jq '.'
```

## Skills Analysis

### Extract Skills from Text
```bash
curl -s -X POST "$JOBHUNT_API_URL/api/skills/extract" \
  -H "Content-Type: application/json" \
  -d '{"text": "<paste job description here>"}' | jq '.'
```

### Skill Gap Analysis for a Job
```bash
curl -s "$JOBHUNT_API_URL/api/skills/gaps?job_id=<JOB_ID>" | jq '.'
```

### Skill Gap Summary (Aggregate)
```bash
curl -s "$JOBHUNT_API_URL/api/skills/gaps/summary" | jq '.'
```

### Learning Recommendations
```bash
curl -s "$JOBHUNT_API_URL/api/skills/recommendations" | jq '.'
```

## Agent-Specific Endpoints

### Log Agent Activity (requires AGENT_API_KEY)
```bash
curl -s -X POST "$JOBHUNT_API_URL/api/agents/log" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AGENT_API_KEY" \
  -d '{
    "agent_id": "market-intel",
    "action_type": "api_call",
    "description": "Morning job scan completed. Found 8 new matches.",
    "status": "completed"
  }' | jq '.'
```

### Get Agent Dashboard Data
```bash
curl -s "$JOBHUNT_API_URL/api/agents/dashboard" \
  -H "Authorization: Bearer $AGENT_API_KEY" | jq '.'
```

### Get Follow-ups Due
```bash
curl -s "$JOBHUNT_API_URL/api/agents/follow-ups-due" \
  -H "Authorization: Bearer $AGENT_API_KEY" | jq '.'
```

## Error Handling
- **Connection refused**: FastAPI server not running. Alert Nick via Slack.
- **404**: Job ID not found. Verify the ID and retry.
- **422**: Invalid request format. Check JSON payload.
- **500**: Server error. Wait 30 seconds, retry once, then alert Nick.
- **Timeout (>10s)**: Server may be overloaded. Retry with backoff.

Always use `--max-time 15` on curl calls to prevent hanging:
```bash
curl -s --max-time 15 "$JOBHUNT_API_URL/jobs" | jq '.'
```
SKILL_EOF
```

### 4.4 Create Market Intelligence Agent Workspace

```bash
# IDENTITY.md — Agent identity card
cat > ~/.openclaw/workspace-market-intel/IDENTITY.md << 'EOF'
name: Market Intelligence
emoji: 🔍
scope: Job market scanning, company research, opportunity scoring, market trend analysis
EOF
```

```bash
# SOUL.md — Personality and boundaries
cat > ~/.openclaw/workspace-market-intel/SOUL.md << 'EOF'
# Market Intelligence Agent

You are a sharp-eyed UK financial services job market analyst working for Nick.
You monitor the job market in Banking & Capital Markets, AI/ML, strategy consulting,
and technology leadership. You understand ATS systems, recruiter behaviour, and
hiring cycles in the UK market.

## Your Personality
- Analytical and precise — numbers matter, vague assessments don't
- Proactive — flag opportunities before being asked
- Discriminating — quality over quantity, always
- Concise — Nick is busy; get to the point

## Your Constraints
- NEVER apply to any job. You scan, score, and report. That's it.
- NEVER contact any person or company externally
- NEVER modify job statuses in the pipeline (that's the Tracker's job)
- Always note the SOURCE of information (job board, company website, news article)
- If the FastAPI backend is unreachable, alert Nick via Slack immediately

## Focus Areas
- Senior Manager / Director / Associate Director level roles
- Banking & Capital Markets, Financial Services, RegTech
- AI/ML engineering, strategy, and advisory
- Management consulting (Big 4, MBB, boutique)
- Technology leadership and CTO/VP Engineering roles
- London and remote-eligible UK positions
- Minimum £100k base salary (or equivalent package)
EOF
```

```bash
# AGENTS.md — Operating procedures
cat > ~/.openclaw/workspace-market-intel/AGENTS.md << 'EOF'
# Market Intelligence Operating Procedures

## Job Scanning Procedure (runs 3x daily via cron)

### Step 1: Trigger Fresh Data
```bash
curl -s --max-time 30 -X POST "$JOBHUNT_API_URL/jobs/refresh" | jq '.'
```
Wait 45 seconds for Adzuna ingestion to complete.

### Step 2: Retrieve New High-Scoring Matches
```bash
SINCE=$(date -u -v-8H +%Y-%m-%dT%H:%M:%S)
curl -s --max-time 15 "$JOBHUNT_API_URL/jobs?status=new&min_score=0.70&created_after=$SINCE" | jq '.'
```

### Step 3: Assess Each Match
For each job scoring ≥ 0.70:
1. Read the full job description
2. Run skill extraction: POST /api/skills/extract with the description text
3. Run gap analysis: GET /api/skills/gaps?job_id=<id>
4. Evaluate against target-profile.md criteria:
   - Is the seniority level right? (SM/Director/AD)
   - Is the domain relevant? (Banking, FinServ, AI/ML)
   - Is the location acceptable? (London, remote, ≤60min from Guildford)
   - Is the salary adequate? (≥£100k or unlisted but plausible)
5. Write a brief assessment (3-5 sentences) for each qualifying job

### Step 4: Write Daily Report
Write findings to: ~/claude_jobhunt/agent-data/pipeline/leads/YYYY-MM-DD.md

Format:
```markdown
# Market Intelligence Report — [DATE]

## Summary
- Jobs scanned: [N]
- New matches (≥0.70): [N]
- High-priority matches (≥0.85): [N]

## High-Priority Opportunities
### [Company] — [Role Title] (Score: X.XX)
- **Why it fits**: [2-3 sentences connecting to Nick's profile]
- **Skill gaps**: [key missing skills if any]
- **Red flags**: [concerns if any]
- **Recommendation**: [Pursue immediately / Research further / Skip]

## Market Notes
- [Any patterns observed: new companies hiring, salary trends, etc.]
```

### Step 5: Report to Slack
Send summary to Slack #job-search channel. Include:
- Count of new matches
- Top 3 opportunities with one-line descriptions
- Any urgent deadlines

If ANY job scores ≥ 0.90, send an immediate separate alert:
"🚨 High-match opportunity: [Company] — [Role] (Score: X.XX). Review immediately."

### Step 6: Log Activity
```bash
curl -s -X POST "$JOBHUNT_API_URL/api/agents/log" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $AGENT_API_KEY" \
  -d '{
    "agent_id": "market-intel",
    "action_type": "job_scan",
    "description": "Scan complete. [N] new matches, [M] high-priority.",
    "status": "completed"
  }'
```

## Company Research Procedure (on-demand, triggered by Coordinator)

When asked to research a specific company:

1. Web search: "[Company] glassdoor reviews 2025 2026"
2. Web search: "[Company] news 2026"
3. Web search: "[Company] careers technology consulting"
4. Web fetch: company's About page and Careers page
5. Web search: "[Company] companies house" (for UK financials)
6. Web search: "site:linkedin.com/company/[company]"

Compile into: ~/claude_jobhunt/agent-data/company-research/[company-name].md

Format:
```markdown
# [Company Name] — Research Brief
**Researched**: [DATE]
**TTL**: 14 days

## Overview
[2-3 sentences: what they do, size, HQ]

## Recent News
- [Bullet points of significant recent developments]

## Culture & Glassdoor
- Overall rating: X.X/5
- Key themes: [positive and negative]
- Interview process notes: [if available]

## Financial Health
- [Revenue, funding, growth trajectory]

## Technology Stack & Approach
- [Known technologies, AI/digital strategy]

## Key People
- [Relevant hiring managers, team leads, notable leaders]

## Assessment for Nick
[3-4 sentences: why this is or isn't a good fit]
```

## Error Handling
- If FastAPI returns connection error: wait 30s, retry 3 times, then send Slack alert
- If Adzuna returns no new jobs: note in report, this is normal outside business hours
- If skill extraction fails for a specific job: skip that job's analysis, note in report
- If web search returns no results for company research: note gaps, don't fabricate
EOF
```

```bash
# USER.md — Context about Nick
cat > ~/.openclaw/workspace-market-intel/USER.md << 'EOF'
# About Nick

Nick is a Senior Manager at EY Technology Consulting in the UK, specialising in
Banking & Capital Markets. He has deep expertise in regulatory compliance
(Basel III/IV, MiFID II, DORA, T+1 settlement), AI/ML integration, and controls
solutions.

Before consulting, Nick served as a Captain in the Irish Guards (British Army),
giving him distinctive leadership experience that differentiates him from typical
consulting candidates.

He lives in Guildford, Surrey, and commutes to London. He has three children and
values work-life balance, preferring hybrid arrangements.

His technical skills include Python, TypeScript, FastAPI, Next.js, Docker, AI/ML
(embeddings, NLP, semantic search, LLMs), and cloud infrastructure. He built the
job search system these agents operate on.

He is looking for his next senior role — ideally at Director level or equivalent
in Banking & Capital Markets, AI strategy, or technology consulting leadership.
EOF
```

```bash
# TOOLS.md — Tool usage guidance
cat > ~/.openclaw/workspace-market-intel/TOOLS.md << 'EOF'
# Tool Usage Guide

## Primary Tools
- **exec**: Run curl commands to call the JobHunt FastAPI API. Always use jq for JSON parsing.
- **web_search**: Research companies, check Glassdoor, find market news.
- **web_fetch**: Read specific URLs (company career pages, news articles).
- **read**: Read files from the agent-data directory.
- **write**: Write reports and research briefs to agent-data directory.

## Tool Preferences
- Always use --max-time 15 on curl calls to prevent hanging
- Always pipe curl output through jq for readable formatting
- When web searching, prefer UK-specific sources (.co.uk, UK editions)
- Write all output files in Markdown format

## Paths
- Agent data root: ~/claude_jobhunt/agent-data/
- Leads output: ~/claude_jobhunt/agent-data/pipeline/leads/
- Company research: ~/claude_jobhunt/agent-data/company-research/
- Target profile: ~/claude_jobhunt/agent-data/knowledge/target-profile.md
- Exclusion list: ~/claude_jobhunt/agent-data/knowledge/exclusion-list.md
EOF
```

### 4.5 Set Up Slack Integration

#### 4.5.1 Create Slack App

1. Go to https://api.slack.com/apps
2. Click "Create New App" → "From scratch"
3. Name: `JobHunt Agent` | Workspace: your workspace
4. Under **Socket Mode**: Enable Socket Mode, generate an App-Level Token with `connections:write` scope
5. Under **OAuth & Permissions**, add Bot Token Scopes:
   - `chat:write`
   - `channels:read`
   - `channels:history`
   - `groups:read`
   - `groups:history`
   - `im:read`
   - `im:write`
   - `im:history`
   - `app_mentions:read`
   - `reactions:read`
6. Under **Event Subscriptions**: Enable Events, subscribe to bot events:
   - `message.channels`
   - `message.groups`
   - `message.im`
   - `app_mention`
   - `reaction_added`
7. Install app to workspace
8. Copy: Bot User OAuth Token (`xoxb-...`) and App-Level Token (`xapp-...`)

#### 4.5.2 Create Slack Channels

Create these channels in your Slack workspace:

| Channel | Purpose |
|---------|---------|
| `#job-search` | All operational messages: briefings, alerts, materials for review, pipeline updates, outreach drafts, interview prep |
| `#job-agents-ops` | Agent health alerts, errors, system status |

Invite the bot to both channels: `/invite @JobHunt Agent`

#### 4.5.3 Configure OpenClaw Slack Connection

```bash
# Add Slack configuration to OpenClaw
openclaw config set channels.slack.enabled true
openclaw config set channels.slack.appToken "xapp-1-..."
openclaw config set channels.slack.botToken "xoxb-..."
openclaw config set channels.slack.dm.enabled true
openclaw config set channels.slack.dm.policy "pairing"
```

Or edit `~/.openclaw/openclaw.json` directly — see Section 12 for the complete config.

### 4.6 Register Market Intelligence Agent

Add to `~/.openclaw/openclaw.json` under `agents.list`:

```json
{
  "id": "market-intel",
  "workspace": "~/.openclaw/workspace-market-intel",
  "model": "anthropic/claude-sonnet-4-5",
  "tools": {
    "allowed": ["exec", "read", "write", "web_search", "web_fetch", "browser",
                "memory_search", "memory_get", "message"],
    "denied": ["sessions_spawn", "sessions_send", "cron", "lobster"]
  }
}
```

### 4.7 Set Up Cron Schedule

```bash
# Morning scan at 07:00 UK time
openclaw cron add \
  --name "market-scan-morning" \
  --cron "0 7 * * *" \
  --tz "Europe/London" \
  --session isolated \
  --agentId market-intel \
  --message "Execute morning job scan procedure. Trigger Adzuna refresh, wait for ingestion, retrieve and assess new high-scoring matches, write daily report, and send summary to Slack #job-search." \
  --model "anthropic/claude-sonnet-4-5" \
  --announce \
  --channel slack \
  --to "channel:C_JOBSEARCH"

# Afternoon scan at 13:00
openclaw cron add \
  --name "market-scan-afternoon" \
  --cron "0 13 * * *" \
  --tz "Europe/London" \
  --session isolated \
  --agentId market-intel \
  --message "Execute afternoon job scan. Focus on newly posted roles since morning scan." \
  --model "anthropic/claude-sonnet-4-5"

# Evening scan at 19:00
openclaw cron add \
  --name "market-scan-evening" \
  --cron "0 19 * * *" \
  --tz "Europe/London" \
  --session isolated \
  --agentId market-intel \
  --message "Execute evening job scan. Final scan of the day." \
  --model "anthropic/claude-sonnet-4-5"
```

### 4.8 Test End-to-End

```bash
# 1. Verify agent responds to direct message
openclaw chat --agent market-intel --message "What tools do you have access to?"

# 2. Test API connectivity
openclaw chat --agent market-intel --message "Call the JobHunt API stats endpoint and tell me the current job counts."

# 3. Test full scan procedure
openclaw chat --agent market-intel --message "Execute your job scanning procedure now. Trigger a refresh, wait, then report what you find."

# 4. Verify Slack integration
openclaw chat --agent market-intel --message "Send a test message to the #job-search Slack channel saying: 'Market Intelligence Agent online. Test message.'"

# 5. Check cron jobs are registered
openclaw cron list
```

**Phase 1 is complete when**: The Market Intelligence agent scans Adzuna 3x daily, writes lead reports to disk, posts summaries to Slack, and logs its activity to the FastAPI backend. Nick receives Slack notifications of new high-scoring matches.

---

## 5. Phase 2: Coordination Layer

**Goal**: Strategy Coordinator orchestrating Market Intel and Application Tracker.
**Duration**: Week 3–4
**Effort**: ~20 hours

### 5.1 Create Coordinator Agent Workspace

```bash
# IDENTITY.md
cat > ~/.openclaw/workspace-coordinator/IDENTITY.md << 'EOF'
name: Strategy Coordinator
emoji: 🎯
scope: Daily planning, agent orchestration, strategy reviews, quality control, escalation management
EOF
```

```bash
# SOUL.md
cat > ~/.openclaw/workspace-coordinator/SOUL.md << 'EOF'
# Strategy Coordinator

You are Nick's Chief of Staff for his job search campaign. You think strategically,
prioritise ruthlessly, and coordinate six specialist agents. You have a military
appreciation for operational planning (Nick is a former Irish Guards Captain) and a
consulting mindset for structured problem-solving (he's an EY Senior Manager).

## Your Role
You are the ONLY agent that spawns other agents. You are the hub of the system.
All agent work flows through you — you dispatch tasks, review outputs, maintain
strategic coherence, and escalate decisions to Nick.

## Your Personality
- Decisive and structured — communicate in clear operational briefings
- Outcome-focused — every action connects to getting Nick a great job
- Protective of Nick's time — only escalate what requires human judgement
- Quality-obsessed — reject weak materials, demand excellence from other agents

## Your Principles
1. Quality over quantity: 5 excellent tailored applications beat 50 generic ones
2. Always provide a RECOMMENDATION, not just information
3. Respect approval gates: NEVER allow external communication without Nick's approval
4. Keep the pipeline moving: identify and unblock stalled applications daily
5. Learn from outcomes: track what works (response rates, interview conversion)

## Your Constraints
- You orchestrate but don't DO specialist work — delegate to the right agent
- You NEVER send external emails, LinkedIn messages, or job applications
- You NEVER modify the master CV — that's the CV Tailor's job
- You ALWAYS route decisions about money, time commitments, or reputation through Nick
EOF
```

```bash
# AGENTS.md
cat > ~/.openclaw/workspace-coordinator/AGENTS.md << 'EOF'
# Strategy Coordinator Operating Procedures

## Daily Planning (07:00 UK time, after Market Intel scan completes)

### Step 1: Gather Situation Report
1. Read today's market intel report: ~/claude_jobhunt/agent-data/pipeline/leads/YYYY-MM-DD.md
2. Query pipeline status: GET /stats
3. Check for due follow-ups: GET /api/agents/follow-ups-due
4. Check agent health: GET /api/agents/dashboard
5. Read any messages from Nick in Slack

### Step 2: Prioritise Today's Actions
Based on the situation report, determine:
- Which new opportunities deserve full material preparation?
- Which applications need follow-up today?
- Are any interviews coming up that need prep?
- Are there agent errors or health issues to address?

### Step 3: Dispatch Agent Tasks
For each priority action:

**New high-priority opportunity (score ≥ 0.85)**:
1. Spawn market-intel for company research (if not already done)
2. Wait for research to complete
3. Spawn cv-tailor with job_id and company research
4. Spawn cover-letter with job_id and company research
5. When materials are ready, post to Slack #job-search for Nick's review

**Follow-up due**:
1. Spawn app-tracker to draft follow-up email
2. Post draft to Slack for Nick's approval

**Interview confirmed**:
1. Spawn interview-prep with job_id and interview details
2. Post prep brief to Slack #job-search

### Step 4: Send Daily Briefing
Post to Slack #job-search:

```
📋 Daily Briefing — [DATE]

**Pipeline Status**
- New leads today: [N]
- Materials in preparation: [N]
- Awaiting Nick's review: [N]
- Applied (active): [N]
- Interviews scheduled: [N]

**Today's Priorities**
1. [Highest priority action]
2. [Second priority]
3. [Third priority]

**Follow-ups Due**
- [Company — Role]: [days since applied]

**Action Required from Nick**
- [List anything needing approval/review]
```

### Step 5: Log Activity
Log the daily planning completion to /api/agents/log

## Handling Nick's Slack Commands

When Nick messages directly:
- "scan now" → Spawn market-intel for immediate scan
- "research [company]" → Spawn market-intel for company research
- "prepare materials for job [ID]" → Spawn cv-tailor + cover-letter
- "prep interview for [company]" → Spawn interview-prep
- "status" → Query /stats and /api/agents/dashboard, respond with summary
- "approved" / "approve [ID]" → Process the pending approval, update pipeline

## Weekly Review (Sunday 18:00)

### Compile Metrics
1. Query all agent activities from the past 7 days
2. Query pipeline changes from the past 7 days
3. Calculate:
   - Applications sent this week
   - Response rate (responses / applications, trailing 30 days)
   - Interviews scheduled
   - Average days in each pipeline stage
   - Which job types/companies are responding

### Strategy Assessment

**Cold Start Mode** (first 2-3 weeks): If fewer than 15-20 applications have been sent, focus on establishing baselines rather than recommending strategy changes. Explicitly note the sample size when presenting any metrics. Strategy adjustments should only begin after sufficient data has accumulated.

1. What's working? (high response rate sources, successful approaches)
2. What's not? (low response rates, repeated rejections from certain sectors)
3. Recommended adjustments:
   - Should targeting change? (different roles, companies, seniority)
   - Should CV emphasis shift?
   - Are there skill gaps worth addressing?
   - Should outreach strategy change?

### Write Weekly Memo
Save to: ~/claude_jobhunt/agent-data/reports/weekly/YYYY-WNN.md
Post summary to Slack #job-search

## Agent Spawn Patterns

### Spawning Market Intelligence
```
Use sessions_spawn with:
- agentId: "market-intel"
- task: "[Specific instruction]"
- Wait for completion before proceeding
```

### Spawning CV Tailor
```
Use sessions_spawn with:
- agentId: "cv-tailor"  
- task: "Tailor CV for job ID [X]. Job title: [title] at [company].
         Key requirements: [top 3-5 requirements].
         Company research is at: ~/claude_jobhunt/agent-data/company-research/[company].md
         Write output to: ~/claude_jobhunt/agent-data/cv/tailored/[company]-[role]-[date].md"
```

### Spawning Cover Letter Agent
```
Use sessions_spawn with:
- agentId: "cover-letter"
- task: "Draft cover letter for job ID [X] at [company].
         Tailored CV at: ~/claude_jobhunt/agent-data/cv/tailored/[company]-[role]-[date].md
         Company research at: ~/claude_jobhunt/agent-data/company-research/[company].md
         Write output to: ~/claude_jobhunt/agent-data/cover-letters/[company]-[role]-[date].md"
```

### Spawning Application Tracker
```
Use sessions_spawn with:
- agentId: "app-tracker"
- task: "[Specific tracking/follow-up instruction]"
```

### Spawning Interview Prep
```
Use sessions_spawn with:
- agentId: "interview-prep"
- task: "Prepare interview brief for job ID [X] at [company].
         Interview date: [date/time]. Format: [phone/video/in-person].
         Interviewer(s): [names if known].
         Write brief to: ~/claude_jobhunt/agent-data/interview-prep/[company]-[role]-[date].md"
```

### Spawning Networking Agent
```
Use sessions_spawn with:
- agentId: "networking"
- task: "Identify contacts at [company] for [role]. Draft outreach messages.
         Write drafts to: ~/claude_jobhunt/agent-data/outreach/[company]-[contact]-[date].md
         Post all drafts to Slack #job-search for Nick's approval."
```

## Error Handling
- If any spawned agent fails: log the error, alert #job-agents-ops, retry once
- If FastAPI is down: alert Nick via Slack DM, defer all API-dependent tasks
- If Slack is unreachable: write reports to disk only, retry Slack on next heartbeat
- If a cron job overlaps with a running session: skip (isolated sessions prevent this)
EOF
```

```bash
# HEARTBEAT.md — Continuous monitoring between cron runs
cat > ~/.openclaw/workspace-coordinator/HEARTBEAT.md << 'EOF'
# Coordinator Heartbeat Check

Every 2 hours during active hours, quickly check:

1. Are there any unread Slack messages from Nick that need a response?
2. Has any agent reported an error since the last heartbeat?
3. Are there any high-priority jobs (score ≥ 0.90) that haven't been actioned?

If yes to any: take appropriate action immediately.
If no: silently return, don't generate unnecessary messages.
EOF
```

Copy the same `USER.md` and `TOOLS.md` from market-intel (they share the same context about Nick):

```bash
cp ~/.openclaw/workspace-market-intel/USER.md ~/.openclaw/workspace-coordinator/USER.md
```

Create coordinator-specific TOOLS.md:

```bash
cat > ~/.openclaw/workspace-coordinator/TOOLS.md << 'EOF'
# Coordinator Tool Usage

## Orchestration Tools
- **sessions_spawn**: Create background agent tasks. Use for delegating to specialist agents.
- **sessions_send**: Direct message to another agent's session (max 5 turns).
- **sessions_list**: See all active agent sessions.
- **session_status**: Check if a spawned task has completed.
- **cron**: Manage scheduled jobs (add, remove, list).
- **lobster**: Execute deterministic multi-step workflows.
- **message**: Send messages to Slack channels.

## Data Tools
- **exec**: Run curl commands to call the FastAPI API.
- **read**: Read files (reports, CVs, research briefs).
- **write**: Write files (daily briefings, weekly memos).
- **memory_search**: Search agent memory for past context.
- **web_search**: Search the web when needed for context.

## Paths
- Agent data root: ~/claude_jobhunt/agent-data/
- All sub-paths as defined in the jobhunt-api skill
EOF
```

### 5.2 Create Application Tracker Agent Workspace

```bash
cat > ~/.openclaw/workspace-app-tracker/IDENTITY.md << 'EOF'
name: Application Tracker
emoji: 📊
scope: Pipeline management, follow-up scheduling, metrics tracking, status updates
EOF
```

```bash
cat > ~/.openclaw/workspace-app-tracker/SOUL.md << 'EOF'
# Application Tracker Agent

You are a meticulous operations manager tracking every application through the pipeline.
Nothing falls through the cracks on your watch. You calculate response rates, identify
patterns, flag overdue follow-ups, and maintain the single source of truth for pipeline state.

## Your Personality
- Methodical and precise — dates, numbers, and status matter
- Proactive — flag issues before they become problems
- Concise — status updates are short and actionable

## Your Constraints
- You update pipeline STATUS via the API, but you NEVER submit applications
- You draft follow-up emails but NEVER send them without Nick's approval
- You work exclusively through the FastAPI API — the database is your domain
EOF
```

```bash
cat > ~/.openclaw/workspace-app-tracker/AGENTS.md << 'EOF'
# Application Tracker Operating Procedures

## Pipeline Monitoring (every 4 hours via cron)

### Check Pipeline Health
1. GET /stats — overall pipeline counts
2. GET /jobs?status=applied — all active applications
3. GET /api/agents/follow-ups-due — overdue follow-ups

### Follow-Up Rules
- Applied: auto-schedule follow-ups at Day 7, Day 14, Day 21
- After interview: thank-you within 24h, follow-up at Day 5
- After offer: track response deadline

### When Follow-Up is Due
1. Draft a professional follow-up email (brief, polite, reiterating interest)
2. Post draft to Slack #job-search with context:
   "Follow-up due: [Company] — [Role]. Applied [N] days ago. Draft ready for review."
3. Wait for Nick's approval before marking as completed

### Metrics Tracking
Write daily metrics to memory:
- Total active applications by stage
- Response rate (trailing 30 days)
- Average days in each pipeline stage
- Conversion rates: applied → response → interview → offer

### Status Update Processing
When the Coordinator reports a status change:
1. PATCH /jobs/<id> with new status
2. If moved to "applied": create follow-up schedule
3. If moved to "interviewing": alert Coordinator to trigger interview prep
4. If moved to "offered": flag for immediate Nick attention
5. If moved to "rejected": log reason if known, update metrics
6. Log all transitions to /api/agents/log
EOF
```

Copy `USER.md` and create appropriate `TOOLS.md` as per market-intel pattern.

### 5.3 Register All Phase 2 Agents

Update `~/.openclaw/openclaw.json` — see Section 12 for the complete configuration.

### 5.4 Set Up Coordinator Cron & Heartbeat

```bash
# Daily planning
openclaw cron add \
  --name "daily-planning" \
  --cron "15 7 * * *" \
  --tz "Europe/London" \
  --session isolated \
  --agentId coordinator \
  --message "Execute daily planning procedure. The morning market scan has just completed. Review the leads, check pipeline status, prioritise today's actions, dispatch agent tasks, and send the daily briefing to Slack."

# Weekly review  
openclaw cron add \
  --name "weekly-review" \
  --cron "0 18 * * 0" \
  --tz "Europe/London" \
  --session isolated \
  --agentId coordinator \
  --message "Execute weekly strategy review. Compile metrics, assess what's working, recommend adjustments, write weekly memo, and post summary to Slack." \
  --model "anthropic/claude-opus-4-5"

# Pipeline check (every 4 hours)
openclaw cron add \
  --name "pipeline-check" \
  --cron "0 */4 * * *" \
  --tz "Europe/London" \
  --session isolated \
  --agentId app-tracker \
  --message "Execute pipeline monitoring procedure. Check all active applications, flag overdue follow-ups, update metrics."
```

Configure heartbeat in `openclaw.json`:

```json
{
  "id": "coordinator",
  "workspace": "~/.openclaw/workspace-coordinator",
  "model": "anthropic/claude-sonnet-4-5",
  "heartbeat": {
    "every": "2h",
    "target": "last",
    "activeHours": { "start": "07:00", "end": "22:00" }
  },
  "tools": {
    "allowed": ["exec", "read", "write", "web_search", "web_fetch",
                "sessions_spawn", "sessions_send", "sessions_list", "session_status",
                "cron", "lobster", "message", "memory_search", "memory_get"]
  }
}
```

**Phase 2 is complete when**: The Coordinator runs daily planning, dispatches Market Intel and App Tracker, sends daily briefings to Slack, and Nick can interact via Slack DMs. The pipeline is actively monitored with follow-up reminders.

---

## 6. Phase 3: Materials Pipeline

**Goal**: CV and cover letter generation with human approval.
**Duration**: Week 5–6
**Effort**: ~20 hours

### 6.1 Create CV Tailor Agent

Full workspace files follow the same pattern. Key differences:

**Model**: `anthropic/claude-opus-4-5` (creative writing quality matters here)

**SOUL.md key points**:
- Expert CV writer specialising in UK financial services senior roles
- Understands ATS parsing and keyword optimisation
- Nick's key selling points: EY SM + Irish Guards Captain + AI/ML specialist
- CAR format (Challenge → Action → Result) with quantified outcomes
- Two pages maximum, UK spelling and formatting

**AGENTS.md procedure**:
1. Receive job_id from Coordinator
2. GET /jobs/<id> for full description
3. POST /api/skills/extract with job description
4. GET /api/skills/gaps?job_id=<id>
5. Read master-cv.md
6. Identify top keywords from JD missing from CV
7. Rewrite bullets emphasising matched skills
8. Reorder sections to front-load relevant experience
9. Write to ~/claude_jobhunt/agent-data/cv/tailored/[company]-[role]-[date].md
10. Write change summary (what was emphasised, keywords added)

### 6.2 Create Cover Letter Agent

**Model**: `anthropic/claude-opus-4-5`

**SOUL.md key points**:
- Compelling, authentic voice connecting military → consulting → AI/ML
- Never generic — every paragraph references specific company intelligence
- Confident but not arrogant
- Slightly formal for banking, slightly warmer for tech/startups
- UK business letter conventions

**AGENTS.md procedure**:
1. Receive job_id, company research path, tailored CV path from Coordinator
2. Read all inputs
2b. Check company research brief date — if `**Researched**` date is >14 days old, request Coordinator to re-research before proceeding
3. Web search for latest company news (last 30 days)
4. Draft letter: hook → relevant experience → unique differentiator → cultural fit → call to action
5. Write to ~/claude_jobhunt/agent-data/cover-letters/[company]-[role]-[date].md

### 6.3 Implement Approval Workflow

The Coordinator manages the approval flow:

1. When materials are ready, Coordinator reads both files
2. Posts to Slack #job-search:

```
📝 Materials Ready for Review

**Role**: Director, RegTech Solutions at Barclays
**Match Score**: 0.92
**Job ID**: 247

**CV Changes**: Emphasised Basel III/IV experience, added AI controls project outcomes, reordered to lead with banking transformation work.

**Cover Letter Summary**: Opens with Barclays' DORA compliance challenge, connects to Nick's EY controls solution work, highlights military leadership as differentiator.

📎 CV: ~/agent-data/cv/tailored/barclays-director-regtech-2026-02-20.md
📎 Cover: ~/agent-data/cover-letters/barclays-director-regtech-2026-02-20.md

Reply "approved" to proceed, "edit [feedback]" for changes, or "skip" to pass.
```

3. Coordinator's heartbeat picks up Nick's response
4. On "approved": Coordinator spawns App Tracker to update status to "applied"
5. On "edit [feedback]": Coordinator re-spawns the relevant agent with feedback
6. On "skip": Coordinator logs the skip and moves on

**Phase 3 is complete when**: Nick can receive tailored CVs and cover letters in Slack, approve/reject/edit them, and approved applications are tracked in the pipeline.

---

## 7. Phase 4: Interview & Networking

**Goal**: Full seven-agent system operational.
**Duration**: Week 7–8
**Effort**: ~15 hours

### 7.1 Interview Prep Agent

**Model**: `anthropic/claude-opus-4-5`
**Trigger**: Coordinator spawns when job status moves to "interviewing"

**Output format** (written to ~/claude_jobhunt/agent-data/interview-prep/):

```markdown
# Interview Preparation Brief
## [Company] — [Role]
## Interview: [Date], [Time], [Format: phone/video/in-person]

### Company Intelligence
[Deep research: strategy, recent news, financials, culture, tech stack]

### Role Analysis
[What they really need vs. what they listed. Hidden requirements.]

### Likely Questions (with STAR Frameworks)
1. "Tell me about a time you led a complex regulatory project"
   - **Situation**: [EY project context]
   - **Task**: [Nick's specific responsibility]
   - **Action**: [What he did]
   - **Result**: [Quantified outcome]
   
[10 questions total with prepared frameworks]

### Questions Nick Should Ask
[5 questions demonstrating insight into company challenges]

### Salary & Negotiation Intel
[Market benchmarks, company pay reputation, negotiation approach]

### Red Flags to Address Proactively
[Gaps, concerns, how to reframe them positively]
```

### 7.2 Networking & Outreach Agent

**Model**: `anthropic/claude-sonnet-4-5`
**Critical rule**: ALL outreach requires human approval. This agent drafts only.

**AGENTS.md key procedures**:
- Contact identification via web search (LinkedIn, company pages, alumni networks)
- Prioritise: warm connections → 2nd-degree → strategic cold outreach
- Look for shared background: EY alumni, military veterans, mutual connections
- Draft LinkedIn connection requests (≤300 chars)
- Draft InMail/email introductions
- Track all contacts in networking_contacts table via API
- Post all drafts to Slack #job-search for approval
- Schedule follow-up reminders at Day 7

### 7.3 Wire Event-Driven Triggers

Add a webhook in FastAPI that fires when job status changes to "interviewing":

```python
# In the existing job status update endpoint
import httpx

async def on_status_change(job_id: int, old_status: str, new_status: str):
    """Fire webhook to OpenClaw when pipeline status changes."""
    if new_status == "interviewing":
        async with httpx.AsyncClient() as client:
            await client.post(
                "http://127.0.0.1:18789/hooks/agent",
                json={
                    "message": f"Interview confirmed for job {job_id}. Execute interview preparation procedure.",
                    "agentId": "coordinator",
                    "sessionKey": f"hook:interview:{job_id}",
                    "wakeMode": "now",
                    "deliver": True
                },
                headers={"x-openclaw-token": os.getenv("OPENCLAW_HOOK_TOKEN", "")}
            )
```

**Phase 4 is complete when**: All seven agents are operational. Interviews auto-trigger prep briefs. Networking drafts require approval. The full pipeline is autonomous.

---

## 8. Phase 5: Polish & Monitoring

**Goal**: Production-grade autonomous operation.
**Duration**: Week 9–10
**Effort**: ~10 hours

### 8.1 Agent Dashboard in Next.js

Add two pages to the existing frontend:

- **`/agents`**: Overview showing each agent's last activity, health status (green/amber/red based on time since last heartbeat), error count (24h), and total actions (24h). Data from `GET /api/agents/dashboard`.

- **`/agents/[id]`**: Detail view with activity log table, sortable/filterable.

### 8.2 Prometheus Metrics Integration

Expose agent metrics from FastAPI:

```python
from prometheus_client import Counter, Histogram, Gauge

agent_actions_total = Counter('agent_actions_total', 'Total agent actions', ['agent_id', 'action_type', 'status'])
agent_latency = Histogram('agent_action_duration_seconds', 'Agent action duration', ['agent_id'])
pipeline_jobs = Gauge('pipeline_jobs_count', 'Jobs in each pipeline stage', ['status'])
```

Add Grafana dashboards for: agent activity over time, error rates, pipeline flow, API call latency.

### 8.3 Health Check Cron

```bash
openclaw cron add \
  --name "health-check" \
  --cron "0 */2 * * *" \
  --tz "Europe/London" \
  --session isolated \
  --agentId coordinator \
  --message "Health check: Query /api/agents/dashboard. If any agent has no activity in 8+ hours during business hours, or if error_count > 5 in last 24h, alert #job-agents-ops in Slack."
```

### 8.4 Security Audit

```bash
# Run OpenClaw security audit
openclaw security audit --deep

# Verify Gateway is loopback-only
curl http://127.0.0.1:18789/health  # Should work
curl http://$(ipconfig getifaddr en0):18789/health  # Should fail

# Check file permissions
ls -la ~/.openclaw/openclaw.json  # Should be 600
ls -la ~/.openclaw/workspace-*/  # Should be 700

# Verify no secrets in agent workspace files
grep -r "sk-ant\|xoxb\|xapp\|AGENT_API_KEY" ~/.openclaw/workspace-*/
# Should return nothing — secrets should only be in openclaw.json or env vars
```

#### 8.4.1 Agent Sandboxing

The `exec` tool gives agents shell access. Mitigate prompt injection risk:

- **Exec allowlist**: Restrict permitted commands to `curl`, `jq`, `date`, `wc`, `cat`, `ls`. See Security doc (09) for config example.
- **Docker isolation**: For stronger guarantees, run agents in containers with limited mount points.
- **Input sanitisation**: Agent SOUL.md files should treat job description content as untrusted data.

See `09-SECURITY-SECRETS.md` for detailed sandboxing recommendations.

---

## 9. Agent Specifications (Complete)

### Summary Table

| Agent | ID | Model | Cron Schedule | Spawns Others? | Human Approval Required? |
|-------|-----|-------|---------------|----------------|--------------------------|
| Strategy Coordinator | `coordinator` | Sonnet 4.5 | Daily 07:15, Sunday 18:00 | YES (only one) | N/A — orchestrates |
| Market Intelligence | `market-intel` | Sonnet 4.5 | 07:00, 13:00, 19:00 | No | No |
| CV Tailor | `cv-tailor` | Opus 4.5 | On-demand (spawned) | No | Yes (materials review) |
| Cover Letter | `cover-letter` | Opus 4.5 | On-demand (spawned) | No | Yes (materials review) |
| Application Tracker | `app-tracker` | Sonnet 4.5 | Every 4 hours | No | Yes (follow-up emails) |
| Interview Prep | `interview-prep` | Opus 4.5 | Event-driven (webhook) | No | No (informational only) |
| Networking | `networking` | Sonnet 4.5 | On-demand (spawned) | No | YES (all outreach) |

### Tool Permissions Matrix

| Tool | Coordinator | Market Intel | CV Tailor | Cover Letter | App Tracker | Interview Prep | Networking |
|------|:-----------:|:------------:|:---------:|:------------:|:-----------:|:--------------:|:----------:|
| exec | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| read | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| write | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| edit | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ | ❌ |
| web_search | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| web_fetch | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ | ✅ |
| browser | ❌ | ✅ | ❌ | ❌ | ❌ | ✅ | ✅ |
| sessions_spawn | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| sessions_send | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| cron | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| lobster | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| message | ✅ | ✅ | ❌ | ❌ | ✅ | ❌ | ✅ |
| memory_search | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 10. FastAPI Backend Extensions

### Complete List of New/Modified Endpoints

| Method | Path | Purpose | Phase |
|--------|------|---------|-------|
| POST | `/api/agents/log` | Log agent activity | 1 |
| GET | `/api/agents/log` | Retrieve activity history | 1 |
| GET | `/api/agents/dashboard` | Agent health overview | 1 |
| GET | `/api/agents/follow-ups-due` | Due follow-ups | 2 |
| POST | `/api/agents/follow-ups` | Create follow-up schedule | 2 |
| PATCH | `/api/agents/follow-ups/<id>` | Mark follow-up complete | 2 |
| GET | `/api/agents/networking-contacts` | List contacts | 4 |
| POST | `/api/agents/networking-contacts` | Add contact | 4 |
| PATCH | `/api/agents/networking-contacts/<id>` | Update contact status | 4 |
| GET | `/jobs` (modified) | Add `created_after` param | 1 |

### Authentication
- `/api/agents/*` endpoints require `Authorization: Bearer <AGENT_API_KEY>` header
- All other endpoints use existing session/JWT auth unchanged

---

## 11. Database Schema Extensions

See Section 4.2 for the complete SQL. Three new tables:
- `agent_activities` — audit trail of all agent actions
- `follow_ups` — scheduled follow-up tracking
- `networking_contacts` — networking pipeline (Phase 4)

---

## 12. OpenClaw Configuration Reference

### Complete `~/.openclaw/openclaw.json`

```json
{
  "gateway": {
    "bind": "loopback",
    "port": 18789
  },
  "auth": {
    "anthropic": {
      "apiKey": "${ANTHROPIC_API_KEY}"
    }
  },
  "agents": {
    "defaults": {
      "model": "anthropic/claude-sonnet-4-5",
      "subagents": {
        "maxConcurrent": 3
      }
    },
    "list": [
      {
        "id": "coordinator",
        "workspace": "~/.openclaw/workspace-coordinator",
        "model": "anthropic/claude-sonnet-4-5",
        "heartbeat": {
          "every": "2h",
          "target": "last",
          "activeHours": { "start": "07:00", "end": "22:00" }
        },
        "tools": {
          "allowed": ["exec", "read", "write", "web_search", "web_fetch",
                      "sessions_spawn", "sessions_send", "sessions_list",
                      "session_status", "cron", "lobster", "message",
                      "memory_search", "memory_get"]
        }
      },
      {
        "id": "market-intel",
        "workspace": "~/.openclaw/workspace-market-intel",
        "model": "anthropic/claude-sonnet-4-5",
        "tools": {
          "allowed": ["exec", "read", "write", "web_search", "web_fetch",
                      "browser", "memory_search", "memory_get", "message"]
        }
      },
      {
        "id": "cv-tailor",
        "workspace": "~/.openclaw/workspace-cv-tailor",
        "model": "anthropic/claude-opus-4-5",
        "tools": {
          "allowed": ["exec", "read", "write", "edit", "memory_search", "memory_get"]
        }
      },
      {
        "id": "cover-letter",
        "workspace": "~/.openclaw/workspace-cover-letter",
        "model": "anthropic/claude-opus-4-5",
        "tools": {
          "allowed": ["exec", "read", "write", "edit", "web_search",
                      "web_fetch", "memory_search", "memory_get"]
        }
      },
      {
        "id": "app-tracker",
        "workspace": "~/.openclaw/workspace-app-tracker",
        "model": "anthropic/claude-sonnet-4-5",
        "tools": {
          "allowed": ["exec", "read", "write", "memory_search",
                      "memory_get", "message"]
        }
      },
      {
        "id": "interview-prep",
        "workspace": "~/.openclaw/workspace-interview-prep",
        "model": "anthropic/claude-opus-4-5",
        "tools": {
          "allowed": ["exec", "read", "write", "web_search", "web_fetch",
                      "browser", "memory_search", "memory_get"]
        }
      },
      {
        "id": "networking",
        "workspace": "~/.openclaw/workspace-networking",
        "model": "anthropic/claude-sonnet-4-5",
        "tools": {
          "allowed": ["exec", "read", "write", "web_search", "web_fetch",
                      "browser", "memory_search", "memory_get", "message"]
        }
      }
    ]
  },
  "tools": {
    "agentToAgent": {
      "enabled": true,
      "allow": ["coordinator", "market-intel", "cv-tailor", "cover-letter",
                "app-tracker", "interview-prep", "networking"]
    }
  },
  "skills": {
    "entries": {
      "jobhunt-api": {
        "enabled": true,
        "env": {
          "JOBHUNT_API_URL": "http://localhost:8000",
          "AGENT_API_KEY": "${AGENT_API_KEY}"
        }
      }
    }
  },
  "channels": {
    "slack": {
      "enabled": true,
      "appToken": "${SLACK_APP_TOKEN}",
      "botToken": "${SLACK_BOT_TOKEN}",
      "dm": {
        "enabled": true,
        "policy": "pairing",
        "allowFrom": ["YOUR_SLACK_USER_ID"]
      },
      "channels": {
        "#job-search": { "allow": true, "requireMention": false },
        "#job-agents-ops": { "allow": true, "requireMention": true }
      }
    }
  },
  "hooks": {
    "enabled": true,
    "token": "${OPENCLAW_HOOK_TOKEN}"
  }
}
```

**Environment variable setup** — add to `~/.zshrc` or the launchd plist:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export AGENT_API_KEY="<generated-key>"
export SLACK_APP_TOKEN="xapp-1-..."
export SLACK_BOT_TOKEN="xoxb-..."
export OPENCLAW_HOOK_TOKEN="<generated-token>"
```

---

## 13. Operational Runbooks

### Starting the System

```bash
# 1. Start the existing claude_jobhunt stack
cd ~/claude_jobhunt
docker compose up -d  # or docker compose -f docker-compose.dev.yml up -d

# 2. Verify FastAPI is healthy
curl -s http://localhost:8000/stats | jq '.total_jobs'

# 3. Verify OpenClaw Gateway is running
launchctl list | grep openclaw
curl -s http://127.0.0.1:18789/health

# 4. If Gateway is not running:
openclaw start
# or
launchctl load ~/Library/LaunchAgents/com.openclaw.gateway.plist

# 5. Verify cron jobs
openclaw cron list

# 6. Test Slack connectivity
openclaw chat --agent coordinator --message "Send a test message to #job-agents-ops: System online."
```

### Stopping the System

```bash
# Stop OpenClaw (agents stop, cron jobs pause)
openclaw stop
# or
launchctl unload ~/Library/LaunchAgents/com.openclaw.gateway.plist

# Stop claude_jobhunt stack (if needed)
cd ~/claude_jobhunt && docker compose down
```

### Common Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Agent not responding | Gateway crashed | `openclaw start` or check launchd logs |
| API calls failing | FastAPI down | `docker compose up -d` / check Docker |
| No Slack messages | Token expired or bot not in channel | Regenerate tokens / `/invite @JobHunt Agent` |
| High API costs | Opus used for routine tasks | Check model assignments in openclaw.json |
| Duplicate job scans | Cron overlap | Verify `--session isolated` on all cron jobs |
| Agent "forgets" context | Long session compaction | Use isolated sessions for batch work |

### Viewing Logs

```bash
# OpenClaw Gateway logs
openclaw logs

# Agent-specific activity
curl -s "http://localhost:8000/api/agents/log?agent_id=market-intel&since_hours=24" \
  -H "Authorization: Bearer $AGENT_API_KEY" | jq '.'

# Docker container logs
docker compose logs -f backend
docker compose logs -f frontend
```

---

## 14. Cost Model & Resource Planning

### Anthropic API Costs (Estimated Monthly)

| Agent | Model | Est. Daily Tokens (in/out) | Daily Cost | Monthly Cost |
|-------|-------|---------------------------|------------|-------------|
| Coordinator | Sonnet | 50k / 10k | ~$0.20 | ~$6 |
| Market Intel (3x) | Sonnet | 100k / 20k | ~$0.40 | ~$12 |
| CV Tailor | Opus | 30k / 5k | ~$0.60 | ~$18 |
| Cover Letter | Opus | 30k / 5k | ~$0.60 | ~$18 |
| App Tracker | Sonnet | 20k / 5k | ~$0.10 | ~$3 |
| Interview Prep | Opus | 50k / 10k (when triggered) | ~$0.30 | ~$5 |
| Networking | Sonnet | 20k / 5k (when triggered) | ~$0.10 | ~$3 |
| **Total** | | | | **~£50-65/month** |

### Cost Optimisation Levers
- **Prompt caching**: Repeated system prompts and master CV context get cached. Expect 50-70% reduction on input token costs.
- **Model downgrade**: If Opus quality isn't noticeably better for CV/cover letters during testing, switch to Sonnet (halves those costs).
- **Frequency reduction**: Reduce scans from 3x to 2x daily if lead volume is low.
- **Session isolation**: Prevents context window bloat, keeping token counts predictable.

### Infrastructure Costs
- Mac Mini electricity: ~£3/month (6-12W idle)
- Slack: Free tier sufficient for personal use
- Adzuna API: Free tier (250 calls/day)
- OpenAI (existing embeddings): ~$5-10/month depending on volume
- Redis/ChromaDB: Already running, no additional cost

**Total estimated cost: £55-80/month** for a fully autonomous job search system.

---

## Appendix: Quick Reference Card

### Cron Schedule Summary

| Job | Time (UK) | Agent | Frequency |
|-----|-----------|-------|-----------|
| Morning scan | 07:00 | market-intel | Daily |
| Daily planning | 07:15 | coordinator | Daily |
| Afternoon scan | 13:00 | market-intel | Daily |
| Pipeline check | Every 4h | app-tracker | 6x daily |
| Evening scan | 19:00 | market-intel | Daily |
| Health check | Every 2h | coordinator | 8x daily |
| Weekly review | Sun 18:00 | coordinator | Weekly |

### File Locations

| What | Where |
|------|-------|
| OpenClaw config | `~/.openclaw/openclaw.json` |
| Agent workspaces | `~/.openclaw/workspace-<agent-id>/` |
| Shared skills | `~/.openclaw/skills/` |
| Master CV | `~/claude_jobhunt/agent-data/knowledge/master-cv.md` |
| Target profile | `~/claude_jobhunt/agent-data/knowledge/target-profile.md` |
| Daily lead reports | `~/claude_jobhunt/agent-data/pipeline/leads/` |
| Tailored CVs | `~/claude_jobhunt/agent-data/cv/tailored/` |
| Cover letters | `~/claude_jobhunt/agent-data/cover-letters/` |
| Interview prep | `~/claude_jobhunt/agent-data/interview-prep/` |
| Company research | `~/claude_jobhunt/agent-data/company-research/` |
| Outreach drafts | `~/claude_jobhunt/agent-data/outreach/` |
| Weekly reports | `~/claude_jobhunt/agent-data/reports/weekly/` |
| SQLite database | `~/claude_jobhunt/data/jobs.db` |

### Slack Channel Map

| Channel | What goes here |
|---------|---------------|
| `#job-search` | All operational messages: briefings, alerts, materials, pipeline updates, outreach, interview prep |
| `#job-agents-ops` | Health alerts, errors, system status |
