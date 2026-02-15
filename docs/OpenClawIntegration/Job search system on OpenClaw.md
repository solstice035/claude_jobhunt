
# Building a seven-agent job search system on OpenClaw

**OpenClaw can serve as the autonomous orchestration layer atop Nick’s existing FastAPI/Next.js job search application, with all seven agents running on a single Gateway process on his Mac Mini.** The platform’s skill system  (SKILL.md files instructing agents to call APIs via `exec` + `curl`), built-in cron/heartbeat scheduling, Lobster workflow engine with approval gates, and native Slack integration  provide every primitive needed. The critical architectural insight: OpenClaw agents don’t run code — they follow natural-language instructions using built-in tools  like `exec`, `web_fetch`, and `browser`. This means each agent is a workspace directory containing Markdown files that define its identity, behavior, and API-calling patterns, all managed by one always-on Node.js Gateway daemon.

OpenClaw (formerly Clawdbot, then Moltbot) is Peter Steinberger’s  MIT-licensed open-source project  with **117,000+ GitHub stars**, making it one of the fastest-growing AI projects of early 2026. It runs as a self-hosted Gateway process that connects messaging platforms to Claude-powered agents capable of executing real-world tasks.   Nick’s existing system — with its FastAPI REST endpoints, hybrid search, ESCO skill extraction, and pipeline management — becomes the “brain” that OpenClaw agents interact with through HTTP calls.

-----

## A. OpenClaw platform architecture and capabilities

### Gateway-centric design

OpenClaw runs a **single Gateway process** (Node.js ≥22, default port 18789) that owns all session state, routing, tools, and scheduling.  The architecture separates the interface layer (messaging platforms) from the agent runtime (where intelligence lives):

```
Slack / Discord / WhatsApp / Telegram / WebChat / iMessage
                    │
                    ▼
        ┌─────────────────────────┐
        │       Gateway           │
        │   (control plane)       │
        │  ws://127.0.0.1:18789   │
        └───────────┬─────────────┘
                    │
            ├── Agent Runtime (Pi)   ← LLM loop + tool execution
            ├── Cron / Heartbeat     ← Scheduled automation
            ├── Lobster Engine       ← Deterministic workflows
            ├── Webhook Handler      ← Event-driven triggers
            └── Session Manager      ← Multi-agent routing
```

On macOS, the Gateway installs as a **launchd daemon**  at `~/Library/LaunchAgents/com.openclaw.gateway.plist`, auto-starting on login and surviving terminal closure.  Installation is straightforward: `npm install -g openclaw@latest && openclaw onboard --install-daemon`. 

### How agents are defined

Each agent is a workspace directory containing Markdown files that shape its behavior. The **entire agent definition is text files** — no code compilation, no class hierarchies:

```
~/.openclaw/workspace-<agentId>/
├── SOUL.md        # Personality, voice, boundaries
├── AGENTS.md      # Behavioral instructions, operating procedures
├── USER.md        # Context about Nick (background, preferences)
├── TOOLS.md       # Tool usage guidance
├── IDENTITY.md    # Name, emoji, role scope
├── HEARTBEAT.md   # Scheduled check-in instructions
├── MEMORY.md      # Curated long-term knowledge
├── memory/        # Date-stamped daily logs (YYYY-MM-DD.md)
└── skills/        # Per-agent custom skills (SKILL.md files)
```

Multi-agent configuration in `~/.openclaw/openclaw.json` registers all agents under one Gateway: 

```json
{
  "agents": {
    "list": [
      { "id": "coordinator", "workspace": "~/.openclaw/workspace-coordinator" },
      { "id": "market-intel", "workspace": "~/.openclaw/workspace-market-intel" },
      { "id": "cv-tailor", "workspace": "~/.openclaw/workspace-cv-tailor" },
      { "id": "cover-letter", "workspace": "~/.openclaw/workspace-cover-letter" },
      { "id": "app-tracker", "workspace": "~/.openclaw/workspace-app-tracker" },
      { "id": "interview-prep", "workspace": "~/.openclaw/workspace-interview-prep" },
      { "id": "networking", "workspace": "~/.openclaw/workspace-networking" }
    ]
  }
}
```

### The three-layer tool system

OpenClaw provides **25 built-in tools** across three layers:  

**Layer 1 — Core tools** give agents the ability to act: `read`, `write`, `edit`, `exec` (run any shell command), `web_search`, `web_fetch`,  and `browser`  (CDP-controlled Chromium with click/fill/screenshot).  The `exec` tool is the workhorse for API integration — agents run `curl` commands to call Nick’s FastAPI endpoints.

**Layer 2 — Automation tools** enable autonomous behavior:  `cron` (scheduled jobs  with 5-field cron expressions), `message` (send to any channel), `sessions_spawn` (create sub-agent sessions), `sessions_send` (agent-to-agent messaging), and `lobster` (deterministic workflow pipelines with approval gates).

**Layer 3 — Skills** (SKILL.md files) are natural-language instructions injected into the agent’s system prompt that teach it how to combine tools for specific tasks.  Skills follow the **AgentSkills spec**  and can be per-agent (workspace), shared (managed), bundled, or installed from ClawHub’s 5,700+ community registry.  

### Multi-agent coordination

Agents communicate through three mechanisms. **`sessions_spawn`** creates non-blocking background sub-agent runs that report results back to the caller  — ideal for the Strategy Coordinator delegating tasks. **`sessions_send`** enables direct message ping-pong between agents  (capped at 5 turns to prevent loops).  **Webhooks** allow external systems (Nick’s FastAPI) to trigger agent actions via HTTP POST. One important limitation: **sub-agents cannot spawn sub-sub-agents**  — only the top-level coordinator should spawn.

Agent-to-agent communication requires explicit enablement:

```json
{
  "tools": {
    "agentToAgent": {
      "enabled": true,
      "allow": ["coordinator", "market-intel", "cv-tailor", "cover-letter",
                "app-tracker", "interview-prep", "networking"]
    }
  }
}
```

### Memory and persistent state

Memory is **plain Markdown files on disk**  — no database, no proprietary format.   MEMORY.md holds curated long-term facts (injected into every session). Date-stamped `memory/YYYY-MM-DD.md` files serve as daily logs accessed on-demand via the `memory_search` tool, which performs hybrid **BM25 + vector similarity** search over a local SQLite index (`~/.openclaw/memory/{agentId}.sqlite`). Context compaction automatically triggers before the context window fills, with a pre-compaction memory flush that writes durable facts to disk.

### MCP support status

Native MCP client support is **not yet merged** into OpenClaw core as of February 2026  (multiple GitHub issues remain open). Community solutions exist: the `openclaw-mcp-plugin` by lunarpulse implements Streamable HTTP transport,  and PR #5121 merged MCP server support so OpenClaw can be called from MCP clients.  For Nick’s use case, the `exec` + `curl` pattern for API calls is more reliable than depending on MCP.

-----

## B. Seven-agent architecture design

### Shared infrastructure: the JobHunt API skill

Every agent that calls Nick’s FastAPI backend needs a common skill. This **shared skill** lives at `~/.openclaw/skills/jobhunt-api/SKILL.md` and is available to all agents:

```markdown
---
name: jobhunt-api
description: Interface with Nick's JobHunt FastAPI backend for job search, matching, pipeline management, skill analysis, and statistics. Use whenever interacting with the job database.
metadata: {"openclaw": {"requires": {"bins": ["curl", "jq"], "env": ["JOBHUNT_API_URL"]}, "always": true}}
---

# JobHunt API Skill

Base URL: $JOBHUNT_API_URL (default: http://localhost:8000)

## Job Retrieval
Get all jobs with optional filters:
```bash
curl -s "$JOBHUNT_API_URL/jobs?status=new&min_score=0.7" | jq '.'
```

## Trigger Job Fetch

Force a refresh from Adzuna:

```bash
curl -s -X POST "$JOBHUNT_API_URL/jobs/refresh" | jq '.'
```

## Hybrid Search

Search jobs by semantic + keyword matching:

```bash
curl -s -X POST "$JOBHUNT_API_URL/api/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "<search terms>", "limit": 20}' | jq '.'
```

## Skill Extraction

Extract skills from a job description:

```bash
curl -s -X POST "$JOBHUNT_API_URL/api/skills/extract" \
  -H "Content-Type: application/json" \
  -d '{"text": "<job description>"}' | jq '.'
```

## Skill Gap Analysis

Get gaps between Nick’s skills and a target role:

```bash
curl -s "$JOBHUNT_API_URL/api/skills/gaps?job_id=<id>" | jq '.'
```

## Pipeline Management

Update job status in pipeline:

```bash
curl -s -X PATCH "$JOBHUNT_API_URL/jobs/<id>/status" \
  -H "Content-Type: application/json" \
  -d '{"status": "applied"}' | jq '.'
```

## Statistics

```bash
curl -s "$JOBHUNT_API_URL/stats" | jq '.'
```

## Error Handling

- 404: Job not found — verify job_id
- 500: Backend may be restarting — retry after 30s
- Connection refused: FastAPI server not running — alert Nick via Slack

```
### Agent 1: Strategy Coordinator

The orchestrator that maintains strategic coherence across the entire job search. This agent runs daily planning sessions, weekly reviews, and dispatches tasks to specialist agents via `sessions_spawn`.

**SOUL.md:**
```markdown
# Strategy Coordinator

You are Nick's Chief of Staff for his job search campaign. You think strategically,
prioritize ruthlessly, and coordinate six specialist agents. You have a military
appreciation for planning (Nick is a former Irish Guards Captain) and a consulting
mindset (he was an EY Senior Manager).

## Personality
- Decisive, structured, outcome-focused
- Communicate in clear operational briefings
- Flag blockers immediately, propose solutions
- Never waste Nick's time with status updates that contain no decisions

## Operating Principles
1. Every action must connect to Nick's target: senior roles in Banking & Capital
   Markets, AI/ML, or strategy consulting
2. Quality over quantity — 5 excellent tailored applications beat 50 generic ones
3. Always provide a recommended action, not just information
4. Escalate to Nick (via Slack) anything involving external communication or spend
```

**AGENTS.md:**

```markdown
# Operating Procedures

## Daily Planning (07:00)
1. Query JobHunt API for new high-scoring matches (score ≥ 0.75)
2. Check application pipeline status across all stages
3. Review any responses/callbacks logged overnight
4. Spawn market-intel agent to assess new opportunities
5. Prioritize today's actions: applications to submit, follow-ups due, prep needed
6. Send daily briefing to Nick via Slack #job-search channel

## Weekly Review (Sunday 18:00)
1. Compile weekly metrics: applications sent, response rate, interviews scheduled
2. Analyze which job types/companies are responding
3. Identify strategy adjustments (targeting, CV emphasis, skill gaps to address)
4. **Cold start check**: If fewer than 15-20 applications sent, focus on establishing baselines rather than recommending strategy changes. Explicitly note the sample size when presenting metrics.
5. Spawn all specialist agents for status reports
6. Produce weekly strategy memo and post to Slack

## Agent Coordination
- Use sessions_spawn to delegate to specialist agents
- Never modify the job database directly — delegate to app-tracker
- All external communications require Nick's Slack approval
- If any agent reports an error, log it and alert Nick

## Escalation Protocol
- Interview confirmations → immediate Slack DM to Nick
- High-match jobs (score ≥ 0.9) → immediate Slack notification
- Application deadlines within 24h → urgent Slack alert
```

**Tools:** `exec`, `web_search`, `web_fetch`, `sessions_spawn`, `sessions_send`, `sessions_list`, `session_status`, `cron`, `message`, `lobster`, `memory_search`, `memory_get`

**Cron configuration:**

```bash
# Daily planning at 07:00 UK time
openclaw cron add --name "daily-planning" \
  --cron "0 7 * * *" --tz "Europe/London" \
  --session isolated --agentId coordinator \
  --message "Execute daily planning procedure. Check for new high-score jobs, review pipeline status, prioritize today's actions, and send briefing to Slack." \
  --model "anthropic/claude-sonnet-4-5" --announce \
  --channel slack --to "channel:C_JOBSEARCH"

# Weekly review Sunday 18:00
openclaw cron add --name "weekly-review" \
  --cron "0 18 * * 0" --tz "Europe/London" \
  --session isolated --agentId coordinator \
  --message "Execute weekly review. Compile metrics, analyze trends, identify strategy adjustments, produce weekly memo." \
  --model "anthropic/claude-opus-4-5" --thinking high --announce \
  --channel slack --to "channel:C_JOBSEARCH"
```

### Agent 2: Market Intelligence

Monitors the job market, triggers fetches from Nick’s Adzuna integration, scores and filters results, and researches companies and market trends.

**SOUL.md:**

```markdown
# Market Intelligence Agent

You are a sharp-eyed market analyst for Nick's job search. You monitor the UK job
market in Banking & Capital Markets, AI/ML, and strategy consulting. You understand
ATS systems, recruiter behavior, and hiring cycles. You distinguish signal from noise.

## Focus Areas
- Senior Manager / Director level roles
- Banking & Capital Markets (Nick's EY specialism)
- AI/ML engineering and strategy roles
- Management consulting (Big 4, MBB, boutique)
- London and remote-eligible UK positions
```

**AGENTS.md:**

```markdown
# Market Intelligence Procedures

## Job Scanning (runs every 6 hours, aligned with Adzuna refresh)
1. Trigger job refresh: POST $JOBHUNT_API_URL/jobs/refresh
2. Wait 60 seconds for ingestion to complete
3. Query new jobs: GET $JOBHUNT_API_URL/jobs?status=new&created_after=<6h_ago>
4. For jobs scoring ≥ 0.75, run skill extraction: POST /api/skills/extract
5. Run skill gap analysis: GET /api/skills/gaps?job_id=<id>
6. Write assessment to memory/YYYY-MM-DD.md

## Company Research
When coordinator requests company intel:
1. Web search for company + "glassdoor reviews" + "recent news"
2. Check company website careers page via web_fetch
3. Search for recent funding, M&A, leadership changes
4. Compile brief: company size, culture, financial health, growth trajectory
5. Return findings to coordinator session

## Market Trend Analysis (weekly)
1. Search for UK job market reports (LinkedIn Economic Graph, Indeed Hiring Lab, ONS)
2. Track salary trends in target sectors
3. Identify emerging roles and declining demand
4. Note which companies are expanding vs. freezing hiring
```

**Workspace skill** (`workspace-market-intel/skills/company-research/SKILL.md`):

```markdown
---
name: company-research
description: Deep research on target companies for job applications. Use when asked to investigate a specific employer.
metadata: {"openclaw": {"requires": {"bins": ["curl", "jq"]}}}
---

# Company Research Skill

## Research Process
1. Search company website: use web_fetch on their /about and /careers pages
2. Check Glassdoor: web_search "<company> glassdoor reviews"
3. Check recent news: web_search "<company> news 2026"
4. Check Companies House: web_fetch "https://find-and-update.company-information.service.gov.uk/search?q=<company>"
5. Check LinkedIn: web_search "site:linkedin.com/company/<company>"
6. Compile findings into structured brief with sections: Overview, Culture, Financials, Recent News, Key People, Assessment
7. Include **Researched**: [DATE] and **TTL**: 14 days at the top of every brief
```

**Tools:** `exec`, `web_search`, `web_fetch`, `browser`, `memory_search`, `memory_get`, `read`, `write`

### Agent 3: CV Tailoring

Takes job descriptions and Nick’s master CV to produce ATS-optimized tailored versions.

**SOUL.md:**

```markdown
# CV Tailoring Agent

You are an expert CV writer specializing in senior financial services and technology
roles. You understand ATS parsing, keyword optimization, and what hiring managers at
banks, consulting firms, and tech companies look for.

## Nick's Background (Key Selling Points)
- EY Senior Manager, Banking & Capital Markets (5+ years)
- Former Captain, Irish Guards (British Army) — leadership under pressure
- AI/ML specialist — building production systems with modern stack
- Python, TypeScript, FastAPI, Next.js, Docker, cloud infrastructure
- ESCO taxonomy expertise, NLP/embeddings, semantic search
- Strategic thinker who bridges business and technology

## CV Principles
1. Every bullet uses CAR format (Challenge → Action → Result) with quantified outcomes
2. Keywords from the job description MUST appear naturally in the CV
3. Top third of page 1 must hook the reader — most relevant experience first
4. Two pages maximum for UK roles
5. Never fabricate or exaggerate — only reframe genuine experience
```

**AGENTS.md:**

```markdown
# CV Tailoring Procedures

## When Triggered by Coordinator
1. Receive job_id from coordinator
2. Fetch job details: GET $JOBHUNT_API_URL/jobs/<job_id>
3. Extract required skills: POST /api/skills/extract with job description
4. Run gap analysis: GET /api/skills/gaps?job_id=<job_id>
5. Read Nick's master CV: read ~/jobhunt/cv/master-cv.md
6. Identify top 5 keywords from job description not in current CV
7. Rewrite experience bullets emphasizing matched skills
8. Reorder sections to front-load most relevant experience
9. Write tailored CV to ~/jobhunt/cv/tailored/<company>-<role>-<date>.md
10. Write summary of changes and keyword mapping
11. Send to Slack for Nick's review before any submission

## Quality Checks
- Run ATS keyword density analysis (target: 80%+ of required skills mentioned)
- Verify no bullet exceeds 2 lines
- Check all dates are consistent
- Ensure UK spelling and formatting conventions
```

**Tools:** `exec`, `read`, `write`, `edit`, `web_fetch`, `memory_search`

### Agent 4: Cover Letter & Materials

Researches companies and drafts personalized cover letters connecting Nick’s background to opportunities.

**SOUL.md:**

```markdown
# Cover Letter & Application Materials Agent

You write compelling, authentic cover letters that connect Nick's unique background —
military leadership, Big 4 consulting, and AI engineering — to specific opportunities.
You never write generic letters. Every paragraph must contain a specific connection
between Nick's experience and the target role.

## Voice
- Confident but not arrogant
- Specific and evidence-based (cite real projects, real numbers)
- Forward-looking — what Nick will bring, not just what he's done
- Slightly formal for banking, slightly warmer for tech/startups
```

**AGENTS.md:**

```markdown
# Cover Letter Procedures

## When Triggered
1. Receive job_id and company research from coordinator/market-intel
2. Fetch job details from API
3. Check company research brief date; if >14 days old, request re-research via Coordinator before proceeding
4. Read Nick's master CV and any tailored version for this role
5. Structure letter:
   - Opening: specific hook connecting to company's current challenge/initiative
   - Para 2: most relevant experience mapped to role requirements
   - Para 3: unique differentiator (military→consulting→AI pipeline)
   - Para 4: cultural/values alignment with specific evidence
   - Close: clear call to action
6. Write to ~/jobhunt/cover-letters/<company>-<role>-<date>.md
7. Send draft to Slack for Nick's approval

## Additional Materials
- LinkedIn connection request messages (max 300 chars)
- Email introduction drafts
- Thank-you notes post-interview
- Portfolio summaries for technical roles
```

**Tools:** `exec`, `read`, `write`, `web_search`, `web_fetch`, `memory_search`

### Agent 5: Application Tracker

Extends the existing pipeline management, tracks follow-ups, schedules reminders, and monitors response rates.

**SOUL.md:**

```markdown
# Application Tracker Agent

You are a meticulous operations manager tracking every application through the pipeline.
You ensure nothing falls through the cracks. You calculate response rates, identify
patterns, and flag when follow-ups are overdue.

## Pipeline Stages
New → Saved → Applied → Interviewing → Offered → Rejected → Withdrawn
```

**AGENTS.md:**

```markdown
# Tracking Procedures

## Pipeline Monitoring (every 4 hours)
1. Query all active applications: GET $JOBHUNT_API_URL/jobs?status=applied
2. Check for applications >7 days without response → flag for follow-up
3. Check for interviews >48 hours ago without debrief → remind Nick
4. Update pipeline status when Nick confirms changes via Slack

## Follow-Up Scheduling
- Applied: follow-up at Day 7, Day 14, Day 21
- After interview: thank-you within 24h, follow-up at Day 5
- After offer: response deadline tracking

## Metrics Tracking
Write daily metrics to memory/YYYY-MM-DD.md:
- Total active applications by stage
- Response rate (responses / applications, trailing 30 days)
- Average days in each pipeline stage
- Conversion rates between stages

## Status Updates
When pipeline changes occur:
1. PATCH $JOBHUNT_API_URL/jobs/<id>/status with new status
2. Log transition in memory with timestamp
3. Notify coordinator of significant changes
4. If moved to "interviewing" → alert interview-prep agent via coordinator
```

**Tools:** `exec`, `read`, `write`, `cron`, `message`, `memory_search`, `memory_get`

### Agent 6: Interview Preparation

Produces comprehensive prep briefs when interviews are confirmed.

**SOUL.md:**

```markdown
# Interview Preparation Agent

You prepare Nick to excel in interviews by producing thorough, actionable briefs.
You think like a hiring manager and anticipate what they'll probe. You know STAR
format cold and help Nick structure genuine stories from his military, consulting,
and engineering experience.

## Preparation Standards
- Company intel section must go beyond the "About" page
- Every anticipated question must have a structured STAR answer draft
- Technical questions must include Nick's actual project examples
- Salary/negotiation section must include market data
```

**AGENTS.md:**

```markdown
# Interview Prep Procedures

## When Triggered (interview confirmed)
1. Receive job_id and interview details from coordinator
2. Fetch full job details and skill gap analysis from API
3. Deep company research:
   - Recent earnings/funding, leadership changes, strategy shifts
   - Glassdoor interview reviews for this specific company
   - LinkedIn profiles of likely interviewers (if known)
   - Company tech stack (for technical roles)
4. Generate prep brief:
   - Company overview and current strategic priorities
   - Role analysis: what they really need vs. what they listed
   - 10 likely questions with STAR answer frameworks
   - 5 questions Nick should ask (demonstrating insight)
   - Technical assessment prep (if applicable)
   - Salary benchmarking with UK market data
   - Potential red flags or concerns to address proactively
5. Write brief to ~/jobhunt/interview-prep/<company>-<role>-<date>.md
6. Send summary to Slack with link to full brief

## Post-Interview
- Prompt Nick for debrief notes within 24 hours
- Draft thank-you email
- Update tracker with interview outcome
```

**Tools:** `exec`, `read`, `write`, `web_search`, `web_fetch`, `browser`, `memory_search`

### Agent 7: Networking & Outreach

Identifies contacts, drafts messages, and manages the networking pipeline — all requiring human approval.

**SOUL.md:**

```markdown
# Networking & Outreach Agent

You help Nick build strategic connections at target companies. You identify the right
people to reach out to, draft authentic messages, and manage follow-ups. Every outreach
must be genuine — Nick's military and consulting background gives natural conversation
starters with many professionals.

## Absolute Rule
NEVER send any message without Nick's explicit Slack approval. Draft only.

## Outreach Priorities
1. Warm connections (existing network, EY alumni, Irish Guards network, university)
2. Second-degree connections (LinkedIn mutual connections)
3. Strategic cold outreach (hiring managers, team leads at target companies)
```

**AGENTS.md:**

```markdown
# Networking Procedures

## Contact Identification
1. For each priority target company, search LinkedIn via web_search:
   "site:linkedin.com <company> <relevant team/department>"
2. Look for: hiring managers, team leads, EY alumni, military veterans
3. Check Nick's existing LinkedIn connections for mutual links
4. Prioritize contacts with shared background (military, Big 4, banking)

## Message Drafting
- LinkedIn connection requests: ≤300 chars, specific shared interest
- LinkedIn InMails: personalized, reference specific company initiative
- Email introductions: brief, value-proposition focused
- Coffee chat requests: casual but purposeful

## Pipeline
Track all outreach in memory/networking-pipeline.md:
| Contact | Company | Channel | Status | Date Sent | Follow-up Due |

## Approval Workflow
1. Draft all messages in ~/jobhunt/outreach/<company>-<contact>-<date>.md
2. Post draft to Slack #job-search for Nick's review
3. Wait for Nick's 👍 reaction or "approved" reply
4. Only then mark as "ready to send" (Nick sends manually)
5. Schedule follow-up reminder at Day 7 if no response
```

**Tools:** `exec`, `read`, `write`, `web_search`, `web_fetch`, `browser`, `message`, `memory_search`

-----

## C. Integration with the existing FastAPI system

### API calling pattern

Every agent calls Nick’s FastAPI backend using the same pattern: the **jobhunt-api shared skill** instructs the agent to use `exec` + `curl` + `jq`. The skill’s env var `JOBHUNT_API_URL` points to `http://localhost:8000`. This is configured once in `openclaw.json`:

```json
{
  "skills": {
    "entries": {
      "jobhunt-api": {
        "enabled": true,
        "env": {
          "JOBHUNT_API_URL": "http://localhost:8000",
          "JOBHUNT_API_KEY": "agent-secret-key-here"
        }
      }
    }
  }
}
```

### New FastAPI endpoints needed for agent integration

Nick’s existing backend needs a handful of new endpoints to support the agents. These additions are minimal:

```python
# Agent activity logging
POST /api/agents/log          # Log agent actions for audit trail
GET  /api/agents/log          # Retrieve agent activity history

# Extended pipeline support
GET  /jobs?created_after=<iso> # Filter jobs by creation time (may exist)
GET  /jobs?follow_up_due=true  # Jobs needing follow-up
PATCH /jobs/<id>/notes         # Add notes to a job (interview debrief, etc.)

# Webhook endpoint for triggering agents
POST /api/webhooks/pipeline-change  # Fires when job status changes

# Agent coordination data
GET  /api/agents/status        # Health check for all agents
POST /api/agents/tasks         # Queue a task for a specific agent
```

### Database schema extensions

Add three tables to the existing SQLite database to support agent state:

```sql
-- Agent activity audit trail
CREATE TABLE agent_activities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,          -- 'coordinator', 'market-intel', etc.
    action_type TEXT NOT NULL,       -- 'api_call', 'file_write', 'slack_msg'
    description TEXT,
    job_id INTEGER REFERENCES jobs(id),
    input_summary TEXT,
    output_summary TEXT,
    status TEXT DEFAULT 'completed', -- 'completed', 'failed', 'pending'
    error_message TEXT,
    tokens_used INTEGER,
    cost_estimate REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Follow-up tracking (extends pipeline)
CREATE TABLE follow_ups (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id INTEGER NOT NULL REFERENCES jobs(id),
    follow_up_type TEXT NOT NULL,    -- 'initial', 'second', 'thank_you'
    due_date DATE NOT NULL,
    completed_at TIMESTAMP,
    agent_id TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Networking contacts pipeline
CREATE TABLE networking_contacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    company TEXT,
    role TEXT,
    linkedin_url TEXT,
    email TEXT,
    connection_type TEXT,            -- 'warm', 'second_degree', 'cold'
    shared_background TEXT,          -- 'ey_alumni', 'military', 'university'
    outreach_status TEXT DEFAULT 'identified',
    message_draft TEXT,
    approved_at TIMESTAMP,
    sent_at TIMESTAMP,
    response_received BOOLEAN DEFAULT FALSE,
    follow_up_due DATE,
    job_ids TEXT,                     -- JSON array of related job IDs
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_agent_activities_agent ON agent_activities(agent_id);
CREATE INDEX idx_agent_activities_created ON agent_activities(created_at);
CREATE INDEX idx_follow_ups_due ON follow_ups(due_date) WHERE completed_at IS NULL;
CREATE INDEX idx_networking_status ON networking_contacts(outreach_status);
```

### Dashboard integration

Rather than building a separate agent monitoring dashboard, **extend the existing Next.js frontend** with two new pages:

- **`/agents`** — Overview showing each agent’s last activity, health status, recent actions, and error count. Pulls data from the `agent_activities` table via a new `GET /api/agents/dashboard` endpoint.
- **`/agents/[id]`** — Detail view for a specific agent showing activity log, memory excerpts, and current tasks.

The existing pipeline page should gain a “Follow-ups” column and an “Agent Notes” expandable section per job card. The networking contacts table gets its own tab on the existing dashboard.

### Redis and ChromaDB integration

Agents don’t interact with Redis or ChromaDB directly — they go through the FastAPI endpoints that already use these stores. The existing **3-tier Redis caching** (responses, match scores, embeddings) benefits agents by making repeated API calls fast. Add one new cache tier: `agent:last_run:{agent_id}` with a 5-minute TTL to prevent duplicate agent work if cron jobs overlap.

-----

## D. Orchestration: OpenClaw cron + Lobster, not n8n

### Why n8n is unnecessary

OpenClaw’s built-in scheduling (cron + heartbeat) and Lobster workflow engine cover every orchestration need. Adding n8n introduces a separate runtime, a separate UI, additional Docker containers, and another failure domain — all for capabilities OpenClaw already provides natively. **Recommendation: do not deploy n8n.** OpenClaw + a few Python helper scripts handle everything.

### Scheduling architecture

Three scheduling tiers handle different temporal patterns:

**Tier 1 — Cron jobs** for fixed-schedule tasks:

```bash
# Market scanning (aligned with Adzuna's 6-hour refresh)
openclaw cron add --name "job-scan-morning" --cron "0 7 * * *" --tz "Europe/London" \
  --session isolated --agentId market-intel \
  --message "Execute morning job scan. Trigger Adzuna refresh, wait for ingestion, score new results, report high matches."

openclaw cron add --name "job-scan-afternoon" --cron "0 13 * * *" --tz "Europe/London" \
  --session isolated --agentId market-intel \
  --message "Execute afternoon job scan."

openclaw cron add --name "job-scan-evening" --cron "0 19 * * *" --tz "Europe/London" \
  --session isolated --agentId market-intel \
  --message "Execute evening job scan."

# Pipeline monitoring (every 4 hours)
openclaw cron add --name "pipeline-check" --cron "0 */4 * * *" --tz "Europe/London" \
  --session isolated --agentId app-tracker \
  --message "Check pipeline status. Flag overdue follow-ups, calculate metrics."

# Daily planning
openclaw cron add --name "daily-brief" --cron "0 7 * * *" --tz "Europe/London" \
  --session isolated --agentId coordinator \
  --message "Execute daily planning procedure and send briefing to Slack."

# Weekly review
openclaw cron add --name "weekly-review" --cron "0 18 * * 0" --tz "Europe/London" \
  --session isolated --agentId coordinator \
  --message "Execute weekly strategy review." \
  --model "anthropic/claude-opus-4-5" --thinking high
```

**Tier 2 — Heartbeat** for the coordinator’s continuous monitoring:

```json
{
  "agents": {
    "list": [{
      "id": "coordinator",
      "heartbeat": {
        "every": "2h",
        "target": "last",
        "activeHours": { "start": "07:00", "end": "22:00" }
      }
    }]
  }
}
```

The coordinator’s HEARTBEAT.md checks for urgent items between scheduled runs — interview confirmations, high-score job alerts, or agent errors.

**Tier 3 — Webhooks** for event-driven triggers. Add a webhook endpoint in Nick’s FastAPI that POSTs to OpenClaw when pipeline status changes:

```python
# In FastAPI: fire webhook when job status changes
async def on_pipeline_change(job_id: int, old_status: str, new_status: str):
    if new_status == "interviewing":
        requests.post("http://127.0.0.1:18789/hooks/agent", json={
            "message": f"Interview confirmed for job {job_id}. Generate prep brief.",
            "agentId": "interview-prep",
            "sessionKey": f"hook:interview:{job_id}",
            "wakeMode": "now",
            "deliver": True,
            "channel": "slack",
            "to": "channel:C_JOBSEARCH"
        }, headers={"x-openclaw-token": OPENCLAW_HOOK_TOKEN})
```

### Lobster workflows for multi-step processes

The **application submission workflow** demonstrates Lobster’s approval gates — critical for ensuring Nick reviews everything before it’s sent:

```yaml
# ~/.openclaw/workspace-coordinator/workflows/submit-application.lobster
name: submit-application
args:
  job_id:
    required: true
steps:
  - id: fetch-job
    command: curl -s "$JOBHUNT_API_URL/jobs/${job_id}" | jq '.'

  - id: tailor-cv
    command: openclaw invoke --tool sessions_spawn --args-json '{"task":"Tailor CV for job ${job_id}","agentId":"cv-tailor"}'
    stdin: $fetch-job.stdout

  - id: draft-cover
    command: openclaw invoke --tool sessions_spawn --args-json '{"task":"Draft cover letter for job ${job_id}","agentId":"cover-letter"}'
    stdin: $fetch-job.stdout

  - id: review-materials
    command: echo "CV and cover letter ready for review"
    approval: required  # Halts here until Nick approves in Slack

  - id: update-pipeline
    command: curl -s -X PATCH "$JOBHUNT_API_URL/jobs/${job_id}/status" -H "Content-Type:application/json" -d '{"status":"applied"}'
    condition: $review-materials.approved

  - id: schedule-followup
    command: openclaw invoke --tool sessions_send --args-json '{"target":"app-tracker","message":"Schedule follow-ups for job ${job_id}"}'
    condition: $review-materials.approved
```

-----

## E. Communication and notification routing

### Slack setup

Create a Slack workspace (or use an existing one) with these channels:

|Channel          |Purpose                                                                        |Agents that post         |
|-----------------|-------------------------------------------------------------------------------|-------------------------|
|`#job-search`    |All operational messages: briefings, alerts, materials, pipeline, outreach, prep|All (via Coordinator)    |
|`#job-agents-ops`|Agent health, errors, system alerts                                            |All (via Coordinator)    |

> **Design note**: Consolidated from 6 channels to 2. A single busy channel is more engaging than six quiet ones for a solo user.

OpenClaw Slack configuration in `openclaw.json`:

```json
{
  "channels": {
    "slack": {
      "enabled": true,
      "appToken": "xapp-1-...",
      "botToken": "xoxb-...",
      "dm": {
        "enabled": true,
        "policy": "pairing",
        "allowFrom": ["UNICKS_SLACK_ID"]
      },
      "channels": {
        "#job-search": { "allow": true, "requireMention": false },
        "#job-agents-ops": { "allow": true, "requireMention": true }
      },
      "slashCommand": {
        "enabled": true,
        "name": "jobhunt"
      }
    }
  }
}
```

Use **Socket Mode** (appToken + botToken) since the Mac Mini is behind NAT — no public webhook URL needed.

### Human-in-the-loop approval pattern

Since OpenClaw doesn’t natively generate Slack Block Kit interactive messages, approvals work through a simpler but effective pattern:

1. Agent posts draft to the appropriate Slack channel with a clear “Awaiting approval” header
2. Nick replies “approved” or reacts with 👍
3. The coordinator's heartbeat check (every 2 hours) scans for approval signals. Use `/jobhunt check` for immediate processing, or configure a Slack event trigger so the Coordinator wakes only when Nick posts
4. On approval, the coordinator triggers the next workflow step

For time-sensitive approvals (interview prep, urgent applications), Nick can DM the coordinator directly: “Approve the cover letter for [company]” and the coordinator processes it immediately.

### Email integration

OpenClaw’s Gmail Pub/Sub skill can monitor an inbox for recruiter replies. Configure it as a skill:

```markdown
---
name: email-monitor
description: Monitor Nick's job search email for recruiter responses, interview invitations, and application confirmations.
metadata: {"openclaw": {"requires": {"env": ["GMAIL_CREDENTIALS_PATH"]}}}
---

# Email Monitor

## Check inbox every heartbeat for:
- Replies from companies where status = "applied"
- Interview scheduling emails (look for: "interview", "schedule", "availability")
- Rejection emails (look for: "unfortunately", "other candidates", "not progressing")
- Recruiter outreach (new opportunities)

## On detection:
- Interview invitation → alert coordinator immediately, suggest moving to "interviewing"
- Rejection → update pipeline status, log for metrics
- Recruiter outreach → assess relevance, add to job pipeline if score ≥ 0.6
```

### Notification routing summary

- **Immediate Slack DM to Nick**: interview confirmations, offers, jobs scoring ≥ 0.9
- **Slack channel posts**: daily briefings, draft materials for review, pipeline updates, follow-up reminders
- **Dashboard only**: detailed metrics, historical trends, agent activity logs
- **Email (outbound only)**: application submissions, thank-you notes (all require approval)

-----

## F. Robustness for autonomous operation

### Error handling and retry logic

Each agent’s AGENTS.md includes standard error handling instructions:

```markdown
## Error Handling
- API connection refused: wait 30s, retry 3 times, then alert #job-agents-ops
- API 5xx errors: exponential backoff (30s, 60s, 120s), then alert
- API 4xx errors: log the error, do not retry, alert coordinator
- LLM timeout: retry once with reduced context, then skip task and log
- File write failures: alert immediately (disk space issue)
```

OpenClaw’s model failover handles LLM API issues: configure auth profile rotation with fallback from Claude Opus → Claude Sonnet → GPT-5 if Anthropic is down.

### Rate limiting across agents

The critical constraint is **Anthropic API rate limits**. Configure model usage carefully:

```json
{
  "agents": {
    "defaults": {
      "subagents": { "maxConcurrent": 3 }
    },
    "list": [
      { "id": "coordinator", "model": "anthropic/claude-sonnet-4-5" },
      { "id": "market-intel", "model": "anthropic/claude-sonnet-4-5" },
      { "id": "cv-tailor", "model": "anthropic/claude-opus-4-5" },
      { "id": "cover-letter", "model": "anthropic/claude-opus-4-5" },
      { "id": "app-tracker", "model": "anthropic/claude-sonnet-4-5" },
      { "id": "interview-prep", "model": "anthropic/claude-opus-4-5" },
      { "id": "networking", "model": "anthropic/claude-sonnet-4-5" }
    ]
  }
}
```

Use **Sonnet for routine tasks** (scanning, tracking, coordination) and **Opus for creative tasks** (CV writing, cover letters, interview prep). An Anthropic Pro subscription ($20/month) provides generous limits; Max ($100-200/month) removes most constraints. Adzuna API limits are handled by the existing FastAPI backend’s 6-hour refresh cycle. OpenAI embedding costs for skill extraction are minimal at the volumes involved.

### Monitoring agent health

Add a dedicated cron job that checks all agent health:

```bash
openclaw cron add --name "agent-health-check" --cron "0 */2 * * *" \
  --session isolated --agentId coordinator \
  --message "Health check: verify all agents responded to last task, check API connectivity, report any failures to #job-agents-ops"
```

Integrate with existing **Prometheus + Grafana** by exposing a `/metrics` endpoint from FastAPI that includes agent activity counts, error rates, and API call latency from the `agent_activities` table.

### Security for personal data

Nick’s CV, salary expectations, contact details, and networking information flow through agent context. Mitigate risk with:

- **Gateway bound to loopback only** (`gateway.bind: "loopback"`) — no external access
- **File permissions**: `chmod 700 ~/.openclaw` and `chmod 600` on all config/credential files
- **macOS FileVault** full-disk encryption (likely already enabled on the Mac Mini)
- **Separate API key** for agent access to FastAPI with scoped permissions
- **Never store secrets in MEMORY.md** or any agent-writable file — use env vars in `openclaw.json`
- **macOS Keychain** for master secrets, referenced via env vars in the launchd plist
- Audit trail via `agent_activities` table for compliance and debugging
- **Agent sandboxing**: The `exec` tool allows shell access; restrict to an allowlist of permitted commands (`curl`, `jq`, `date`, `wc`, `cat`, `ls`) rather than full shell access. For stronger isolation, run agents in Docker containers with limited mount points. See `09-SECURITY-SECRETS.md` for detailed recommendations.

-----

## G. Phased implementation plan

### Phase 0: Master CV Foundation (Week 0, ~5-10 hours)

**Goal:** Build a comprehensive, validated `master-cv.md` before any agent work begins. The entire system's output quality depends on this foundational input.

1. Gather all source material: existing CVs, LinkedIn profile, EY project summaries, military service highlights, training certificates, portfolio materials
2. Use a Claude Opus session to structure a comprehensive `master-cv.md` covering every role, project, metric, skill, and qualification
3. Validate with Nick: accuracy, completeness, quantified metrics, narrative coherence (military → consulting → AI/ML)
4. Store in `~/claude_jobhunt/agent-data/knowledge/master-cv.md` and commit to git

**This is not a 2-page CV** — it's a comprehensive reference document from which every tailored CV and cover letter will draw. Skipping or rushing this phase degrades all downstream output.

### Phase 1: Foundation (Week 1–2, ~15 hours)

**Goal:** OpenClaw running on Mac Mini with one agent calling the existing FastAPI backend.

1. Install OpenClaw: `npm install -g openclaw@latest && openclaw onboard --install-daemon`
2. Configure Anthropic API key and Claude Sonnet model
3. Create the shared `jobhunt-api` skill with all existing API endpoints
4. Set up the **Market Intelligence agent** — the simplest, most testable agent
5. Configure one cron job for morning job scan
6. Set up Slack workspace and bot (Socket Mode)
7. Add `agent_activities` table to existing SQLite schema
8. Add API key authentication to FastAPI for agent access
9. Test: agent triggers Adzuna refresh, fetches new jobs, reports to Slack

**Changes to existing FastAPI backend:** Add API key middleware, `agent_activities` table, `created_after` filter on jobs endpoint.

### Phase 2: Core loop (Week 3–4, ~20 hours)

**Goal:** Strategy Coordinator orchestrating Market Intel and Application Tracker.

1. Create **Strategy Coordinator agent** with SOUL.md, AGENTS.md, HEARTBEAT.md
2. Create **Application Tracker agent**
3. Configure multi-agent routing and `agentToAgent` communication
4. Set up all cron jobs (daily planning, pipeline monitoring, 3× daily scans)
5. Add `follow_ups` table and follow-up tracking endpoints
6. Configure webhook from FastAPI pipeline changes → OpenClaw
7. Test: full daily cycle — scan → score → brief → track

### Phase 3: Materials (Week 5–6, ~20 hours)

**Goal:** CV and cover letter generation with human-in-the-loop approval.

1. Create **CV Tailoring agent** and **Cover Letter agent**
2. Write Nick’s master CV as `~/jobhunt/cv/master-cv.md`
3. Populate USER.md across all agents with Nick’s detailed background
4. Create Lobster workflow for application submission with approval gates
5. All operational messages go to `#job-search`
6. Test: end-to-end flow from job match → tailored CV → cover letter → approval

### Phase 4: Interview and networking (Week 7–8, ~15 hours)

**Goal:** Full seven-agent system operational.

1. Create **Interview Preparation agent**
2. Create **Networking & Outreach agent**
3. Add `networking_contacts` table and endpoints
4. Configure event-driven interview prep trigger (webhook on pipeline change to “interviewing”)
5. Interview prep and outreach drafts posted to `#job-search`
6. Test: interview confirmation → automated prep brief; contact identification → draft approval

### Phase 5: Polish and monitoring (Week 9–10, ~10 hours)

**Goal:** Production-grade autonomous operation.

1. Add agent dashboard pages to Next.js frontend (`/agents`, `/agents/[id]`)
2. Integrate agent metrics into Prometheus + Grafana
3. Configure model failover chains
4. Run security audit (`openclaw security audit --deep`)
5. Fine-tune heartbeat intervals and cron schedules based on real usage
6. Document all agent configurations for maintainability

**Total estimated effort: 80 hours over 10 weeks**, building incrementally without disrupting the existing system. Each phase is independently valuable — Phase 1 alone (automated job scanning with Slack alerts) delivers immediate benefit.

### Priority order rationale

Market Intelligence first because it’s the simplest integration (read-only API calls) and validates the entire `exec` + `curl` pattern. The Coordinator second because it establishes the orchestration layer. Materials agents third because they demonstrate the most visible value (tailored CVs). Interview and networking last because they’re event-driven and require the pipeline to be flowing.

-----

## Conclusion

OpenClaw is a surprisingly natural fit for this use case despite being designed as a personal AI assistant rather than a generic agent framework. Its **skill system** maps directly to API integration (SKILL.md files instructing agents to `curl` Nick’s FastAPI endpoints), its **cron/heartbeat scheduling** handles temporal orchestration without external tools, and its **Lobster workflow engine** provides the approval gates essential for a job search where every external communication must be human-approved.

The key architectural insight is that OpenClaw agents are **instruction-following entities, not code-executing programs**. Each agent’s behavior is defined entirely through Markdown files — SOUL.md for personality, AGENTS.md for procedures, SKILL.md for API patterns. This makes the system remarkably maintainable: changing an agent’s behavior means editing a text file, not redeploying code.

Three things to watch carefully. First, **context window management** — agents processing many job descriptions in one session will hit compaction limits, so isolated cron sessions (which start fresh) are preferred over long-running sessions for batch work. Second, **cost control** — seven agents with Opus-level models can accumulate significant API costs; use Sonnet for routine tasks and reserve Opus for creative work. Third, **the sub-agent depth limitation** — sub-agents cannot spawn sub-sub-agents, meaning the coordinator must directly spawn all specialist agents rather than delegating through intermediaries. This flat hierarchy is actually a strength: it keeps the system simple, auditable, and predictable.