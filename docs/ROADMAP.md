# Job Hunt System — Product Roadmap
**Version:** 1.0  
**Last Updated:** 2026-02-25  
**Status:** Phases 1-3 Complete, Phase 4 Planning

---

## Vision

An **autonomous, intelligent job search operating system** that finds, evaluates, and prepares Nick for high-quality opportunities while learning his preferences and optimizing over time.

**Core Principles:**
1. **Signal over noise** — 3-5 great opportunities, not 50 mediocre ones
2. **Autonomous daily operations** — Runs itself, Nick reviews and decides
3. **Multi-source intelligence** — Job boards + LinkedIn + network + inbox signals
4. **Deep research capability** — Perplexity for company intelligence, not just surface-level search
5. **Learning loop** — Gets smarter based on approve/reject/skip patterns

---

## Phase 1-3: Core Job Search Agents ✅ COMPLETE

**Timeline:** Jan 8 - Feb 15, 2026  
**Status:** Deployed and operational  
**Branch:** `feature/openclaw-agents-phase1-3`

### What Was Built

| Component | Status | Description |
|-----------|--------|-------------|
| **Coordinator** | ✅ Deployed | Daily planning, agent orchestration, briefing generation |
| **Market Intelligence** | ✅ Deployed | Adzuna job scanning, match scoring, lead prioritization |
| **Application Tracker** | ✅ Deployed | Pipeline monitoring, follow-up scheduling, status tracking |
| **CV Tailor** | ✅ Built | On-demand CV customization (Opus model) |
| **Cover Letter** | ✅ Built | On-demand cover letter generation (Opus model) |
| **FastAPI Backend** | ✅ Deployed | Job data API, agent logging, profile management |
| **PostgreSQL Database** | ✅ Migrated | 4,899 jobs, profile data, agent logs |
| **Slack Integration** | ✅ Working | Bot posting to `#job-search` channel |

### Key Achievements
- 40+ QA tests, 8 UAT scenarios (all passed)
- Real CV loaded (Nick Solly profile)
- Brave Search API integrated (web search enabled)
- Comprehensive documentation (IMPLEMENTATION_GUIDE.md, API.md, OPERATIONS.md)

### Estimated Effort: 80-100 hours
### Actual Cost: ~$40-60/month (ongoing Anthropic API)

---

## Phase 3.5: Daily Operations Infrastructure ✅ COMPLETE

**Timeline:** Feb 25, 2026  
**Status:** Configured, first runs scheduled

### What Was Built

- **6 cron jobs** — Coordinator (daily + weekly), Market Intel (3x daily), App Tracker (4x daily)
- **Perplexity integration** — Deep research capability for company intelligence (~$5-6/month)
- **Interview Prep workspace** — Agent ready for on-demand spawning

### Schedule
- **Morning:** Market Intel 07:00 → Coordinator 07:15
- **Midday:** Market Intel 13:00, App Tracker 12:00
- **Evening:** Market Intel 19:00, App Tracker 16:00 + 20:00
- **Weekly:** Coordinator strategic review (Sunday 18:00, Opus model)

### Estimated Effort: 12-15 hours
### Cost Impact: +$5-6/month (Perplexity API)

---

## Phase 4: Interview Prep & Networking (IN PROGRESS)

**Timeline:** Mar 1 - Mar 21, 2026 (3 weeks)  
**Status:** Planning → Implementation  
**Priority:** High

### Phase 4A: Interview Preparation ✅ READY

**What's Included:**
- ✅ **Interview Prep agent** — Auto-triggers when job status → "interviewing"
- ✅ **Company deep-dive** — Perplexity-powered research (strategy, culture, recent news)
- ✅ **Interviewer research** — LinkedIn profiles, shared connections, recent posts
- ✅ **STAR frameworks** — Pre-built answer structures from Nick's experience
- ✅ **Strategic questions** — Demonstrate domain expertise and genuine interest
- ✅ **Negotiation intel** — Salary ranges, company pay reputation, leverage points

**Deliverable:** Comprehensive prep brief saved to `agent-data/interview-prep/{company}-{role}-{date}.md`

**Status:** Workspace created, agent persona defined, ready to spawn on-demand

**Estimated Effort:** 0 hours (already complete)

---

### Phase 4B: Networking & Contact Management 🔄 IN PROGRESS

**What's Needed:**

**1. Contact Management UI** (Priority 1)
- Mission Control module for `networking_contacts` table
- Features: priority ranking, tags, relationship warmth, last contacted, next action
- Drag-to-reorder, filters, click-to-expand details
- Obsidian integration (optional): Dataview table for quick reference

**2. Networking Agent Design** (Priority 2)
- Contact identification from job postings (hiring managers, team members, shared connections)
- Draft-only outreach (LinkedIn, email) — all messages require Nick's approval
- Response tracking, engagement optimization
- Learning: response rates → improved targeting

**3. Backend Extension** (Priority 3)
- Schema updates: `priority`, `tags`, `last_contacted`, `next_action`, `warmth`, `source`
- API endpoints: CRUD + prioritization + filtering
- Slack integration: Daily contact summary, approval workflow

**Dependencies:**
- Mission Control codebase (Vue 3 + FastAPI)
- `networking_contacts` table (already exists)

**Deliverables:**
- Mission Control module: `frontend/src/modules/networking/`
- Backend routes: `backend/app/api/networking.py`
- Agent workspace: `~/.openclaw/workspace-networking/`
- Documentation: `docs/NETWORKING-AGENT-DESIGN.md`

**Estimated Effort:** 2-3 weeks
- UI development: 10-12 hours
- Backend extension: 4-6 hours
- Agent design + testing: 4-6 hours
- Documentation: 2-3 hours
- **Total:** 20-27 hours

**Cost Impact:** Minimal (~$2-3/month for agent operations)

---

## Phase 5: Multi-Source Intelligence & Learning

**Timeline:** Apr 1 - Apr 30, 2026 (4 weeks)  
**Status:** Planning  
**Priority:** Medium

### Phase 5A: LinkedIn Email Integration

**Problem:** LinkedIn sends high-quality job recommendations to Nick's inbox, but they're not captured by the system.

**Solution:** Email parsing pipeline
- Forward LinkedIn notification emails → dedicated inbox (e.g., `jobhunt@jeevesbot.io`)
- Parse HTML structure (job title, company, link, match reasons)
- Ingest via same pipeline as Adzuna (deduplication, match scoring)
- Track source: LinkedIn vs Adzuna for quality comparison

**Design Decisions:**
- ✅ **Email parsing** (not browser automation) — Lower risk, no LinkedIn ToS issues
- ✅ **Forwarding rule** from `nicksolly@gmail.com` → monitoring agent
- ✅ **Predictable HTML structure** — LinkedIn emails are consistent

**Dependencies:**
- Email monitoring agent (himalaya CLI or IMAP)
- HTML parser (BeautifulSoup)
- Ingestion API (`/api/jobs/ingest`)

**Deliverables:**
- Parser script: `scripts/parse_linkedin_emails.py`
- Monitoring agent: Cron job or inbox watcher
- Documentation: `docs/LINKEDIN-PARSER-DESIGN.md`

**Estimated Effort:** 1 week (6-8 hours)  
**Cost Impact:** $0 (no API costs)

---

### Phase 5B: Learning Algorithm & Prioritisation

**Problem:** Match scoring is static. Nick's approve/reject patterns contain signal about what he actually wants.

**Solution:** Learning loop
- **Capture signals:** Save/skip/reject actions with optional reasons
- **Feature extraction:** Role type, company tier, salary range, keywords, job board source
- **Score adjustment:** Boost/penalize similar jobs based on historical patterns
- **Threshold tuning:** Learn optimal match score cutoff (start conservative, relax over time)

**Example:**
- Nick rejects 3 "Senior Manager" roles → penalize seniority keyword
- Nick saves 5 "Associate Director" roles in FinServ → boost that combo
- Nick skips all jobs <£100k → raise salary floor threshold

**Implementation:**
- Simple scoring model first (weighted features, linear adjustments)
- SQLite table: `learning_signals` (job_id, action, reason, timestamp)
- Weekly batch: recalculate match score weights
- Dashboard: Show learned preferences, allow manual overrides

**Dependencies:**
- User action tracking (already exists: saved_jobs, applications)
- Feature extraction pipeline
- Score recalculation logic

**Deliverables:**
- Learning pipeline: `backend/app/services/learning.py`
- Cron job: Weekly model retraining
- Dashboard view: Learned preferences + manual tuning
- Documentation: `docs/LEARNING-ALGORITHM.md`

**Estimated Effort:** 2-3 weeks (15-20 hours)  
**Cost Impact:** $0 (no API costs, local ML)

---

### Phase 5C: Multi-Model Review Panel (OPTIONAL)

**Problem:** Single-model scoring may miss nuances. Cost-efficient consensus could improve quality.

**Solution:** 3-stage review pipeline
- **Stage 1 (Haiku):** Fast screen — obvious rejects (~$0.0003/job)
- **Stage 2 (Sonnet):** Deep analysis — mid-tier candidates (~$0.003/job)
- **Stage 3 (Opus):** Strategic assessment — top 5-10% only (~$0.015/job)

**Evaluation Dimensions:**
- Role fit (skills, experience, seniority match)
- Company fit (culture, values, mission alignment)
- Growth potential (career trajectory, learning opportunities)
- Red flags (unrealistic requirements, poor Glassdoor reviews)

**Cost Optimization:**
- 100 jobs/day → 100 Haiku + 30 Sonnet + 5 Opus = ~$0.20/day = ~$6/month
- Only use Opus for jobs that pass both Haiku + Sonnet screens

**Dependencies:**
- Multi-model orchestration logic
- Consensus scoring algorithm (weighted average, veto rules)

**Deliverables:**
- Review pipeline: `backend/app/services/review_panel.py`
- Scoring logic: Configurable weights + veto thresholds
- Documentation: `docs/REVIEW-PANEL-DESIGN.md`

**Estimated Effort:** 1-2 weeks (10-12 hours)  
**Cost Impact:** +$6-10/month  
**Priority:** Low (quality enhancement, not core workflow)

---

## Phase 6: Advanced Features (BACKLOG)

**Timeline:** TBD (May 2026+)  
**Status:** Ideas / Future Consideration  
**Priority:** Low

### Potential Features

**6A: Company Intelligence Dashboard**
- Visual timeline: funding rounds, leadership changes, product launches
- Competitive landscape: Who else is hiring? What tech stacks?
- Culture signals: Glassdoor sentiment, employee LinkedIn activity

**6B: Salary Negotiation Assistant**
- Market rate analysis (Glassdoor, Payscale, network data)
- Leverage point identification (competing offers, urgency signals)
- Script generation: Negotiation conversation frameworks

**6C: Application Materials Library**
- Version control for CVs/cover letters
- Template management (by role type, seniority, industry)
- A/B testing: Track which materials get responses

**6D: Network Graph Visualization**
- Map connections: who knows whom at target companies?
- Warm intro paths: optimal routes to hiring managers
- Relationship health tracking: last contact, engagement level

**6E: Interview Performance Tracking**
- Post-interview debrief capture
- Common question tracking (what gets asked frequently?)
- Improvement suggestions based on feedback patterns

---

## Effort Summary

| Phase | Timeline | Effort (Hours) | Cost/Month | Status |
|-------|----------|----------------|------------|--------|
| **1-3: Core Agents** | Jan-Feb 2026 | 80-100 | $40-60 | ✅ Complete |
| **3.5: Daily Ops** | Feb 25 | 12-15 | +$5-6 | ✅ Complete |
| **4A: Interview Prep** | (Done) | 0 | $0 | ✅ Ready |
| **4B: Networking** | Mar 1-21 | 20-27 | +$2-3 | 🔄 In Progress |
| **5A: LinkedIn** | Apr 1-7 | 6-8 | $0 | 📋 Planned |
| **5B: Learning** | Apr 8-28 | 15-20 | $0 | 📋 Planned |
| **5C: Review Panel** | TBD | 10-12 | +$6-10 | 📋 Optional |
| **6: Advanced** | TBD | TBD | TBD | 💡 Backlog |

**Total (through Phase 5B):** 133-170 hours  
**Total Cost (Phase 5B):** $47-69/month

---

## Success Metrics

### Operational Health
- ✅ **Uptime:** Cron jobs run reliably (>95% success rate)
- ✅ **Latency:** Morning briefing arrives by 07:20 UK daily
- ✅ **Coverage:** 3+ job scans/day, no overnight gaps >12h

### Signal Quality
- 🎯 **Match accuracy:** >80% of briefed jobs marked "worth reviewing" by Nick
- 🎯 **False positives:** <20% of briefed jobs immediately rejected
- 🎯 **False negatives:** <5% of saved jobs not surfaced by agents

### Engagement
- 🎯 **Nick reviews briefings:** >80% of daily briefings opened within 2 hours
- 🎯 **Agent-sourced applications:** >50% of applications come from agent leads (not manual search)
- 🎯 **Interview conversion:** Agent-sourced applications → interviews at >10% rate

### Efficiency
- 🎯 **Time saved:** Nick spends <30min/day on job search (down from 2+ hours manual)
- 🎯 **Cost per lead:** <£2 per high-quality opportunity (API costs / briefed jobs)
- 🎯 **ROI:** System pays for itself if it accelerates job search by >2 weeks (time value of salary increase)

---

## Dependencies & Risks

### External Dependencies
- **Adzuna API:** Rate limits (300 calls/day), data quality (truncated descriptions)
- **LinkedIn:** Email structure changes could break parser
- **Perplexity API:** Cost per query (~$0.02-0.05), rate limits
- **Anthropic API:** Model pricing changes, availability

### Technical Risks
- **Cron reliability:** Gateway restarts, system sleep, timezone changes
- **Agent failure:** Error handling, fallback strategies, alerting
- **Data drift:** Job market changes, keyword trends, scoring relevance
- **Cost overrun:** Runaway agent spawning, expensive model calls

### Mitigation Strategies
- ✅ **Isolated sessions:** Prevent context window bloat
- ✅ **Announce mode:** Failures post to Slack, not silent
- ✅ **Model tiers:** Haiku for cheap ops, Opus for high-value tasks
- 🔄 **Cost monitoring:** Add Perplexity spend tracking to dashboard
- 🔄 **Learning loop:** Adapt scoring as job market evolves

---

## Review Cadence

- **Daily:** Monitor cron job outputs in Slack `#job-search`
- **Weekly:** Review metrics (match quality, time saved, cost)
- **Monthly:** Roadmap retrospective (what shipped, what didn't, why)
- **Quarterly:** Strategic pivot point (Phase priorities, new features, cost/benefit)

**Next formal review:** 2026-03-04 (after 1 week of daily operations)

---

## Change Log

| Date | Version | Changes |
|------|---------|---------|
| 2026-02-25 | 1.0 | Initial roadmap — Phases 1-5 breakdown, effort estimates, success metrics |

---

**Owner:** Jeeves 🫖  
**Approver:** Nick Solly  
**Last Review:** 2026-02-25
