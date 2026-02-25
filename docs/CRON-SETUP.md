# OpenClaw Cron Job Configuration - Job Hunt Project

> **Status**: ✅ All cron jobs configured and scheduled  
> **Date**: 2026-02-25  
> **Configured by**: Cron Configuration Specialist (subagent)

---

## Overview

The Job Hunt project uses 6 automated cron jobs to orchestrate daily operations. All jobs run in **isolated sessions** to avoid context window bloat and use the UK timezone (`Europe/London`).

---

## Configured Jobs

### 1. 🎯 Coordinator Daily Planning
**Schedule**: Every day at 07:15 UK  
**Cron Expression**: `15 7 * * *`  
**Job ID**: `ca10071d-f42a-4dd4-bfd7-3b14f94a797d`  
**Model**: `anthropic/claude-sonnet-4-5` (default)

**Purpose**: Orchestrates daily operations, reviews leads from Market Intel, prioritizes jobs, and spawns CV Tailor and Cover Letter agents as needed.

**Tasks**:
1. Read today's lead report from Market Intel (`agent-data/pipeline/leads/`)
2. Check pipeline stats via `GET /stats`
3. Check follow-ups due via `GET /api/agents/follow-ups-due`
4. Prioritize 3 high-score jobs + 2 follow-ups
5. Spawn CV Tailor + Cover Letter agents for top job if needed
6. Post daily briefing to `#job-search` Slack channel

**Workspace**: `~/.openclaw/workspace-coordinator/`

---

### 2. 🎯 Coordinator Weekly Review
**Schedule**: Every Sunday at 18:00 UK  
**Cron Expression**: `0 18 * * 0`  
**Job ID**: `786ea000-cf28-44ac-8fe9-7ad8cb767a47`  
**Model**: `anthropic/claude-opus-4-5` (Opus for deeper analysis)

**Purpose**: Conducts weekly strategy review with focus on establishing baselines during cold start phase (first 2-3 weeks).

**Tasks**:
1. Review past week's pipeline data via `GET /stats`
2. Analyze application:interview:offer conversion rates
3. Review quality of materials produced
4. Identify patterns in successful vs unsuccessful applications
5. Write weekly strategy memo to `agent-data/reports/weekly-review-YYYY-MM-DD.md`
6. Post summary to `#job-search` Slack channel

**Cold Start Mode**: First 2-3 weeks focus on establishing baselines, not recommending changes. Strategy adjustments kick in only after 15-20 applications sent.

**Workspace**: `~/.openclaw/workspace-coordinator/`

---

### 3. 🔍 Market Intel Morning Scan
**Schedule**: Every day at 07:00 UK  
**Cron Expression**: `0 7 * * *`  
**Job ID**: `9d37483a-575c-43ce-9890-fc5fa1ae15be`  
**Model**: `anthropic/claude-sonnet-4-5` (default)

**Purpose**: Morning job market scan to catch overnight postings.

**Tasks**:
1. Trigger Adzuna refresh via `POST /jobs/refresh`
2. Wait 45 seconds for fetch to complete
3. Fetch new jobs via `GET /jobs?status=new&min_score=0.75`
4. Score and assess each match against `target-profile.md`
5. Write lead report to `agent-data/pipeline/leads/YYYY-MM-DD-morning.md`
6. Post summary to `#job-search` Slack channel for jobs scoring ≥0.90

**Workspace**: `~/.openclaw/workspace-market-intel/`

---

### 4. 🔍 Market Intel Afternoon Scan
**Schedule**: Every day at 13:00 UK  
**Cron Expression**: `0 13 * * *`  
**Job ID**: `cca74bf9-207c-44a5-bfc7-8ad9e00f2653`  
**Model**: `anthropic/claude-sonnet-4-5` (default)

**Purpose**: Midday job market scan to catch lunchtime postings.

**Tasks**: Same as morning scan, writes to `YYYY-MM-DD-afternoon.md`

**Workspace**: `~/.openclaw/workspace-market-intel/`

---

### 5. 🔍 Market Intel Evening Scan
**Schedule**: Every day at 19:00 UK  
**Cron Expression**: `0 19 * * *`  
**Job ID**: `dec45403-3046-455b-96fe-7446599a3b30`  
**Model**: `anthropic/claude-sonnet-4-5` (default)

**Purpose**: Evening job market scan to catch end-of-day postings.

**Tasks**: Same as morning scan, writes to `YYYY-MM-DD-evening.md`

**Workspace**: `~/.openclaw/workspace-market-intel/`

---

### 6. 📊 App Tracker Pipeline Monitor
**Schedule**: Every 4 hours at 08:00, 12:00, 16:00, 20:00 UK  
**Cron Expression**: `0 8,12,16,20 * * *`  
**Job ID**: `479b188f-b603-4b1a-8902-6b50796c97bb`  
**Model**: `anthropic/claude-sonnet-4-5` (default)

**Purpose**: Monitors active applications, schedules follow-ups, tracks metrics.

**Tasks**:
1. Fetch follow-ups due via `GET /api/agents/follow-ups-due`
2. Check for applications needing Day 7/14/21 follow-ups
3. Draft follow-up emails (save to `agent-data/follow-ups/`)
4. Post to `#job-search` Slack channel with follow-up drafts requiring Nick's approval
5. Track response rates and conversion metrics
6. Alert if interviews confirmed (triggers Interview Prep via Coordinator)

**Workspace**: `~/.openclaw/workspace-app-tracker/`

---

## Daily Timeline

```
07:00 UK → Market Intel: Morning Scan
07:15 UK → Coordinator: Daily Planning & Briefing
08:00 UK → App Tracker: Pipeline Monitor (1/4)
12:00 UK → App Tracker: Pipeline Monitor (2/4)
13:00 UK → Market Intel: Afternoon Scan
16:00 UK → App Tracker: Pipeline Monitor (3/4)
19:00 UK → Market Intel: Evening Scan
20:00 UK → App Tracker: Pipeline Monitor (4/4)

Sunday 18:00 UK → Coordinator: Weekly Review
```

---

## Verifying Jobs Are Running

### List All JobHunt Cron Jobs
```bash
openclaw cron list | grep -i jobhunt
```

### View Specific Job Details
```bash
openclaw cron status <job-id>
```

### Check Recent Run History
```bash
openclaw cron runs <job-id> --limit 10
```

### Check Next Run Time
```bash
openclaw cron list --json | jq '.[] | select(.name | contains("JobHunt")) | {name, nextRunAtMs}'
```

---

## Troubleshooting

### Job Not Running?

1. **Check if job is enabled**:
   ```bash
   openclaw cron list | grep <job-id>
   ```
   Look for "Status: enabled" vs "disabled"

2. **Check recent run history**:
   ```bash
   openclaw cron runs <job-id> --limit 5
   ```
   Look for errors, timeouts, or "skipped" status

3. **Check Gateway status**:
   ```bash
   openclaw gateway status
   ```
   Ensure Gateway is running

4. **Manually trigger a job (for testing)**:
   ```bash
   openclaw cron run <job-id>
   ```

### Job Running But Failing?

1. **Check logs**:
   ```bash
   openclaw cron runs <job-id> --limit 1
   ```
   Look for error messages, API failures, or timeout issues

2. **Common issues**:
   - **API endpoint not responding**: Check Mission Control FastAPI is running
   - **Workspace files missing**: Ensure `target-profile.md` exists in workspace
   - **Slack channel not configured**: Verify `#job-search` channel exists
   - **Session timeout**: Default 30s may be too short for complex operations

3. **Test the job manually with verbose output**:
   ```bash
   openclaw cron run <job-id> --json | jq
   ```

### Job Taking Too Long?

Default timeout is 30 seconds. Extend if needed:

```bash
openclaw cron edit <job-id> --timeout 60000  # 60 seconds
```

---

## Disabling/Enabling Jobs

### Disable a Job (stops scheduling, preserves config)
```bash
openclaw cron disable <job-id>
```

### Re-enable a Job
```bash
openclaw cron enable <job-id>
```

### Remove a Job Permanently
```bash
openclaw cron rm <job-id>
```

⚠️ **Warning**: Removing is permanent. Use `disable` if you might want to re-enable later.

---

## Editing Job Configuration

### Change schedule
```bash
openclaw cron edit <job-id> --cron "0 8 * * *"
```

### Change model
```bash
openclaw cron edit <job-id> --model "anthropic/claude-opus-4-5"
```

### Change message/prompt
```bash
openclaw cron edit <job-id> --message "New prompt here"
```

### View current config
```bash
openclaw cron list --json | jq '.[] | select(.id=="<job-id>")'
```

---

## Key Design Decisions

1. **Isolated Sessions**: All jobs use `--session isolated` to prevent context window bloat and ensure clean execution.

2. **Announce Mode**: All jobs use `--announce` to post summaries to the last active channel (typically Slack `#job-search`).

3. **UK Timezone**: All schedules use `Europe/London` timezone for consistency with Nick's location.

4. **Opus for Weekly Review**: Weekly review uses Claude Opus (`anthropic/claude-opus-4-5`) for deeper strategic analysis.

5. **Staggered Market Intel Scans**: Three daily scans (07:00, 13:00, 19:00) ensure comprehensive coverage of job postings throughout the day.

6. **App Tracker Frequency**: Every 4 hours (08:00, 12:00, 16:00, 20:00) provides timely follow-up tracking without excessive polling.

---

## Integration with Other Components

### Mission Control FastAPI
All agents interact with the Mission Control API:
- `POST /jobs/refresh` - Trigger Adzuna fetch
- `GET /jobs?status=new&min_score=0.75` - Fetch new job matches
- `GET /stats` - Pipeline statistics
- `GET /api/agents/follow-ups-due` - Applications needing follow-up
- `PATCH /jobs/<id>` - Update job status

### Slack Integration
All agents post updates to:
- `#job-search` - Main coordination channel
- `#briefing` - Daily briefings
- `#daily` - Daily operations log

### Agent Workspaces
Each agent has an isolated workspace:
- `~/.openclaw/workspace-coordinator/`
- `~/.openclaw/workspace-market-intel/`
- `~/.openclaw/workspace-cv-tailor/`
- `~/.openclaw/workspace-cover-letter/`
- `~/.openclaw/workspace-app-tracker/`
- `~/.openclaw/workspace-interview-prep/`
- `~/.openclaw/workspace-networking/`

---

## Monitoring & Maintenance

### Weekly Checklist
- [ ] Review weekly strategy memo from Coordinator
- [ ] Check cron run history for failures: `openclaw cron runs <job-id>`
- [ ] Verify all jobs completed in the past week
- [ ] Check Slack `#job-search` for agent outputs
- [ ] Review pipeline stats via Mission Control dashboard

### Monthly Review
- [ ] Analyze cron job effectiveness (are scans finding good leads?)
- [ ] Review and optimize cron schedules based on job posting patterns
- [ ] Check for stale company research briefs (>14 days old)
- [ ] Review follow-up conversion rates
- [ ] Adjust model allocations (Sonnet vs Opus) based on budget

---

## Future Enhancements

Potential improvements to consider:

1. **Dynamic Scheduling**: Adjust scan times based on when jobs are typically posted
2. **Smart Follow-Up Timing**: Use ML to optimize follow-up schedules based on response patterns
3. **Proactive Company Research**: Trigger research for trending companies before jobs appear
4. **Interview Correlation**: Correlate job posting patterns with interview scheduling
5. **Budget Optimization**: Auto-switch models based on task complexity and token budget

---

## Emergency Contacts

- **Gateway Issues**: `openclaw gateway restart`
- **Mission Control Down**: Check `http://localhost:5055/api/chat/health`
- **Slack Integration Broken**: Verify channel IDs in OpenClaw config
- **Database Issues**: Check PostgreSQL status: `brew services list | grep postgresql`

---

## References

- Agent Specification Cards: `~/projects/claude_jobhunt/docs/OpenClawIntegration/06-AGENT-SPECIFICATION-CARDS.md`
- Daily Operations Cycle: `~/projects/claude_jobhunt/docs/OpenClawIntegration/03-DAILY-OPERATIONAL-CYCLE.mermaid`
- Daily Sequence Diagram: `~/projects/claude_jobhunt/docs/OpenClawIntegration/03b-DAILY-SEQUENCE.mermaid`
- OpenClaw Cron Docs: https://docs.openclaw.ai/cli/cron

---

**Last Updated**: 2026-02-25  
**Maintainer**: Strategy Coordinator Agent  
**Status**: ✅ Production Ready
