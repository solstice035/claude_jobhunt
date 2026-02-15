# Operational Checklist

> Maintenance tasks to keep the job search agent system running smoothly.
> Automated items are handled by agents. Manual items need Nick's attention.

---

## Daily Checklist (2 minutes)

**Automated** (verify these happened via Slack):
- [ ] Morning job scan completed (`#job-search` briefing posted by ~07:30)
- [ ] Daily planning briefing received from Coordinator
- [ ] Afternoon scan completed (~13:15)
- [ ] Evening scan completed (~19:15)
- [ ] Pipeline checks ran (every 4 hours)

**Manual** (Nick's actions):
- [ ] Review any materials, follow-up drafts, and outreach drafts awaiting approval in `#job-search`
- [ ] Respond to any agent health alerts in `#job-agents-ops`
- [ ] Update pipeline with any external changes (e.g., received email response manually, interview confirmed via phone)

---

## Weekly Checklist (Sunday, 15 minutes)

**Automated** (verify via Slack):
- [ ] Weekly strategy review posted by Coordinator (~18:00 Sunday)
- [ ] Metrics are tracking (response rates, conversion rates)

**Manual**:
- [ ] Read the weekly strategy memo
- [ ] Check if past cold-start threshold (15-20 applications sent) — if not, expect baseline-only analysis from the Coordinator
- [ ] Update `target-profile.md` if strategy needs adjustment
- [ ] Update `exclusion-list.md` if any companies/roles should be excluded
- [ ] Review agent activity in dashboard (`http://localhost:3000/agents`)
- [ ] Check API costs:
  ```bash
  # Anthropic usage
  # Check at console.anthropic.com/billing
  
  # OpenAI usage  
  # Check at platform.openai.com/usage
  ```
- [ ] Clear old lead reports if disk space is a concern:
  ```bash
  # Keep last 30 days, archive older
  find ~/claude_jobhunt/agent-data/pipeline/leads/ -name "*.md" -mtime +30 -exec mv {} ~/claude_jobhunt/agent-data/archive/leads/ \;
  ```

---

## Monthly Checklist (30 minutes)

### Security
- [ ] Rotate `AGENT_API_KEY` (see Security doc for procedure)
- [ ] Run `openclaw security audit --deep`
- [ ] Verify file permissions:
  ```bash
  chmod 600 ~/.openclaw/openclaw.json
  chmod -R 700 ~/.openclaw/workspace-*/
  chmod 600 ~/claude_jobhunt/.env
  ```
- [ ] Check for secrets in agent files:
  ```bash
  grep -r "sk-ant\|xoxb\|xapp\|AGENT_API_KEY" ~/.openclaw/workspace-*/
  # Should return nothing
  ```
- [ ] Review agent_activities log for anomalies:
  ```bash
  curl -s "http://localhost:8000/api/agents/log?since_hours=720&status=failed" \
    -H "Authorization: Bearer $AGENT_API_KEY" | jq 'length'
  ```

### System Health
- [ ] Update OpenClaw to latest version:
  ```bash
  npm update -g openclaw
  openclaw --version
  ```
- [ ] Update Docker images:
  ```bash
  cd ~/claude_jobhunt
  docker compose pull
  docker compose up -d --build
  ```
- [ ] Check Mac Mini disk space:
  ```bash
  df -h /
  # Ensure >20% free
  ```
- [ ] Check SQLite database size:
  ```bash
  ls -lh ~/claude_jobhunt/data/jobs.db
  # If >500MB, consider archiving old data
  ```
- [ ] Verify macOS auto-updates haven't broken anything:
  ```bash
  node --version   # Should be 22+
  python3 --version # Should be 3.12+
  ```

### Content Refresh
- [ ] Update `master-cv.md` with any new experience, projects, or skills
- [ ] Review and update `target-profile.md` based on market learnings
- [ ] Archive completed/rejected applications older than 90 days
- [ ] Prune company research files for companies no longer targeted

### Performance Review
- [ ] Review trailing 30-day metrics:
  - Application count
  - Response rate
  - Interview conversion rate
  - Average time in each pipeline stage
  - Cost per application (API costs / applications sent)
- [ ] Compare response rates by: source, company type, role level
- [ ] Assess whether agent outputs (CVs, cover letters) need prompt tuning
- [ ] Evaluate if model assignments are correct (Opus vs Sonnet per agent)

---

## Quarterly Checklist (1 hour)

- [ ] Rotate `ANTHROPIC_API_KEY` (see Security doc)
- [ ] Rotate `OPENAI_API_KEY`
- [ ] Rotate `ADZUNA_API_KEY`
- [ ] Rotate `COHERE_API_KEY` (if used)
- [ ] Review and update agent SOUL.md files based on performance data
- [ ] Review Slack app permissions — remove any unused scopes
- [ ] Back up SQLite database:
  ```bash
  cp ~/claude_jobhunt/data/jobs.db ~/claude_jobhunt/backups/jobs-$(date +%Y%m%d).db
  ```
- [ ] Back up agent workspaces:
  ```bash
  tar -czf ~/backups/openclaw-workspaces-$(date +%Y%m%d).tar.gz ~/.openclaw/workspace-*/
  ```
- [ ] Review OpenClaw changelog for new features that could improve the system
- [ ] Assess whether n8n should be introduced (if OpenClaw scheduling proves insufficient)

---

## Incident Response

### FastAPI Backend Down

**Symptoms**: Agents report "connection refused" in `#job-agents-ops`

**Fix**:
```bash
cd ~/claude_jobhunt
docker compose ps            # Check container status
docker compose logs backend  # Check for errors
docker compose up -d         # Restart
curl http://localhost:8000/stats | jq '.'  # Verify
```

### OpenClaw Gateway Down

**Symptoms**: No Slack messages, no cron jobs firing

**Fix**:
```bash
launchctl list | grep openclaw    # Check daemon status
openclaw start                     # Restart
curl http://127.0.0.1:18789/health # Verify
openclaw cron list                 # Verify cron jobs intact
```

### Agent Stuck / Looping

**Symptoms**: Agent posting repeated messages or consuming high API tokens

**Fix**:
```bash
openclaw sessions list              # Find stuck session
openclaw sessions kill <session-id> # Terminate it
# Review the agent's AGENTS.md for logic that might cause loops
```

### Slack Connection Lost

**Symptoms**: Agents operational but no Slack messages appearing

**Fix**:
```bash
# Check Slack app status at api.slack.com/apps
# Verify tokens haven't expired
openclaw restart  # Reconnects Socket Mode
# Send test message
openclaw chat --agent coordinator --message "Send test to #job-agents-ops"
```

### Mac Mini Restarted / Power Loss

**Expected behaviour**: Everything should auto-recover:
1. macOS starts → launchd starts OpenClaw Gateway
2. Docker Desktop starts → containers come up (FastAPI, Redis, etc.)
3. OpenClaw cron jobs resume on schedule

**Verify after restart**:
```bash
docker compose ps                    # All containers running
curl http://localhost:8000/stats     # FastAPI healthy
curl http://127.0.0.1:18789/health  # OpenClaw healthy
openclaw cron list                   # Cron jobs registered
```

### High API Costs

**Symptoms**: Unexpected charges on Anthropic/OpenAI billing

**Investigate**:
```bash
# Check agent token usage
curl -s "http://localhost:8000/api/agents/log?since_hours=168" \
  -H "Authorization: Bearer $AGENT_API_KEY" | \
  jq '[group_by(.agent_id)[] | {agent: .[0].agent_id, total_tokens: [.[].tokens_used] | add}]'

# Common causes:
# - Agent using Opus for routine tasks (check model assignments)
# - Agent in retry loop (check error counts)  
# - Context window bloat (ensure --session isolated on cron jobs)
```

---

## Backup Strategy

| What | Frequency | Method | Location |
|------|-----------|--------|----------|
| SQLite database | Weekly | `cp jobs.db backups/` | `~/claude_jobhunt/backups/` |
| Agent workspaces | Monthly | `tar -czf` | `~/backups/` |
| Master CV | On change | Git commit | GitHub repo |
| OpenClaw config | On change | Git commit (secrets excluded) | GitHub repo |
| Lead reports | Keep 90 days | Auto-archive older | `agent-data/archive/` |
| Weekly reports | Keep all | No rotation | `agent-data/reports/weekly/` |

### Automated Backup Script

```bash
#!/bin/bash
# ~/scripts/backup-jobhunt.sh
# Run weekly via cron: 0 2 * * 0 ~/scripts/backup-jobhunt.sh

BACKUP_DIR=~/backups/jobhunt
DATE=$(date +%Y%m%d)

mkdir -p $BACKUP_DIR

# SQLite database
cp ~/claude_jobhunt/data/jobs.db $BACKUP_DIR/jobs-$DATE.db

# Agent workspaces (exclude memory sqlite files — large and regenerable)
tar -czf $BACKUP_DIR/workspaces-$DATE.tar.gz \
  --exclude='*.sqlite' \
  ~/.openclaw/workspace-*/

# Agent data (reports and knowledge files)
tar -czf $BACKUP_DIR/agent-data-$DATE.tar.gz \
  ~/claude_jobhunt/agent-data/knowledge/ \
  ~/claude_jobhunt/agent-data/reports/

# Prune backups older than 90 days
find $BACKUP_DIR -name "*.db" -mtime +90 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +90 -delete

echo "Backup complete: $BACKUP_DIR/*-$DATE.*"
```

```bash
# Make executable and schedule
chmod +x ~/scripts/backup-jobhunt.sh
crontab -e
# Add: 0 2 * * 0 ~/scripts/backup-jobhunt.sh
```
