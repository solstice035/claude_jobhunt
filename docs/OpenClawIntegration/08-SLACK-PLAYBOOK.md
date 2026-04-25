---
aliases: [Slack Interaction Playbook]
linter-yaml-title-alias: Slack Interaction Playbook
date created: 2026-02-15 14:32:10 pm
date modified: 2026-03-31 12:45:18 pm
---

# Slack Interaction Playbook

> Complete reference for all Slack interactions between Nick and the agent system.

---

## Channel Map

| Channel           | Purpose                                                                                                                                                                    | Who Posts                    | Nick's Role                         |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------- | ----------------------------------- |
| `#job-search`     | All operational messages: daily briefings, high-match alerts, strategy memos, pipeline updates, materials for review, outreach drafts, interview prep, follow-up reminders | All agents (via Coordinator) | Read, command, **review & approve** |
| `#job-agents-ops` | Health alerts, errors, system status                                                                                                                                       | All (via Coordinator)        | Monitor                             |
| DM with bot       | Direct commands, urgent approvals                                                                                                                                          | Coordinator                  | Command & control                   |

> **Design note**: Consolidated from 6 channels to 2. Six channels for one person creates noise without benefit—a single busy channel is more engaging than six quiet ones. Split out channels later only if volume justifies it.

---

## Commands Nick Can Issue

### Via DM to @JobHunt Agent

| Command | What Happens |
|---------|-------------|
| `status` | Coordinator queries /stats and /api/agents/dashboard, responds with pipeline summary |
| `scan now` | Coordinator spawns Market Intel for immediate scan |
| `research [company]` | Coordinator spawns Market Intel for company deep-dive |
| `prepare materials for job [ID]` | Coordinator spawns CV Tailor + Cover Letter for the specified job |
| `prep interview for [company]` | Coordinator spawns Interview Prep with available context |
| `find contacts at [company]` | Coordinator spawns Networking for contact identification |
| `weekly review` | Coordinator runs the weekly review procedure immediately |
| `pause agents` | Coordinator acknowledges (manual: stop cron jobs via OpenClaw CLI) |
| `resume agents` | Coordinator acknowledges (manual: restart cron jobs) |

### Via Channel Replies

| Context | Nick's Response | What Happens |
|---------|----------------|-------------|
| Materials posted in `#job-search` | `approved` | Coordinator processes → App Tracker logs application |
| Materials posted in `#job-search` | `edit: [feedback]` | Coordinator re-spawns agent with feedback |
| Materials posted in `#job-search` | `skip` | Coordinator logs skip, moves to next priority |
| Follow-up draft in `#job-search` | `approved` | App Tracker marks follow-up ready to send |
| Follow-up draft in `#job-search` | `skip` | Follow-up deferred |
| Outreach draft in `#job-search` | `approved` or 👍 reaction | Coordinator marks contact as approved |
| Outreach draft in `#job-search` | `edit: [feedback]` | Networking agent revises draft |
| Outreach draft in `#job-search` | `skip` | Contact marked as skipped |

---

## Message Formats

### Daily Briefing (`#job-search`, 07:30 UK)

```
📋 Daily Briefing — 15 Feb 2026

📊 Pipeline Status
• New leads today: 8
• Materials in preparation: 2
• Awaiting your review: 1 (see below)
• Applied (active): 6
• Interviews scheduled: 1 (Barclays, 18 Feb)

🎯 Today's Priorities
1. Review materials for Goldman Sachs VP Technology role (score: 0.91)
2. Follow-up due: HSBC Senior Manager — applied 7 days ago
3. New high-match: Capco Director, Controls — score 0.88

⚡ Action Required
• Approve/reject Goldman Sachs materials (posted below)
• Approve follow-up email for HSBC (posted below)
```

### High-Match Alert (`#job-search`, immediate)

```
🚨 High-Match Opportunity

Director, RegTech Solutions — Barclays
Score: 0.92 | London (Hybrid) | £130k-£150k

Why it fits: Direct match to Basel III/IV experience, AI/ML integration
focus aligns with your controls solution work, Director-level seniority.

Skill gaps: Minimal — DORA experience listed as "preferred" not required.

Shall I prepare materials? Reply "prepare" or "skip".
```

### Materials Review (`#job-search`)

```
📝 Materials Ready for Review

🏢 Barclays — Director, RegTech Solutions
📊 Match Score: 0.92
🔗 Job ID: #247

CV Changes Made:
• Led with Basel III/IV regulatory compliance experience
• Added AI controls aggregation project outcomes (30% efficiency gain)
• Emphasised team leadership (10 direct reports → mapped to "15+ team")
• Added keywords: DORA, RegTech, controls automation, risk taxonomy

Cover Letter Approach:
• Opens with Barclays' published DORA compliance timeline challenge
• Connects EY controls solution work directly to their needs
• Military leadership as differentiator for Director-level presence

📄 Files:
• CV: agent-data/cv/tailored/barclays-director-regtech-2026-02-15.md
• Cover: agent-data/cover-letters/barclays-director-regtech-2026-02-15.md

Reply: "approved" | "edit: [your feedback]" | "skip"
```

### Follow-Up Reminder (`#job-search`)

```
📬 Follow-Up Due

HSBC — Senior Manager, Digital Banking
Applied: 8 Feb 2026 (7 days ago)
Status: No response received

Draft follow-up email:
---
Subject: Following up — Senior Manager, Digital Banking application

Dear Hiring Team,

I wanted to follow up on my application for the Senior Manager,
Digital Banking role submitted on 8 February. I remain very
interested in this opportunity and believe my experience in
banking technology transformation at EY would be a strong fit.

I'd welcome the chance to discuss how I could contribute to
HSBC's digital strategy. Please don't hesitate to reach out
if you need any additional information.

Kind regards,
Nick
---

Reply: "approved" | "edit: [your feedback]" | "skip"
```

### Outreach Draft (`#job-search`)

```
🤝 Networking Outreach Draft

Contact: Sarah Chen
Company: Barclays | Role: VP Technology
Connection: 2nd degree (EY alumni)
Related to: Job #247 (Director, RegTech Solutions)

LinkedIn Connection Request (287 chars):
---
Hi Sarah — I noticed we share an EY background (I was a Senior Manager
in Banking & Capital Markets TC). I'm exploring Director-level RegTech
opportunities and would love to learn about your experience at Barclays.
Would you be open to a brief chat?
---

🛑 This will NOT be sent without your approval.
Reply: "approved" | "edit: [your feedback]" | "skip"
```

### Interview Alert (`#job-search`)

```
🎤 Interview Prep Brief Ready

Barclays — Director, RegTech Solutions
📅 18 Feb 2026, 10:00 AM | 📍 Video (Teams)
👤 Interviewer: James Morrison (MD, RegTech)

Full prep brief: agent-data/interview-prep/barclays-director-regtech-2026-02-18.md

Key Points:
• Barclays just announced £200M DORA compliance programme
• James Morrison was previously at Oliver Wyman — values structured thinking
• Expect case question on controls automation ROI
• Salary benchmarks suggest £135k-£155k for this level

Top 3 questions to prepare:
1. "How would you approach building a RegTech controls framework?"
2. "Tell me about leading a large-scale regulatory programme"
3. "How do you see AI/ML transforming compliance?"
```

### Agent Health Alert (`#job-agents-ops`)

```
⚠️ Agent Health Alert

Agent: market-intel
Issue: FastAPI connection refused (3 consecutive attempts)
Last successful action: 2026-02-15T06:45:00Z
Impact: Morning job scan incomplete

Suggested action: Check if Docker containers are running.
Run: docker compose up -d
```

### Weekly Strategy Memo (`#job-search`, Sunday 18:00)

```
📊 Weekly Strategy Review — Week 7, 2026

📈 This Week's Numbers
• Applications sent: 4
• Responses received: 2 (50% response rate)
• Interviews scheduled: 1 (Barclays, 18 Feb)
• New leads identified: 34
• High-priority leads: 8

📊 Trailing 30-Day Metrics
• Total applications: 14
• Response rate: 43%
• Interview conversion: 21%
• Avg. days to response: 5.2

🔍 What's Working
• Banking-specific roles: 60% response rate
• Direct company applications: higher response than job board applies
• Cover letters referencing specific company initiatives: 2x response rate

⚠️ What's Not
• Consulting firm applications: 20% response rate (saturated market?)
• Roles listing "10+ years financial services" — getting filtered

🎯 Recommended Adjustments
1. Shift 60% of effort to direct bank applications (vs. 40% consulting)
2. Add "regulatory transformation" to search keywords
3. Consider targeting Tier 2 fintechs — Thought Machine and 10x Banking
   both expanding compliance teams

📅 Next Week's Focus
• Barclays interview prep (18 Feb)
• Follow-ups due: Goldman Sachs (Day 14), JPMC (Day 7)
• New batch of materials for 3 shortlisted roles
```

---

## Approval Flow Timing

The Coordinator checks for approvals via its heartbeat (every 2 hours, 07:00–22:00). This means:

- **Best case**: Nick approves, action taken within ~2 hours
- **Worst case**: Nick approves just after a check, action taken near 2 hours later
- **Outside hours**: Approvals queue until 07:00 next day

For **urgent** items (interview confirmations, offers), Nick can DM the bot directly for immediate processing.

> **Tip**: Use `/jobhunt check` slash command to trigger the Coordinator immediately instead of waiting for the next heartbeat. Alternatively, consider configuring a Slack event trigger so the Coordinator wakes only when Nick posts in the channel, eliminating polling entirely.

---

## Emoji Reactions

Agents monitor these reactions as approval signals:

| Reaction | Meaning |
|----------|---------|
| 👍 | Approved (same as replying "approved") |
| ❌ | Rejected/skip (same as replying "skip") |
| ✏️ | Needs editing (agent will ask for feedback) |
| 🔥 | Priority—process next |
| ⏸️ | Hold—don't action yet |
