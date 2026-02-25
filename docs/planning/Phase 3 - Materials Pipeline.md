# Phase 3 — Materials Pipeline

**Status**: 🟡 Scaffolded (awaiting master CV)
**Date**: 2026-02-15

## What Was Built

### New Agents
1. **CV Tailor** (`cv-tailor`) — Expert CV writer, ATS-optimised, CAR format, 2-page UK standard
   - Model: `anthropic/claude-opus-4-5` (creative quality)
   - Workspace: `~/.openclaw/workspace-cv-tailor/`
2. **Cover Letter Writer** (`cover-letter`) — Bespoke cover letters, company-specific narrative
   - Model: `anthropic/claude-opus-4-5`
   - Workspace: `~/.openclaw/workspace-cover-letter/`

### Coordinator Updates
- Approval workflow added to Coordinator's AGENTS.md
- Flow: CV Tailor → Cover Letter → Slack approval post → Nick responds → action
- Responses: "approved" / "edit [feedback]" / "skip"

### Config Changes
- Both agents added to `openclaw.json`
- Coordinator can spawn: market-intel, app-tracker, cv-tailor, cover-letter
- Main agent can spawn all five specialist agents

### Output Directories
- `~/projects/claude_jobhunt/agent-data/cv/tailored/`
- `~/projects/claude_jobhunt/agent-data/cover-letters/`
- `~/projects/claude_jobhunt/agent-data/company-research/`

## Integration Tests (2026-02-15)
- ✅ Docker (Colima) running
- ✅ API auth working (cookie JWT)
- ✅ `POST /skills/extract` — returns skills from job text
- ✅ `GET /skills/gaps` — works but needs CV profile text
- ✅ Slack posting to #briefing
- ❌ Slack channel creation — bot lacks `channels:manage` scope
- ✅ All 5 agent workspaces coherent (SOUL, AGENTS, IDENTITY, USER, skills, memory)

## Blockers
- **Master CV**: Needed before CV Tailor can operate. Everything else is ready.
- **Slack channel**: Can't create #applications — need to add `channels:manage` scope to bot, or create manually
- **Gateway restart needed**: Config changes won't take effect until `openclaw gateway restart`

## Next Steps
1. Nick provides master CV → save to `agent-data/cv/master-cv.md`
2. Restart gateway to pick up new agent config
3. Test end-to-end: pick a job → CV Tailor → Cover Letter → Slack approval
4. Add `channels:manage` to Slack bot (optional — can just use #briefing)