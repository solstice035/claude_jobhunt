# JobHunt Multi-Agent System — Documentation Index

> Complete documentation suite for the OpenClaw-based autonomous job search system.
> 11 files covering architecture, agent specifications, API contracts, and operations.

---

## Documents

### Architecture & Flow Diagrams (Mermaid)

| # | File | Purpose |
|---|------|---------|
| 1 | `diagrams/01-SYSTEM-ARCHITECTURE.mermaid` | Full system topology — OpenClaw Gateway, 7 agents, FastAPI stack, external services |
| 2 | `diagrams/02-AGENT-COMMUNICATION-FLOW.mermaid` | Who spawns whom, message routing, approval gates, autonomous vs. gated actions |
| 3 | `diagrams/03-DAILY-OPERATIONAL-CYCLE.mermaid` | Gantt chart of the daily agent schedule (07:00–20:00 UK time) |
| 3b | `diagrams/03b-DAILY-SEQUENCE.mermaid` | Sequence diagram showing the full morning cycle: scan → plan → prepare → approve → apply |
| 4 | `diagrams/04-PIPELINE-STATE-MACHINE.mermaid` | All application status transitions with agent ownership and human approval gates |
| 5 | `diagrams/05-DATA-FLOW.mermaid` | Where data lives (SQLite, filesystem, ChromaDB, Redis, Slack) and how it moves |

### Reference Documents

| # | File | Purpose |
|---|------|---------|
| 6 | `reference/06-AGENT-SPECIFICATION-CARDS.md` | One-page-per-agent quick reference: identity, model, tools, triggers, inputs/outputs, constraints |
| 7 | `reference/07-API-CONTRACT.md` | Complete specification of all new/modified FastAPI endpoints with schemas |
| 8 | `reference/08-SLACK-PLAYBOOK.md` | Every Slack command, message format, approval workflow, and channel map |
| 9 | `reference/09-SECURITY-SECRETS.md` | Secrets inventory, storage methods, access scoping, rotation procedures, audit checklist |
| 10 | `reference/10-OPERATIONAL-CHECKLIST.md` | Daily/weekly/monthly/quarterly maintenance tasks, incident response, backup strategy |

### Master Implementation Guide

| File | Purpose |
|------|---------|
| `IMPLEMENTATION_GUIDE.md` | The comprehensive step-by-step build guide with all code, config, and phased plan |

---

## How to Use These Documents

**Building the system**: Start with `IMPLEMENTATION_GUIDE.md`. Follow the phases sequentially. Reference the diagrams for architectural context and the API contract for endpoint specifications.

**During an OpenClaw/Claude Code session**: Point the session at `IMPLEMENTATION_GUIDE.md` for the relevant phase. Agent spec cards (`06`) provide quick context for each agent being built.

**Day-to-day operation**: Use the Slack Playbook (`08`) for interaction patterns and the Operational Checklist (`10`) for maintenance routines.

**Security reviews**: Use the Security doc (`09`) for audits and secret rotation.

**Understanding the design**: The five Mermaid diagrams provide visual understanding at different levels — from full system architecture down to individual data flows.

---

## Rendering Mermaid Diagrams

Mermaid files can be rendered in:
- **GitHub**: Automatically renders `.mermaid` files in repos
- **VS Code**: Install the "Mermaid Preview" extension
- **Mermaid Live Editor**: Paste content at https://mermaid.live
- **Obsidian**: Native Mermaid support in code blocks
- **Any Markdown viewer**: Wrap in ```mermaid code blocks
