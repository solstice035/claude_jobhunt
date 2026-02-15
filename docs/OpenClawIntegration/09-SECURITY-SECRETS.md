# Security & Secrets Reference

> Complete guide to secrets management, access control, and data protection
> for the JobHunt multi-agent system.

---

## Secrets Inventory

| Secret | Purpose | Storage Location | Used By | Rotation |
|--------|---------|-----------------|---------|----------|
| `ANTHROPIC_API_KEY` | Claude model access | macOS Keychain → env var | OpenClaw Gateway → all agents | Quarterly or on suspected compromise |
| `AGENT_API_KEY` | Agent → FastAPI auth | `.env` file (chmod 600) | All agents (via skill env) | Monthly |
| `OPENAI_API_KEY` | Embeddings for matching | `.env` file (chmod 600) | FastAPI backend | Quarterly |
| `ADZUNA_APP_ID` | Job board API access | `.env` file (chmod 600) | FastAPI backend | N/A (static) |
| `ADZUNA_API_KEY` | Job board API auth | `.env` file (chmod 600) | FastAPI backend | Quarterly |
| `COHERE_API_KEY` | Re-ranking (optional) | `.env` file (chmod 600) | FastAPI backend | Quarterly |
| `SLACK_APP_TOKEN` | Socket Mode connection | `openclaw.json` (chmod 600) | OpenClaw Gateway | On regeneration only |
| `SLACK_BOT_TOKEN` | Bot posting/reading | `openclaw.json` (chmod 600) | OpenClaw Gateway | On regeneration only |
| `OPENCLAW_HOOK_TOKEN` | Webhook authentication | `.env` + `openclaw.json` | FastAPI → OpenClaw | Monthly |
| `SECRET_KEY` | JWT session encryption | `.env` file (chmod 600) | FastAPI backend | Quarterly |
| `APP_PASSWORD` | Human login to dashboard | `.env` file (chmod 600) | Nick (browser) | As needed |

---

## Storage Methods

### macOS Keychain (Preferred for Master Secrets)

Store the Anthropic API key in Keychain for maximum security:

```bash
# Add to Keychain
security add-generic-password \
  -a "openclaw" \
  -s "anthropic-api-key" \
  -w "sk-ant-..." \
  -T "" \
  ~/Library/Keychains/login.keychain-db

# Retrieve in scripts
ANTHROPIC_API_KEY=$(security find-generic-password -a "openclaw" -s "anthropic-api-key" -w)
```

Reference from the launchd plist for the OpenClaw Gateway:
```xml
<key>EnvironmentVariables</key>
<dict>
  <key>ANTHROPIC_API_KEY</key>
  <string>$(security find-generic-password -a "openclaw" -s "anthropic-api-key" -w)</string>
</dict>
```

### Environment Variables (via .env files)

For the FastAPI backend, secrets live in `~/claude_jobhunt/.env`:

```bash
# File permissions — CRITICAL
chmod 600 ~/claude_jobhunt/.env

# Verify
ls -la ~/claude_jobhunt/.env
# Expected: -rw------- 1 nick staff ... .env
```

### OpenClaw Configuration

Secrets in `~/.openclaw/openclaw.json` use `${ENV_VAR}` syntax to reference environment variables rather than storing values directly:

```json
{
  "auth": {
    "anthropic": {
      "apiKey": "${ANTHROPIC_API_KEY}"
    }
  },
  "channels": {
    "slack": {
      "appToken": "${SLACK_APP_TOKEN}",
      "botToken": "${SLACK_BOT_TOKEN}"
    }
  }
}
```

```bash
# File permissions
chmod 600 ~/.openclaw/openclaw.json
```

---

## Access Scoping (Principle of Least Privilege)

### Agent → Secret Access Matrix

| Agent | ANTHROPIC | AGENT_API_KEY | OPENAI | ADZUNA | SLACK | COHERE |
|-------|:---------:|:-------------:|:------:|:------:|:-----:|:------:|
| Coordinator | ✅ (via Gateway) | ✅ (API calls) | ❌ | ❌ | ✅ (via Gateway) | ❌ |
| Market Intel | ✅ (via Gateway) | ✅ (API calls) | ❌ | ❌ | ✅ (via Gateway) | ❌ |
| CV Tailor | ✅ (via Gateway) | ✅ (API calls) | ❌ | ❌ | ❌ | ❌ |
| Cover Letter | ✅ (via Gateway) | ✅ (API calls) | ❌ | ❌ | ❌ | ❌ |
| App Tracker | ✅ (via Gateway) | ✅ (API calls) | ❌ | ❌ | ✅ (via Gateway) | ❌ |
| Interview Prep | ✅ (via Gateway) | ✅ (API calls) | ❌ | ❌ | ❌ | ❌ |
| Networking | ✅ (via Gateway) | ✅ (API calls) | ❌ | ❌ | ✅ (via Gateway) | ❌ |

**Key insight**: Agents never see raw API keys. The Anthropic key is consumed by the OpenClaw Gateway process. The AGENT_API_KEY is injected via the shared skill's `env` configuration. Agents call `curl` with `$AGENT_API_KEY` which the Gateway resolves at runtime.

### Network Access Scoping

```
┌─────────────────────────────────────────────────┐
│ OpenClaw Gateway: bound to 127.0.0.1:18789      │
│ → NOT accessible from other devices on network  │
│                                                   │
│ FastAPI: bound to 0.0.0.0:8000                   │
│ → Accessible on LAN (for Next.js frontend)      │
│ → Add IP allowlist if concerned                  │
│                                                   │
│ Next.js: bound to 0.0.0.0:3000                   │
│ → Accessible on LAN (for Nick's browser)        │
│                                                   │
│ Slack: Outbound WebSocket only (Socket Mode)     │
│ → No inbound ports opened                        │
└─────────────────────────────────────────────────┘
```

**Gateway bind configuration** in `openclaw.json`:
```json
{
  "gateway": {
    "bind": "loopback"
  }
}
```

---

## Sensitive Data Handling

### Data Classification

| Data Type | Sensitivity | Where It Lives | Agent Access |
|-----------|-------------|---------------|-------------|
| Master CV | High | `agent-data/knowledge/master-cv.md` | CV Tailor, Cover Letter, Interview Prep (read) |
| Salary expectations | High | `agent-data/knowledge/target-profile.md` | Coordinator, Market Intel (read) |
| Contact details | High | `networking_contacts` table | Networking, App Tracker (via API) |
| Tailored CVs | Medium | `agent-data/cv/tailored/` | Created by CV Tailor, read by Coordinator |
| Cover letters | Medium | `agent-data/cover-letters/` | Created by Cover Letter, read by Coordinator |
| Job descriptions | Low | `jobs` table | All agents (via API) |
| Agent logs | Low | `agent_activities` table | All agents (write), Coordinator (read) |

### Rules for Agents

These rules are embedded in each agent's SOUL.md:

1. **NEVER write secrets** to MEMORY.md, daily logs, or any agent-writable file
2. **NEVER include salary expectations** in cover letters or outreach messages
3. **NEVER share Nick's contact details** (phone, personal email, home address) in any agent output
4. **NEVER log full CV content** in agent_activities — use summaries only
5. **NEVER store raw job application credentials** (login passwords for job portals) anywhere in the agent system
6. **ALWAYS use the `$AGENT_API_KEY` env var** — never hardcode the key value

### Filesystem Permissions

```bash
# Agent workspaces — owner-only access
chmod -R 700 ~/.openclaw/workspace-*/

# Config file — owner read/write only
chmod 600 ~/.openclaw/openclaw.json

# Agent data directory — owner-only
chmod -R 700 ~/claude_jobhunt/agent-data/

# Verify no world-readable sensitive files
find ~/.openclaw ~/claude_jobhunt/agent-data -perm -o=r -type f 2>/dev/null
# Should return nothing
```

### Full Disk Encryption

Verify macOS FileVault is enabled:
```bash
fdesetup status
# Expected: FileVault is On.

# If not enabled:
sudo fdesetup enable
```

---

## Secret Rotation Procedures

### Rotating AGENT_API_KEY (Monthly)

```bash
# 1. Generate new key
NEW_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")

# 2. Update FastAPI .env
sed -i '' "s/AGENT_API_KEY=.*/AGENT_API_KEY=$NEW_KEY/" ~/claude_jobhunt/.env

# 3. Update OpenClaw skill config
# Edit ~/.openclaw/openclaw.json → skills.entries.jobhunt-api.env.AGENT_API_KEY

# 4. Restart FastAPI
cd ~/claude_jobhunt && docker compose restart backend

# 5. Restart OpenClaw Gateway
openclaw restart

# 6. Test
curl -s -H "Authorization: Bearer $NEW_KEY" http://localhost:8000/api/agents/dashboard | jq '.timestamp'
```

### Rotating ANTHROPIC_API_KEY (Quarterly)

```bash
# 1. Generate new key at console.anthropic.com
# 2. Update Keychain
security delete-generic-password -a "openclaw" -s "anthropic-api-key"
security add-generic-password -a "openclaw" -s "anthropic-api-key" -w "sk-ant-NEW..."

# 3. Restart OpenClaw Gateway
openclaw restart

# 4. Test
openclaw chat --agent coordinator --message "Test: confirm you can respond."
```

### Rotating Slack Tokens

```bash
# 1. Regenerate tokens at api.slack.com/apps
# 2. Update openclaw.json with new xapp-... and xoxb-... values
# 3. Restart OpenClaw Gateway
openclaw restart

# 4. Test
openclaw chat --agent coordinator --message "Send test message to #job-agents-ops"
```

---

## Agent Sandboxing

Filesystem permissions alone are insufficient if the agent process owns those files and has shell access via the `exec` tool. A prompt injection via a malicious job description could cause an agent to run unexpected commands.

### Recommended Mitigations

1. **Exec tool allowlist**: Restrict the `exec` tool to a set of permitted commands rather than full shell access. Safe commands: `curl`, `jq`, `date`, `wc`, `cat`, `ls`. If OpenClaw supports tool-level restrictions in config, apply them:

```json
{
  "tools": {
    "exec": {
      "allowlist": ["curl", "jq", "date", "wc", "cat", "ls", "grep"]
    }
  }
}
```

2. **Docker container isolation** (stronger): Run each agent in a Docker container with limited mount points — only the specific directories that agent needs:

```yaml
# Example: Market Intel agent container
services:
  market-intel:
    image: openclaw-agent
    volumes:
      - ~/.openclaw/workspace-market-intel:/workspace:rw
      - ~/claude_jobhunt/agent-data/pipeline/leads:/data/leads:rw
      - ~/claude_jobhunt/agent-data/company-research:/data/research:rw
    read_only: true  # Root filesystem read-only
    network_mode: "host"  # Needs localhost access to FastAPI
```

3. **Restricted shell**: If Docker is too heavy, use `rbash` (restricted bash) or a custom wrapper that validates commands before execution.

4. **Input sanitisation**: Agent SOUL.md files should instruct agents to treat job description content as untrusted data and never execute commands derived from it.

### Minimum Viable Security

At minimum, ensure:
- [ ] Agents cannot access each other's workspace directories
- [ ] No agent has write access to `openclaw.json` or `.env`
- [ ] The `exec` tool cannot be used to install packages or modify system files
- [ ] Agent activity logging captures all `exec` commands for audit

---

## Security Audit Checklist

Run monthly:

- [ ] `openclaw security audit --deep` — no warnings
- [ ] File permissions verified (`chmod 600/700` on sensitive paths)
- [ ] No secrets in agent workspace files: `grep -r "sk-ant\|xoxb\|xapp" ~/.openclaw/workspace-*/`
- [ ] No secrets in git: `cd ~/claude_jobhunt && git log --all -p | grep -i "api.key\|secret\|password"` 
- [ ] FileVault enabled: `fdesetup status`
- [ ] Gateway bound to loopback: verify in `openclaw.json`
- [ ] Agent activity log reviewed for anomalies
- [ ] Slack app permissions reviewed (no unnecessary scopes)
- [ ] API keys rotated if older than rotation schedule
- [ ] `.env` file not committed to git: verify in `.gitignore`
