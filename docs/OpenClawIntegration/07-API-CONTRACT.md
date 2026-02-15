# API Contract Document

> Complete specification of all new and modified FastAPI endpoints required for agent integration.
> Base URL: `http://localhost:8000`
> Agent endpoints require: `Authorization: Bearer <AGENT_API_KEY>`

---

## Existing Endpoints (Modified)

### GET /jobs

**Modification**: Add `created_after` query parameter.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `status` | string | No | Filter by pipeline status |
| `min_score` | float | No | Minimum match score (0–100) |
| `created_after` | string (ISO 8601) | No | **NEW** — Only return jobs created after this timestamp |
| `limit` | int | No | Max results (default 50) |
| `skip` | int | No | Pagination offset |

**Example**:
```bash
# Jobs from last 6 hours scoring 75+
GET /jobs?status=new&min_score=75&created_after=2026-02-15T01:00:00Z
```

**Response** (unchanged):
```json
[
  {
    "id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
    "title": "Director, RegTech Solutions",
    "company": "Barclays",
    "location": "London",
    "salary_min": 120000,
    "salary_max": 150000,
    "description": "...",
    "url": "https://...",
    "source": "adzuna",
    "status": "new",
    "match_score": 92.0,
    "created_at": "2026-02-15T07:05:23Z"
  }
]
```

---

## New Endpoints — Agent Activity

### POST /api/agents/log

Log an agent action for the audit trail.

**Auth**: Required (Bearer token)

**Request Body**:
```json
{
  "agent_id": "market-intel",
  "action_type": "job_scan",
  "description": "Morning scan complete. 8 new matches found, 3 high-priority.",
  "job_id": null,
  "status": "completed",
  "error_message": null,
  "tokens_used": 12500
}
```

| Field | Type | Required | Values |
|-------|------|----------|--------|
| `agent_id` | string | Yes | `coordinator`, `market-intel`, `cv-tailor`, `cover-letter`, `app-tracker`, `interview-prep`, `networking` |
| `action_type` | string | Yes | `job_scan`, `company_research`, `cv_tailor`, `cover_letter`, `pipeline_update`, `follow_up`, `interview_prep`, `outreach_draft`, `slack_msg`, `error`, `health_check` |
| `description` | string | Yes | Human-readable description of the action |
| `job_id` | string (UUID) | No | Related job ID if applicable |
| `status` | string | No | `completed` (default), `failed`, `pending` |
| `error_message` | string | No | Error details if status is `failed` |
| `tokens_used` | int | No | Estimated tokens consumed |

**Response**: `201 Created`
```json
{
  "status": "logged",
  "id": 456
}
```

---

### GET /api/agents/log

Retrieve recent agent activity.

**Auth**: Required (Bearer token)

**Query Parameters**:
| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `agent_id` | string | No | — | Filter by specific agent |
| `since_hours` | int | No | 24 | Hours to look back (max 168 / 7 days) |
| `action_type` | string | No | — | Filter by action type |
| `status` | string | No | — | Filter by status |
| `limit` | int | No | 50 | Max results (max 200) |

**Response**: `200 OK`
```json
[
  {
    "id": 456,
    "agent_id": "market-intel",
    "action_type": "job_scan",
    "description": "Morning scan complete. 8 new matches found.",
    "job_id": null,
    "status": "completed",
    "error_message": null,
    "tokens_used": 12500,
    "cost_estimate": 0.04,
    "created_at": "2026-02-15T07:12:45Z"
  }
]
```

---

### GET /api/agents/dashboard

Aggregated overview for agent monitoring.

**Auth**: Required (Bearer token)

**Response**: `200 OK`
```json
{
  "agents": [
    {
      "agent_id": "market-intel",
      "last_active": "2026-02-15T07:12:45Z",
      "total_actions": 12,
      "error_count": 0
    },
    {
      "agent_id": "coordinator",
      "last_active": "2026-02-15T07:35:02Z",
      "total_actions": 8,
      "error_count": 0
    }
  ],
  "pipeline": [
    { "status": "new", "count": 45 },
    { "status": "saved", "count": 12 },
    { "status": "applied", "count": 8 },
    { "status": "interviewing", "count": 2 },
    { "status": "offered", "count": 0 },
    { "status": "rejected", "count": 15 }
  ],
  "timestamp": "2026-02-15T07:40:00Z"
}
```

---

## New Endpoints — Follow-up Tracking

### GET /api/agents/follow-ups-due

Get all follow-ups that are due (today or overdue).

**Auth**: Required (Bearer token)

**Response**: `200 OK`
```json
[
  {
    "id": 12,
    "job_id": "b2c3d4e5-f6a7-8901-bcde-f12345678901",
    "title": "Senior Manager, AI Strategy",
    "company": "Goldman Sachs",
    "follow_up_type": "initial",
    "due_date": "2026-02-15",
    "completed_at": null,
    "agent_id": null,
    "notes": null,
    "created_at": "2026-02-08T14:30:00Z"
  }
]
```

---

### POST /api/agents/follow-ups

Create a follow-up schedule for an application.

**Auth**: Required (Bearer token)

**Request Body**:
```json
{
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "follow_ups": [
    { "type": "initial", "due_date": "2026-02-22" },
    { "type": "second", "due_date": "2026-03-01" },
    { "type": "final", "due_date": "2026-03-08" }
  ]
}
```

**Response**: `201 Created`
```json
{
  "status": "scheduled",
  "follow_up_ids": [15, 16, 17]
}
```

---

### PATCH /api/agents/follow-ups/{id}

Mark a follow-up as completed.

**Auth**: Required (Bearer token)

**Request Body**:
```json
{
  "completed_at": "2026-02-22T10:30:00Z",
  "agent_id": "app-tracker",
  "notes": "Follow-up email sent. Nick approved."
}
```

**Response**: `200 OK`
```json
{
  "status": "updated",
  "id": 15
}
```

---

## New Endpoints — Networking Contacts (Phase 4)

### GET /api/agents/networking-contacts

List all networking contacts.

**Auth**: Required (Bearer token)

**Query Parameters**:
| Parameter | Type | Required | Default |
|-----------|------|----------|---------|
| `company` | string | No | — |
| `outreach_status` | string | No | — |
| `connection_type` | string | No | — |
| `limit` | int | No | 50 |

**Response**: `200 OK`
```json
[
  {
    "id": 3,
    "name": "Sarah Chen",
    "company": "Barclays",
    "role": "VP Technology",
    "linkedin_url": "https://linkedin.com/in/sarahchen",
    "email": null,
    "connection_type": "second_degree",
    "shared_background": "ey_alumni",
    "outreach_status": "draft_ready",
    "message_draft": "Hi Sarah, I noticed we share an EY background...",
    "approved_at": null,
    "sent_at": null,
    "response_received": false,
    "follow_up_due": null,
    "job_ids": "[\"a1b2c3d4-e5f6-7890-abcd-ef1234567890\"]",
    "created_at": "2026-02-15T08:00:00Z"
  }
]
```

---

### POST /api/agents/networking-contacts

Add a new contact.

**Auth**: Required (Bearer token)

**Request Body**:
```json
{
  "name": "Sarah Chen",
  "company": "Barclays",
  "role": "VP Technology",
  "linkedin_url": "https://linkedin.com/in/sarahchen",
  "connection_type": "second_degree",
  "shared_background": "ey_alumni",
  "job_ids": "[\"a1b2c3d4-e5f6-7890-abcd-ef1234567890\"]"
}
```

---

### PATCH /api/agents/networking-contacts/{id}

Update contact status or details.

**Auth**: Required (Bearer token)

**Request Body** (all fields optional):
```json
{
  "outreach_status": "approved",
  "approved_at": "2026-02-15T12:00:00Z",
  "message_draft": "Updated message...",
  "sent_at": "2026-02-15T12:30:00Z",
  "response_received": true,
  "follow_up_due": "2026-02-22"
}
```

**Outreach Status Values**: `identified` → `researched` → `draft_ready` → `approved` → `sent` → `responded` | `no_response` → `follow_up_sent` → `closed`

---

## Webhook Endpoint (FastAPI → OpenClaw)

### POST http://127.0.0.1:18789/hooks/agent

**Purpose**: Trigger OpenClaw agents from FastAPI when pipeline events occur.

**Auth**: `x-openclaw-token` header

**When to fire**:
| Event | agentId | Message |
|-------|---------|---------|
| Status → "interviewing" | coordinator | "Interview confirmed for job {id}. Trigger interview prep." |
| Status → "offered" | coordinator | "Offer received for job {id} at {company}! Alert Nick immediately." |
| New job score ≥ 95 | coordinator | "Exceptional match found: job {id} scores 95+. Prioritise immediately." |

**Request Body**:
```json
{
  "message": "Interview confirmed for job 247. Execute interview preparation procedure.",
  "agentId": "coordinator",
  "sessionKey": "hook:interview:247",
  "wakeMode": "now",
  "deliver": true,
  "channel": "slack",
  "to": "channel:C_JOBSEARCH"
}
```

---

## Error Responses (All Endpoints)

| Status Code | Meaning | Agent Action |
|-------------|---------|-------------|
| 200/201 | Success | Process response |
| 400 | Bad request / validation error | Check request format, log error |
| 401 | Missing auth token | Check AGENT_API_KEY env var |
| 403 | Invalid auth token | Regenerate AGENT_API_KEY |
| 404 | Resource not found | Verify ID, skip this item |
| 422 | Validation error (Pydantic) | Check JSON payload types |
| 429 | Rate limited | Wait 60s, retry once |
| 500 | Server error | Wait 30s, retry 3x with backoff, then alert |
| Connection refused | FastAPI not running | Alert Nick via Slack immediately |
| Timeout (>15s) | Server overloaded | Retry once, then skip and alert |
