# API Documentation

This document provides detailed documentation for all API endpoints in the AI-Powered Job Search Agent.

## Base URL

- Development: `http://localhost:8000`
- Production: Configure via `NEXT_PUBLIC_API_URL`

## Authentication

All endpoints (except `/auth/login`) require a valid session cookie obtained from login.

### POST /auth/login

Login with password to obtain session cookie.

**Request Body:**
```json
{
  "password": "your-password"
}
```

**Response:**
```json
{
  "message": "Login successful"
}
```

Sets an httpOnly cookie `session_token` containing a JWT.

### POST /auth/logout

Clear session cookie.

**Response:**
```json
{
  "message": "Logged out"
}
```

---

## Jobs API

### GET /jobs

List jobs with optional filtering.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `status` | string | Filter by status: `new`, `saved`, `applied`, `interviewing`, `offered`, `rejected` |
| `min_score` | number | Minimum match score (0-100) |
| `source` | string | Filter by job source (e.g., `adzuna`) |
| `search` | string | Search in title and company |
| `skip` | number | Pagination offset (default: 0) |
| `limit` | number | Results per page (default: 50, max: 100) |

**Response:**
```json
{
  "jobs": [
    {
      "id": "uuid",
      "title": "Senior Python Developer",
      "company": "Tech Corp",
      "location": "London",
      "description": "...",
      "url": "https://...",
      "source": "adzuna",
      "status": "new",
      "match_score": 85.5,
      "created_at": "2024-01-15T10:30:00Z",
      "notes": ""
    }
  ],
  "total": 150
}
```

### GET /jobs/{id}

Get detailed job information.

**Response:**
```json
{
  "id": "uuid",
  "title": "Senior Python Developer",
  "company": "Tech Corp",
  "location": "London",
  "description": "Full job description...",
  "url": "https://...",
  "source": "adzuna",
  "status": "new",
  "match_score": 85.5,
  "match_reasons": [
    "Strong semantic match (0.82)",
    "Skills match: python, fastapi, aws",
    "Seniority alignment: senior"
  ],
  "created_at": "2024-01-15T10:30:00Z",
  "notes": ""
}
```

### PATCH /jobs/{id}

Update job status or notes.

**Request Body:**
```json
{
  "status": "saved",
  "notes": "Good match, will apply tomorrow"
}
```

**Response:** Updated job object.

### POST /jobs/refresh

Trigger manual job fetch from all sources.

**Response:**
```json
{
  "message": "Job refresh started",
  "task_id": "uuid"
}
```

---

## Profile API

### GET /profile

Get user profile and CV.

**Response:**
```json
{
  "id": "uuid",
  "cv_text": "Your CV content...",
  "preferred_locations": ["London", "Remote"],
  "target_seniority": "senior",
  "target_roles": ["Backend Developer", "Python Engineer"],
  "excluded_companies": ["Example Corp"],
  "updated_at": "2024-01-15T10:30:00Z"
}
```

### PUT /profile

Update profile and CV. Triggers re-scoring of all jobs.

**Request Body:**
```json
{
  "cv_text": "Your updated CV...",
  "preferred_locations": ["London", "Manchester", "Remote"],
  "target_seniority": "senior",
  "target_roles": ["Backend Developer"],
  "excluded_companies": []
}
```

**Response:** Updated profile object.

---

## Stats API

### GET /stats

Get dashboard statistics.

**Response:**
```json
{
  "total_jobs": 500,
  "new_jobs": 150,
  "saved_jobs": 25,
  "applied_jobs": 10,
  "avg_match_score": 72.5,
  "top_companies": [
    {"name": "Tech Corp", "count": 15},
    {"name": "Startup Inc", "count": 12}
  ],
  "jobs_by_status": {
    "new": 150,
    "saved": 25,
    "applied": 10,
    "interviewing": 3,
    "offered": 1,
    "rejected": 5
  },
  "last_refresh": "2024-01-15T10:30:00Z"
}
```

---

## Skills API (Phase 2)

### POST /api/skills/extract

Extract skills from text using LLM (GPT-4o-mini).

**Request Body:**
```json
{
  "text": "Looking for a Python developer with AWS experience and strong communication skills"
}
```

**Response:**
```json
{
  "skills": [
    {
      "name": "Python",
      "category": "technical",
      "required": true,
      "confidence": "high"
    },
    {
      "name": "AWS",
      "category": "technical",
      "required": true,
      "confidence": "high"
    },
    {
      "name": "communication",
      "category": "soft",
      "required": true,
      "confidence": "high"
    }
  ],
  "count": 3
}
```

**Categories:**
- `technical` - Programming languages, frameworks, platforms
- `soft` - Communication, leadership, teamwork
- `domain` - Industry-specific knowledge
- `tool` - Specific software tools

### GET /api/skills/search

Search the ESCO skills database (13,890+ standardized skills).

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | string | Search query (min 2 chars) |
| `limit` | number | Max results (default: 20, max: 100) |

**Response:**
```json
[
  {
    "uri": "http://data.europa.eu/esco/skill/...",
    "preferred_label": "Python programming",
    "alt_labels": ["Python", "python3", "Python 3"],
    "description": "Write and maintain programs using Python...",
    "skill_type": "skill"
  }
]
```

### GET /api/skills/gaps

Analyze skill gaps between your CV and target jobs.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `limit` | number | Max gaps to return (default: 20, max: 50) |

**Response:**
```json
[
  {
    "skill": "kubernetes",
    "frequency": 0.85,
    "importance": "critical",
    "category": "technical",
    "related_skills_present": ["docker", "containerization"]
  },
  {
    "skill": "terraform",
    "frequency": 0.65,
    "importance": "important",
    "category": "technical",
    "related_skills_present": ["aws", "infrastructure"]
  }
]
```

**Importance Levels:**
- `critical` - Appears in >70% of target jobs
- `important` - Appears in 40-70% of target jobs
- `nice-to-have` - Appears in <40% of target jobs

### GET /api/skills/gaps/summary

Get aggregated skill gap statistics.

**Response:**
```json
{
  "total_gaps": 15,
  "critical_gaps": 3,
  "technical_gaps": 10,
  "soft_gaps": 5,
  "top_gaps": [...],
  "coverage_score": 65.5
}
```

The `coverage_score` (0-100) indicates what percentage of required skills in target jobs you already have.

### GET /api/skills/recommendations

Get personalized learning recommendations based on skill gaps.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `max_skills` | number | Max recommendations (default: 5, max: 10) |

**Response:**
```json
[
  {
    "skill": "kubernetes",
    "category": "technical",
    "importance": "critical",
    "rationale": "Required in 85% of target jobs; Related to skills you have: Docker"
  }
]
```

### GET /api/skills/infer

Infer additional skills from explicit skills using the skill relationship graph.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `skills` | string | Comma-separated list of skills |
| `include_related` | boolean | Include weaker related skills (default: false) |

**Example:**
```
GET /api/skills/infer?skills=kubernetes,python
```

**Response:**
```json
["containerization", "docker", "kubernetes", "programming", "python"]
```

### GET /api/skills/esco/{uri}

Get a specific ESCO skill by its URI.

**Response:**
```json
{
  "uri": "http://data.europa.eu/esco/skill/...",
  "preferred_label": "Python programming",
  "alt_labels": ["Python", "python3"],
  "description": "...",
  "skill_type": "skill"
}
```

---

## Search API (Phase 3)

### POST /api/search/hybrid

Perform hybrid search combining BM25 keyword search and semantic embedding search.

**Request Body:**
```json
{
  "query_text": "Python backend developer with AWS experience",
  "query_embedding": [...],  // Optional: pre-computed 1536-dim embedding
  "top_k": 50,
  "bm25_weight": 0.5,
  "semantic_weight": 0.5,
  "use_rrf": true,
  "use_reranker": true,
  "required_skills": ["python", "aws"]  // Optional: must-have skills
}
```

**Parameters:**
| Field | Type | Description |
|-------|------|-------------|
| `query_text` | string | Text query for BM25 search (required) |
| `query_embedding` | float[] | Pre-computed embedding (optional) |
| `top_k` | number | Max results (default: 50, max: 200) |
| `bm25_weight` | float | Weight for keyword search (0-1, default: 0.5) |
| `semantic_weight` | float | Weight for semantic search (0-1, default: 0.5) |
| `use_rrf` | boolean | Use Reciprocal Rank Fusion (default: true) |
| `use_reranker` | boolean | Apply cross-encoder re-ranking (default: true) |
| `required_skills` | string[] | Skills that must appear in job description |

**Response:**
```json
{
  "results": [
    {
      "job_id": "uuid",
      "title": "Senior Python Developer",
      "company": "Tech Corp",
      "location": "London",
      "match_score": 85.5,
      "hybrid_score": 0.92,
      "rerank_score": 0.89,
      "description_preview": "We are looking for..."
    }
  ],
  "total": 50,
  "query_text": "Python backend developer...",
  "search_config": {
    "bm25_weight": 0.5,
    "semantic_weight": 0.5,
    "use_rrf": true,
    "use_reranker": true
  }
}
```

### POST /api/search/rerank

Re-rank existing job results using a cross-encoder model.

**Request Body:**
```json
{
  "query": "Your CV text or job preferences...",
  "job_ids": ["uuid1", "uuid2", "uuid3"],
  "top_k": 20,
  "provider": "local"  // or "cohere"
}
```

**Providers:**
- `local` - Uses sentence-transformers CrossEncoder (free, runs locally)
- `cohere` - Uses Cohere Rerank API (paid, higher quality)

**Response:**
```json
{
  "results": [
    {
      "job_id": "uuid",
      "title": "Senior Python Developer",
      "company": "Tech Corp",
      "relevance_score": 0.95
    }
  ],
  "total": 20,
  "provider_used": "local"
}
```

### GET /api/search/status

Get search service health and configuration.

**Response:**
```json
{
  "status": "healthy",
  "bm25_index_size": 500,
  "embedding_count": 500,
  "hybrid_search_available": true,
  "reranker_available": true,
  "config": {
    "embedding_provider": "openai",
    "reranker_provider": "local",
    "hybrid_bm25_weight": 0.5,
    "hybrid_semantic_weight": 0.5,
    "hybrid_use_rrf": true,
    "retrieval_candidates": 200,
    "rerank_top_k": 50
  }
}
```

### POST /api/search/rebuild-index

Rebuild the hybrid search index. Call after bulk job updates.

**Response:**
```json
{
  "status": "success",
  "message": "Search index rebuilt"
}
```

---

## Error Responses

All endpoints return standard error responses:

```json
{
  "detail": "Error message describing what went wrong"
}
```

**HTTP Status Codes:**
| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 401 | Unauthorized - Missing or invalid session |
| 404 | Not Found - Resource doesn't exist |
| 422 | Validation Error - Request body validation failed |
| 500 | Internal Server Error - Server-side error |

---

## Rate Limiting

Currently no rate limiting is implemented. For production deployments, consider adding rate limiting via nginx or a reverse proxy.

---

## Caching (Phase 4)

The API implements a 3-tier Redis caching strategy:

| Layer | TTL | Purpose |
|-------|-----|---------|
| L1 Response | 5 min | Full API responses |
| L2 Match Score | 1 hour | Job-profile match calculations |
| L3 Embedding | 24 hours | OpenAI embedding vectors |

Cache is automatically invalidated when:
- Profile/CV is updated (clears match scores and embeddings)
- Jobs are refreshed (clears response cache)
- Manual invalidation via admin endpoints

---

## Dependencies by Phase

### Phase 1 (Core)
- `fastapi`, `sqlalchemy`, `openai`, `apscheduler`

### Phase 2 (Skills Intelligence)
- `networkx` - Skill relationship graphs

### Phase 3 (Advanced ML)
- `rank-bm25` - BM25 keyword search
- `sentence-transformers` - Local cross-encoder models
- `nltk` - Text tokenization
- `cohere` - Optional cloud re-ranking API

### Phase 4 (Infrastructure)
- `redis` - Caching layer
- `celery` - Background task processing
- `chromadb` - Vector database
- `prometheus-client` - Metrics export
