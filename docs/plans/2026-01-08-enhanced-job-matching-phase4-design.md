# Enhanced Job Matching - Phase 4 Design: Infrastructure & Scale

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Production-ready infrastructure supporting high-volume matching with sub-second response times, efficient caching, and comprehensive monitoring.

**Architecture:** Vector database for similarity search, Redis caching layer, async processing pipeline, observability stack.

**Tech Stack:** Python, FastAPI, pgvector/ChromaDB, Redis, Celery, Prometheus, Grafana

**Prerequisites:** Phase 1 complete, Phase 2-3 recommended

---

## 1. Vector Database

### Why Dedicated Vector Storage?

Current approach stores embeddings as JSON arrays in PostgreSQL. Problems at scale:

| Scale | Embeddings | Storage | Query Time |
|-------|------------|---------|------------|
| 1K jobs | 6MB | OK | ~50ms |
| 10K jobs | 60MB | OK | ~200ms |
| 100K jobs | 600MB | Slow | ~2s |
| 1M jobs | 6GB | Very slow | ~20s |

### Options Comparison

| Solution | Type | ANN Index | Filtering | Cost |
|----------|------|-----------|-----------|------|
| **pgvector** | PostgreSQL extension | HNSW, IVFFlat | SQL WHERE | Free |
| **ChromaDB** | Embedded | HNSW | Metadata | Free |
| **Pinecone** | Managed SaaS | Proprietary | Metadata | $70+/mo |
| **Weaviate** | Self-hosted | HNSW | GraphQL | Free |
| **Qdrant** | Self-hosted | HNSW | Payload | Free |

**Recommendation:** pgvector for simplicity (already using PostgreSQL), ChromaDB for development.

### pgvector Setup

```sql
-- Enable extension
CREATE EXTENSION vector;

-- Add vector column to jobs table
ALTER TABLE jobs ADD COLUMN embedding vector(1536);

-- Create HNSW index for fast similarity search
CREATE INDEX ON jobs USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Example query: find similar jobs
SELECT id, title, company,
       1 - (embedding <=> $1) as similarity
FROM jobs
WHERE status = 'new'
  AND match_score > 50
ORDER BY embedding <=> $1
LIMIT 50;
```

### ChromaDB Setup (Development)

```python
import chromadb
from chromadb.config import Settings

# Initialize persistent client
client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(anonymized_telemetry=False)
)

# Create collection
job_collection = client.get_or_create_collection(
    name="jobs",
    metadata={"hnsw:space": "cosine"}
)

# Add jobs
job_collection.add(
    ids=[job.id for job in jobs],
    embeddings=[job.embedding for job in jobs],
    metadatas=[{
        "title": job.title,
        "company": job.company,
        "status": job.status,
        "match_score": job.match_score
    } for job in jobs],
    documents=[job.description for job in jobs]
)

# Query similar jobs
results = job_collection.query(
    query_embeddings=[cv_embedding],
    n_results=50,
    where={"status": "new", "match_score": {"$gt": 50}}
)
```

---

## 2. Caching Strategy

### Cache Layers

```
Request
    │
    ▼
┌─────────────────────────┐
│ L1: Response Cache      │  TTL: 5 min
│ (full API responses)    │  Key: endpoint + params hash
└───────────┬─────────────┘
            │ miss
            ▼
┌─────────────────────────┐
│ L2: Computation Cache   │  TTL: 1 hour
│ (match scores, skills)  │  Key: job_id + profile_hash
└───────────┬─────────────┘
            │ miss
            ▼
┌─────────────────────────┐
│ L3: Embedding Cache     │  TTL: 24 hours
│ (job/CV embeddings)     │  Key: content_hash
└───────────┬─────────────┘
            │ miss
            ▼
        Compute
```

### Redis Implementation

```python
import redis.asyncio as redis
import hashlib
import json

class MatchCache:
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(redis_url)

    def _hash(self, *args) -> str:
        content = json.dumps(args, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    # L1: Response cache
    async def get_response(self, endpoint: str, params: dict) -> Optional[dict]:
        key = f"resp:{endpoint}:{self._hash(params)}"
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None

    async def set_response(self, endpoint: str, params: dict, response: dict):
        key = f"resp:{endpoint}:{self._hash(params)}"
        await self.redis.setex(key, 300, json.dumps(response))  # 5 min TTL

    # L2: Match score cache
    async def get_match_score(self, job_id: str, profile_hash: str) -> Optional[Tuple[float, List[str]]]:
        key = f"match:{job_id}:{profile_hash}"
        cached = await self.redis.get(key)
        if cached:
            data = json.loads(cached)
            return data["score"], data["reasons"]
        return None

    async def set_match_score(self, job_id: str, profile_hash: str, score: float, reasons: List[str]):
        key = f"match:{job_id}:{profile_hash}"
        await self.redis.setex(key, 3600, json.dumps({"score": score, "reasons": reasons}))

    # L3: Embedding cache
    async def get_embedding(self, content_hash: str) -> Optional[List[float]]:
        key = f"emb:{content_hash}"
        cached = await self.redis.get(key)
        return json.loads(cached) if cached else None

    async def set_embedding(self, content_hash: str, embedding: List[float]):
        key = f"emb:{content_hash}"
        await self.redis.setex(key, 86400, json.dumps(embedding))  # 24h TTL
```

### Cache Invalidation

```python
async def invalidate_profile_cache(profile_id: str):
    """Invalidate all caches when profile changes."""
    # Get profile hash
    profile = await get_profile(profile_id)
    profile_hash = hash_profile(profile)

    # Invalidate match scores (pattern delete)
    keys = await cache.redis.keys(f"match:*:{profile_hash}")
    if keys:
        await cache.redis.delete(*keys)

    # Invalidate CV embedding
    cv_hash = hashlib.sha256(profile.cv_text.encode()).hexdigest()[:16]
    await cache.redis.delete(f"emb:{cv_hash}")
```

---

## 3. Async Processing Pipeline

### Background Job Processing

Use Celery for CPU-intensive and long-running tasks:

```python
from celery import Celery

celery_app = Celery(
    "job_matching",
    broker=settings.redis_url,
    backend=settings.redis_url
)

@celery_app.task(bind=True, max_retries=3)
def process_new_jobs(self, job_ids: List[str]):
    """Background task to process newly fetched jobs."""
    try:
        for job_id in job_ids:
            # Generate embedding
            job = get_job(job_id)
            embedding = get_embedding(job.description)

            # Calculate match scores for all profiles
            profiles = get_all_profiles()
            for profile in profiles:
                score, reasons = calculate_match_score(job, profile)
                update_job_score(job_id, profile.id, score, reasons)

            # Update vector index
            update_vector_index(job_id, embedding)

    except Exception as exc:
        self.retry(exc=exc, countdown=60)

@celery_app.task
def recalculate_all_scores(profile_id: str):
    """Recalculate scores when profile changes."""
    profile = get_profile(profile_id)
    jobs = get_all_active_jobs()

    for job in jobs:
        score, reasons = calculate_match_score(job, profile)
        update_job_score(job.id, profile_id, score, reasons)
```

### Job Queue Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Scheduler    │────►│ Redis Queue  │────►│ Celery       │
│ (APScheduler)│     │              │     │ Workers (x3) │
└──────────────┘     └──────────────┘     └──────────────┘
                            │
                            ▼
                     ┌──────────────┐
                     │ Task Types:  │
                     │ - fetch_jobs │
                     │ - embed_job  │
                     │ - score_job  │
                     │ - reindex    │
                     └──────────────┘
```

### Rate Limiting

```python
from celery import chain, group
from celery.exceptions import RateLimitExceeded

# Rate limit OpenAI calls
@celery_app.task(rate_limit="100/m")  # 100 per minute
def generate_embedding(text: str) -> List[float]:
    return openai_embed(text)

# Batch processing with rate limiting
def process_jobs_batch(jobs: List[Job]):
    # Group embedding tasks (rate limited)
    embedding_tasks = group(
        generate_embedding.s(job.description)
        for job in jobs
    )

    # Chain: embeddings -> scoring
    workflow = chain(
        embedding_tasks,
        calculate_scores_batch.s(job_ids=[j.id for j in jobs])
    )

    workflow.apply_async()
```

---

## 4. Monitoring & Observability

### Metrics to Track

| Category | Metric | Alert Threshold |
|----------|--------|-----------------|
| **Latency** | API response time (p50, p95, p99) | p95 > 500ms |
| **Throughput** | Requests per second | < 10 rps |
| **Errors** | Error rate by endpoint | > 1% |
| **Cache** | Hit rate by layer | < 80% |
| **Queue** | Task queue depth | > 1000 |
| **ML** | Embedding generation time | > 2s |
| **ML** | Match score calculation time | > 100ms |

### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency",
    ["method", "endpoint", "status"]
)

REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

# Cache metrics
CACHE_HITS = Counter(
    "cache_hits_total",
    "Cache hit count",
    ["layer"]  # L1, L2, L3
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Cache miss count",
    ["layer"]
)

# ML metrics
EMBEDDING_LATENCY = Histogram(
    "embedding_generation_seconds",
    "Time to generate embeddings",
    ["provider"]
)

MATCH_SCORE_LATENCY = Histogram(
    "match_score_calculation_seconds",
    "Time to calculate match scores"
)

# Queue metrics
QUEUE_DEPTH = Gauge(
    "celery_queue_depth",
    "Number of tasks in queue",
    ["queue_name"]
)
```

### FastAPI Middleware

```python
from fastapi import Request
import time

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()

    response = await call_next(request)

    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).observe(duration)

    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()

    return response
```

### Grafana Dashboards

```yaml
# Dashboard panels (pseudo-config)
panels:
  - title: "API Latency (p95)"
    query: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

  - title: "Cache Hit Rate"
    query: rate(cache_hits_total[5m]) / (rate(cache_hits_total[5m]) + rate(cache_misses_total[5m]))

  - title: "Queue Depth"
    query: celery_queue_depth

  - title: "Error Rate"
    query: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m])
```

---

## 5. Scaling Architecture

### Current (Single Server)

```
┌────────────────────────────────────────┐
│              Single Server             │
│  ┌──────────┐  ┌──────────┐  ┌──────┐ │
│  │ FastAPI  │  │PostgreSQL│  │Redis │ │
│  └──────────┘  └──────────┘  └──────┘ │
└────────────────────────────────────────┘
```

### Target (Scalable)

```
                    ┌─────────────┐
                    │ Load        │
                    │ Balancer    │
                    └──────┬──────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ FastAPI (1)  │  │ FastAPI (2)  │  │ FastAPI (3)  │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  PostgreSQL  │ │    Redis     │ │   Celery     │
│  (Primary)   │ │   Cluster    │ │   Workers    │
│      │       │ └──────────────┘ └──────────────┘
│      ▼       │
│  (Replica)   │
└──────────────┘
```

### Docker Compose (Development)

```yaml
version: "3.8"

services:
  api:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/jobhunt
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  db:
    image: pgvector/pgvector:pg16
    volumes:
      - postgres_data:/var/lib/postgresql/data
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=jobhunt

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  celery:
    build: ./backend
    command: celery -A app.celery worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/jobhunt
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  postgres_data:
  redis_data:
  grafana_data:
```

---

## 6. Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/app/services/vector_db.py` | pgvector/ChromaDB interface |
| `backend/app/services/cache.py` | Multi-layer caching |
| `backend/app/celery.py` | Celery app configuration |
| `backend/app/tasks/` | Background task definitions |
| `backend/app/middleware/metrics.py` | Prometheus middleware |
| `docker-compose.yml` | Multi-service development setup |
| `prometheus.yml` | Metrics collection config |
| `grafana/dashboards/` | Pre-built dashboard JSON |

---

## 7. Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| API latency (p95) | ~500ms | <200ms |
| Cache hit rate | 0% | >85% |
| Embedding cost | $X/month | -50% (via caching) |
| Max concurrent users | ~10 | 100+ |
| Jobs searchable | 1K | 100K+ |

---

## 8. Dependencies

```
Phase 1 ──► Phase 4 (can start immediately)
              │
              ├── Vector DB (pgvector)
              ├── Redis caching
              ├── Celery workers
              └── Monitoring stack

Phase 2 ──► Phase 4 (benefits from caching)
Phase 3 ──► Phase 4 (re-ranking needs queue)
```

Phase 4 can begin in parallel with Phase 2/3. Infrastructure improvements benefit all phases.
