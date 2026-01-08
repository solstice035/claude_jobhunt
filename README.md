# AI-Powered Job Search Agent

An intelligent job search application that uses AI to match your CV against job postings and rank them by relevance.

## Features

### Core Features
- **AI-Powered Matching**: Uses OpenAI embeddings to semantically compare your CV with job descriptions
- **Multi-Factor Scoring**: Combines semantic similarity, skills overlap, seniority alignment, and location matching
- **Automated Job Fetching**: Background scheduler fetches jobs from Adzuna every 6 hours
- **Pipeline Management**: Track jobs through stages: New -> Saved -> Applied -> Interviewing -> Offered/Rejected
- **Real-Time Filtering**: Filter by match score, status, source, and search terms

### Enhanced Job Matching (Phase 2-4)

#### Skills Intelligence (Phase 2)
- **ESCO Skills Taxonomy**: 13,890+ standardized skills from the European Skills/Competences, Qualifications and Occupations database
- **LLM-Powered Skill Extraction**: Uses GPT-4o-mini to extract skills from job descriptions and CVs with category classification (technical, soft, domain, tool)
- **Skill Gap Analysis**: Identifies missing skills between your CV and target jobs, prioritized by importance
- **Learning Recommendations**: Personalized suggestions for which skills to learn based on job market demand
- **Skill Inference**: Expands skill sets using relationship graphs (e.g., knowing Docker implies container knowledge)

#### Advanced ML Matching (Phase 3)
- **Hybrid Search**: Combines BM25 keyword search with semantic embedding search using Reciprocal Rank Fusion (RRF)
- **Two-Stage Retrieval**: Fast recall stage (200 candidates) followed by precise cross-encoder re-ranking (top 50)
- **Cross-Encoder Re-ranking**: Uses sentence-transformers or Cohere API for deep semantic comparison
- **Configurable Weights**: Tune the balance between keyword and semantic matching

#### Infrastructure & Scale (Phase 4)
- **Redis Caching**: 3-tier cache (responses 5min, match scores 1hr, embeddings 24hr)
- **ChromaDB Vector Store**: HNSW-indexed persistent vector database for fast similarity search
- **Background Processing**: Celery with Redis broker for async job processing
- **Prometheus Metrics**: Production-ready observability

## Tech Stack

### Backend
- **FastAPI** - Async Python web framework
- **SQLAlchemy** (async) - ORM with SQLite
- **OpenAI API** - text-embedding-3-small for semantic search
- **APScheduler** - Background job scheduling
- **Adzuna API** - UK job listings
- **Redis** - Caching and Celery broker
- **ChromaDB** - Vector database for embeddings
- **Cohere** (optional) - High-quality re-ranking API
- **sentence-transformers** - Local cross-encoder models

### Frontend
- **Next.js 16** - React framework with App Router
- **TypeScript** - Type safety
- **Tailwind CSS** - Utility-first styling
- **shadcn/ui** - Component library

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Next.js UI    │────▶│   FastAPI API   │────▶│    SQLite DB    │
│   (Port 3000)   │     │   (Port 8000)   │     │                 │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                       │
        ┌────────────────────────┼───────────────────────┤
        │                        │                       │
        ▼                        ▼                       ▼
┌──────────────┐   ┌───────────────────┐   ┌───────────────────┐
│    Redis     │   │     ChromaDB      │   │    APScheduler    │
│  (Caching)   │   │  (Vector Store)   │   │    (6 hr jobs)    │
└──────────────┘   └───────────────────┘   └───────────────────┘
                                 │
        ┌────────────────────────┼────────────────────────┐
        │                        │                        │
        ▼                        ▼                        ▼
┌──────────────┐   ┌───────────────────┐   ┌───────────────────┐
│   Adzuna     │   │  OpenAI / Local   │   │ Cohere (optional) │
│     API      │   │    Embeddings     │   │    Re-ranking     │
└──────────────┘   └───────────────────┘   └───────────────────┘
```

### Search Pipeline

```
                    ┌─────────────────────────────────────┐
                    │         User Query / CV             │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │        Skill Extraction (LLM)       │
                    │    GPT-4o-mini extracts skills      │
                    └─────────────────┬───────────────────┘
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
          ▼                           ▼                           ▼
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  BM25 Keyword   │       │ Semantic Search │       │  Skill Filter   │
│     Search      │       │  (Embeddings)   │       │  (ESCO Match)   │
└────────┬────────┘       └────────┬────────┘       └────────┬────────┘
         │                         │                         │
         └────────────┬────────────┘                         │
                      ▼                                      │
          ┌─────────────────┐                                │
          │   RRF Fusion    │◀───────────────────────────────┘
          │ (200 candidates)│
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Cross-Encoder  │
          │   Re-ranking    │
          │  (Top 50 jobs)  │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │  Final Ranked   │
          │     Results     │
          └─────────────────┘
```

## Match Score Algorithm

Jobs are scored 0-100 based on weighted factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| Semantic | 30% | Cosine similarity between CV and job embeddings |
| Skills | 30% | Overlap of tech keywords (python, react, aws, etc.) |
| Seniority | 25% | Alignment of job level (junior → executive) |
| Location | 15% | Match with preferred locations |

## Quick Start

### Docker (Recommended)

1. Copy `.env.example` to `.env` and fill in your API keys
2. Run: `docker compose -f docker-compose.dev.yml up --build`
3. Frontend: http://localhost:3000
4. Backend API: http://localhost:8000/docs

### Manual Setup

#### Prerequisites
- Python 3.12+
- Node.js 20+
- OpenAI API key
- Adzuna API credentials

#### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run server
uvicorn app.main:app --reload
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.example .env.local
# Ensure NEXT_PUBLIC_API_URL=http://localhost:8000

# Run development server
npm run dev
```

### Access the App

1. Open http://localhost:3000
2. Login with your configured password
3. Go to Profile page and paste your CV
4. Jobs will be fetched and scored automatically

## API Endpoints

### Authentication
- `POST /auth/login` - Login with password
- `POST /auth/logout` - Clear session

### Jobs
- `GET /jobs` - List jobs (with filtering)
- `GET /jobs/{id}` - Get job details
- `PATCH /jobs/{id}` - Update job status/notes
- `POST /jobs/refresh` - Trigger manual fetch

### Profile
- `GET /profile` - Get user profile
- `PUT /profile` - Update profile/CV

### Stats
- `GET /stats` - Dashboard statistics

### Skills (Phase 2)
- `POST /api/skills/extract` - Extract skills from text using LLM
- `GET /api/skills/search` - Search ESCO skills database
- `GET /api/skills/gaps` - Get skill gaps for current profile
- `GET /api/skills/gaps/summary` - Get aggregated skill gap statistics
- `GET /api/skills/recommendations` - Get personalized learning recommendations
- `GET /api/skills/infer` - Infer additional skills from explicit skills
- `GET /api/skills/esco/{uri}` - Get ESCO skill by URI

### Search (Phase 3)
- `POST /api/search/hybrid` - Hybrid search with BM25 + semantic + re-ranking
- `POST /api/search/rerank` - Re-rank existing results with cross-encoder
- `GET /api/search/status` - Get search service health and configuration
- `POST /api/search/rebuild-index` - Rebuild the hybrid search index

For detailed API documentation, see [docs/API.md](docs/API.md).

## Environment Variables

### Backend (.env)

#### Required
```bash
SECRET_KEY=your-secret-key              # JWT session encryption (32+ chars)
APP_PASSWORD=your-login-password        # Login password
OPENAI_API_KEY=sk-...                   # OpenAI API key for embeddings
ADZUNA_APP_ID=your-app-id               # Adzuna developer app ID
ADZUNA_API_KEY=your-api-key             # Adzuna API key
```

#### Optional - Database & Storage
```bash
DATABASE_URL=sqlite:///./data/jobs.db   # SQLite connection URL
REDIS_URL=redis://localhost:6379        # Redis for caching (optional)
CHROMA_PERSIST_DIRECTORY=./data/chroma_db  # ChromaDB storage path
```

#### Optional - ML Configuration
```bash
# Embedding provider: "openai" (default) or "local"
EMBEDDING_PROVIDER=openai
LOCAL_EMBEDDING_MODEL=nomic-ai/nomic-embed-text-v1.5

# Re-ranker provider: "local" (default) or "cohere"
RERANKER_PROVIDER=local
LOCAL_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2
COHERE_API_KEY=                         # Required if RERANKER_PROVIDER=cohere

# Hybrid search weights (0-1, must sum to ~1)
HYBRID_BM25_WEIGHT=0.5
HYBRID_SEMANTIC_WEIGHT=0.5
HYBRID_USE_RRF=true                     # Use Reciprocal Rank Fusion

# Two-stage retrieval settings
RETRIEVAL_CANDIDATES=200                # Stage 1: candidates to retrieve
RERANK_TOP_K=50                         # Stage 2: final results after reranking
```

#### Optional - Background Processing
```bash
SCRAPE_INTERVAL_HOURS=6                 # Job fetch interval
CELERY_BROKER_URL=redis://localhost:6379/0
CELERY_RESULT_BACKEND=redis://localhost:6379/0
```

### Frontend (.env.local)
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Project Structure

```
claude_jobhunt/
├── backend/
│   └── app/
│       ├── api/                    # FastAPI route handlers
│       │   ├── skills.py           # Skills extraction & gap analysis
│       │   └── search.py           # Hybrid search endpoints
│       ├── models/                 # SQLAlchemy ORM models
│       │   └── esco.py             # ESCO skills taxonomy model
│       ├── schemas/                # Pydantic request/response schemas
│       ├── services/               # Business logic
│       │   ├── embeddings.py       # OpenAI embedding client
│       │   ├── matcher.py          # Match score calculator
│       │   ├── scrapers/           # Job source integrations
│       │   ├── esco.py             # ESCO skills database service
│       │   ├── skill_extractor.py  # LLM-powered skill extraction
│       │   ├── skill_gaps.py       # Skill gap analysis
│       │   ├── skill_graph.py      # Skill relationship inference
│       │   ├── hybrid_search.py    # BM25 + semantic search
│       │   ├── reranker.py         # Cross-encoder re-ranking
│       │   ├── cache.py            # Redis caching service
│       │   └── vector_db.py        # ChromaDB vector store
│       ├── auth.py                 # Authentication logic
│       ├── scheduler.py            # Background job scheduler
│       └── main.py                 # Application entry point
│
├── docs/
│   ├── API.md                      # Detailed API documentation
│   └── plans/                      # Implementation design docs
│
└── frontend/
    └── src/
        ├── app/           # Next.js pages (App Router)
        ├── components/    # React components
        ├── lib/           # Utilities and API client
        └── types/         # TypeScript type definitions
```

## API Keys Needed

- **OpenAI**: https://platform.openai.com/api-keys (required)
- **Adzuna**: https://developer.adzuna.com/ (required)
- **Cohere**: https://dashboard.cohere.com/api-keys (optional, for enhanced re-ranking)

## Production Deployment

1. Configure `.env` with production values
2. Set up SSL certificates in `nginx/ssl/`
3. Run: `docker compose up -d --build`

## Security Notes

- Passwords are compared using constant-time comparison (`secrets.compare_digest`)
- Session tokens are JWT with httpOnly cookies
- CORS restricted to localhost:3000 by default
- API keys should never be committed to version control

## License

MIT
