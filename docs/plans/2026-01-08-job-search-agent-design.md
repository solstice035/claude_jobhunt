# AI-Powered Job Search Agent - Design Document

## Overview

A personal-use application that aggregates job listings from UK data sources, applies intelligent matching against your profile/CV, and presents ranked opportunities through a clean dashboard.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Deployment | Cloud VPS + Docker Compose | Always-on, accessible anywhere |
| Backend | FastAPI | Async scrapers, auto OpenAPI docs, modern Python |
| Frontend | Next.js 16.1 + ShadCN UI | Latest App Router, customizable components |
| Database | SQLite | Simple, easy backup, sufficient for personal use |
| Embeddings | OpenAI text-embedding-3-small | Best quality, negligible cost at this scale |
| Auth | Simple password wall | Single user, session cookie |
| Initial data source | Adzuna API | Free tier, official API, extensible architecture |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        VPS (Docker Compose)                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │   Nginx     │───▶│  Frontend   │    │   Backend   │     │
│  │  (proxy)    │    │  Next.js    │───▶│  FastAPI    │     │
│  └─────────────┘    │  :3000      │    │  :8000      │     │
│                     └─────────────┘    └──────┬──────┘     │
│                                               │             │
│                     ┌─────────────┐    ┌──────▼──────┐     │
│                     │  Scheduler  │    │   SQLite    │     │
│                     │ (APScheduler│───▶│   (jobs.db) │     │
│                     │  in-process)│    └─────────────┘     │
│                     └─────────────┘                         │
│                            │                                │
│                     ┌──────▼──────┐    ┌─────────────┐     │
│                     │   Adzuna    │    │   OpenAI    │     │
│                     │   API       │    │   API       │     │
│                     └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## Backend Structure

```
/backend
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app, startup/shutdown events
│   ├── config.py            # Pydantic Settings (env vars)
│   ├── auth.py              # Simple password auth middleware
│   ├── database.py          # SQLAlchemy async setup
│   │
│   ├── models/              # SQLAlchemy models
│   │   ├── job.py           # Job listings
│   │   ├── profile.py       # User profile + CV
│   │   └── application.py   # Application tracking
│   │
│   ├── schemas/             # Pydantic request/response models
│   │   ├── job.py
│   │   ├── profile.py
│   │   └── application.py
│   │
│   ├── api/                 # Route handlers
│   │   ├── jobs.py          # CRUD + filters
│   │   ├── profile.py       # Profile management
│   │   ├── auth.py          # Login endpoint
│   │   └── stats.py         # Dashboard stats
│   │
│   ├── services/
│   │   ├── scrapers/
│   │   │   ├── base.py      # Abstract scraper interface
│   │   │   └── adzuna.py    # Adzuna implementation
│   │   ├── matcher.py       # Scoring algorithm
│   │   └── embeddings.py    # OpenAI embedding calls
│   │
│   └── scheduler.py         # APScheduler job definitions
│
├── alembic/                 # Database migrations
├── requirements.txt
├── Dockerfile
└── .env.example
```

## Frontend Structure

```
/frontend
├── src/
│   ├── app/
│   │   ├── layout.tsx           # Root layout + auth check
│   │   ├── page.tsx             # Dashboard home
│   │   ├── login/
│   │   │   └── page.tsx         # Password login form
│   │   ├── jobs/
│   │   │   ├── page.tsx         # Job listings with filters
│   │   │   └── [id]/
│   │   │       └── page.tsx     # Job detail + match breakdown
│   │   ├── profile/
│   │   │   └── page.tsx         # CV + preferences config
│   │   └── applications/
│   │       └── page.tsx         # Application tracker
│   │
│   ├── components/
│   │   ├── ui/                  # ShadCN components
│   │   ├── layout/
│   │   │   ├── Sidebar.tsx
│   │   │   └── Header.tsx
│   │   ├── jobs/
│   │   │   ├── JobCard.tsx
│   │   │   ├── JobDetail.tsx
│   │   │   ├── FilterPanel.tsx
│   │   │   └── MatchBreakdown.tsx
│   │   └── profile/
│   │       └── ProfileForm.tsx
│   │
│   ├── lib/
│   │   ├── api.ts
│   │   └── utils.ts
│   │
│   └── types/
│       └── index.ts
│
├── tailwind.config.ts
├── next.config.js
├── Dockerfile
└── package.json
```

## Matching Algorithm

```
                    ┌─────────────────┐
                    │   New Job       │
                    │   Listing       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Generate       │
                    │  Embedding      │◄──── OpenAI text-embedding-3-small
                    └────────┬────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
┌───────▼───────┐  ┌─────────▼────────┐  ┌───────▼───────┐
│ Semantic      │  │ Keyword          │  │ Structured    │
│ Similarity    │  │ Extraction       │  │ Matching      │
│ (30%)         │  │ (30%)            │  │ (40%)         │
└───────┬───────┘  └─────────┬────────┘  └───────┬───────┘
        │                    │                    │
        │  cosine(cv_emb,    │  skills overlap,   │  seniority (25%)
        │  job_emb)          │  tech matches      │  location (15%)
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Composite      │
                    │  Score (0-100)  │
                    └─────────────────┘
```

### Scoring Weights (Configurable)

| Component | Weight | Method |
|-----------|--------|--------|
| Semantic similarity | 30% | Cosine similarity of embeddings |
| Skills match | 30% | Keyword extraction + fuzzy matching |
| Seniority alignment | 25% | Title parsing |
| Location match | 15% | Exact/fuzzy against preferences |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/auth/login` | Password login, returns session cookie |
| `POST` | `/api/auth/logout` | Clear session |
| `GET` | `/api/jobs` | List jobs (filters: status, score_min, source, search) |
| `GET` | `/api/jobs/:id` | Job detail with full match breakdown |
| `PATCH` | `/api/jobs/:id` | Update status, notes |
| `POST` | `/api/jobs/refresh` | Manual scrape trigger |
| `GET` | `/api/profile` | Get profile + preferences |
| `PUT` | `/api/profile` | Update profile (triggers CV re-embedding) |
| `GET` | `/api/stats` | Counts by status, avg score, sources breakdown |

## Data Models

### Job
```typescript
interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  salaryMin?: number;
  salaryMax?: number;
  description: string;
  url: string;
  source: 'adzuna';  // Extensible
  postedAt: Date;
  closingDate?: Date;
  matchScore: number;
  matchReasons: string[];
  status: 'new' | 'saved' | 'applied' | 'interviewing' | 'offered' | 'rejected' | 'archived';
  notes?: string;
  createdAt: Date;
  updatedAt: Date;
}
```

### Profile
```typescript
interface Profile {
  id: string;
  cvText: string;
  cvEmbedding: number[];
  targetRoles: string[];
  targetSectors: string[];
  locations: string[];
  salaryMin?: number;
  salaryTarget?: number;
  excludeKeywords: string[];
  scoreWeights: {
    semantic: number;
    skills: number;
    seniority: number;
    location: number;
  };
}
```

## Docker & Deployment

### Environment Variables
```bash
OPENAI_API_KEY=sk-...
ADZUNA_APP_ID=...
ADZUNA_API_KEY=...
APP_PASSWORD=your-secure-password
SECRET_KEY=random-secret-for-sessions
```

### Deployment
```bash
# On VPS
git clone <repo>
cp .env.example .env
# Edit .env with your keys
docker compose up -d
```

## Implementation Phases

1. **Project Scaffolding** - Monorepo, Docker Compose, basic apps
2. **Core Backend** - Models, migrations, auth, CRUD
3. **Data Collection** - Adzuna client, scheduler, deduplication
4. **AI Matching** - Embeddings, scoring, match reasons
5. **Frontend** - All pages and components
6. **Polish & Deploy** - Nginx, SSL, production config
