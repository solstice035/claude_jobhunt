# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered job search agent that aggregates UK job listings from Adzuna, matches them against your CV using OpenAI embeddings, and presents ranked opportunities in a dashboard.

## Tech Stack

- **Backend**: Python 3.12+, FastAPI, SQLAlchemy (async), APScheduler
- **Frontend**: Next.js 16.1, TypeScript, TailwindCSS, ShadCN UI
- **Database**: SQLite (via aiosqlite)
- **AI**: OpenAI text-embedding-3-small for semantic matching
- **Deployment**: Docker Compose on VPS with Nginx reverse proxy

## Development Commands

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

### Docker (Full Stack)
```bash
# Development (with hot reload)
docker compose -f docker-compose.dev.yml up --build

# Production
docker compose up -d --build
```

## Architecture

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Nginx     │───▶│  Frontend   │    │   Backend   │
│  (proxy)    │    │  Next.js    │───▶│  FastAPI    │
└─────────────┘    │  :3000      │    │  :8000      │
                   └─────────────┘    └──────┬──────┘
                                             │
                   ┌─────────────┐    ┌──────▼──────┐
                   │  Scheduler  │───▶│   SQLite    │
                   │ (APScheduler)    │   (jobs.db) │
                   └──────┬──────┘    └─────────────┘
                          │
                   ┌──────▼──────┐    ┌─────────────┐
                   │   Adzuna    │    │   OpenAI    │
                   │   API       │    │   API       │
                   └─────────────┘    └─────────────┘
```

## Key Files

### Backend
- `backend/app/main.py` - FastAPI app with lifespan events
- `backend/app/services/scrapers/adzuna.py` - Adzuna API client
- `backend/app/services/matcher.py` - AI matching algorithm
- `backend/app/services/embeddings.py` - OpenAI embedding calls
- `backend/app/scheduler.py` - Background job scheduling

### Frontend
- `frontend/src/app/jobs/page.tsx` - Main job listings
- `frontend/src/app/jobs/[id]/page.tsx` - Job detail view
- `frontend/src/app/profile/page.tsx` - CV and preferences
- `frontend/src/lib/api.ts` - API client wrapper

## Environment Variables

Required in `.env`:
```bash
OPENAI_API_KEY=sk-...          # For embeddings
ADZUNA_APP_ID=...              # From developer.adzuna.com
ADZUNA_API_KEY=...             # From developer.adzuna.com
APP_PASSWORD=...               # Login password
SECRET_KEY=...                 # JWT session encryption (32+ chars)
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/auth/login` | Password login |
| POST | `/auth/logout` | Clear session |
| GET | `/jobs` | List jobs with filters |
| GET | `/jobs/:id` | Job detail |
| PATCH | `/jobs/:id` | Update status/notes |
| POST | `/jobs/refresh` | Trigger manual scrape |
| GET | `/profile` | Get profile |
| PUT | `/profile` | Update profile |
| GET | `/stats` | Dashboard stats |

## Adding New Job Sources

1. Create new scraper in `backend/app/services/scrapers/`
2. Extend `BaseScraper` class
3. Implement `fetch_jobs()` method returning `List[JobCreate]`
4. Register in `backend/app/scheduler.py`

## Matching Algorithm

The match score (0-100) is calculated from:
- **Semantic similarity** (30%): Cosine similarity of CV and job embeddings
- **Skills match** (30%): Keyword extraction and overlap
- **Seniority alignment** (25%): Title parsing for level match
- **Location match** (15%): Preference matching
