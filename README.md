# AI-Powered Job Search Agent

An intelligent job search application that uses AI to match your CV against job postings and rank them by relevance.

## Features

- **AI-Powered Matching**: Uses OpenAI embeddings to semantically compare your CV with job descriptions
- **Multi-Factor Scoring**: Combines semantic similarity, skills overlap, seniority alignment, and location matching
- **Automated Job Fetching**: Background scheduler fetches jobs from Adzuna every 6 hours
- **Pipeline Management**: Track jobs through stages: New → Saved → Applied → Interviewing → Offered/Rejected
- **Real-Time Filtering**: Filter by match score, status, source, and search terms

## Tech Stack

### Backend
- **FastAPI** - Async Python web framework
- **SQLAlchemy** (async) - ORM with SQLite
- **OpenAI API** - text-embedding-3-small for semantic search
- **APScheduler** - Background job scheduling
- **Adzuna API** - UK job listings

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
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
              ┌──────────┐ ┌──────────┐ ┌──────────┐
              │  Adzuna  │ │  OpenAI  │ │Scheduler │
              │   API    │ │Embeddings│ │ (6 hrs)  │
              └──────────┘ └──────────┘ └──────────┘
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

## Environment Variables

### Backend (.env)
```
SECRET_KEY=your-secret-key
APP_PASSWORD=your-login-password
DATABASE_URL=sqlite:///./data/jobs.db
OPENAI_API_KEY=sk-...
ADZUNA_APP_ID=your-app-id
ADZUNA_API_KEY=your-api-key
SCRAPE_INTERVAL_HOURS=6
```

### Frontend (.env.local)
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Project Structure

```
claude_jobhunt/
├── backend/
│   └── app/
│       ├── api/           # FastAPI route handlers
│       ├── models/        # SQLAlchemy ORM models
│       ├── schemas/       # Pydantic request/response schemas
│       ├── services/      # Business logic
│       │   ├── embeddings.py   # OpenAI embedding client
│       │   ├── matcher.py      # Match score calculator
│       │   └── scrapers/       # Job source integrations
│       ├── auth.py        # Authentication logic
│       ├── scheduler.py   # Background job scheduler
│       └── main.py        # Application entry point
│
└── frontend/
    └── src/
        ├── app/           # Next.js pages (App Router)
        ├── components/    # React components
        ├── lib/           # Utilities and API client
        └── types/         # TypeScript type definitions
```

## API Keys Needed

- **OpenAI**: https://platform.openai.com/api-keys
- **Adzuna**: https://developer.adzuna.com/

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
