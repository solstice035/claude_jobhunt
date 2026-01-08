# Job Search Agent Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an AI-powered job search agent that aggregates UK jobs from Adzuna, matches them against a CV using OpenAI embeddings, and presents ranked opportunities in a dashboard.

**Architecture:** FastAPI backend with SQLite database, Next.js 16.1 frontend with ShadCN UI, APScheduler for background job polling, all containerized with Docker Compose for VPS deployment.

**Tech Stack:** Python 3.12+, FastAPI, SQLAlchemy, APScheduler, OpenAI API, Next.js 16.1, TypeScript, TailwindCSS, ShadCN UI, Docker Compose, Nginx

---

## Phase 1: Project Scaffolding

### Task 1.1: Create Project Root Structure

**Files:**
- Create: `.gitignore`
- Create: `.env.example`
- Create: `docker-compose.yml`
- Create: `docker-compose.dev.yml`
- Create: `README.md`

**Step 1: Create .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
.env

# Node
node_modules/
.next/
out/

# IDE
.idea/
.vscode/
*.swp

# Data
data/*.db
*.sqlite

# OS
.DS_Store
Thumbs.db

# Logs
*.log
```

**Step 2: Create .env.example**

```bash
# OpenAI
OPENAI_API_KEY=sk-your-key-here

# Adzuna (get from https://developer.adzuna.com/)
ADZUNA_APP_ID=your-app-id
ADZUNA_API_KEY=your-api-key

# Auth
APP_PASSWORD=your-secure-password
SECRET_KEY=generate-random-32-char-string

# Optional
SCRAPE_INTERVAL_HOURS=6
```

**Step 3: Create docker-compose.yml**

```yaml
services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - frontend
      - backend
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    environment:
      - NEXT_PUBLIC_API_URL=/api
    expose:
      - "3000"
    restart: unless-stopped

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=sqlite:///./data/jobs.db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ADZUNA_APP_ID=${ADZUNA_APP_ID}
      - ADZUNA_API_KEY=${ADZUNA_API_KEY}
      - APP_PASSWORD=${APP_PASSWORD}
      - SECRET_KEY=${SECRET_KEY}
      - SCRAPE_INTERVAL_HOURS=${SCRAPE_INTERVAL_HOURS:-6}
    volumes:
      - ./data:/app/data
    expose:
      - "8000"
    restart: unless-stopped
```

**Step 4: Create docker-compose.dev.yml**

```yaml
services:
  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.dev
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
    ports:
      - "3000:3000"
    volumes:
      - ./frontend/src:/app/src
      - ./frontend/public:/app/public

  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.dev
    environment:
      - DATABASE_URL=sqlite:///./data/jobs.db
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ADZUNA_APP_ID=${ADZUNA_APP_ID}
      - ADZUNA_API_KEY=${ADZUNA_API_KEY}
      - APP_PASSWORD=${APP_PASSWORD}
      - SECRET_KEY=${SECRET_KEY}
      - SCRAPE_INTERVAL_HOURS=${SCRAPE_INTERVAL_HOURS:-6}
    ports:
      - "8000:8000"
    volumes:
      - ./backend/app:/app/app
      - ./data:/app/data
```

**Step 5: Create README.md**

```markdown
# AI Job Search Agent

Personal job search aggregator with AI-powered matching.

## Quick Start (Development)

1. Copy `.env.example` to `.env` and fill in your API keys
2. Run: `docker compose -f docker-compose.dev.yml up --build`
3. Frontend: http://localhost:3000
4. Backend API: http://localhost:8000/docs

## Production Deployment

1. Configure `.env` with production values
2. Set up SSL certificates in `nginx/ssl/`
3. Run: `docker compose up -d --build`

## API Keys Needed

- **OpenAI**: https://platform.openai.com/api-keys
- **Adzuna**: https://developer.adzuna.com/
```

**Step 6: Commit**

```bash
git add .gitignore .env.example docker-compose.yml docker-compose.dev.yml README.md
git commit -m ":tada: chore: Initialize project structure with Docker Compose"
```

---

### Task 1.2: Create Backend Scaffolding

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/Dockerfile.dev`
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`

**Step 1: Create backend/requirements.txt**

```txt
fastapi==0.115.6
uvicorn[standard]==0.34.0
sqlalchemy[asyncio]==2.0.36
aiosqlite==0.20.0
pydantic==2.10.4
pydantic-settings==2.7.1
python-multipart==0.0.20
httpx==0.28.1
openai==1.59.5
apscheduler==3.10.4
numpy==2.2.1
python-jose[cryptography]==3.3.0
passlib==1.7.4
alembic==1.14.0
pytest==8.3.4
pytest-asyncio==0.25.2
```

**Step 2: Create backend/app/config.py**

```python
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/jobs.db"
    openai_api_key: str
    adzuna_app_id: str
    adzuna_api_key: str
    app_password: str
    secret_key: str
    scrape_interval_hours: int = 6

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
```

**Step 3: Create backend/app/__init__.py**

```python
# Job Search Agent Backend
```

**Step 4: Create backend/app/main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Job Search Agent API",
    description="AI-powered job matching API",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Step 5: Create backend/Dockerfile**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 6: Create backend/Dockerfile.dev**

```dockerfile
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p /app/data

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
```

**Step 7: Commit**

```bash
git add backend/
git commit -m ":sparkles: feat(backend): Add FastAPI scaffolding with config"
```

---

### Task 1.3: Create Frontend Scaffolding

**Files:**
- Create: `frontend/` (via create-next-app)
- Modify: `frontend/package.json`
- Create: `frontend/Dockerfile`
- Create: `frontend/Dockerfile.dev`

**Step 1: Create Next.js app**

```bash
cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt
npx create-next-app@latest frontend --typescript --tailwind --eslint --app --src-dir --import-alias "@/*" --use-npm
```

**Step 2: Install ShadCN UI**

```bash
cd frontend
npx shadcn@latest init -d
npx shadcn@latest add button card input label badge tabs textarea select dialog dropdown-menu separator skeleton toast
```

**Step 3: Create frontend/Dockerfile**

```dockerfile
FROM node:22-alpine AS builder

WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

EXPOSE 3000
CMD ["node", "server.js"]
```

**Step 4: Create frontend/Dockerfile.dev**

```dockerfile
FROM node:22-alpine

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .

EXPOSE 3000
CMD ["npm", "run", "dev"]
```

**Step 5: Update frontend/next.config.ts for standalone output**

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
};

export default nextConfig;
```

**Step 6: Create frontend/src/lib/api.ts**

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = baseUrl;
  }

  async fetch<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      credentials: "include",
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
    });

    if (!response.ok) {
      if (response.status === 401) {
        window.location.href = "/login";
      }
      throw new Error(`API error: ${response.status}`);
    }

    return response.json();
  }

  get<T>(endpoint: string) {
    return this.fetch<T>(endpoint);
  }

  post<T>(endpoint: string, data: unknown) {
    return this.fetch<T>(endpoint, {
      method: "POST",
      body: JSON.stringify(data),
    });
  }

  patch<T>(endpoint: string, data: unknown) {
    return this.fetch<T>(endpoint, {
      method: "PATCH",
      body: JSON.stringify(data),
    });
  }

  put<T>(endpoint: string, data: unknown) {
    return this.fetch<T>(endpoint, {
      method: "PUT",
      body: JSON.stringify(data),
    });
  }
}

export const api = new ApiClient(API_URL);
```

**Step 7: Create frontend/src/types/index.ts**

```typescript
export type JobStatus =
  | "new"
  | "saved"
  | "applied"
  | "interviewing"
  | "offered"
  | "rejected"
  | "archived";

export type JobSource = "adzuna";

export interface Job {
  id: string;
  title: string;
  company: string;
  location: string;
  salary_min?: number;
  salary_max?: number;
  description: string;
  url: string;
  source: JobSource;
  posted_at: string;
  closing_date?: string;
  match_score: number;
  match_reasons: string[];
  status: JobStatus;
  notes?: string;
  created_at: string;
  updated_at: string;
}

export interface Profile {
  id: string;
  cv_text: string;
  target_roles: string[];
  target_sectors: string[];
  locations: string[];
  salary_min?: number;
  salary_target?: number;
  exclude_keywords: string[];
  score_weights: {
    semantic: number;
    skills: number;
    seniority: number;
    location: number;
  };
}

export interface Stats {
  total_jobs: number;
  new_jobs: number;
  saved_jobs: number;
  applied_jobs: number;
  avg_match_score: number;
  jobs_by_source: Record<string, number>;
}
```

**Step 8: Commit**

```bash
git add frontend/
git commit -m ":sparkles: feat(frontend): Add Next.js 16.1 scaffolding with ShadCN UI"
```

---

### Task 1.4: Create Nginx Config

**Files:**
- Create: `nginx/nginx.conf`
- Create: `data/.gitkeep`

**Step 1: Create nginx/nginx.conf**

```nginx
events {
    worker_connections 1024;
}

http {
    upstream frontend {
        server frontend:3000;
    }

    upstream backend {
        server backend:8000;
    }

    server {
        listen 80;
        server_name _;

        # API routes
        location /api/ {
            proxy_pass http://backend/;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }

        # Frontend routes
        location / {
            proxy_pass http://frontend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection 'upgrade';
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_cache_bypass $http_upgrade;
        }
    }
}
```

**Step 2: Create data directory**

```bash
mkdir -p data nginx/ssl
touch data/.gitkeep nginx/ssl/.gitkeep
```

**Step 3: Commit**

```bash
git add nginx/ data/
git commit -m ":wrench: config: Add Nginx reverse proxy configuration"
```

---

## Phase 2: Core Backend

### Task 2.1: Database Models

**Files:**
- Create: `backend/app/database.py`
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/job.py`
- Create: `backend/app/models/profile.py`

**Step 1: Create backend/app/database.py**

```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

settings = get_settings()

# Convert sqlite:/// to sqlite+aiosqlite:///
database_url = settings.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")

engine = create_async_engine(database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        yield session


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
```

**Step 2: Create backend/app/models/__init__.py**

```python
from app.models.job import Job
from app.models.profile import Profile

__all__ = ["Job", "Profile"]
```

**Step 3: Create backend/app/models/job.py**

```python
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base
import uuid


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(500), nullable=False)
    company = Column(String(500), nullable=False)
    location = Column(String(500), nullable=False)
    salary_min = Column(Integer, nullable=True)
    salary_max = Column(Integer, nullable=True)
    description = Column(Text, nullable=False)
    url = Column(String(2000), nullable=False, unique=True)
    url_hash = Column(String(64), nullable=False, unique=True, index=True)
    source = Column(String(50), nullable=False, default="adzuna")
    posted_at = Column(DateTime, nullable=True)
    closing_date = Column(DateTime, nullable=True)
    match_score = Column(Float, nullable=False, default=0.0)
    match_reasons = Column(JSON, nullable=False, default=list)
    embedding = Column(JSON, nullable=True)  # Store as JSON array
    status = Column(String(20), nullable=False, default="new", index=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
```

**Step 4: Create backend/app/models/profile.py**

```python
from sqlalchemy import Column, String, Integer, Text, JSON, DateTime
from sqlalchemy.sql import func
from app.database import Base


class Profile(Base):
    __tablename__ = "profiles"

    id = Column(String, primary_key=True, default="default")
    cv_text = Column(Text, nullable=False, default="")
    cv_embedding = Column(JSON, nullable=True)  # Store as JSON array
    target_roles = Column(JSON, nullable=False, default=list)
    target_sectors = Column(JSON, nullable=False, default=list)
    locations = Column(JSON, nullable=False, default=list)
    salary_min = Column(Integer, nullable=True)
    salary_target = Column(Integer, nullable=True)
    exclude_keywords = Column(JSON, nullable=False, default=list)
    score_weights = Column(
        JSON,
        nullable=False,
        default=lambda: {
            "semantic": 0.30,
            "skills": 0.30,
            "seniority": 0.25,
            "location": 0.15,
        },
    )
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
```

**Step 5: Update backend/app/main.py to init DB**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Job Search Agent API",
    description="AI-powered job matching API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Step 6: Commit**

```bash
git add backend/app/
git commit -m ":sparkles: feat(backend): Add SQLAlchemy models for Job and Profile"
```

---

### Task 2.2: Pydantic Schemas

**Files:**
- Create: `backend/app/schemas/__init__.py`
- Create: `backend/app/schemas/job.py`
- Create: `backend/app/schemas/profile.py`
- Create: `backend/app/schemas/auth.py`

**Step 1: Create backend/app/schemas/__init__.py**

```python
from app.schemas.job import JobCreate, JobUpdate, JobResponse, JobListResponse
from app.schemas.profile import ProfileUpdate, ProfileResponse
from app.schemas.auth import LoginRequest, LoginResponse

__all__ = [
    "JobCreate",
    "JobUpdate",
    "JobResponse",
    "JobListResponse",
    "ProfileUpdate",
    "ProfileResponse",
    "LoginRequest",
    "LoginResponse",
]
```

**Step 2: Create backend/app/schemas/job.py**

```python
from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class JobBase(BaseModel):
    title: str
    company: str
    location: str
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    description: str
    url: str
    source: str = "adzuna"
    posted_at: Optional[datetime] = None
    closing_date: Optional[datetime] = None


class JobCreate(JobBase):
    pass


class JobUpdate(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None


class JobResponse(JobBase):
    id: str
    match_score: float
    match_reasons: list[str]
    status: str
    notes: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int
    page: int
    per_page: int
```

**Step 3: Create backend/app/schemas/profile.py**

```python
from pydantic import BaseModel
from typing import Optional


class ScoreWeights(BaseModel):
    semantic: float = 0.30
    skills: float = 0.30
    seniority: float = 0.25
    location: float = 0.15


class ProfileUpdate(BaseModel):
    cv_text: Optional[str] = None
    target_roles: Optional[list[str]] = None
    target_sectors: Optional[list[str]] = None
    locations: Optional[list[str]] = None
    salary_min: Optional[int] = None
    salary_target: Optional[int] = None
    exclude_keywords: Optional[list[str]] = None
    score_weights: Optional[ScoreWeights] = None


class ProfileResponse(BaseModel):
    id: str
    cv_text: str
    target_roles: list[str]
    target_sectors: list[str]
    locations: list[str]
    salary_min: Optional[int] = None
    salary_target: Optional[int] = None
    exclude_keywords: list[str]
    score_weights: ScoreWeights

    class Config:
        from_attributes = True
```

**Step 4: Create backend/app/schemas/auth.py**

```python
from pydantic import BaseModel


class LoginRequest(BaseModel):
    password: str


class LoginResponse(BaseModel):
    success: bool
    message: str
```

**Step 5: Commit**

```bash
git add backend/app/schemas/
git commit -m ":sparkles: feat(backend): Add Pydantic schemas for API validation"
```

---

### Task 2.3: Authentication

**Files:**
- Create: `backend/app/auth.py`
- Create: `backend/app/api/__init__.py`
- Create: `backend/app/api/auth.py`

**Step 1: Create backend/app/auth.py**

```python
from datetime import datetime, timedelta
from typing import Optional
from fastapi import Request, HTTPException, status
from jose import JWTError, jwt
from app.config import get_settings

settings = get_settings()

ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30
COOKIE_NAME = "session_token"


def create_session_token() -> str:
    expire = datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS)
    to_encode = {"exp": expire, "authenticated": True}
    return jwt.encode(to_encode, settings.secret_key, algorithm=ALGORITHM)


def verify_session_token(token: str) -> bool:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload.get("authenticated", False)
    except JWTError:
        return False


def verify_password(password: str) -> bool:
    return password == settings.app_password


async def get_current_user(request: Request) -> bool:
    token = request.cookies.get(COOKIE_NAME)
    if not token or not verify_session_token(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )
    return True
```

**Step 2: Create backend/app/api/__init__.py**

```python
from fastapi import APIRouter
from app.api import auth, jobs, profile, stats

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(profile.router, prefix="/profile", tags=["profile"])
api_router.include_router(stats.router, prefix="/stats", tags=["stats"])
```

**Step 3: Create backend/app/api/auth.py**

```python
from fastapi import APIRouter, Response, HTTPException, status
from app.schemas import LoginRequest, LoginResponse
from app.auth import verify_password, create_session_token, COOKIE_NAME

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest, response: Response):
    if not verify_password(request.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid password",
        )

    token = create_session_token()
    response.set_cookie(
        key=COOKIE_NAME,
        value=token,
        httponly=True,
        max_age=30 * 24 * 60 * 60,  # 30 days
        samesite="lax",
    )
    return LoginResponse(success=True, message="Logged in successfully")


@router.post("/logout", response_model=LoginResponse)
async def logout(response: Response):
    response.delete_cookie(COOKIE_NAME)
    return LoginResponse(success=True, message="Logged out successfully")


@router.get("/check")
async def check_auth():
    return {"authenticated": True}
```

**Step 4: Commit**

```bash
git add backend/app/auth.py backend/app/api/
git commit -m ":lock: feat(backend): Add password authentication with JWT sessions"
```

---

### Task 2.4: Jobs API

**Files:**
- Create: `backend/app/api/jobs.py`

**Step 1: Create backend/app/api/jobs.py**

```python
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import load_only
from typing import Optional
from app.database import get_db
from app.models import Job
from app.schemas import JobResponse, JobListResponse, JobUpdate
from app.auth import get_current_user

router = APIRouter()


@router.get("", response_model=JobListResponse)
async def list_jobs(
    status: Optional[str] = Query(None),
    source: Optional[str] = Query(None),
    score_min: Optional[float] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    query = select(Job)
    count_query = select(func.count(Job.id))

    if status:
        query = query.where(Job.status == status)
        count_query = count_query.where(Job.status == status)

    if source:
        query = query.where(Job.source == source)
        count_query = count_query.where(Job.source == source)

    if score_min is not None:
        query = query.where(Job.match_score >= score_min)
        count_query = count_query.where(Job.match_score >= score_min)

    if search:
        search_filter = Job.title.ilike(f"%{search}%") | Job.company.ilike(f"%{search}%")
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)

    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar() or 0

    # Get paginated results, ordered by match_score desc, then created_at desc
    query = query.order_by(Job.match_score.desc(), Job.created_at.desc())
    query = query.offset((page - 1) * per_page).limit(per_page)

    result = await db.execute(query)
    jobs = result.scalars().all()

    return JobListResponse(
        jobs=[JobResponse.model_validate(job) for job in jobs],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse.model_validate(job)


@router.patch("/{job_id}", response_model=JobResponse)
async def update_job(
    job_id: str,
    update: JobUpdate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    update_data = update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(job, field, value)

    await db.commit()
    await db.refresh(job)

    return JobResponse.model_validate(job)


@router.post("/refresh")
async def trigger_refresh(
    _: bool = Depends(get_current_user),
):
    # Will be implemented with scheduler
    return {"message": "Job refresh triggered", "status": "queued"}
```

**Step 2: Commit**

```bash
git add backend/app/api/jobs.py
git commit -m ":sparkles: feat(backend): Add Jobs CRUD API with filtering"
```

---

### Task 2.5: Profile and Stats API

**Files:**
- Create: `backend/app/api/profile.py`
- Create: `backend/app/api/stats.py`
- Modify: `backend/app/main.py`

**Step 1: Create backend/app/api/profile.py**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import Profile
from app.schemas import ProfileResponse, ProfileUpdate
from app.auth import get_current_user

router = APIRouter()

DEFAULT_PROFILE_ID = "default"


async def get_or_create_profile(db: AsyncSession) -> Profile:
    result = await db.execute(select(Profile).where(Profile.id == DEFAULT_PROFILE_ID))
    profile = result.scalar_one_or_none()

    if not profile:
        profile = Profile(id=DEFAULT_PROFILE_ID)
        db.add(profile)
        await db.commit()
        await db.refresh(profile)

    return profile


@router.get("", response_model=ProfileResponse)
async def get_profile(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    profile = await get_or_create_profile(db)
    return ProfileResponse.model_validate(profile)


@router.put("", response_model=ProfileResponse)
async def update_profile(
    update: ProfileUpdate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    profile = await get_or_create_profile(db)

    update_data = update.model_dump(exclude_unset=True)

    # Convert score_weights to dict if present
    if "score_weights" in update_data and update_data["score_weights"]:
        update_data["score_weights"] = update_data["score_weights"].model_dump() if hasattr(update_data["score_weights"], "model_dump") else update_data["score_weights"]

    cv_changed = "cv_text" in update_data and update_data["cv_text"] != profile.cv_text

    for field, value in update_data.items():
        setattr(profile, field, value)

    # Clear embedding if CV changed (will be regenerated on next match)
    if cv_changed:
        profile.cv_embedding = None

    await db.commit()
    await db.refresh(profile)

    return ProfileResponse.model_validate(profile)
```

**Step 2: Create backend/app/api/stats.py**

```python
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from app.database import get_db
from app.models import Job
from app.auth import get_current_user

router = APIRouter()


@router.get("")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    # Total jobs
    total_result = await db.execute(select(func.count(Job.id)))
    total_jobs = total_result.scalar() or 0

    # Jobs by status
    status_counts = {}
    for status in ["new", "saved", "applied", "interviewing", "offered", "rejected", "archived"]:
        result = await db.execute(
            select(func.count(Job.id)).where(Job.status == status)
        )
        status_counts[status] = result.scalar() or 0

    # Average match score
    avg_result = await db.execute(select(func.avg(Job.match_score)))
    avg_match_score = round(avg_result.scalar() or 0, 1)

    # Jobs by source
    source_query = select(Job.source, func.count(Job.id)).group_by(Job.source)
    source_result = await db.execute(source_query)
    jobs_by_source = {row[0]: row[1] for row in source_result.all()}

    return {
        "total_jobs": total_jobs,
        "new_jobs": status_counts["new"],
        "saved_jobs": status_counts["saved"],
        "applied_jobs": status_counts["applied"],
        "interviewing_jobs": status_counts["interviewing"],
        "avg_match_score": avg_match_score,
        "jobs_by_source": jobs_by_source,
    }
```

**Step 3: Update backend/app/main.py to include routers**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.api import api_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(
    title="Job Search Agent API",
    description="AI-powered job matching API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Step 4: Commit**

```bash
git add backend/app/api/ backend/app/main.py
git commit -m ":sparkles: feat(backend): Add Profile and Stats API endpoints"
```

---

## Phase 3: Data Collection

### Task 3.1: Adzuna Scraper

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/scrapers/__init__.py`
- Create: `backend/app/services/scrapers/base.py`
- Create: `backend/app/services/scrapers/adzuna.py`

**Step 1: Create backend/app/services/__init__.py**

```python
# Services module
```

**Step 2: Create backend/app/services/scrapers/__init__.py**

```python
from app.services.scrapers.base import BaseScraper
from app.services.scrapers.adzuna import AdzunaScraper

__all__ = ["BaseScraper", "AdzunaScraper"]
```

**Step 3: Create backend/app/services/scrapers/base.py**

```python
from abc import ABC, abstractmethod
from typing import List
from app.schemas import JobCreate


class BaseScraper(ABC):
    """Base class for job scrapers"""

    source: str = "unknown"

    @abstractmethod
    async def fetch_jobs(self, search_query: str, location: str = "uk") -> List[JobCreate]:
        """Fetch jobs from the source"""
        pass
```

**Step 4: Create backend/app/services/scrapers/adzuna.py**

```python
import httpx
import hashlib
from datetime import datetime
from typing import List, Optional
from app.services.scrapers.base import BaseScraper
from app.schemas import JobCreate
from app.config import get_settings

settings = get_settings()


class AdzunaScraper(BaseScraper):
    source = "adzuna"
    base_url = "https://api.adzuna.com/v1/api/jobs/gb/search"

    def __init__(self):
        self.app_id = settings.adzuna_app_id
        self.api_key = settings.adzuna_api_key

    async def fetch_jobs(
        self,
        search_query: str,
        location: str = "uk",
        results_per_page: int = 50,
        max_pages: int = 3,
    ) -> List[JobCreate]:
        jobs = []

        async with httpx.AsyncClient() as client:
            for page in range(1, max_pages + 1):
                params = {
                    "app_id": self.app_id,
                    "app_key": self.api_key,
                    "results_per_page": results_per_page,
                    "page": page,
                    "what": search_query,
                    "where": location,
                    "sort_by": "date",
                    "max_days_old": 30,
                }

                try:
                    response = await client.get(
                        f"{self.base_url}/{page}",
                        params=params,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                    for result in data.get("results", []):
                        job = self._parse_job(result)
                        if job:
                            jobs.append(job)

                    # Stop if we got fewer results than requested
                    if len(data.get("results", [])) < results_per_page:
                        break

                except httpx.HTTPError as e:
                    print(f"Adzuna API error on page {page}: {e}")
                    break

        return jobs

    def _parse_job(self, data: dict) -> Optional[JobCreate]:
        try:
            # Parse salary
            salary_min = None
            salary_max = None
            if data.get("salary_min"):
                salary_min = int(data["salary_min"])
            if data.get("salary_max"):
                salary_max = int(data["salary_max"])

            # Parse posted date
            posted_at = None
            if data.get("created"):
                posted_at = datetime.fromisoformat(data["created"].replace("Z", "+00:00"))

            return JobCreate(
                title=data.get("title", "Unknown Title"),
                company=data.get("company", {}).get("display_name", "Unknown Company"),
                location=data.get("location", {}).get("display_name", "UK"),
                salary_min=salary_min,
                salary_max=salary_max,
                description=data.get("description", ""),
                url=data.get("redirect_url", ""),
                source=self.source,
                posted_at=posted_at,
            )
        except Exception as e:
            print(f"Error parsing Adzuna job: {e}")
            return None


def generate_url_hash(url: str) -> str:
    """Generate a hash of the URL for deduplication"""
    return hashlib.sha256(url.encode()).hexdigest()
```

**Step 5: Commit**

```bash
git add backend/app/services/
git commit -m ":sparkles: feat(backend): Add Adzuna API scraper"
```

---

### Task 3.2: Embeddings Service

**Files:**
- Create: `backend/app/services/embeddings.py`

**Step 1: Create backend/app/services/embeddings.py**

```python
from openai import AsyncOpenAI
from typing import List, Optional
import numpy as np
from app.config import get_settings

settings = get_settings()
client = AsyncOpenAI(api_key=settings.openai_api_key)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


async def get_embedding(text: str) -> List[float]:
    """Get embedding for a single text"""
    text = text.replace("\n", " ").strip()
    if not text:
        return [0.0] * EMBEDDING_DIMENSIONS

    response = await client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
    )
    return response.data[0].embedding


async def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts in a single API call"""
    cleaned_texts = [t.replace("\n", " ").strip() for t in texts]

    # Filter out empty texts but track their indices
    non_empty_indices = [i for i, t in enumerate(cleaned_texts) if t]
    non_empty_texts = [cleaned_texts[i] for i in non_empty_indices]

    if not non_empty_texts:
        return [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]

    # Batch in groups of 100 (OpenAI limit is 2048)
    batch_size = 100
    all_embeddings = []

    for i in range(0, len(non_empty_texts), batch_size):
        batch = non_empty_texts[i:i + batch_size]
        response = await client.embeddings.create(
            input=batch,
            model=EMBEDDING_MODEL,
        )
        all_embeddings.extend([d.embedding for d in response.data])

    # Reconstruct full list with zero vectors for empty texts
    result = [[0.0] * EMBEDDING_DIMENSIONS for _ in texts]
    for idx, emb in zip(non_empty_indices, all_embeddings):
        result[idx] = emb

    return result


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    a = np.array(vec1)
    b = np.array(vec2)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))
```

**Step 2: Commit**

```bash
git add backend/app/services/embeddings.py
git commit -m ":sparkles: feat(backend): Add OpenAI embeddings service"
```

---

### Task 3.3: Matching Engine

**Files:**
- Create: `backend/app/services/matcher.py`

**Step 1: Create backend/app/services/matcher.py**

```python
import re
from typing import List, Tuple, Optional
from app.services.embeddings import cosine_similarity

# Seniority keywords for level detection
SENIORITY_LEVELS = {
    "executive": ["ceo", "cto", "cfo", "cio", "chief", "president", "vp", "vice president"],
    "director": ["director", "head of", "vp of"],
    "senior": ["senior", "lead", "principal", "staff", "architect"],
    "mid": ["manager", "specialist", "analyst", "engineer", "developer", "consultant"],
    "junior": ["junior", "associate", "assistant", "trainee", "graduate", "entry", "intern"],
}

# Common tech skills for keyword matching
TECH_SKILLS = [
    "python", "javascript", "typescript", "java", "c#", "go", "rust", "ruby",
    "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring",
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "machine learning", "ai", "data science", "deep learning",
    "agile", "scrum", "devops", "ci/cd", "microservices",
]


def extract_seniority(title: str) -> str:
    """Extract seniority level from job title"""
    title_lower = title.lower()

    for level, keywords in SENIORITY_LEVELS.items():
        for keyword in keywords:
            if keyword in title_lower:
                return level

    return "mid"  # Default to mid-level


def extract_skills_from_text(text: str) -> List[str]:
    """Extract tech skills from text"""
    text_lower = text.lower()
    found_skills = []

    for skill in TECH_SKILLS:
        # Use word boundaries for accurate matching
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return found_skills


def match_location(job_location: str, preferred_locations: List[str]) -> float:
    """Calculate location match score (0-1)"""
    if not preferred_locations:
        return 1.0  # No preference means all locations match

    job_loc_lower = job_location.lower()

    for pref in preferred_locations:
        pref_lower = pref.lower()
        if pref_lower in job_loc_lower or job_loc_lower in pref_lower:
            return 1.0
        if "remote" in pref_lower and "remote" in job_loc_lower:
            return 1.0

    return 0.0


def match_seniority(job_title: str, target_roles: List[str]) -> float:
    """Calculate seniority match score (0-1)"""
    if not target_roles:
        return 0.5  # Neutral if no preference

    job_seniority = extract_seniority(job_title)

    # Extract target seniority levels
    target_levels = set()
    for role in target_roles:
        target_levels.add(extract_seniority(role))

    if job_seniority in target_levels:
        return 1.0

    # Partial match for adjacent levels
    level_order = ["junior", "mid", "senior", "director", "executive"]
    job_idx = level_order.index(job_seniority) if job_seniority in level_order else 2

    for target in target_levels:
        if target in level_order:
            target_idx = level_order.index(target)
            diff = abs(job_idx - target_idx)
            if diff == 1:
                return 0.5

    return 0.0


def calculate_match_score(
    job_embedding: List[float],
    job_description: str,
    job_title: str,
    job_location: str,
    cv_embedding: List[float],
    cv_text: str,
    target_roles: List[str],
    preferred_locations: List[str],
    score_weights: dict,
) -> Tuple[float, List[str]]:
    """
    Calculate composite match score and generate match reasons.

    Returns:
        Tuple of (score 0-100, list of match reasons)
    """
    reasons = []

    # 1. Semantic similarity (embedding comparison)
    semantic_score = cosine_similarity(cv_embedding, job_embedding)
    semantic_score = max(0, min(1, semantic_score))  # Clamp to 0-1

    # 2. Skills match
    cv_skills = set(extract_skills_from_text(cv_text))
    job_skills = set(extract_skills_from_text(job_description))

    if cv_skills and job_skills:
        common_skills = cv_skills & job_skills
        skills_score = len(common_skills) / max(len(job_skills), 1)
        skills_score = min(1.0, skills_score)  # Cap at 1.0

        if common_skills:
            top_skills = list(common_skills)[:3]
            reasons.append(f"Skills: {', '.join(top_skills)}")
    else:
        skills_score = 0.5  # Neutral if no skills detected

    # 3. Seniority match
    seniority_score = match_seniority(job_title, target_roles)
    if seniority_score == 1.0:
        reasons.append(f"Seniority: {extract_seniority(job_title).title()} level match")

    # 4. Location match
    location_score = match_location(job_location, preferred_locations)
    if location_score == 1.0 and preferred_locations:
        reasons.append(f"Location: {job_location}")

    # Calculate weighted composite score
    weights = score_weights
    composite = (
        semantic_score * weights.get("semantic", 0.30) +
        skills_score * weights.get("skills", 0.30) +
        seniority_score * weights.get("seniority", 0.25) +
        location_score * weights.get("location", 0.15)
    )

    # Convert to 0-100 scale
    final_score = round(composite * 100, 1)

    # Add semantic match reason if high
    if semantic_score > 0.7:
        reasons.insert(0, "Strong CV match")
    elif semantic_score > 0.5:
        reasons.insert(0, "Good CV match")

    return final_score, reasons[:5]  # Limit to 5 reasons
```

**Step 2: Commit**

```bash
git add backend/app/services/matcher.py
git commit -m ":sparkles: feat(backend): Add AI matching engine with scoring"
```

---

### Task 3.4: Scheduler and Job Processing

**Files:**
- Create: `backend/app/scheduler.py`
- Modify: `backend/app/main.py`
- Modify: `backend/app/api/jobs.py`

**Step 1: Create backend/app/scheduler.py**

```python
import asyncio
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import async_session
from app.models import Job, Profile
from app.services.scrapers import AdzunaScraper
from app.services.scrapers.adzuna import generate_url_hash
from app.services.embeddings import get_embedding, get_embeddings_batch
from app.services.matcher import calculate_match_score
from app.config import get_settings

settings = get_settings()
scheduler = AsyncIOScheduler()

# Search queries to run (customizable via profile in future)
DEFAULT_SEARCH_QUERIES = [
    "technology director",
    "head of technology",
    "principal consultant",
    "CTO",
    "engineering director",
]


async def fetch_and_process_jobs():
    """Main job that fetches jobs and processes them"""
    print(f"[{datetime.now()}] Starting job fetch...")

    scraper = AdzunaScraper()
    all_jobs = []

    # Fetch jobs for each search query
    for query in DEFAULT_SEARCH_QUERIES:
        try:
            jobs = await scraper.fetch_jobs(query, location="uk")
            all_jobs.extend(jobs)
            print(f"  Fetched {len(jobs)} jobs for query: {query}")
        except Exception as e:
            print(f"  Error fetching jobs for {query}: {e}")

    if not all_jobs:
        print("  No jobs fetched")
        return

    async with async_session() as db:
        # Get profile for matching
        profile = await db.execute(select(Profile).where(Profile.id == "default"))
        profile = profile.scalar_one_or_none()

        if not profile or not profile.cv_text:
            print("  No profile/CV configured, skipping matching")
            cv_embedding = None
        else:
            # Get or generate CV embedding
            if not profile.cv_embedding:
                profile.cv_embedding = await get_embedding(profile.cv_text)
                await db.commit()
            cv_embedding = profile.cv_embedding

        # Deduplicate and insert new jobs
        new_jobs_count = 0
        jobs_to_embed = []
        job_objects = []

        for job_data in all_jobs:
            url_hash = generate_url_hash(job_data.url)

            # Check if job already exists
            existing = await db.execute(
                select(Job).where(Job.url_hash == url_hash)
            )
            if existing.scalar_one_or_none():
                continue

            job = Job(
                title=job_data.title,
                company=job_data.company,
                location=job_data.location,
                salary_min=job_data.salary_min,
                salary_max=job_data.salary_max,
                description=job_data.description,
                url=job_data.url,
                url_hash=url_hash,
                source=job_data.source,
                posted_at=job_data.posted_at,
                status="new",
            )

            db.add(job)
            jobs_to_embed.append(job_data.description)
            job_objects.append(job)
            new_jobs_count += 1

        if job_objects:
            await db.commit()

            # Generate embeddings for new jobs
            print(f"  Generating embeddings for {len(job_objects)} new jobs...")
            embeddings = await get_embeddings_batch(jobs_to_embed)

            # Calculate match scores
            for job, embedding in zip(job_objects, embeddings):
                job.embedding = embedding

                if cv_embedding and profile:
                    score, reasons = calculate_match_score(
                        job_embedding=embedding,
                        job_description=job.description,
                        job_title=job.title,
                        job_location=job.location,
                        cv_embedding=cv_embedding,
                        cv_text=profile.cv_text,
                        target_roles=profile.target_roles or [],
                        preferred_locations=profile.locations or [],
                        score_weights=profile.score_weights or {},
                    )
                    job.match_score = score
                    job.match_reasons = reasons

            await db.commit()

        print(f"  Added {new_jobs_count} new jobs")


async def trigger_manual_refresh():
    """Trigger an immediate job refresh"""
    await fetch_and_process_jobs()


def start_scheduler():
    """Start the background scheduler"""
    scheduler.add_job(
        fetch_and_process_jobs,
        trigger=IntervalTrigger(hours=settings.scrape_interval_hours),
        id="fetch_jobs",
        replace_existing=True,
    )
    scheduler.start()
    print(f"Scheduler started: fetching jobs every {settings.scrape_interval_hours} hours")


def stop_scheduler():
    """Stop the background scheduler"""
    scheduler.shutdown()
```

**Step 2: Update backend/app/main.py**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import init_db
from app.api import api_router
from app.scheduler import start_scheduler, stop_scheduler, fetch_and_process_jobs
import asyncio


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    start_scheduler()
    # Run initial fetch after startup
    asyncio.create_task(fetch_and_process_jobs())
    yield
    stop_scheduler()


app = FastAPI(
    title="Job Search Agent API",
    description="AI-powered job matching API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

**Step 3: Update backend/app/api/jobs.py refresh endpoint**

```python
# Add at the end of backend/app/api/jobs.py

from app.scheduler import trigger_manual_refresh
import asyncio


@router.post("/refresh")
async def refresh_jobs(
    _: bool = Depends(get_current_user),
):
    asyncio.create_task(trigger_manual_refresh())
    return {"message": "Job refresh triggered", "status": "processing"}
```

**Step 4: Commit**

```bash
git add backend/app/scheduler.py backend/app/main.py backend/app/api/jobs.py
git commit -m ":sparkles: feat(backend): Add APScheduler for automated job fetching"
```

---

## Phase 4: Frontend Implementation

### Task 4.1: Layout and Navigation

**Files:**
- Modify: `frontend/src/app/layout.tsx`
- Create: `frontend/src/components/layout/Sidebar.tsx`
- Create: `frontend/src/components/layout/Header.tsx`
- Create: `frontend/src/app/globals.css` (update)

**Step 1: Create frontend/src/components/layout/Sidebar.tsx**

```tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import {
  Briefcase,
  User,
  BarChart3,
  Settings,
  LogOut,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";

const navItems = [
  { href: "/jobs", label: "Jobs", icon: Briefcase },
  { href: "/applications", label: "Applications", icon: BarChart3 },
  { href: "/profile", label: "Profile", icon: User },
];

export function Sidebar() {
  const pathname = usePathname();

  const handleLogout = async () => {
    await api.post("/auth/logout", {});
    window.location.href = "/login";
  };

  return (
    <aside className="w-64 border-r bg-card h-screen flex flex-col">
      <div className="p-6">
        <h1 className="text-xl font-bold">Job Search</h1>
        <p className="text-sm text-muted-foreground">AI-Powered Agent</p>
      </div>

      <nav className="flex-1 px-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname.startsWith(item.href);

          return (
            <Link key={item.href} href={item.href}>
              <div
                className={cn(
                  "flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                  isActive
                    ? "bg-primary text-primary-foreground"
                    : "hover:bg-accent"
                )}
              >
                <Icon className="h-4 w-4" />
                {item.label}
              </div>
            </Link>
          );
        })}
      </nav>

      <div className="p-4 border-t">
        <Button
          variant="ghost"
          className="w-full justify-start gap-3"
          onClick={handleLogout}
        >
          <LogOut className="h-4 w-4" />
          Logout
        </Button>
      </div>
    </aside>
  );
}
```

**Step 2: Create frontend/src/components/layout/Header.tsx**

```tsx
"use client";

import { RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { useState } from "react";

interface HeaderProps {
  title: string;
  showRefresh?: boolean;
}

export function Header({ title, showRefresh = false }: HeaderProps) {
  const [refreshing, setRefreshing] = useState(false);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await api.post("/jobs/refresh", {});
    } finally {
      setTimeout(() => setRefreshing(false), 2000);
    }
  };

  return (
    <header className="h-16 border-b flex items-center justify-between px-6">
      <h2 className="text-lg font-semibold">{title}</h2>
      {showRefresh && (
        <Button
          variant="outline"
          size="sm"
          onClick={handleRefresh}
          disabled={refreshing}
        >
          <RefreshCw
            className={cn("h-4 w-4 mr-2", refreshing && "animate-spin")}
          />
          {refreshing ? "Refreshing..." : "Refresh Jobs"}
        </Button>
      )}
    </header>
  );
}

function cn(...classes: (string | boolean | undefined)[]) {
  return classes.filter(Boolean).join(" ");
}
```

**Step 3: Update frontend/src/app/layout.tsx**

```tsx
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Job Search Agent",
  description: "AI-powered job search and matching",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
```

**Step 4: Commit**

```bash
git add frontend/src/
git commit -m ":sparkles: feat(frontend): Add layout with sidebar navigation"
```

---

### Task 4.2: Login Page

**Files:**
- Create: `frontend/src/app/login/page.tsx`
- Modify: `frontend/src/app/page.tsx`

**Step 1: Create frontend/src/app/login/page.tsx**

```tsx
"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { api } from "@/lib/api";

export default function LoginPage() {
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      await api.post("/auth/login", { password });
      router.push("/jobs");
    } catch (err) {
      setError("Invalid password");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <Card className="w-full max-w-md">
        <CardHeader className="text-center">
          <CardTitle className="text-2xl">Job Search Agent</CardTitle>
          <CardDescription>
            Enter your password to access the dashboard
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <Input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Enter password"
                required
              />
            </div>
            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? "Logging in..." : "Login"}
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
```

**Step 2: Update frontend/src/app/page.tsx**

```tsx
import { redirect } from "next/navigation";

export default function Home() {
  redirect("/jobs");
}
```

**Step 3: Commit**

```bash
git add frontend/src/app/
git commit -m ":sparkles: feat(frontend): Add login page with password auth"
```

---

### Task 4.3: Jobs List Page

**Files:**
- Create: `frontend/src/app/jobs/layout.tsx`
- Create: `frontend/src/app/jobs/page.tsx`
- Create: `frontend/src/components/jobs/JobCard.tsx`
- Create: `frontend/src/components/jobs/FilterPanel.tsx`
- Create: `frontend/src/components/jobs/MatchBadge.tsx`

**Step 1: Create frontend/src/app/jobs/layout.tsx**

```tsx
import { Sidebar } from "@/components/layout/Sidebar";

export default function JobsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
```

**Step 2: Create frontend/src/components/jobs/MatchBadge.tsx**

```tsx
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

interface MatchBadgeProps {
  score: number;
  size?: "sm" | "md" | "lg";
}

export function MatchBadge({ score, size = "md" }: MatchBadgeProps) {
  const getColor = () => {
    if (score >= 80) return "bg-green-500 hover:bg-green-600";
    if (score >= 60) return "bg-amber-500 hover:bg-amber-600";
    return "bg-red-500 hover:bg-red-600";
  };

  const getSize = () => {
    switch (size) {
      case "sm":
        return "text-xs px-2 py-0.5";
      case "lg":
        return "text-lg px-4 py-1";
      default:
        return "text-sm px-3 py-1";
    }
  };

  return (
    <Badge className={cn(getColor(), getSize(), "text-white font-semibold")}>
      {score}%
    </Badge>
  );
}
```

**Step 3: Create frontend/src/components/jobs/JobCard.tsx**

```tsx
import Link from "next/link";
import { Card, CardContent, CardHeader } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MatchBadge } from "./MatchBadge";
import { Job, JobStatus } from "@/types";
import { Bookmark, Archive, ExternalLink, MapPin, Building2, Calendar } from "lucide-react";
import { api } from "@/lib/api";

interface JobCardProps {
  job: Job;
  onStatusChange: () => void;
}

export function JobCard({ job, onStatusChange }: JobCardProps) {
  const formatDate = (date: string) => {
    return new Date(date).toLocaleDateString("en-GB", {
      day: "numeric",
      month: "short",
    });
  };

  const formatSalary = () => {
    if (!job.salary_min && !job.salary_max) return null;
    const min = job.salary_min ? `£${(job.salary_min / 1000).toFixed(0)}k` : "";
    const max = job.salary_max ? `£${(job.salary_max / 1000).toFixed(0)}k` : "";
    if (min && max) return `${min} - ${max}`;
    return min || max;
  };

  const handleStatusChange = async (status: JobStatus) => {
    await api.patch(`/jobs/${job.id}`, { status });
    onStatusChange();
  };

  return (
    <Card className="hover:shadow-md transition-shadow">
      <CardHeader className="pb-2">
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <Link href={`/jobs/${job.id}`}>
              <h3 className="font-semibold text-lg hover:text-primary truncate">
                {job.title}
              </h3>
            </Link>
            <div className="flex items-center gap-4 mt-1 text-sm text-muted-foreground">
              <span className="flex items-center gap-1">
                <Building2 className="h-3 w-3" />
                {job.company}
              </span>
              <span className="flex items-center gap-1">
                <MapPin className="h-3 w-3" />
                {job.location}
              </span>
            </div>
          </div>
          <MatchBadge score={job.match_score} />
        </div>
      </CardHeader>
      <CardContent>
        <div className="flex flex-wrap gap-2 mb-3">
          {job.match_reasons.slice(0, 3).map((reason, idx) => (
            <Badge key={idx} variant="secondary" className="text-xs">
              {reason}
            </Badge>
          ))}
        </div>

        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3 text-sm text-muted-foreground">
            {formatSalary() && <span>{formatSalary()}</span>}
            <span className="flex items-center gap-1">
              <Calendar className="h-3 w-3" />
              {formatDate(job.posted_at)}
            </span>
            <Badge variant="outline" className="text-xs">
              {job.source}
            </Badge>
          </div>

          <div className="flex items-center gap-2">
            {job.status === "new" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleStatusChange("saved")}
                title="Save"
              >
                <Bookmark className="h-4 w-4" />
              </Button>
            )}
            {job.status !== "archived" && (
              <Button
                variant="ghost"
                size="sm"
                onClick={() => handleStatusChange("archived")}
                title="Archive"
              >
                <Archive className="h-4 w-4" />
              </Button>
            )}
            <Button variant="ghost" size="sm" asChild>
              <a href={job.url} target="_blank" rel="noopener noreferrer">
                <ExternalLink className="h-4 w-4" />
              </a>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
```

**Step 4: Create frontend/src/components/jobs/FilterPanel.tsx**

```tsx
"use client";

import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Search } from "lucide-react";

interface FilterPanelProps {
  status: string;
  minScore: string;
  search: string;
  onStatusChange: (value: string) => void;
  onMinScoreChange: (value: string) => void;
  onSearchChange: (value: string) => void;
}

export function FilterPanel({
  status,
  minScore,
  search,
  onStatusChange,
  onMinScoreChange,
  onSearchChange,
}: FilterPanelProps) {
  return (
    <div className="flex flex-wrap gap-4 p-4 bg-card border rounded-lg">
      <div className="flex-1 min-w-[200px]">
        <Label htmlFor="search" className="text-xs text-muted-foreground">
          Search
        </Label>
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            id="search"
            placeholder="Search jobs..."
            value={search}
            onChange={(e) => onSearchChange(e.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      <div className="w-[150px]">
        <Label htmlFor="status" className="text-xs text-muted-foreground">
          Status
        </Label>
        <Select value={status} onValueChange={onStatusChange}>
          <SelectTrigger id="status">
            <SelectValue placeholder="All" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="all">All</SelectItem>
            <SelectItem value="new">New</SelectItem>
            <SelectItem value="saved">Saved</SelectItem>
            <SelectItem value="applied">Applied</SelectItem>
            <SelectItem value="interviewing">Interviewing</SelectItem>
            <SelectItem value="archived">Archived</SelectItem>
          </SelectContent>
        </Select>
      </div>

      <div className="w-[150px]">
        <Label htmlFor="score" className="text-xs text-muted-foreground">
          Min Score
        </Label>
        <Select value={minScore} onValueChange={onMinScoreChange}>
          <SelectTrigger id="score">
            <SelectValue placeholder="Any" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="0">Any</SelectItem>
            <SelectItem value="50">50%+</SelectItem>
            <SelectItem value="60">60%+</SelectItem>
            <SelectItem value="70">70%+</SelectItem>
            <SelectItem value="80">80%+</SelectItem>
          </SelectContent>
        </Select>
      </div>
    </div>
  );
}
```

**Step 5: Create frontend/src/app/jobs/page.tsx**

```tsx
"use client";

import { useEffect, useState, useCallback } from "react";
import { Header } from "@/components/layout/Header";
import { JobCard } from "@/components/jobs/JobCard";
import { FilterPanel } from "@/components/jobs/FilterPanel";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Job } from "@/types";

interface JobsResponse {
  jobs: Job[];
  total: number;
  page: number;
  per_page: number;
}

export default function JobsPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);

  // Filters
  const [status, setStatus] = useState("new");
  const [minScore, setMinScore] = useState("0");
  const [search, setSearch] = useState("");

  const fetchJobs = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (status !== "all") params.set("status", status);
      if (minScore !== "0") params.set("score_min", minScore);
      if (search) params.set("search", search);
      params.set("page", String(page));

      const data = await api.get<JobsResponse>(`/jobs?${params}`);
      setJobs(data.jobs);
      setTotal(data.total);
    } catch (error) {
      console.error("Failed to fetch jobs:", error);
    } finally {
      setLoading(false);
    }
  }, [status, minScore, search, page]);

  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  // Debounce search
  useEffect(() => {
    const timer = setTimeout(() => {
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  return (
    <div className="flex flex-col h-full">
      <Header title="Jobs" showRefresh />
      <div className="flex-1 overflow-auto p-6 space-y-4">
        <FilterPanel
          status={status}
          minScore={minScore}
          search={search}
          onStatusChange={(v) => {
            setStatus(v);
            setPage(1);
          }}
          onMinScoreChange={(v) => {
            setMinScore(v);
            setPage(1);
          }}
          onSearchChange={setSearch}
        />

        <div className="text-sm text-muted-foreground">
          {total} jobs found
        </div>

        {loading ? (
          <div className="space-y-4">
            {[...Array(5)].map((_, i) => (
              <Skeleton key={i} className="h-32 w-full" />
            ))}
          </div>
        ) : jobs.length === 0 ? (
          <div className="text-center py-12 text-muted-foreground">
            No jobs found. Try adjusting your filters.
          </div>
        ) : (
          <div className="space-y-4">
            {jobs.map((job) => (
              <JobCard key={job.id} job={job} onStatusChange={fetchJobs} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
```

**Step 6: Commit**

```bash
git add frontend/src/
git commit -m ":sparkles: feat(frontend): Add jobs list page with filters"
```

---

### Task 4.4: Job Detail Page

**Files:**
- Create: `frontend/src/app/jobs/[id]/page.tsx`
- Create: `frontend/src/components/jobs/MatchBreakdown.tsx`

**Step 1: Create frontend/src/components/jobs/MatchBreakdown.tsx**

```tsx
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";

interface MatchBreakdownProps {
  score: number;
  reasons: string[];
}

export function MatchBreakdown({ score, reasons }: MatchBreakdownProps) {
  return (
    <div className="space-y-4">
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium">Match Score</span>
          <span className="text-2xl font-bold">{score}%</span>
        </div>
        <Progress value={score} className="h-3" />
      </div>

      {reasons.length > 0 && (
        <div>
          <span className="text-sm font-medium">Match Reasons</span>
          <div className="flex flex-wrap gap-2 mt-2">
            {reasons.map((reason, idx) => (
              <Badge key={idx} variant="secondary">
                {reason}
              </Badge>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

**Step 2: Create frontend/src/app/jobs/[id]/page.tsx**

```tsx
"use client";

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import { Header } from "@/components/layout/Header";
import { MatchBreakdown } from "@/components/jobs/MatchBreakdown";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Job, JobStatus } from "@/types";
import {
  ArrowLeft,
  ExternalLink,
  Building2,
  MapPin,
  Calendar,
  Banknote,
} from "lucide-react";

const STATUS_OPTIONS: { value: JobStatus; label: string }[] = [
  { value: "new", label: "New" },
  { value: "saved", label: "Saved" },
  { value: "applied", label: "Applied" },
  { value: "interviewing", label: "Interviewing" },
  { value: "offered", label: "Offered" },
  { value: "rejected", label: "Rejected" },
  { value: "archived", label: "Archived" },
];

export default function JobDetailPage() {
  const params = useParams();
  const router = useRouter();
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [notes, setNotes] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    const fetchJob = async () => {
      try {
        const data = await api.get<Job>(`/jobs/${params.id}`);
        setJob(data);
        setNotes(data.notes || "");
      } catch (error) {
        console.error("Failed to fetch job:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchJob();
  }, [params.id]);

  const handleStatusChange = async (status: JobStatus) => {
    if (!job) return;
    setSaving(true);
    try {
      const updated = await api.patch<Job>(`/jobs/${job.id}`, { status });
      setJob(updated);
    } finally {
      setSaving(false);
    }
  };

  const handleSaveNotes = async () => {
    if (!job) return;
    setSaving(true);
    try {
      const updated = await api.patch<Job>(`/jobs/${job.id}`, { notes });
      setJob(updated);
    } finally {
      setSaving(false);
    }
  };

  const formatSalary = () => {
    if (!job?.salary_min && !job?.salary_max) return null;
    const min = job.salary_min
      ? `£${(job.salary_min / 1000).toFixed(0)}k`
      : "";
    const max = job.salary_max
      ? `£${(job.salary_max / 1000).toFixed(0)}k`
      : "";
    if (min && max) return `${min} - ${max}`;
    return min || max;
  };

  if (loading) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Job Details" />
        <div className="p-6 space-y-4">
          <Skeleton className="h-8 w-48" />
          <Skeleton className="h-64 w-full" />
        </div>
      </div>
    );
  }

  if (!job) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Job Details" />
        <div className="p-6 text-center">Job not found</div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Job Details" />
      <div className="flex-1 overflow-auto p-6">
        <Button
          variant="ghost"
          className="mb-4"
          onClick={() => router.back()}
        >
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back to Jobs
        </Button>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Main content */}
          <div className="lg:col-span-2 space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-2xl">{job.title}</CardTitle>
                    <CardDescription className="flex items-center gap-4 mt-2">
                      <span className="flex items-center gap-1">
                        <Building2 className="h-4 w-4" />
                        {job.company}
                      </span>
                      <span className="flex items-center gap-1">
                        <MapPin className="h-4 w-4" />
                        {job.location}
                      </span>
                    </CardDescription>
                  </div>
                  <Button asChild>
                    <a
                      href={job.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      Apply
                      <ExternalLink className="h-4 w-4 ml-2" />
                    </a>
                  </Button>
                </div>

                <div className="flex flex-wrap gap-3 mt-4">
                  {formatSalary() && (
                    <Badge variant="outline" className="text-sm">
                      <Banknote className="h-3 w-3 mr-1" />
                      {formatSalary()}
                    </Badge>
                  )}
                  <Badge variant="outline" className="text-sm">
                    <Calendar className="h-3 w-3 mr-1" />
                    Posted{" "}
                    {new Date(job.posted_at).toLocaleDateString("en-GB")}
                  </Badge>
                  <Badge variant="outline" className="text-sm">
                    {job.source}
                  </Badge>
                </div>
              </CardHeader>
              <CardContent>
                <h3 className="font-semibold mb-3">Job Description</h3>
                <div className="prose prose-sm max-w-none whitespace-pre-wrap text-muted-foreground">
                  {job.description}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Match Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <MatchBreakdown
                  score={job.match_score}
                  reasons={job.match_reasons}
                />
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Status</CardTitle>
              </CardHeader>
              <CardContent>
                <Select
                  value={job.status}
                  onValueChange={handleStatusChange}
                  disabled={saving}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {STATUS_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Notes</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <Textarea
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Add your notes..."
                  rows={4}
                />
                <Button
                  onClick={handleSaveNotes}
                  disabled={saving || notes === (job.notes || "")}
                  className="w-full"
                >
                  {saving ? "Saving..." : "Save Notes"}
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
```

**Step 3: Add Progress component (ShadCN)**

```bash
cd frontend && npx shadcn@latest add progress
```

**Step 4: Commit**

```bash
git add frontend/src/
git commit -m ":sparkles: feat(frontend): Add job detail page with match breakdown"
```

---

### Task 4.5: Profile Page

**Files:**
- Create: `frontend/src/app/profile/layout.tsx`
- Create: `frontend/src/app/profile/page.tsx`

**Step 1: Create frontend/src/app/profile/layout.tsx**

```tsx
import { Sidebar } from "@/components/layout/Sidebar";

export default function ProfileLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
```

**Step 2: Create frontend/src/app/profile/page.tsx**

```tsx
"use client";

import { useEffect, useState } from "react";
import { Header } from "@/components/layout/Header";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Profile } from "@/types";
import { Save } from "lucide-react";

export default function ProfilePage() {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);

  // Form state
  const [cvText, setCvText] = useState("");
  const [targetRoles, setTargetRoles] = useState("");
  const [targetSectors, setTargetSectors] = useState("");
  const [locations, setLocations] = useState("");
  const [salaryMin, setSalaryMin] = useState("");
  const [salaryTarget, setSalaryTarget] = useState("");
  const [excludeKeywords, setExcludeKeywords] = useState("");

  useEffect(() => {
    const fetchProfile = async () => {
      try {
        const data = await api.get<Profile>("/profile");
        setProfile(data);
        setCvText(data.cv_text || "");
        setTargetRoles(data.target_roles?.join(", ") || "");
        setTargetSectors(data.target_sectors?.join(", ") || "");
        setLocations(data.locations?.join(", ") || "");
        setSalaryMin(data.salary_min?.toString() || "");
        setSalaryTarget(data.salary_target?.toString() || "");
        setExcludeKeywords(data.exclude_keywords?.join(", ") || "");
      } catch (error) {
        console.error("Failed to fetch profile:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchProfile();
  }, []);

  const handleSave = async () => {
    setSaving(true);
    try {
      const updated = await api.put<Profile>("/profile", {
        cv_text: cvText,
        target_roles: targetRoles
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        target_sectors: targetSectors
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        locations: locations
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
        salary_min: salaryMin ? parseInt(salaryMin) : null,
        salary_target: salaryTarget ? parseInt(salaryTarget) : null,
        exclude_keywords: excludeKeywords
          .split(",")
          .map((s) => s.trim())
          .filter(Boolean),
      });
      setProfile(updated);
    } catch (error) {
      console.error("Failed to save profile:", error);
    } finally {
      setSaving(false);
    }
  };

  if (loading) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Profile" />
        <div className="p-6 space-y-4">
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-32 w-full" />
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Profile" />
      <div className="flex-1 overflow-auto p-6 space-y-6">
        <Card>
          <CardHeader>
            <CardTitle>Your CV</CardTitle>
            <CardDescription>
              Paste your CV text for AI matching. This will be used to calculate
              match scores against job descriptions.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Textarea
              value={cvText}
              onChange={(e) => setCvText(e.target.value)}
              placeholder="Paste your CV content here (markdown or plain text)..."
              rows={12}
              className="font-mono text-sm"
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Job Preferences</CardTitle>
            <CardDescription>
              Define what you're looking for. Separate multiple values with
              commas.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="roles">Target Roles</Label>
                <Input
                  id="roles"
                  value={targetRoles}
                  onChange={(e) => setTargetRoles(e.target.value)}
                  placeholder="Director, Head of Technology, CTO"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="sectors">Target Sectors</Label>
                <Input
                  id="sectors"
                  value={targetSectors}
                  onChange={(e) => setTargetSectors(e.target.value)}
                  placeholder="FinTech, Consulting, SaaS"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="locations">Locations</Label>
                <Input
                  id="locations"
                  value={locations}
                  onChange={(e) => setLocations(e.target.value)}
                  placeholder="London, Remote, UK"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="exclude">Exclude Keywords</Label>
                <Input
                  id="exclude"
                  value={excludeKeywords}
                  onChange={(e) => setExcludeKeywords(e.target.value)}
                  placeholder="junior, intern, contract"
                />
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="salaryMin">Minimum Salary (£)</Label>
                <Input
                  id="salaryMin"
                  type="number"
                  value={salaryMin}
                  onChange={(e) => setSalaryMin(e.target.value)}
                  placeholder="80000"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="salaryTarget">Target Salary (£)</Label>
                <Input
                  id="salaryTarget"
                  type="number"
                  value={salaryTarget}
                  onChange={(e) => setSalaryTarget(e.target.value)}
                  placeholder="120000"
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="flex justify-end">
          <Button onClick={handleSave} disabled={saving} size="lg">
            <Save className="h-4 w-4 mr-2" />
            {saving ? "Saving..." : "Save Profile"}
          </Button>
        </div>
      </div>
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add frontend/src/app/profile/
git commit -m ":sparkles: feat(frontend): Add profile page for CV and preferences"
```

---

### Task 4.6: Applications Page

**Files:**
- Create: `frontend/src/app/applications/layout.tsx`
- Create: `frontend/src/app/applications/page.tsx`

**Step 1: Create frontend/src/app/applications/layout.tsx**

```tsx
import { Sidebar } from "@/components/layout/Sidebar";

export default function ApplicationsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-screen">
      <Sidebar />
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
```

**Step 2: Create frontend/src/app/applications/page.tsx**

```tsx
"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Header } from "@/components/layout/Header";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { api } from "@/lib/api";
import { Job } from "@/types";
import { Building2, MapPin, ExternalLink } from "lucide-react";

const PIPELINE_STAGES = [
  { key: "saved", label: "Saved", color: "bg-blue-500" },
  { key: "applied", label: "Applied", color: "bg-yellow-500" },
  { key: "interviewing", label: "Interviewing", color: "bg-purple-500" },
  { key: "offered", label: "Offered", color: "bg-green-500" },
  { key: "rejected", label: "Rejected", color: "bg-red-500" },
] as const;

export default function ApplicationsPage() {
  const [jobsByStatus, setJobsByStatus] = useState<Record<string, Job[]>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchJobs = async () => {
      try {
        const grouped: Record<string, Job[]> = {};
        for (const stage of PIPELINE_STAGES) {
          const data = await api.get<{ jobs: Job[] }>(
            `/jobs?status=${stage.key}&per_page=50`
          );
          grouped[stage.key] = data.jobs;
        }
        setJobsByStatus(grouped);
      } catch (error) {
        console.error("Failed to fetch jobs:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchJobs();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col h-full">
        <Header title="Applications" />
        <div className="p-6 grid grid-cols-5 gap-4">
          {PIPELINE_STAGES.map((stage) => (
            <Skeleton key={stage.key} className="h-96 w-full" />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      <Header title="Applications" />
      <div className="flex-1 overflow-auto p-6">
        <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 min-h-[calc(100vh-8rem)]">
          {PIPELINE_STAGES.map((stage) => (
            <div key={stage.key} className="flex flex-col">
              <div className="flex items-center gap-2 mb-3">
                <div className={`w-3 h-3 rounded-full ${stage.color}`} />
                <h3 className="font-semibold">{stage.label}</h3>
                <Badge variant="secondary" className="ml-auto">
                  {jobsByStatus[stage.key]?.length || 0}
                </Badge>
              </div>

              <div className="flex-1 space-y-3 overflow-auto">
                {jobsByStatus[stage.key]?.map((job) => (
                  <Link key={job.id} href={`/jobs/${job.id}`}>
                    <Card className="cursor-pointer hover:shadow-md transition-shadow">
                      <CardContent className="p-3">
                        <h4 className="font-medium text-sm truncate">
                          {job.title}
                        </h4>
                        <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                          <Building2 className="h-3 w-3" />
                          <span className="truncate">{job.company}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-1 text-xs text-muted-foreground">
                          <MapPin className="h-3 w-3" />
                          <span className="truncate">{job.location}</span>
                        </div>
                        <div className="flex items-center justify-between mt-2">
                          <Badge
                            variant={
                              job.match_score >= 80
                                ? "default"
                                : job.match_score >= 60
                                ? "secondary"
                                : "outline"
                            }
                            className="text-xs"
                          >
                            {job.match_score}%
                          </Badge>
                        </div>
                      </CardContent>
                    </Card>
                  </Link>
                ))}

                {(!jobsByStatus[stage.key] ||
                  jobsByStatus[stage.key].length === 0) && (
                  <div className="text-center py-8 text-sm text-muted-foreground border-2 border-dashed rounded-lg">
                    No jobs
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

**Step 3: Commit**

```bash
git add frontend/src/app/applications/
git commit -m ":sparkles: feat(frontend): Add applications kanban board"
```

---

## Phase 5: Final Polish

### Task 5.1: Auth Middleware for Frontend

**Files:**
- Create: `frontend/src/middleware.ts`

**Step 1: Create frontend/src/middleware.ts**

```typescript
import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // Skip auth check for login page and static assets
  if (
    pathname === "/login" ||
    pathname.startsWith("/_next") ||
    pathname.startsWith("/api") ||
    pathname.includes(".")
  ) {
    return NextResponse.next();
  }

  // Check for session cookie
  const sessionToken = request.cookies.get("session_token");

  if (!sessionToken) {
    return NextResponse.redirect(new URL("/login", request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
```

**Step 2: Commit**

```bash
git add frontend/src/middleware.ts
git commit -m ":lock: feat(frontend): Add auth middleware for protected routes"
```

---

### Task 5.2: Create CLAUDE.md

**Files:**
- Create: `CLAUDE.md`

**Step 1: Create CLAUDE.md**

```markdown
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
# Development
docker compose -f docker-compose.dev.yml up --build

# Production
docker compose up -d --build
```

### Tests
```bash
cd backend && pytest
cd frontend && npm run test
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
- `OPENAI_API_KEY` - For embeddings
- `ADZUNA_APP_ID` - From developer.adzuna.com
- `ADZUNA_API_KEY` - From developer.adzuna.com
- `APP_PASSWORD` - Login password
- `SECRET_KEY` - JWT session encryption

## Adding New Job Sources

1. Create new scraper in `backend/app/services/scrapers/`
2. Extend `BaseScraper` class
3. Implement `fetch_jobs()` method returning `List[JobCreate]`
4. Register in `backend/app/scheduler.py`
```

**Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m ":memo: docs: Add CLAUDE.md with project guidance"
```

---

### Task 5.3: Final Integration Test

**Steps:**

1. Start development environment:
```bash
cp .env.example .env
# Edit .env with real API keys
docker compose -f docker-compose.dev.yml up --build
```

2. Verify backend health:
```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy"}
```

3. Verify API docs:
```bash
open http://localhost:8000/docs
```

4. Verify frontend:
```bash
open http://localhost:3000
# Should redirect to /login
```

5. Test login and job fetching flow

**Final Commit:**

```bash
git add -A
git commit -m ":rocket: chore: Complete MVP implementation"
```

---

## Summary

This plan implements a complete job search agent MVP with:

- **6 phases**, **20+ tasks**, each with atomic steps
- Full TDD approach where applicable
- Commits after each logical unit
- Production-ready Docker deployment
- Extensible scraper architecture

Total estimated implementation: ~150 atomic steps
