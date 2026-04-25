"""
Job Search Agent API - Main Application Entry Point

This module initializes the FastAPI application with:
- Database connection and schema initialization
- Background job scheduler for periodic job fetching
- CORS middleware for frontend communication
- API router registration

Architecture:
    FastAPI App
    ├── Lifespan Management (startup/shutdown)
    ├── CORS Middleware (localhost:3000)
    └── API Router
        ├── /auth - Authentication endpoints
        ├── /jobs - Job CRUD operations
        ├── /profile - User profile management
        └── /stats - Dashboard statistics
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from app.database import init_db
from app.api import api_router
from app.scheduler import start_scheduler, stop_scheduler
from app.middleware import setup_metrics
from app.config import get_settings

logger = logging.getLogger(__name__)


def _fire_and_forget(coro):
    """Create a background task with error logging."""
    task = asyncio.create_task(coro)
    task.add_done_callback(_log_task_exception)
    return task


def _log_task_exception(task: asyncio.Task):
    if task.cancelled():
        return
    exc = task.exception()
    if exc:
        logger.error("Background task failed: %s", exc, exc_info=exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()

    if settings.app_password == "changeme":
        logger.warning("APP_PASSWORD is set to default 'changeme' — change it for any non-local use")
    if settings.secret_key == "dev-secret-key-change-in-production":
        logger.warning("SECRET_KEY is set to the default — change it for any non-local use")

    await init_db()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(
    title="Job Search Agent API",
    description="AI-powered job matching API",
    version="0.1.0",
    lifespan=lifespan,
)

settings = get_settings()
cors_origins = [
    o.strip()
    for o in settings.cors_origins.split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

# Setup Prometheus metrics endpoint and middleware
setup_metrics(app)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
