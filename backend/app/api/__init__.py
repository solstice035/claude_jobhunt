from fastapi import APIRouter
from app.api import auth, jobs, profile, stats, skills, search

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["jobs"])
api_router.include_router(profile.router, prefix="/profile", tags=["profile"])
api_router.include_router(stats.router, prefix="/stats", tags=["stats"])
api_router.include_router(skills.router, prefix="/skills", tags=["skills"])
api_router.include_router(search.router, tags=["search"])
