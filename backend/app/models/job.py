"""
Job Model - SQLAlchemy ORM model for job postings

Stores job listings scraped from external sources (Adzuna, etc.)
along with AI-generated match scores and user-assigned status.

Status Flow:
    new → saved → applied → interviewing → offered/rejected → archived
"""

from sqlalchemy import Column, String, Integer, Float, Text, DateTime, JSON
from sqlalchemy.sql import func
from app.database import Base
import uuid


class Job(Base):
    """
    Job posting entity with AI match scoring.

    Attributes:
        id: UUID primary key
        title: Job title (max 500 chars)
        company: Company name
        location: Job location
        salary_min/max: Salary range (nullable)
        description: Full job description text
        url: Original job posting URL (unique)
        url_hash: SHA-256 hash for deduplication (indexed)
        source: Data source (e.g., "adzuna")
        match_score: AI-calculated relevance (0-100)
        match_reasons: JSON list of human-readable match explanations
        embedding: 1536-dim OpenAI embedding vector (JSON)
        status: Pipeline stage (indexed)
        notes: User notes
    """

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
