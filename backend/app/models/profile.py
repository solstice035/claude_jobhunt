"""
Profile Model - User job search preferences and CV storage

Singleton model (id="default") storing the user's CV text, embedding,
and match scoring preferences.

Default Score Weights:
    - semantic: 0.30 (CV-job description similarity)
    - skills: 0.30 (keyword overlap)
    - seniority: 0.25 (job level alignment)
    - location: 0.15 (geographic match)
"""

from sqlalchemy import Column, String, Integer, Text, JSON, DateTime
from sqlalchemy.sql import func
from app.database import Base


class Profile(Base):
    """
    User profile for job matching configuration.

    Note: Single-user MVP uses id="default" as the sole profile.

    Attributes:
        cv_text: Full CV/resume text for skill extraction
        cv_embedding: Pre-computed 1536-dim embedding of CV
        target_roles: List of desired job titles
        target_sectors: List of preferred industries
        locations: List of preferred work locations
        salary_min/target: Salary expectations
        exclude_keywords: Negative keywords to filter jobs
        score_weights: Dict configuring match score composition
    """

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
