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
