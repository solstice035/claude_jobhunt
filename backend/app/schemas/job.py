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
