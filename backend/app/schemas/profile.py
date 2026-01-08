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
