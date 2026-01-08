"""
Skills API - Endpoints for skill extraction, search, and gap analysis.

Provides REST endpoints for:
- Extracting skills from text using LLM
- Searching ESCO skills database
- Analyzing skill gaps between CV and target jobs
- Getting skill recommendations

All endpoints require authentication (inherited from profile endpoints).
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from openai import AsyncOpenAI

from app.database import get_db
from app.config import get_settings
from app.models.profile import Profile
from app.api.profile import get_profile
from app.services.esco import ESCOService
from app.services.skill_extractor import SkillExtractor, ExtractedSkill
from app.services.skill_gaps import SkillGapAnalyzer, SkillGap, SkillGapSummary
from app.services.skill_graph import get_default_skill_graph

router = APIRouter()
settings = get_settings()


# ==============================================================================
# Request/Response Models
# ==============================================================================

class ExtractSkillsRequest(BaseModel):
    """Request body for skill extraction."""
    text: str = Field(..., min_length=10, description="Text to extract skills from")


class ExtractedSkillResponse(BaseModel):
    """Response model for an extracted skill."""
    name: str
    category: str
    required: bool
    confidence: str


class ExtractSkillsResponse(BaseModel):
    """Response for skill extraction endpoint."""
    skills: List[ExtractedSkillResponse]
    count: int


class ESCOSkillResponse(BaseModel):
    """Response model for an ESCO skill."""
    uri: str
    preferred_label: str
    alt_labels: List[str]
    description: Optional[str]
    skill_type: str


class SkillGapResponse(BaseModel):
    """Response model for a skill gap."""
    skill: str
    frequency: float
    importance: str
    category: str
    related_skills_present: List[str]


class SkillGapSummaryResponse(BaseModel):
    """Response model for skill gap summary."""
    total_gaps: int
    critical_gaps: int
    technical_gaps: int
    soft_gaps: int
    top_gaps: List[SkillGapResponse]
    coverage_score: float


class LearningRecommendation(BaseModel):
    """Response model for a learning recommendation."""
    skill: str
    category: str
    importance: str
    rationale: str


# ==============================================================================
# Helper Functions
# ==============================================================================

def get_openai_client() -> AsyncOpenAI:
    """Get configured OpenAI client."""
    return AsyncOpenAI(api_key=settings.openai_api_key)


def get_skill_extractor() -> SkillExtractor:
    """Get configured skill extractor service."""
    client = get_openai_client()
    return SkillExtractor(openai_client=client)


# ==============================================================================
# API Endpoints
# ==============================================================================

@router.post("/extract", response_model=ExtractSkillsResponse)
async def extract_skills(
    request: ExtractSkillsRequest,
    db: AsyncSession = Depends(get_db)
) -> ExtractSkillsResponse:
    """
    Extract skills from text using LLM.

    Takes any text (job description, CV, etc.) and extracts structured
    skill information including category and requirement status.

    Args:
        request: ExtractSkillsRequest with text to analyze

    Returns:
        ExtractSkillsResponse with list of extracted skills

    Example:
        POST /api/skills/extract
        {"text": "Looking for a Python developer with AWS experience"}

        Response:
        {
            "skills": [
                {"name": "Python", "category": "technical", "required": true, "confidence": "high"},
                {"name": "AWS", "category": "technical", "required": true, "confidence": "high"}
            ],
            "count": 2
        }
    """
    extractor = get_skill_extractor()
    skills = await extractor.extract_skills(request.text)

    return ExtractSkillsResponse(
        skills=[
            ExtractedSkillResponse(
                name=s.name,
                category=s.category,
                required=s.required,
                confidence=s.confidence
            )
            for s in skills
        ],
        count=len(skills)
    )


@router.get("/search", response_model=List[ESCOSkillResponse])
async def search_skills(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(20, ge=1, le=100, description="Maximum results"),
    db: AsyncSession = Depends(get_db)
) -> List[ESCOSkillResponse]:
    """
    Search ESCO skills database.

    Performs fuzzy search on skill names and descriptions.

    Args:
        q: Search query (minimum 2 characters)
        limit: Maximum number of results (default 20, max 100)
        db: Database session

    Returns:
        List of matching ESCO skills

    Example:
        GET /api/skills/search?q=python

        Response:
        [
            {
                "uri": "http://data.europa.eu/esco/skill/...",
                "preferred_label": "Python programming",
                "alt_labels": ["Python", "python3"],
                "description": "Programming using Python...",
                "skill_type": "skill"
            }
        ]
    """
    service = ESCOService(db)
    skills = await service.search_skills(q, limit=limit)

    return [
        ESCOSkillResponse(
            uri=s.uri,
            preferred_label=s.preferred_label,
            alt_labels=s.alt_labels or [],
            description=s.description,
            skill_type=s.skill_type
        )
        for s in skills
    ]


@router.get("/gaps", response_model=List[SkillGapResponse])
async def get_skill_gaps(
    limit: int = Query(20, ge=1, le=50, description="Maximum gaps to return"),
    db: AsyncSession = Depends(get_db)
) -> List[SkillGapResponse]:
    """
    Get skill gaps for current profile.

    Analyzes the gap between the profile's CV skills and skills required
    by recent matching jobs.

    Args:
        limit: Maximum number of gaps to return (default 20)
        db: Database session

    Returns:
        List of skill gaps sorted by importance

    Example:
        GET /api/skills/gaps

        Response:
        [
            {
                "skill": "Kubernetes",
                "frequency": 0.85,
                "importance": "critical",
                "category": "technical",
                "related_skills_present": ["Docker"]
            }
        ]
    """
    profile = await get_profile(db)

    if not profile.cv_text:
        raise HTTPException(
            status_code=400,
            detail="Profile CV text is required for gap analysis"
        )

    # Get recent jobs for analysis
    from sqlalchemy import select
    from app.models.job import Job

    jobs_query = (
        select(Job)
        .where(Job.status.in_(["new", "saved"]))
        .order_by(Job.match_score.desc())
        .limit(50)
    )
    result = await db.execute(jobs_query)
    jobs = result.scalars().all()

    if not jobs:
        return []

    # Create analyzer and calculate gaps
    extractor = get_skill_extractor()
    analyzer = SkillGapAnalyzer(skill_extractor=extractor)

    target_jobs = [{"description": job.description} for job in jobs]
    gaps = await analyzer.calculate_gaps(
        cv_text=profile.cv_text,
        target_jobs=target_jobs,
        limit=limit
    )

    return [
        SkillGapResponse(
            skill=g.skill,
            frequency=g.frequency,
            importance=g.importance,
            category=g.category,
            related_skills_present=g.related_skills_present
        )
        for g in gaps
    ]


@router.get("/gaps/summary", response_model=SkillGapSummaryResponse)
async def get_skill_gap_summary(
    db: AsyncSession = Depends(get_db)
) -> SkillGapSummaryResponse:
    """
    Get summary of skill gap analysis.

    Provides aggregated statistics about skill gaps including
    coverage score and breakdown by category.

    Args:
        db: Database session

    Returns:
        SkillGapSummaryResponse with aggregated statistics

    Example:
        GET /api/skills/gaps/summary

        Response:
        {
            "total_gaps": 15,
            "critical_gaps": 3,
            "technical_gaps": 10,
            "soft_gaps": 5,
            "coverage_score": 65.5,
            "top_gaps": [...]
        }
    """
    profile = await get_profile(db)

    if not profile.cv_text:
        raise HTTPException(
            status_code=400,
            detail="Profile CV text is required for gap analysis"
        )

    # Get recent jobs
    from sqlalchemy import select
    from app.models.job import Job

    jobs_query = (
        select(Job)
        .where(Job.status.in_(["new", "saved"]))
        .order_by(Job.match_score.desc())
        .limit(50)
    )
    result = await db.execute(jobs_query)
    jobs = result.scalars().all()

    if not jobs:
        return SkillGapSummaryResponse(
            total_gaps=0,
            critical_gaps=0,
            technical_gaps=0,
            soft_gaps=0,
            top_gaps=[],
            coverage_score=100.0
        )

    extractor = get_skill_extractor()
    analyzer = SkillGapAnalyzer(skill_extractor=extractor)

    target_jobs = [{"description": job.description} for job in jobs]
    summary = await analyzer.get_summary(
        cv_text=profile.cv_text,
        target_jobs=target_jobs
    )

    return SkillGapSummaryResponse(
        total_gaps=summary.total_gaps,
        critical_gaps=summary.critical_gaps,
        technical_gaps=summary.technical_gaps,
        soft_gaps=summary.soft_gaps,
        top_gaps=[
            SkillGapResponse(
                skill=g.skill,
                frequency=g.frequency,
                importance=g.importance,
                category=g.category,
                related_skills_present=g.related_skills_present
            )
            for g in summary.top_gaps
        ],
        coverage_score=summary.coverage_score
    )


@router.get("/recommendations", response_model=List[LearningRecommendation])
async def get_learning_recommendations(
    max_skills: int = Query(5, ge=1, le=10, description="Number of skills to recommend"),
    db: AsyncSession = Depends(get_db)
) -> List[LearningRecommendation]:
    """
    Get personalized learning recommendations.

    Based on skill gap analysis, recommends which skills to learn
    prioritized by impact and learnability.

    Args:
        max_skills: Maximum skills to recommend (default 5)
        db: Database session

    Returns:
        List of learning recommendations with rationale

    Example:
        GET /api/skills/recommendations

        Response:
        [
            {
                "skill": "Kubernetes",
                "category": "technical",
                "importance": "critical",
                "rationale": "Required in 85% of target jobs; Related to skills you have: Docker"
            }
        ]
    """
    profile = await get_profile(db)

    if not profile.cv_text:
        raise HTTPException(
            status_code=400,
            detail="Profile CV text is required for recommendations"
        )

    # Get recent jobs
    from sqlalchemy import select
    from app.models.job import Job

    jobs_query = (
        select(Job)
        .where(Job.status.in_(["new", "saved"]))
        .order_by(Job.match_score.desc())
        .limit(50)
    )
    result = await db.execute(jobs_query)
    jobs = result.scalars().all()

    if not jobs:
        return []

    extractor = get_skill_extractor()
    analyzer = SkillGapAnalyzer(skill_extractor=extractor)

    target_jobs = [{"description": job.description} for job in jobs]
    gaps = await analyzer.calculate_gaps(
        cv_text=profile.cv_text,
        target_jobs=target_jobs,
        limit=20
    )

    recommendations = await analyzer.recommend_learning_path(
        gaps=gaps,
        max_skills=max_skills
    )

    return [
        LearningRecommendation(
            skill=r["skill"],
            category=r["category"],
            importance=r["importance"],
            rationale=r["rationale"]
        )
        for r in recommendations
    ]


@router.get("/infer", response_model=List[str])
async def infer_skills(
    skills: str = Query(..., description="Comma-separated list of skills"),
    include_related: bool = Query(False, description="Include related skills")
) -> List[str]:
    """
    Infer additional skills from explicit skills.

    Uses skill relationship graph to expand a skill set with
    implied/related skills.

    Args:
        skills: Comma-separated list of explicit skills
        include_related: Whether to include related skills (weaker inference)

    Returns:
        List of all skills (explicit + inferred)

    Example:
        GET /api/skills/infer?skills=kubernetes,python

        Response:
        ["kubernetes", "python", "docker", "containerization", "programming"]
    """
    skill_list = [s.strip().lower() for s in skills.split(",") if s.strip()]

    if not skill_list:
        return []

    graph = get_default_skill_graph()
    inferred = graph.infer_skills(
        set(skill_list),
        include_related=include_related
    )

    return sorted(list(inferred))


@router.get("/esco/{uri:path}", response_model=ESCOSkillResponse)
async def get_esco_skill(
    uri: str,
    db: AsyncSession = Depends(get_db)
) -> ESCOSkillResponse:
    """
    Get ESCO skill by URI.

    Args:
        uri: ESCO skill URI
        db: Database session

    Returns:
        ESCOSkillResponse with full skill details

    Raises:
        HTTPException 404 if skill not found
    """
    service = ESCOService(db)
    skill = await service.find_skill_by_uri(uri)

    if not skill:
        raise HTTPException(status_code=404, detail="Skill not found")

    return ESCOSkillResponse(
        uri=skill.uri,
        preferred_label=skill.preferred_label,
        alt_labels=skill.alt_labels or [],
        description=skill.description,
        skill_type=skill.skill_type
    )
