from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Optional
import asyncio
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
async def refresh_jobs(
    _: bool = Depends(get_current_user),
):
    # Import here to avoid circular import
    from app.scheduler import trigger_manual_refresh
    asyncio.create_task(trigger_manual_refresh())
    return {"message": "Job refresh triggered", "status": "processing"}
