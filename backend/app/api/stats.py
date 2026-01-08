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
