"""
Agent API endpoints - Activity logging, dashboard, follow-ups, and networking contacts.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, case
from typing import Optional
from datetime import datetime, timedelta

from app.database import get_db
from app.models.agent import AgentLog, FollowUp, NetworkingContact
from app.schemas.agent import (
    AgentLogCreate, AgentLogResponse, AgentLogListResponse,
    AgentDashboardResponse,
    FollowUpCreate, FollowUpUpdate, FollowUpResponse,
    NetworkingContactCreate, NetworkingContactUpdate, NetworkingContactResponse,
)
from app.auth import get_current_user

router = APIRouter(prefix="/api/agents", tags=["agents"])


# ==================== Agent Log ====================

@router.post("/log", response_model=AgentLogResponse)
async def create_log(
    entry: AgentLogCreate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    log = AgentLog(
        agent_name=entry.agent_name,
        action=entry.action,
        detail=entry.detail,
        metadata_=entry.metadata,
        status=entry.status,
        job_id=entry.job_id,
    )
    db.add(log)
    await db.commit()
    await db.refresh(log)
    return _log_to_response(log)


@router.get("/log", response_model=AgentLogListResponse)
async def list_logs(
    agent_name: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    query = select(AgentLog)
    count_query = select(func.count(AgentLog.id))

    if agent_name:
        query = query.where(AgentLog.agent_name == agent_name)
        count_query = count_query.where(AgentLog.agent_name == agent_name)
    if status:
        query = query.where(AgentLog.status == status)
        count_query = count_query.where(AgentLog.status == status)

    total = (await db.execute(count_query)).scalar() or 0
    query = query.order_by(AgentLog.created_at.desc()).offset(offset).limit(limit)
    result = await db.execute(query)
    logs = result.scalars().all()

    return AgentLogListResponse(
        logs=[_log_to_response(l) for l in logs],
        total=total,
    )


def _log_to_response(log: AgentLog) -> AgentLogResponse:
    return AgentLogResponse(
        id=log.id,
        agent_name=log.agent_name,
        action=log.action,
        detail=log.detail,
        metadata=log.metadata_,
        status=log.status,
        job_id=log.job_id,
        created_at=log.created_at,
    )


# ==================== Dashboard ====================

@router.get("/dashboard", response_model=AgentDashboardResponse)
async def get_dashboard(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    now = datetime.utcnow()
    today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Total actions
    total = (await db.execute(select(func.count(AgentLog.id)))).scalar() or 0

    # Actions today
    today_count = (await db.execute(
        select(func.count(AgentLog.id)).where(AgentLog.created_at >= today_start)
    )).scalar() or 0

    # Distinct active agents
    agents_result = await db.execute(select(AgentLog.agent_name).distinct())
    agents_active = [r[0] for r in agents_result.all()]

    # Recent errors (last 10)
    errors_result = await db.execute(
        select(AgentLog)
        .where(AgentLog.status == "error")
        .order_by(AgentLog.created_at.desc())
        .limit(10)
    )
    recent_errors = [_log_to_response(l) for l in errors_result.scalars().all()]

    # Actions by agent
    by_agent_result = await db.execute(
        select(AgentLog.agent_name, func.count(AgentLog.id))
        .group_by(AgentLog.agent_name)
    )
    actions_by_agent = {name: count for name, count in by_agent_result.all()}

    # Actions by status
    by_status_result = await db.execute(
        select(AgentLog.status, func.count(AgentLog.id))
        .group_by(AgentLog.status)
    )
    actions_by_status = {status: count for status, count in by_status_result.all()}

    return AgentDashboardResponse(
        total_actions=total,
        actions_today=today_count,
        agents_active=agents_active,
        recent_errors=recent_errors,
        actions_by_agent=actions_by_agent,
        actions_by_status=actions_by_status,
    )


# ==================== Follow-ups ====================

@router.get("/follow-ups-due", response_model=list[FollowUpResponse])
async def get_follow_ups_due(
    days_ahead: int = Query(7, ge=0, le=90),
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    cutoff = datetime.utcnow() + timedelta(days=days_ahead)
    result = await db.execute(
        select(FollowUp)
        .where(FollowUp.completed == False)
        .where(FollowUp.due_date <= cutoff)
        .order_by(FollowUp.due_date.asc())
    )
    return [FollowUpResponse.model_validate(f) for f in result.scalars().all()]


@router.post("/follow-ups", response_model=FollowUpResponse)
async def create_follow_up(
    entry: FollowUpCreate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    follow_up = FollowUp(
        job_id=entry.job_id,
        action=entry.action,
        due_date=entry.due_date,
        notes=entry.notes,
    )
    db.add(follow_up)
    await db.commit()
    await db.refresh(follow_up)
    return FollowUpResponse.model_validate(follow_up)


@router.patch("/follow-ups/{follow_up_id}", response_model=FollowUpResponse)
async def update_follow_up(
    follow_up_id: str,
    update: FollowUpUpdate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    result = await db.execute(select(FollowUp).where(FollowUp.id == follow_up_id))
    follow_up = result.scalar_one_or_none()
    if not follow_up:
        raise HTTPException(status_code=404, detail="Follow-up not found")

    update_data = update.model_dump(exclude_unset=True)
    if update_data.get("completed") and not follow_up.completed:
        update_data["completed_at"] = datetime.utcnow()

    for field, value in update_data.items():
        setattr(follow_up, field, value)

    await db.commit()
    await db.refresh(follow_up)
    return FollowUpResponse.model_validate(follow_up)


# ==================== Networking Contacts ====================

@router.get("/networking-contacts", response_model=list[NetworkingContactResponse])
async def list_networking_contacts(
    status: Optional[str] = Query(None),
    warmth: Optional[str] = Query(None),
    priority: Optional[int] = Query(None),
    source: Optional[str] = Query(None),
    response_status: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    """List all networking contacts with optional filters."""
    query = select(NetworkingContact).order_by(NetworkingContact.created_at.desc())
    
    if status:
        query = query.where(NetworkingContact.status == status)
    if warmth:
        query = query.where(NetworkingContact.warmth == warmth)
    if priority is not None:
        query = query.where(NetworkingContact.priority == priority)
    if source:
        query = query.where(NetworkingContact.source == source)
    if response_status:
        query = query.where(NetworkingContact.response_status == response_status)
    
    result = await db.execute(query)
    return [NetworkingContactResponse.model_validate(c) for c in result.scalars().all()]


@router.get("/networking-contacts/pipeline", response_model=dict)
async def get_networking_pipeline(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    """Get pipeline summary stats by warmth and status."""
    # Count by warmth
    warmth_result = await db.execute(
        select(
            NetworkingContact.warmth,
            func.count(NetworkingContact.id)
        ).group_by(NetworkingContact.warmth)
    )
    by_warmth = {warmth or "unknown": count for warmth, count in warmth_result.all()}
    
    # Count by status
    status_result = await db.execute(
        select(
            NetworkingContact.status,
            func.count(NetworkingContact.id)
        ).group_by(NetworkingContact.status)
    )
    by_status = {status: count for status, count in status_result.all()}
    
    # Count by priority
    priority_result = await db.execute(
        select(
            NetworkingContact.priority,
            func.count(NetworkingContact.id)
        ).group_by(NetworkingContact.priority)
    )
    by_priority = {priority or "unset": count for priority, count in priority_result.all()}
    
    # Count by response_status
    response_result = await db.execute(
        select(
            NetworkingContact.response_status,
            func.count(NetworkingContact.id)
        ).group_by(NetworkingContact.response_status)
    )
    by_response_status = {resp or "none": count for resp, count in response_result.all()}
    
    # Total count
    total = (await db.execute(select(func.count(NetworkingContact.id)))).scalar() or 0
    
    return {
        "total": total,
        "by_warmth": by_warmth,
        "by_status": by_status,
        "by_priority": by_priority,
        "by_response_status": by_response_status,
    }


@router.get("/networking-contacts/{contact_id}", response_model=NetworkingContactResponse)
async def get_networking_contact(
    contact_id: str,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    """Get a single networking contact by ID."""
    result = await db.execute(select(NetworkingContact).where(NetworkingContact.id == contact_id))
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    return NetworkingContactResponse.model_validate(contact)


@router.post("/networking-contacts", response_model=NetworkingContactResponse)
async def create_networking_contact(
    entry: NetworkingContactCreate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    contact = NetworkingContact(**entry.model_dump())
    db.add(contact)
    await db.commit()
    await db.refresh(contact)
    return NetworkingContactResponse.model_validate(contact)


@router.patch("/networking-contacts/{contact_id}", response_model=NetworkingContactResponse)
async def update_networking_contact(
    contact_id: str,
    update: NetworkingContactUpdate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    result = await db.execute(select(NetworkingContact).where(NetworkingContact.id == contact_id))
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")

    for field, value in update.model_dump(exclude_unset=True).items():
        setattr(contact, field, value)

    await db.commit()
    await db.refresh(contact)
    return NetworkingContactResponse.model_validate(contact)


@router.delete("/networking-contacts/{contact_id}", status_code=204)
async def delete_networking_contact(
    contact_id: str,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    """Delete a networking contact by ID."""
    result = await db.execute(select(NetworkingContact).where(NetworkingContact.id == contact_id))
    contact = result.scalar_one_or_none()
    if not contact:
        raise HTTPException(status_code=404, detail="Contact not found")
    
    await db.delete(contact)
    await db.commit()
    return None
