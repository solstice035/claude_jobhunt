"""Pydantic schemas for agent API endpoints."""

from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# --- Agent Log ---

class AgentLogCreate(BaseModel):
    agent_name: str
    action: str
    detail: Optional[str] = None
    metadata: Optional[dict] = None
    status: str = "success"
    job_id: Optional[str] = None


class AgentLogResponse(BaseModel):
    id: str
    agent_name: str
    action: str
    detail: Optional[str] = None
    metadata: Optional[dict] = None
    status: str
    job_id: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AgentLogListResponse(BaseModel):
    logs: list[AgentLogResponse]
    total: int


# --- Dashboard ---

class AgentDashboardResponse(BaseModel):
    total_actions: int
    actions_today: int
    agents_active: list[str]
    recent_errors: list[AgentLogResponse]
    actions_by_agent: dict[str, int]
    actions_by_status: dict[str, int]


# --- Follow-ups ---

class FollowUpCreate(BaseModel):
    job_id: str
    action: str
    due_date: datetime
    notes: Optional[str] = None


class FollowUpUpdate(BaseModel):
    completed: Optional[bool] = None
    notes: Optional[str] = None
    action: Optional[str] = None
    due_date: Optional[datetime] = None


class FollowUpResponse(BaseModel):
    id: str
    job_id: str
    action: str
    due_date: datetime
    completed: bool
    completed_at: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# --- Networking Contacts ---

class NetworkingContactCreate(BaseModel):
    name: str
    company: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    context: Optional[str] = None
    job_id: Optional[str] = None
    status: str = "identified"
    notes: Optional[str] = None


class NetworkingContactUpdate(BaseModel):
    name: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    context: Optional[str] = None
    status: Optional[str] = None
    notes: Optional[str] = None


class NetworkingContactResponse(BaseModel):
    id: str
    name: str
    company: Optional[str] = None
    role: Optional[str] = None
    email: Optional[str] = None
    linkedin_url: Optional[str] = None
    context: Optional[str] = None
    job_id: Optional[str] = None
    status: str
    last_contact: Optional[datetime] = None
    notes: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True
