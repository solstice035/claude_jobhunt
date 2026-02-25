"""
Agent Models - SQLAlchemy ORM models for agent activity tracking

Tables:
    agent_log: Audit trail of agent actions
    follow_ups: Scheduled follow-up tasks for job applications
    networking_contacts: Professional networking contacts
"""

from sqlalchemy import Column, String, Integer, Text, DateTime, JSON, Boolean
from sqlalchemy.sql import func
from app.database import Base
import uuid


class AgentLog(Base):
    """Agent activity log entry."""
    __tablename__ = "jobhunt_agent_log"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_name = Column(String(100), nullable=False, index=True)
    action = Column(String(200), nullable=False)
    detail = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSON, nullable=True)
    status = Column(String(20), nullable=False, default="success")  # success, error, warning
    job_id = Column(String(36), nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.now())


class FollowUp(Base):
    """Scheduled follow-up for a job application."""
    __tablename__ = "jobhunt_follow_ups"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id = Column(String(36), nullable=False, index=True)
    action = Column(String(200), nullable=False)
    due_date = Column(DateTime, nullable=False)
    completed = Column(Boolean, nullable=False, default=False)
    completed_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class NetworkingContact(Base):
    """Professional networking contact."""
    __tablename__ = "networking_contacts"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(200), nullable=False)
    company = Column(String(200), nullable=True)
    role = Column(String(200), nullable=True)
    email = Column(String(200), nullable=True)
    linkedin_url = Column(String(500), nullable=True)
    context = Column(Text, nullable=True)  # How/why we know them
    job_id = Column(String(36), nullable=True)  # Related job if any
    status = Column(String(50), nullable=False, default="identified")  # identified, contacted, responded, meeting_scheduled, active
    last_contact = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)
    
    # Phase 4 extensions
    priority = Column(Integer, nullable=True, default=3)  # 1=high, 2=medium, 3=low
    tags = Column(Text, nullable=True)  # Comma-separated or JSON
    next_action = Column(Text, nullable=True)  # What to do next
    next_action_date = Column(DateTime, nullable=True)  # When to do it
    source = Column(String(100), nullable=True)  # linkedin, referral, company_site, etc.
    warmth = Column(String(20), nullable=True, default="cold")  # cold, warm, hot
    response_status = Column(String(50), nullable=True)  # pending, responded, no_response
    shared_background = Column(Text, nullable=True)  # Common ground, mutual connections
    message_draft = Column(Text, nullable=True)  # Draft message to send
    approved_at = Column(DateTime, nullable=True)  # When message was approved
    sent_at = Column(DateTime, nullable=True)  # When message was sent
    follow_up_due = Column(DateTime, nullable=True)  # When to follow up
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
