"""
ESCO Skills Model - European Skills, Competences, Qualifications and Occupations

This module provides the SQLAlchemy ORM model for storing ESCO skills data.
ESCO provides a multilingual classification of 13,890 skills with hierarchical
relationships, synonyms, and skill-to-occupation mappings.

Data Source: https://esco.ec.europa.eu/
License: CC BY 4.0

Schema Design:
    - URI is the primary key (stable ESCO identifier)
    - alt_labels stored as JSONB for efficient array queries
    - Relationship URIs stored as JSONB arrays for graph traversal
"""

from sqlalchemy import Column, String, Text, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy import JSON
from app.database import Base


class ESCOSkill(Base):
    """
    ESCO skill entity for intelligent skill matching.

    The ESCO classification provides standardized skill definitions used across
    the European Union, enabling consistent skill matching regardless of how
    skills are phrased in CVs or job descriptions.

    Attributes:
        uri: Unique ESCO identifier (e.g., "http://data.europa.eu/esco/skill/...")
        preferred_label: Canonical skill name in English
        alt_labels: Alternative names and synonyms (avg 5-10 per skill)
        description: Full skill definition text
        skill_type: Classification ("skill" or "knowledge")
        broader_skills: Parent skill URIs (more general concepts)
        narrower_skills: Child skill URIs (more specific concepts)
        related_skills: Horizontally related skill URIs

    Relationships:
        - broader_skills: Points to more general skills (e.g., Python -> Programming)
        - narrower_skills: Points to more specific skills (e.g., Programming -> Python)
        - related_skills: Points to similar/complementary skills (e.g., React -> JavaScript)

    Example:
        >>> skill = ESCOSkill(
        ...     uri="http://data.europa.eu/esco/skill/python",
        ...     preferred_label="Python programming",
        ...     alt_labels=["Python", "Python3", "py"],
        ...     skill_type="skill",
        ...     broader_skills=["http://data.europa.eu/esco/skill/programming"]
        ... )
    """

    __tablename__ = "esco_skills"

    # Primary identifier from ESCO
    uri = Column(String(255), primary_key=True)

    # Skill names and descriptions
    preferred_label = Column(String(255), nullable=False, index=True)
    alt_labels = Column(JSON, nullable=False, default=list)
    description = Column(Text, nullable=True)

    # Classification
    skill_type = Column(String(50), nullable=False, default="skill")

    # Hierarchical relationships (stored as URI arrays)
    broader_skills = Column(JSON, nullable=False, default=list)
    narrower_skills = Column(JSON, nullable=False, default=list)
    related_skills = Column(JSON, nullable=False, default=list)

    def __repr__(self) -> str:
        return f"<ESCOSkill(uri='{self.uri}', label='{self.preferred_label}')>"

    def to_dict(self) -> dict:
        """Convert skill to dictionary for API responses."""
        return {
            "uri": self.uri,
            "preferred_label": self.preferred_label,
            "alt_labels": self.alt_labels or [],
            "description": self.description,
            "skill_type": self.skill_type,
            "broader_skills": self.broader_skills or [],
            "narrower_skills": self.narrower_skills or [],
            "related_skills": self.related_skills or [],
        }


# Note: For PostgreSQL, these indexes would use gin_trgm_ops for fuzzy matching
# For SQLite compatibility, we use standard indexes
# In production with PostgreSQL, add:
# Index('idx_esco_label_trgm', ESCOSkill.preferred_label, postgresql_using='gin',
#       postgresql_ops={'preferred_label': 'gin_trgm_ops'})
