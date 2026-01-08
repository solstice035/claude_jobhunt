"""
ESCO Service - Skill Lookup and Search Operations

This service provides methods for querying the ESCO skills database,
including exact matching, fuzzy search, and relationship traversal.

The ESCO database contains 13,890 skills with:
- Preferred labels (canonical names)
- Alternative labels (synonyms, abbreviations)
- Descriptions (skill definitions)
- Hierarchical relationships (broader/narrower)
- Horizontal relationships (related skills)

Usage:
    async with get_db() as db:
        service = ESCOService(db)
        skill = await service.find_skill_by_label("Python")
        related = await service.get_related_skills("Python")
"""

from typing import List, Optional
from sqlalchemy import select, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.esco import ESCOSkill


class ESCOService:
    """
    Service for ESCO skill database operations.

    Provides methods for:
    - Finding skills by exact label or synonym
    - Fuzzy/similarity search
    - Traversing skill relationships
    - Bulk skill lookups

    Attributes:
        session: AsyncSession for database operations
    """

    def __init__(self, session: AsyncSession):
        """
        Initialize ESCO service with database session.

        Args:
            session: SQLAlchemy async session for database queries
        """
        self.session = session

    async def find_skill_by_label(self, label: str) -> Optional[ESCOSkill]:
        """
        Find a skill by exact label match (preferred or alternative).

        First checks preferred_label, then searches alt_labels array.
        Case-insensitive matching is used.

        Args:
            label: Skill name to search for

        Returns:
            ESCOSkill if found, None otherwise

        Example:
            >>> skill = await service.find_skill_by_label("Python")
            >>> skill.preferred_label
            'Python programming'
        """
        label_lower = label.lower().strip()

        # First, try exact match on preferred_label
        query = select(ESCOSkill).where(
            func.lower(ESCOSkill.preferred_label) == label_lower
        )
        result = await self.session.execute(query)
        skill = result.scalars().first()

        if skill:
            return skill

        # If not found, search in alt_labels
        # For SQLite, we need to use JSON functions
        # For PostgreSQL, we'd use @> operator
        all_skills = await self.session.execute(select(ESCOSkill))
        for skill in all_skills.scalars():
            if skill.alt_labels:
                alt_labels_lower = [alt.lower() for alt in skill.alt_labels]
                if label_lower in alt_labels_lower:
                    return skill

        return None

    async def find_skill_by_uri(self, uri: str) -> Optional[ESCOSkill]:
        """
        Find a skill by its URI.

        Args:
            uri: ESCO URI identifier

        Returns:
            ESCOSkill if found, None otherwise
        """
        query = select(ESCOSkill).where(ESCOSkill.uri == uri)
        result = await self.session.execute(query)
        return result.scalars().first()

    async def find_skills_by_uris(self, uris: List[str]) -> List[ESCOSkill]:
        """
        Find multiple skills by their URIs.

        Args:
            uris: List of ESCO URI identifiers

        Returns:
            List of found ESCOSkill objects
        """
        if not uris:
            return []

        query = select(ESCOSkill).where(ESCOSkill.uri.in_(uris))
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def get_related_skills(self, label: str) -> List[ESCOSkill]:
        """
        Get skills related to the given skill.

        Finds the skill by label, then fetches all related skills
        based on the related_skills relationship.

        Args:
            label: Skill name to find relationships for

        Returns:
            List of related ESCOSkill objects

        Example:
            >>> related = await service.get_related_skills("Python")
            >>> [s.preferred_label for s in related]
            ['Django', 'Flask', 'NumPy']
        """
        skill = await self.find_skill_by_label(label)
        if not skill or not skill.related_skills:
            return []

        return await self.find_skills_by_uris(skill.related_skills)

    async def get_broader_skills(self, label: str) -> List[ESCOSkill]:
        """
        Get broader (parent) skills for the given skill.

        Broader skills represent more general concepts.
        E.g., "Python programming" -> "Programming languages"

        Args:
            label: Skill name to find broader skills for

        Returns:
            List of broader ESCOSkill objects
        """
        skill = await self.find_skill_by_label(label)
        if not skill or not skill.broader_skills:
            return []

        return await self.find_skills_by_uris(skill.broader_skills)

    async def get_narrower_skills(self, label: str) -> List[ESCOSkill]:
        """
        Get narrower (child) skills for the given skill.

        Narrower skills represent more specific concepts.
        E.g., "Programming languages" -> ["Python", "JavaScript", ...]

        Args:
            label: Skill name to find narrower skills for

        Returns:
            List of narrower ESCOSkill objects
        """
        skill = await self.find_skill_by_label(label)
        if not skill or not skill.narrower_skills:
            return []

        return await self.find_skills_by_uris(skill.narrower_skills)

    async def search_skills(
        self,
        query: str,
        limit: int = 20
    ) -> List[ESCOSkill]:
        """
        Search skills by text query (fuzzy matching).

        Searches both preferred_label and description fields.
        For production PostgreSQL, would use pg_trgm for similarity.

        Args:
            query: Search text
            limit: Maximum number of results

        Returns:
            List of matching ESCOSkill objects, ordered by relevance
        """
        query_lower = query.lower().strip()

        # Basic LIKE search for SQLite compatibility
        # In production with PostgreSQL, use similarity() function
        search_pattern = f"%{query_lower}%"

        stmt = (
            select(ESCOSkill)
            .where(
                or_(
                    func.lower(ESCOSkill.preferred_label).like(search_pattern),
                    func.lower(ESCOSkill.description).like(search_pattern)
                )
            )
            .limit(limit)
        )

        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def get_all_skills(self, limit: int = 1000) -> List[ESCOSkill]:
        """
        Get all skills from the database.

        Args:
            limit: Maximum number of skills to return

        Returns:
            List of ESCOSkill objects
        """
        query = select(ESCOSkill).limit(limit)
        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def normalize_skill_name(self, name: str) -> Optional[str]:
        """
        Normalize a skill name to its ESCO preferred label.

        Useful for standardizing skill names from different sources.

        Args:
            name: Skill name to normalize

        Returns:
            Normalized preferred_label if found, None otherwise

        Example:
            >>> await service.normalize_skill_name("k8s")
            'Kubernetes'
            >>> await service.normalize_skill_name("py")
            'Python programming'
        """
        skill = await self.find_skill_by_label(name)
        return skill.preferred_label if skill else None

    async def bulk_normalize_skills(
        self,
        names: List[str]
    ) -> dict[str, Optional[str]]:
        """
        Normalize multiple skill names to ESCO preferred labels.

        Args:
            names: List of skill names to normalize

        Returns:
            Dict mapping input names to normalized labels (or None)
        """
        results = {}
        for name in names:
            results[name] = await self.normalize_skill_name(name)
        return results
