"""
Skill Gap Analysis Service

This service identifies missing skills between a candidate's CV and their
target jobs, enabling:
- Better job targeting
- Learning recommendations
- CV improvement suggestions

The analysis considers:
- Frequency of skill requirements across target jobs
- Whether skills are required vs preferred
- Skill categories (technical, soft, domain)
- Skill relationships (inferred skills count as partial matches)

Usage:
    analyzer = SkillGapAnalyzer(skill_extractor=extractor)
    gaps = await analyzer.calculate_gaps(
        cv_text="Python developer with Django experience",
        target_jobs=[{"description": "Need Python, K8s, AWS"}]
    )
    # Returns: [SkillGap(skill="Kubernetes", frequency=1.0, importance="critical")]
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Any, Set
import logging

from app.services.skill_extractor import SkillExtractor, ExtractedSkill
from app.services.skill_graph import SkillGraph, get_default_skill_graph

logger = logging.getLogger(__name__)


@dataclass
class SkillGap:
    """
    Represents a missing skill with analysis metadata.

    Attributes:
        skill: Skill name
        frequency: How often this skill appears in target jobs (0-1)
        importance: Priority level (critical, important, nice-to-have)
        category: Skill category (technical, soft, domain, tool)
        learning_resources: Optional list of learning resource URLs
        related_skills_present: Skills the candidate has that are related
    """
    skill: str
    frequency: float
    importance: str
    category: str
    learning_resources: List[str] = field(default_factory=list)
    related_skills_present: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "skill": self.skill,
            "frequency": round(self.frequency, 2),
            "importance": self.importance,
            "category": self.category,
            "learning_resources": self.learning_resources,
            "related_skills_present": self.related_skills_present,
        }


@dataclass
class SkillGapSummary:
    """
    Summary of skill gap analysis.

    Attributes:
        total_gaps: Number of missing skills
        critical_gaps: Number of critical missing skills
        technical_gaps: Number of technical skills missing
        soft_gaps: Number of soft skills missing
        top_gaps: Top priority skill gaps
        coverage_score: Percentage of required skills covered (0-100)
    """
    total_gaps: int
    critical_gaps: int
    technical_gaps: int
    soft_gaps: int
    top_gaps: List[SkillGap]
    coverage_score: float

    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "total_gaps": self.total_gaps,
            "critical_gaps": self.critical_gaps,
            "technical_gaps": self.technical_gaps,
            "soft_gaps": self.soft_gaps,
            "top_gaps": [g.to_dict() for g in self.top_gaps],
            "coverage_score": round(self.coverage_score, 1),
        }


class SkillGapAnalyzer:
    """
    Analyzes skill gaps between candidate CVs and target jobs.

    Uses LLM-based skill extraction and skill graph inference to
    provide comprehensive gap analysis.

    Attributes:
        skill_extractor: Service for extracting skills from text
        skill_graph: Graph for skill relationship inference
    """

    def __init__(
        self,
        skill_extractor: SkillExtractor,
        skill_graph: Optional[SkillGraph] = None
    ):
        """
        Initialize skill gap analyzer.

        Args:
            skill_extractor: Configured skill extraction service
            skill_graph: Optional skill relationship graph (uses default if None)
        """
        self.skill_extractor = skill_extractor
        self.skill_graph = skill_graph or get_default_skill_graph()

    async def calculate_gaps(
        self,
        cv_text: str,
        target_jobs: List[dict],
        limit: int = 20,
        include_inferred: bool = True
    ) -> List[SkillGap]:
        """
        Calculate skill gaps between CV and target jobs.

        Args:
            cv_text: Candidate's CV text
            target_jobs: List of job dicts with "description" key
            limit: Maximum number of gaps to return
            include_inferred: Whether to use skill inference

        Returns:
            List of SkillGap objects, sorted by importance

        Example:
            >>> gaps = await analyzer.calculate_gaps(
            ...     cv_text="Python developer",
            ...     target_jobs=[{"description": "Python + K8s required"}]
            ... )
            >>> gaps[0].skill
            'Kubernetes'
        """
        # Extract CV skills
        cv_skills = await self.skill_extractor.extract_skills(cv_text)
        cv_skill_names = {s.name.lower() for s in cv_skills}

        # Expand CV skills with inferences
        if include_inferred:
            cv_skill_names = self.skill_graph.infer_skills(
                cv_skill_names, include_related=False
            )

        # Extract skills from all target jobs
        job_skill_counts: Counter = Counter()
        job_skill_categories: dict[str, str] = {}
        total_jobs = len(target_jobs)

        for job in target_jobs:
            description = job.get("description", "")
            if not description:
                continue

            job_skills = await self.skill_extractor.extract_skills(description)

            for skill in job_skills:
                skill_lower = skill.name.lower()
                # Weight required skills higher
                weight = 2 if skill.required else 1
                job_skill_counts[skill_lower] += weight

                # Store category (last seen wins, but usually consistent)
                if hasattr(skill, 'category'):
                    job_skill_categories[skill_lower] = skill.category

        # Identify gaps (skills in jobs but not in CV)
        gaps = []
        for skill, count in job_skill_counts.most_common():
            if skill not in cv_skill_names:
                # Calculate frequency (normalized by job count)
                frequency = count / (total_jobs * 2)  # *2 because required skills count as 2
                frequency = min(1.0, frequency)

                # Determine importance based on frequency
                if frequency > 0.7:
                    importance = "critical"
                elif frequency > 0.4:
                    importance = "important"
                else:
                    importance = "nice-to-have"

                # Get category
                category = job_skill_categories.get(skill, "technical")

                # Find related skills the candidate has
                related_present = self._find_related_present(skill, cv_skill_names)

                gap = SkillGap(
                    skill=skill,
                    frequency=frequency,
                    importance=importance,
                    category=category,
                    learning_resources=[],
                    related_skills_present=related_present
                )
                gaps.append(gap)

        # Sort by importance, then frequency
        importance_order = {"critical": 0, "important": 1, "nice-to-have": 2}
        gaps.sort(key=lambda g: (importance_order.get(g.importance, 2), -g.frequency))

        return gaps[:limit]

    def _find_related_present(
        self,
        missing_skill: str,
        cv_skills: Set[str]
    ) -> List[str]:
        """
        Find CV skills related to a missing skill.

        Useful for showing candidates they have transferable skills.

        Args:
            missing_skill: The skill that's missing
            cv_skills: Set of skills the candidate has

        Returns:
            List of related skills the candidate has
        """
        related = []
        similar = self.skill_graph.get_similar_skills(missing_skill)

        for skill in similar:
            if skill in cv_skills:
                related.append(skill)

        return related[:3]  # Limit to top 3

    async def get_summary(
        self,
        cv_text: str,
        target_jobs: List[dict]
    ) -> SkillGapSummary:
        """
        Get a summary of skill gap analysis.

        Args:
            cv_text: Candidate's CV text
            target_jobs: List of job dicts

        Returns:
            SkillGapSummary with aggregated statistics
        """
        gaps = await self.calculate_gaps(
            cv_text=cv_text,
            target_jobs=target_jobs,
            limit=50  # Get more for full analysis
        )

        # Calculate statistics
        critical_gaps = sum(1 for g in gaps if g.importance == "critical")
        technical_gaps = sum(1 for g in gaps if g.category == "technical")
        soft_gaps = sum(1 for g in gaps if g.category == "soft")

        # Calculate coverage score
        cv_skills = await self.skill_extractor.extract_skills(cv_text)
        cv_skill_count = len(cv_skills)

        all_job_skills: Set[str] = set()
        for job in target_jobs:
            job_skills = await self.skill_extractor.extract_skills(
                job.get("description", "")
            )
            all_job_skills.update(s.name.lower() for s in job_skills if s.required)

        if all_job_skills:
            matched = sum(1 for s in cv_skills if s.name.lower() in all_job_skills)
            coverage_score = (matched / len(all_job_skills)) * 100
        else:
            coverage_score = 100.0

        return SkillGapSummary(
            total_gaps=len(gaps),
            critical_gaps=critical_gaps,
            technical_gaps=technical_gaps,
            soft_gaps=soft_gaps,
            top_gaps=gaps[:10],
            coverage_score=coverage_score
        )

    async def recommend_learning_path(
        self,
        gaps: List[SkillGap],
        max_skills: int = 5
    ) -> List[dict]:
        """
        Generate a recommended learning path based on skill gaps.

        Prioritizes:
        1. Critical skills
        2. Skills with related skills present (easier to learn)
        3. High-frequency skills

        Args:
            gaps: List of skill gaps from calculate_gaps
            max_skills: Maximum skills to include in path

        Returns:
            List of learning recommendations with rationale
        """
        recommendations = []

        # Prioritize critical skills with related skills present
        sorted_gaps = sorted(
            gaps,
            key=lambda g: (
                0 if g.importance == "critical" else 1,
                -len(g.related_skills_present),
                -g.frequency
            )
        )

        for gap in sorted_gaps[:max_skills]:
            rationale = []

            if gap.importance == "critical":
                rationale.append(f"Required in {gap.frequency*100:.0f}% of target jobs")

            if gap.related_skills_present:
                rationale.append(
                    f"Related to skills you have: {', '.join(gap.related_skills_present)}"
                )

            recommendations.append({
                "skill": gap.skill,
                "category": gap.category,
                "importance": gap.importance,
                "rationale": "; ".join(rationale) if rationale else "Common requirement",
                "learning_resources": gap.learning_resources
            })

        return recommendations


def categorize_skill(skill_name: str) -> str:
    """
    Categorize a skill based on common patterns.

    Args:
        skill_name: Skill name to categorize

    Returns:
        Category string (technical, soft, domain, tool)
    """
    skill_lower = skill_name.lower()

    # Technical patterns
    technical_patterns = [
        "python", "java", "javascript", "typescript", "go", "rust",
        "react", "angular", "vue", "django", "flask", "spring",
        "aws", "azure", "gcp", "docker", "kubernetes", "terraform",
        "postgresql", "mysql", "mongodb", "redis",
        "api", "rest", "graphql", "microservices",
        "machine learning", "deep learning", "nlp", "data science"
    ]

    soft_patterns = [
        "communication", "leadership", "management", "teamwork",
        "problem solving", "analytical", "presentation", "mentoring",
        "collaboration", "negotiation", "stakeholder"
    ]

    tool_patterns = [
        "git", "jira", "confluence", "slack", "jenkins", "github",
        "figma", "sketch", "photoshop", "excel", "tableau", "powerbi"
    ]

    for pattern in technical_patterns:
        if pattern in skill_lower:
            return "technical"

    for pattern in soft_patterns:
        if pattern in skill_lower:
            return "soft"

    for pattern in tool_patterns:
        if pattern in skill_lower:
            return "tool"

    return "domain"  # Default to domain knowledge
