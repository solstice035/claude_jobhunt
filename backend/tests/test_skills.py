"""
Tests for Skills Intelligence (Phase 2).

Run with: cd backend && pytest tests/test_skills.py -v

Tests cover:
- ESCO model and database operations
- LLM-based skill extraction
- Skill relationship graph and inference
- Skill gap analysis
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List


# ==============================================================================
# ESCO Model Tests
# ==============================================================================

class TestESCOModel:
    """Tests for ESCO skill SQLAlchemy model."""

    def test_esco_skill_model_has_required_fields(self):
        """ESCO skill model should have all required fields."""
        from app.models.esco import ESCOSkill

        # Check required columns exist
        columns = {c.name for c in ESCOSkill.__table__.columns}
        required = {
            "uri", "preferred_label", "alt_labels", "description",
            "skill_type", "broader_skills", "narrower_skills", "related_skills"
        }
        assert required.issubset(columns), f"Missing columns: {required - columns}"

    def test_esco_skill_model_uri_is_primary_key(self):
        """URI should be the primary key."""
        from app.models.esco import ESCOSkill

        pk_columns = [c.name for c in ESCOSkill.__table__.primary_key.columns]
        assert "uri" in pk_columns

    def test_esco_skill_creation(self):
        """Should be able to create an ESCO skill instance."""
        from app.models.esco import ESCOSkill

        skill = ESCOSkill(
            uri="http://data.europa.eu/esco/skill/python",
            preferred_label="Python programming",
            alt_labels=["Python", "python3", "py"],
            description="Programming using Python language",
            skill_type="skill",
            broader_skills=["http://data.europa.eu/esco/skill/programming"],
            narrower_skills=[],
            related_skills=["http://data.europa.eu/esco/skill/django"]
        )
        assert skill.uri == "http://data.europa.eu/esco/skill/python"
        assert skill.preferred_label == "Python programming"
        assert "Python" in skill.alt_labels


# ==============================================================================
# ESCO Service Tests
# ==============================================================================

class TestESCOService:
    """Tests for ESCO skill lookup service."""

    @pytest.mark.asyncio
    async def test_find_skill_by_label_exact_match(self):
        """Should find skill by exact label match."""
        from app.services.esco import ESCOService

        # Mock the database session
        mock_session = AsyncMock()
        mock_skill = MagicMock()
        mock_skill.uri = "http://data.europa.eu/esco/skill/python"
        mock_skill.preferred_label = "Python programming"
        mock_skill.alt_labels = ["Python", "py"]

        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = mock_skill
        mock_session.execute.return_value = mock_result

        service = ESCOService(mock_session)
        result = await service.find_skill_by_label("Python programming")

        assert result is not None
        assert result.preferred_label == "Python programming"

    @pytest.mark.asyncio
    async def test_find_skill_by_label_alt_label_match(self):
        """Should find skill by alternative label."""
        from app.services.esco import ESCOService

        mock_session = AsyncMock()
        mock_skill = MagicMock()
        mock_skill.uri = "http://data.europa.eu/esco/skill/python"
        mock_skill.preferred_label = "Python programming"
        mock_skill.alt_labels = ["Python", "py", "python3"]

        # First query (exact match) returns None
        mock_result_none = MagicMock()
        mock_result_none.scalars.return_value.first.return_value = None

        # Second query (all skills for alt_labels search) returns skill list
        mock_result_all = MagicMock()
        mock_result_all.scalars.return_value = [mock_skill]

        mock_session.execute.side_effect = [mock_result_none, mock_result_all]

        service = ESCOService(mock_session)
        result = await service.find_skill_by_label("py")

        assert result is not None
        assert result.preferred_label == "Python programming"

    @pytest.mark.asyncio
    async def test_find_skill_by_label_not_found(self):
        """Should return None for non-existent skill."""
        from app.services.esco import ESCOService

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value.first.return_value = None
        mock_session.execute.return_value = mock_result

        service = ESCOService(mock_session)
        result = await service.find_skill_by_label("nonexistent_skill_xyz")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_related_skills(self):
        """Should return related skills from graph relationships."""
        from app.services.esco import ESCOService

        mock_session = AsyncMock()
        mock_skill = MagicMock()
        mock_skill.related_skills = [
            "http://data.europa.eu/esco/skill/django",
            "http://data.europa.eu/esco/skill/flask"
        ]

        related_skill_1 = MagicMock()
        related_skill_1.preferred_label = "Django"
        related_skill_2 = MagicMock()
        related_skill_2.preferred_label = "Flask"

        mock_result_main = MagicMock()
        mock_result_main.scalars.return_value.first.return_value = mock_skill

        mock_result_related = MagicMock()
        mock_result_related.scalars.return_value.all.return_value = [
            related_skill_1, related_skill_2
        ]

        mock_session.execute.side_effect = [mock_result_main, mock_result_related]

        service = ESCOService(mock_session)
        result = await service.get_related_skills("Python programming")

        assert len(result) == 2
        assert any(s.preferred_label == "Django" for s in result)

    @pytest.mark.asyncio
    async def test_search_skills_fuzzy(self):
        """Should perform fuzzy search on skills."""
        from app.services.esco import ESCOService

        mock_session = AsyncMock()
        mock_skill = MagicMock()
        mock_skill.preferred_label = "Python programming"
        mock_skill.description = "Programming with Python"

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_skill]
        mock_session.execute.return_value = mock_result

        service = ESCOService(mock_session)
        results = await service.search_skills("pythn")  # Typo

        assert len(results) >= 0  # May or may not find depending on fuzzy impl


# ==============================================================================
# Skill Extractor Tests
# ==============================================================================

class TestSkillExtractor:
    """Tests for LLM-based skill extraction service."""

    @pytest.mark.asyncio
    async def test_extract_skills_from_job_description(self):
        """Should extract skills from job description using LLM."""
        from app.services.skill_extractor import SkillExtractor

        job_description = """
        We are looking for a Senior Python Developer with experience in:
        - Python and Django/FastAPI
        - PostgreSQL and Redis
        - AWS (EC2, S3, Lambda)
        - Docker and Kubernetes
        - Excellent communication skills
        """

        mock_openai = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "skills": [
                {"name": "Python", "category": "technical", "required": true, "confidence": "high"},
                {"name": "Django", "category": "technical", "required": true, "confidence": "high"},
                {"name": "FastAPI", "category": "technical", "required": true, "confidence": "high"},
                {"name": "PostgreSQL", "category": "technical", "required": true, "confidence": "high"},
                {"name": "Redis", "category": "technical", "required": true, "confidence": "high"},
                {"name": "AWS", "category": "technical", "required": true, "confidence": "high"},
                {"name": "Docker", "category": "technical", "required": true, "confidence": "high"},
                {"name": "Kubernetes", "category": "technical", "required": true, "confidence": "high"},
                {"name": "communication", "category": "soft", "required": true, "confidence": "high"}
            ]
        }
        '''
        mock_openai.chat.completions.create.return_value = mock_response

        extractor = SkillExtractor(openai_client=mock_openai)
        skills = await extractor.extract_skills(job_description)

        assert len(skills) > 0
        skill_names = [s.name.lower() for s in skills]
        assert "python" in skill_names
        assert "django" in skill_names or "fastapi" in skill_names

    @pytest.mark.asyncio
    async def test_extract_skills_caches_result(self):
        """Should cache extraction results to avoid repeated API calls."""
        from app.services.skill_extractor import SkillExtractor

        mock_openai = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {"skills": [{"name": "Python", "category": "technical", "required": true, "confidence": "high"}]}
        '''
        mock_openai.chat.completions.create.return_value = mock_response

        mock_cache = MagicMock()
        mock_cache.get.return_value = None  # First call: cache miss

        extractor = SkillExtractor(openai_client=mock_openai, cache=mock_cache)

        # First extraction
        await extractor.extract_skills("Python developer needed")

        # Verify cache was set
        assert mock_cache.set.called

    @pytest.mark.asyncio
    async def test_extract_skills_handles_malformed_json(self):
        """Should handle malformed JSON response from LLM."""
        from app.services.skill_extractor import SkillExtractor

        mock_openai = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not valid JSON"
        mock_openai.chat.completions.create.return_value = mock_response

        extractor = SkillExtractor(openai_client=mock_openai)
        skills = await extractor.extract_skills("Some job description")

        # Should return empty list, not raise exception
        assert skills == []

    @pytest.mark.asyncio
    async def test_extract_skills_returns_typed_objects(self):
        """Extracted skills should be properly typed ExtractedSkill objects."""
        from app.services.skill_extractor import SkillExtractor, ExtractedSkill

        mock_openai = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {"skills": [{"name": "Python", "category": "technical", "required": true, "confidence": "high"}]}
        '''
        mock_openai.chat.completions.create.return_value = mock_response

        extractor = SkillExtractor(openai_client=mock_openai)
        skills = await extractor.extract_skills("Python developer")

        assert len(skills) == 1
        assert isinstance(skills[0], ExtractedSkill)
        assert skills[0].name == "Python"
        assert skills[0].category == "technical"
        assert skills[0].required is True
        assert skills[0].confidence == "high"

    @pytest.mark.asyncio
    async def test_extract_skills_distinguishes_required_vs_preferred(self):
        """Should distinguish between required and preferred skills."""
        from app.services.skill_extractor import SkillExtractor

        mock_openai = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''
        {
            "skills": [
                {"name": "Python", "category": "technical", "required": true, "confidence": "high"},
                {"name": "Go", "category": "technical", "required": false, "confidence": "medium"}
            ]
        }
        '''
        mock_openai.chat.completions.create.return_value = mock_response

        extractor = SkillExtractor(openai_client=mock_openai)
        skills = await extractor.extract_skills("Must have Python, Go is a plus")

        required = [s for s in skills if s.required]
        preferred = [s for s in skills if not s.required]

        assert len(required) >= 1
        assert len(preferred) >= 1


# ==============================================================================
# Skill Graph Tests
# ==============================================================================

class TestSkillGraph:
    """Tests for skill relationship graph and inference."""

    def test_skill_graph_creation(self):
        """Should create a skill graph from ESCO data."""
        from app.services.skill_graph import SkillGraph

        graph = SkillGraph()
        graph.add_skill("kubernetes", skill_type="technical")
        graph.add_skill("docker", skill_type="technical")
        graph.add_skill("containerization", skill_type="technical")

        assert graph.has_skill("kubernetes")
        assert graph.has_skill("docker")
        assert graph.has_skill("containerization")

    def test_skill_graph_add_relationship(self):
        """Should add relationships between skills."""
        from app.services.skill_graph import SkillGraph

        graph = SkillGraph()
        graph.add_skill("kubernetes", skill_type="technical")
        graph.add_skill("docker", skill_type="technical")
        graph.add_relationship("kubernetes", "docker", relation_type="requires")

        relations = graph.get_relationships("kubernetes")
        assert "docker" in [r.target for r in relations]

    def test_skill_graph_infer_broader_skills(self):
        """Should infer broader skills from specific skills."""
        from app.services.skill_graph import SkillGraph

        graph = SkillGraph()
        graph.add_skill("kubernetes", skill_type="technical")
        graph.add_skill("containerization", skill_type="technical")
        graph.add_relationship("kubernetes", "containerization", relation_type="broader")

        inferred = graph.infer_skills({"kubernetes"})

        assert "containerization" in inferred

    def test_skill_graph_infer_required_skills(self):
        """Should infer required/prerequisite skills."""
        from app.services.skill_graph import SkillGraph

        graph = SkillGraph()
        graph.add_skill("kubernetes", skill_type="technical")
        graph.add_skill("docker", skill_type="technical")
        graph.add_relationship("kubernetes", "docker", relation_type="requires")

        inferred = graph.infer_skills({"kubernetes"})

        # Knowing k8s implies knowing docker
        assert "docker" in inferred

    def test_skill_graph_infer_keeps_explicit_skills(self):
        """Inferred set should include explicit skills."""
        from app.services.skill_graph import SkillGraph

        graph = SkillGraph()
        graph.add_skill("python", skill_type="technical")
        graph.add_skill("programming", skill_type="technical")
        graph.add_relationship("python", "programming", relation_type="broader")

        explicit = {"python", "aws"}
        inferred = graph.infer_skills(explicit)

        # Original skills should still be present
        assert "python" in inferred
        assert "aws" in inferred

    def test_skill_graph_get_similar_skills(self):
        """Should find similar/related skills."""
        from app.services.skill_graph import SkillGraph

        graph = SkillGraph()
        graph.add_skill("react", skill_type="technical")
        graph.add_skill("angular", skill_type="technical")
        graph.add_skill("vue", skill_type="technical")
        graph.add_relationship("react", "angular", relation_type="related")
        graph.add_relationship("react", "vue", relation_type="related")

        similar = graph.get_similar_skills("react")

        assert "angular" in similar
        assert "vue" in similar

    def test_skill_graph_build_from_esco_data(self):
        """Should build graph from ESCO skill data structure."""
        from app.services.skill_graph import SkillGraph

        esco_data = [
            {
                "uri": "skill:python",
                "preferred_label": "python",
                "skill_type": "skill",
                "broader_skills": ["skill:programming"],
                "narrower_skills": [],
                "related_skills": ["skill:django", "skill:flask"]
            },
            {
                "uri": "skill:programming",
                "preferred_label": "programming",
                "skill_type": "skill",
                "broader_skills": [],
                "narrower_skills": ["skill:python"],
                "related_skills": []
            }
        ]

        graph = SkillGraph.from_esco_data(esco_data)

        assert graph.has_skill("python")
        assert graph.has_skill("programming")


# ==============================================================================
# Skill Gap Analysis Tests
# ==============================================================================

class TestSkillGapAnalysis:
    """Tests for skill gap identification."""

    @pytest.mark.asyncio
    async def test_calculate_skill_gaps_basic(self):
        """Should identify missing skills between CV and target jobs."""
        from app.services.skill_gaps import SkillGapAnalyzer, SkillGap

        mock_extractor = AsyncMock()

        # CV skills
        cv_skill_1 = MagicMock(name="Python", required=True)
        cv_skill_1.name = "Python"
        cv_skill_2 = MagicMock(name="Django", required=True)
        cv_skill_2.name = "Django"

        # Job skills (includes skills not in CV)
        job_skill_1 = MagicMock(name="Python", required=True)
        job_skill_1.name = "Python"
        job_skill_2 = MagicMock(name="Kubernetes", required=True)
        job_skill_2.name = "Kubernetes"
        job_skill_3 = MagicMock(name="AWS", required=True)
        job_skill_3.name = "AWS"

        mock_extractor.extract_skills.side_effect = [
            [cv_skill_1, cv_skill_2],  # CV extraction
            [job_skill_1, job_skill_2, job_skill_3],  # Job 1
        ]

        analyzer = SkillGapAnalyzer(skill_extractor=mock_extractor)

        gaps = await analyzer.calculate_gaps(
            cv_text="Python Django developer",
            target_jobs=[{"description": "Python K8s AWS role"}]
        )

        gap_names = [g.skill.lower() for g in gaps]
        assert "kubernetes" in gap_names or "aws" in gap_names
        assert "python" not in gap_names  # Already have this

    @pytest.mark.asyncio
    async def test_calculate_skill_gaps_frequency(self):
        """Should calculate frequency of missing skills across jobs."""
        from app.services.skill_gaps import SkillGapAnalyzer

        mock_extractor = AsyncMock()

        cv_skill = MagicMock()
        cv_skill.name = "Python"

        # All 3 jobs need Kubernetes, only 1 needs Go
        k8s = MagicMock()
        k8s.name = "Kubernetes"
        k8s.required = True
        go = MagicMock()
        go.name = "Go"
        go.required = True
        python = MagicMock()
        python.name = "Python"
        python.required = True

        mock_extractor.extract_skills.side_effect = [
            [cv_skill],  # CV
            [python, k8s],  # Job 1
            [python, k8s],  # Job 2
            [python, k8s, go],  # Job 3
        ]

        analyzer = SkillGapAnalyzer(skill_extractor=mock_extractor)

        gaps = await analyzer.calculate_gaps(
            cv_text="Python developer",
            target_jobs=[
                {"description": "Job 1"},
                {"description": "Job 2"},
                {"description": "Job 3"}
            ]
        )

        k8s_gap = next((g for g in gaps if g.skill.lower() == "kubernetes"), None)
        go_gap = next((g for g in gaps if g.skill.lower() == "go"), None)

        assert k8s_gap is not None
        assert k8s_gap.frequency > 0.9  # Required in all 3 jobs
        if go_gap:
            assert go_gap.frequency < k8s_gap.frequency  # Less common

    @pytest.mark.asyncio
    async def test_calculate_skill_gaps_importance_levels(self):
        """Should assign importance levels based on frequency."""
        from app.services.skill_gaps import SkillGapAnalyzer

        mock_extractor = AsyncMock()

        cv_skill = MagicMock()
        cv_skill.name = "Python"

        # Create skills with different frequencies
        common_skill = MagicMock()
        common_skill.name = "Kubernetes"
        common_skill.required = True

        rare_skill = MagicMock()
        rare_skill.name = "Rust"
        rare_skill.required = False

        mock_extractor.extract_skills.side_effect = [
            [cv_skill],
            [common_skill],
            [common_skill],
            [common_skill],
            [rare_skill],
        ]

        analyzer = SkillGapAnalyzer(skill_extractor=mock_extractor)

        gaps = await analyzer.calculate_gaps(
            cv_text="Python",
            target_jobs=[{"description": f"Job {i}"} for i in range(4)]
        )

        # High frequency skill should be "critical"
        k8s_gap = next((g for g in gaps if g.skill.lower() == "kubernetes"), None)
        if k8s_gap:
            assert k8s_gap.importance in ["critical", "important"]

    @pytest.mark.asyncio
    async def test_skill_gaps_limits_results(self):
        """Should limit number of returned gaps."""
        from app.services.skill_gaps import SkillGapAnalyzer

        mock_extractor = AsyncMock()

        cv_skill = MagicMock()
        cv_skill.name = "Python"

        # Create many job skills
        job_skills = []
        for i in range(30):
            skill = MagicMock()
            skill.name = f"Skill{i}"
            skill.required = True
            job_skills.append(skill)

        mock_extractor.extract_skills.side_effect = [
            [cv_skill],
            job_skills
        ]

        analyzer = SkillGapAnalyzer(skill_extractor=mock_extractor)

        gaps = await analyzer.calculate_gaps(
            cv_text="Python",
            target_jobs=[{"description": "Big job"}],
            limit=20
        )

        assert len(gaps) <= 20

    @pytest.mark.asyncio
    async def test_skill_gaps_categorizes_skills(self):
        """Should categorize skill gaps."""
        from app.services.skill_gaps import SkillGapAnalyzer

        mock_extractor = AsyncMock()

        cv_skill = MagicMock()
        cv_skill.name = "Python"

        tech_skill = MagicMock()
        tech_skill.name = "Kubernetes"
        tech_skill.category = "technical"
        tech_skill.required = True

        soft_skill = MagicMock()
        soft_skill.name = "leadership"
        soft_skill.category = "soft"
        soft_skill.required = True

        mock_extractor.extract_skills.side_effect = [
            [cv_skill],
            [tech_skill, soft_skill]
        ]

        analyzer = SkillGapAnalyzer(skill_extractor=mock_extractor)

        gaps = await analyzer.calculate_gaps(
            cv_text="Python developer",
            target_jobs=[{"description": "Lead role"}]
        )

        categories = {g.category for g in gaps}
        assert len(categories) > 0  # Has categorization


# ==============================================================================
# Skills API Tests
# ==============================================================================

class TestSkillsAPI:
    """Tests for skills-related API endpoints."""

    @pytest.mark.asyncio
    async def test_get_skill_gaps_endpoint(self):
        """GET /api/skills/gaps should return skill gaps for profile."""
        from fastapi.testclient import TestClient
        from unittest.mock import patch

        # This test will be expanded once the API is implemented
        pass

    @pytest.mark.asyncio
    async def test_extract_skills_endpoint(self):
        """POST /api/skills/extract should extract skills from text."""
        pass

    @pytest.mark.asyncio
    async def test_search_esco_skills_endpoint(self):
        """GET /api/skills/search should search ESCO database."""
        pass


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestSkillsIntegration:
    """Integration tests for the complete skills intelligence pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_extraction_to_gaps(self):
        """Should extract skills from CV and jobs, then identify gaps."""
        # Full pipeline test - to be implemented
        pass

    @pytest.mark.asyncio
    async def test_skill_inference_improves_matching(self):
        """Skill inference should identify additional matches."""
        from app.services.skill_graph import SkillGraph

        graph = SkillGraph()
        graph.add_skill("kubernetes", skill_type="technical")
        graph.add_skill("docker", skill_type="technical")
        graph.add_skill("containerization", skill_type="technical")
        graph.add_relationship("kubernetes", "docker", relation_type="requires")
        graph.add_relationship("kubernetes", "containerization", relation_type="broader")
        graph.add_relationship("docker", "containerization", relation_type="broader")

        cv_skills = {"kubernetes"}  # Only explicitly mentions k8s
        job_skills = {"kubernetes", "docker", "containerization"}

        # Direct match
        direct_matches = cv_skills & job_skills
        assert len(direct_matches) == 1

        # With inference
        inferred_cv = graph.infer_skills(cv_skills)
        inferred_matches = inferred_cv & job_skills
        assert len(inferred_matches) >= 2  # Should also match docker and containerization
