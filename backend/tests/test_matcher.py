"""
Tests for job matching service.

Run with: cd backend && pytest tests/test_matcher.py -v
"""
import pytest


class TestSkillsTaxonomy:
    """Tests for expanded skills taxonomy with synonyms."""

    def test_skills_taxonomy_has_required_categories(self):
        """Taxonomy should have all 13 categories."""
        from app.services.matcher import SKILLS_TAXONOMY

        expected_categories = {
            # Technical
            "languages", "frontend", "backend", "cloud", "data", "ai_ml",
            # Professional
            "consulting", "project_management", "leadership", "strategy",
            "communication", "delivery", "commercial"
        }
        assert set(SKILLS_TAXONOMY.keys()) == expected_categories

    def test_skills_taxonomy_has_synonyms(self):
        """Each skill should map to a list of synonyms."""
        from app.services.matcher import SKILLS_TAXONOMY

        # Check python has synonyms
        assert "python" in SKILLS_TAXONOMY["languages"]
        assert isinstance(SKILLS_TAXONOMY["languages"]["python"], list)
        assert "py" in SKILLS_TAXONOMY["languages"]["python"]

    def test_skills_taxonomy_all_categories_have_skills(self):
        """Each category should contain at least one skill."""
        from app.services.matcher import SKILLS_TAXONOMY

        for category, skills in SKILLS_TAXONOMY.items():
            assert len(skills) > 0, f"Category '{category}' has no skills"
            for skill, synonyms in skills.items():
                assert isinstance(synonyms, list), f"Skill '{skill}' synonyms should be a list"

    def test_tech_skills_backwards_compatibility(self):
        """TECH_SKILLS should still exist for backwards compatibility."""
        from app.services.matcher import TECH_SKILLS, SKILLS_TAXONOMY

        # TECH_SKILLS should be a flat list
        assert isinstance(TECH_SKILLS, list)
        # Should contain skills from taxonomy
        assert "python" in TECH_SKILLS
        assert "react" in TECH_SKILLS
        assert "aws" in TECH_SKILLS

    def test_extract_skills_finds_primary_skill(self):
        """Should find primary skill names."""
        from app.services.matcher import extract_skills_with_synonyms

        text = "Looking for a Python developer with React experience"
        result = extract_skills_with_synonyms(text)

        assert "languages" in result
        assert "python" in result["languages"]
        assert "frontend" in result
        assert "react" in result["frontend"]

    def test_extract_skills_finds_synonyms(self):
        """Should find skills via synonyms."""
        from app.services.matcher import extract_skills_with_synonyms

        text = "Must have k8s and nodejs experience"
        result = extract_skills_with_synonyms(text)

        assert "cloud" in result
        assert "kubernetes" in result["cloud"]  # Found via "k8s"
        assert "backend" in result
        assert "express" in result["backend"]  # Found via "nodejs"

    def test_extract_skills_word_boundary(self):
        """Should not match partial words."""
        from app.services.matcher import extract_skills_with_synonyms

        text = "Experience with Juniper Networks required"
        result = extract_skills_with_synonyms(text)

        # "junior" should NOT be matched from "Juniper"
        assert "leadership" not in result or "junior" not in result.get("leadership", [])

    def test_extract_skills_case_insensitive(self):
        """Should match regardless of case."""
        from app.services.matcher import extract_skills_with_synonyms

        text = "AWS, PYTHON, React, KUBERNETES"
        result = extract_skills_with_synonyms(text)

        assert "python" in result.get("languages", [])
        assert "aws" in result.get("cloud", [])
        assert "react" in result.get("frontend", [])
        assert "kubernetes" in result.get("cloud", [])

    def test_extract_skills_empty_text(self):
        """Should return empty dict for empty text."""
        from app.services.matcher import extract_skills_with_synonyms

        result = extract_skills_with_synonyms("")
        assert result == {}

    def test_extract_skills_finds_professional_skills(self):
        """Should find professional/soft skills."""
        from app.services.matcher import extract_skills_with_synonyms

        text = "Strong stakeholder management and agile delivery experience required. P&L ownership essential."
        result = extract_skills_with_synonyms(text)

        assert "consulting" in result
        assert "stakeholder engagement" in result["consulting"]
        assert "project_management" in result
        assert "agile delivery" in result["project_management"]
        assert "strategy" in result
        assert "p&l ownership" in result["strategy"]


class TestSalaryMatching:
    """Tests for salary matching logic."""
    pass


class TestExclusionKeywords:
    """Tests for exclusion keyword filtering."""
    pass


class TestGraduatedLocation:
    """Tests for graduated UK location matching."""
    pass


class TestCalculateMatchScore:
    """Integration tests for the full scoring function."""
    pass
