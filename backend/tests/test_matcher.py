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
