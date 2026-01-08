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

    def test_salary_meets_target(self):
        """Job at or above target should score 1.0."""
        from app.services.matcher import match_salary

        score, reason = match_salary(
            job_min=100000, job_max=120000,
            profile_min=80000, profile_target=100000
        )
        assert score == 1.0
        assert "meets target" in reason.lower()

    def test_salary_above_minimum(self):
        """Job between min and target should score 0.5-1.0."""
        from app.services.matcher import match_salary

        score, reason = match_salary(
            job_min=85000, job_max=95000,  # Midpoint 90k
            profile_min=80000, profile_target=100000
        )
        assert 0.5 < score < 1.0
        assert "above minimum" in reason.lower()

    def test_salary_below_minimum(self):
        """Job below minimum should score < 0.5."""
        from app.services.matcher import match_salary

        score, reason = match_salary(
            job_min=50000, job_max=60000,  # Midpoint 55k
            profile_min=80000, profile_target=100000
        )
        assert score < 0.5
        assert "below minimum" in reason.lower()

    def test_salary_no_job_data(self):
        """Missing job salary should return neutral 0.5."""
        from app.services.matcher import match_salary

        score, reason = match_salary(
            job_min=None, job_max=None,
            profile_min=80000, profile_target=100000
        )
        assert score == 0.5
        assert reason is None

    def test_salary_no_profile_preference(self):
        """No profile salary preference should return neutral 0.5."""
        from app.services.matcher import match_salary

        score, reason = match_salary(
            job_min=100000, job_max=120000,
            profile_min=None, profile_target=None
        )
        assert score == 0.5
        assert reason is None

    def test_salary_only_job_min(self):
        """Should work with only job_min provided."""
        from app.services.matcher import match_salary

        score, reason = match_salary(
            job_min=100000, job_max=None,
            profile_min=80000, profile_target=100000
        )
        assert score == 1.0

    def test_salary_only_profile_min(self):
        """Should work with only profile_min (no target)."""
        from app.services.matcher import match_salary

        score, reason = match_salary(
            job_min=100000, job_max=120000,
            profile_min=80000, profile_target=None
        )
        assert score == 1.0  # Above minimum = meets target when no target set


class TestExclusionKeywords:
    """Tests for exclusion keyword filtering."""

    def test_exclusion_found_in_title(self):
        """Should detect exclusion keyword in job title."""
        from app.services.matcher import check_exclusions

        should_exclude, matched = check_exclusions(
            job_title="Junior Developer",
            job_description="Entry level role",
            exclude_keywords=["junior"]
        )
        assert should_exclude is True
        assert matched == "junior"

    def test_exclusion_found_in_description(self):
        """Should detect exclusion keyword in description."""
        from app.services.matcher import check_exclusions

        should_exclude, matched = check_exclusions(
            job_title="Software Developer",
            job_description="This is a 6-month contract role",
            exclude_keywords=["contract"]
        )
        assert should_exclude is True
        assert matched == "contract"

    def test_exclusion_word_boundary(self):
        """Should not match partial words."""
        from app.services.matcher import check_exclusions

        should_exclude, matched = check_exclusions(
            job_title="Network Engineer at Juniper",
            job_description="Working with Juniper Networks equipment",
            exclude_keywords=["junior"]
        )
        assert should_exclude is False
        assert matched is None

    def test_exclusion_case_insensitive(self):
        """Should match regardless of case."""
        from app.services.matcher import check_exclusions

        should_exclude, matched = check_exclusions(
            job_title="JUNIOR Developer",
            job_description="Entry level role",
            exclude_keywords=["junior"]
        )
        assert should_exclude is True

    def test_exclusion_no_keywords(self):
        """Should not exclude when no keywords provided."""
        from app.services.matcher import check_exclusions

        should_exclude, matched = check_exclusions(
            job_title="Junior Developer",
            job_description="Entry level role",
            exclude_keywords=[]
        )
        assert should_exclude is False
        assert matched is None

    def test_exclusion_multiple_keywords(self):
        """Should check all keywords and return first match."""
        from app.services.matcher import check_exclusions

        should_exclude, matched = check_exclusions(
            job_title="PHP Developer",
            job_description="WordPress development",
            exclude_keywords=["junior", "PHP", "recruitment"]
        )
        assert should_exclude is True
        assert matched == "PHP"

    def test_exclusion_empty_keyword_ignored(self):
        """Should ignore empty strings in keyword list."""
        from app.services.matcher import check_exclusions

        should_exclude, matched = check_exclusions(
            job_title="Software Developer",
            job_description="Great role",
            exclude_keywords=["", "  ", "junior"]
        )
        assert should_exclude is False


class TestGraduatedLocation:
    """Tests for graduated UK location matching."""

    def test_location_exact_match(self):
        """Exact location match should score 1.0."""
        from app.services.matcher import match_location_graduated

        score, reason = match_location_graduated(
            job_location="London",
            preferred_locations=["London"]
        )
        assert score == 1.0
        assert "London" in reason

    def test_location_remote_preferred(self):
        """Remote job when remote preferred should score 1.0."""
        from app.services.matcher import match_location_graduated

        score, reason = match_location_graduated(
            job_location="Fully Remote",
            preferred_locations=["Remote"]
        )
        assert score == 1.0
        assert "Remote" in reason

    def test_location_remote_not_preferred(self):
        """Remote job when not preferred should score 0.8."""
        from app.services.matcher import match_location_graduated

        score, reason = match_location_graduated(
            job_location="Remote - UK",
            preferred_locations=["London"]
        )
        assert score == 0.8
        assert "Remote" in reason

    def test_location_hybrid(self):
        """Hybrid should score 0.9 for remote/hybrid preferences."""
        from app.services.matcher import match_location_graduated

        score, reason = match_location_graduated(
            job_location="Hybrid - London (2 days office)",
            preferred_locations=["Remote"]
        )
        assert score == 0.9
        assert "Hybrid" in reason

    def test_location_same_region(self):
        """Same UK region should score 0.8."""
        from app.services.matcher import match_location_graduated

        score, reason = match_location_graduated(
            job_location="Reading",
            preferred_locations=["Brighton"]  # Both South East
        )
        assert score == 0.8
        assert "region" in reason.lower()

    def test_location_no_preference(self):
        """No location preference should score 1.0."""
        from app.services.matcher import match_location_graduated

        score, reason = match_location_graduated(
            job_location="Edinburgh",
            preferred_locations=[]
        )
        assert score == 1.0
        assert reason is None

    def test_location_no_match(self):
        """No match should score 0.0."""
        from app.services.matcher import match_location_graduated

        score, reason = match_location_graduated(
            job_location="Edinburgh",
            preferred_locations=["London"]
        )
        assert score == 0.0
        assert reason is None

    def test_location_london_region(self):
        """London variations should be recognized."""
        from app.services.matcher import match_location_graduated

        score, reason = match_location_graduated(
            job_location="Central London",
            preferred_locations=["London"]
        )
        assert score == 1.0


class TestCalculateMatchScore:
    """Integration tests for the full scoring function."""

    def test_calculate_match_score_exclusion_returns_zero(self):
        """Excluded jobs should return score 0."""
        from app.services.matcher import calculate_match_score

        score, reasons = calculate_match_score(
            job_embedding=[0.1] * 1536,
            job_description="Junior PHP developer needed",
            job_title="Junior Developer",
            job_location="London",
            job_salary_min=30000,
            job_salary_max=40000,
            cv_embedding=[0.1] * 1536,
            cv_text="Senior Python developer with 10 years experience",
            target_roles=["CTO"],
            preferred_locations=["London"],
            exclude_keywords=["junior"],
            salary_min=80000,
            salary_target=100000,
            score_weights={"semantic": 0.25, "skills": 0.25, "seniority": 0.20, "location": 0.15, "salary": 0.15},
        )
        assert score == 0.0
        assert any("Excluded" in r for r in reasons)

    def test_calculate_match_score_uses_new_skills(self):
        """Should use expanded skills taxonomy."""
        from app.services.matcher import calculate_match_score

        score, reasons = calculate_match_score(
            job_embedding=[0.5] * 1536,
            job_description="Looking for k8s and nodejs expert with stakeholder management skills",
            job_title="Senior Engineer",
            job_location="London",
            job_salary_min=100000,
            job_salary_max=120000,
            cv_embedding=[0.5] * 1536,
            cv_text="Expert in kubernetes, node.js, and stakeholder engagement",
            target_roles=["Senior Engineer"],
            preferred_locations=["London"],
            exclude_keywords=[],
            salary_min=80000,
            salary_target=100000,
            score_weights={"semantic": 0.25, "skills": 0.25, "seniority": 0.20, "location": 0.15, "salary": 0.15},
        )
        # Should find kubernetes (via k8s), express (via nodejs), stakeholder engagement
        skill_reasons = [r for r in reasons if r.startswith("Skills:")]
        assert len(skill_reasons) > 0

    def test_calculate_match_score_includes_salary(self):
        """Should include salary in scoring."""
        from app.services.matcher import calculate_match_score

        score, reasons = calculate_match_score(
            job_embedding=[0.5] * 1536,
            job_description="Python developer role",
            job_title="Developer",
            job_location="London",
            job_salary_min=100000,
            job_salary_max=120000,
            cv_embedding=[0.5] * 1536,
            cv_text="Python developer",
            target_roles=["Developer"],
            preferred_locations=["London"],
            exclude_keywords=[],
            salary_min=80000,
            salary_target=100000,
            score_weights={"semantic": 0.25, "skills": 0.25, "seniority": 0.20, "location": 0.15, "salary": 0.15},
        )
        salary_reasons = [r for r in reasons if "Salary" in r]
        assert len(salary_reasons) > 0

    def test_calculate_match_score_default_weights(self):
        """Should work with missing weights (use defaults)."""
        from app.services.matcher import calculate_match_score

        score, reasons = calculate_match_score(
            job_embedding=[0.5] * 1536,
            job_description="Python developer",
            job_title="Developer",
            job_location="London",
            job_salary_min=None,
            job_salary_max=None,
            cv_embedding=[0.5] * 1536,
            cv_text="Python developer",
            target_roles=[],
            preferred_locations=[],
            exclude_keywords=[],
            salary_min=None,
            salary_target=None,
            score_weights={},  # Empty weights - should use defaults
        )
        assert 0 <= score <= 100
