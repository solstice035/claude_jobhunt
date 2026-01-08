# Enhanced Job Matching Phase 1 - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform job matching from 4 dimensions to 5, expand skills coverage from 41 to 150+, and utilize unused profile fields (salary, exclusions).

**Architecture:** TDD approach - write failing tests first, implement minimal code to pass, refactor. Each function is independently testable without database or API calls.

**Tech Stack:** Python 3.12, pytest, FastAPI, regex

---

## Task 1: Set Up Test Infrastructure

**Files:**
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/test_matcher.py`
- Create: `backend/pytest.ini`

**Step 1: Create test directory and files**

```bash
mkdir -p backend/tests
touch backend/tests/__init__.py
```

**Step 2: Create pytest.ini**

```ini
# backend/pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
```

**Step 3: Create initial test file with imports**

```python
# backend/tests/test_matcher.py
"""
Tests for job matching service.

Run with: cd backend && pytest tests/test_matcher.py -v
"""
import pytest


class TestSkillsTaxonomy:
    """Tests for expanded skills taxonomy with synonyms."""
    pass


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
```

**Step 4: Verify pytest runs**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py -v`
Expected: 0 tests collected (empty test classes)

**Step 5: Commit**

```bash
git add backend/tests/ backend/pytest.ini
git commit -m ":white_check_mark: tests: add test infrastructure for matcher"
```

---

## Task 2: Implement Skills Taxonomy

**Files:**
- Modify: `backend/app/services/matcher.py`
- Modify: `backend/tests/test_matcher.py`

**Step 1: Write failing test for SKILLS_TAXONOMY structure**

Add to `backend/tests/test_matcher.py`:

```python
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestSkillsTaxonomy -v`
Expected: FAIL with "cannot import name 'SKILLS_TAXONOMY'"

**Step 3: Implement SKILLS_TAXONOMY in matcher.py**

Replace `TECH_SKILLS` list in `backend/app/services/matcher.py` with:

```python
# Expanded skills taxonomy with synonyms
# Structure: category -> skill -> [synonyms]
SKILLS_TAXONOMY = {
    # Technical categories
    "languages": {
        "python": ["py", "python3", "cpython"],
        "javascript": ["js", "ecmascript", "es6", "es2015"],
        "typescript": ["ts", "tsx"],
        "java": ["jvm", "j2ee", "jakarta"],
        "csharp": ["c#", ".net", "dotnet", "c sharp"],
        "go": ["golang"],
        "rust": ["rustlang"],
        "ruby": ["ror"],
        "php": ["laravel", "symfony"],
        "scala": ["play framework"],
        "kotlin": ["android kotlin"],
        "swift": ["ios swift", "swiftui"],
        "sql": ["structured query language"],
    },
    "frontend": {
        "react": ["reactjs", "react.js", "jsx", "next.js", "nextjs"],
        "angular": ["angularjs", "ng"],
        "vue": ["vuejs", "vue.js", "nuxt", "nuxtjs"],
        "svelte": ["sveltekit"],
        "html": ["html5"],
        "css": ["css3", "sass", "scss", "less", "tailwind", "tailwindcss", "bootstrap"],
    },
    "backend": {
        "django": ["drf", "django rest", "django rest framework"],
        "flask": ["flask-restful"],
        "fastapi": ["starlette"],
        "spring": ["spring boot", "spring mvc", "springboot"],
        "express": ["expressjs", "express.js", "node.js", "nodejs", "node"],
        "rails": ["ruby on rails", "ror"],
    },
    "cloud": {
        "aws": ["amazon web services", "ec2", "s3", "lambda", "eks", "ecs", "amazon"],
        "azure": ["microsoft azure", "azure devops"],
        "gcp": ["google cloud", "google cloud platform", "bigquery", "cloud run"],
        "docker": ["containerization", "dockerfile", "containers"],
        "kubernetes": ["k8s", "helm", "kubectl", "k8"],
        "terraform": ["iac", "infrastructure as code", "terragrunt"],
    },
    "data": {
        "postgresql": ["postgres", "psql"],
        "mysql": ["mariadb"],
        "mongodb": ["nosql", "document db", "mongo"],
        "redis": ["caching", "in-memory"],
        "elasticsearch": ["elastic", "opensearch", "elk"],
        "kafka": ["event streaming", "message queue", "apache kafka"],
        "spark": ["pyspark", "databricks", "apache spark"],
    },
    "ai_ml": {
        "machine learning": ["ml", "predictive modeling", "statistical modeling"],
        "deep learning": ["neural networks", "dl", "neural nets"],
        "nlp": ["natural language processing", "text analytics", "language models"],
        "computer vision": ["cv", "image recognition", "image processing"],
        "pytorch": ["torch"],
        "tensorflow": ["tf", "keras"],
        "scikit-learn": ["sklearn", "scikit"],
        "langchain": ["llm", "rag", "large language models"],
        "data science": ["data scientist", "analytics"],
    },
    # Professional categories
    "consulting": {
        "client management": ["client-facing", "client engagement", "client relations"],
        "stakeholder engagement": ["stakeholder management", "stakeholder relations"],
        "business development": ["bd", "new business", "sales"],
        "proposal writing": ["rfp", "bid writing", "tender"],
        "solution design": ["solutioning", "solution architecture"],
        "discovery workshops": ["discovery", "workshops"],
        "requirements gathering": ["requirements analysis", "business analysis"],
        "pre-sales": ["presales", "pre sales"],
        "account management": ["account manager", "client accounts"],
    },
    "project_management": {
        "agile delivery": ["agile", "scrum", "sprint planning", "sprints"],
        "scrum master": ["sm", "scrum"],
        "product owner": ["po", "product management"],
        "programme management": ["program management", "programme manager"],
        "waterfall": ["waterfall methodology"],
        "prince2": ["prince 2", "prince ii"],
        "pmp": ["project management professional"],
        "budgeting": ["budget management", "financial planning"],
        "resource planning": ["resource management", "capacity planning"],
        "risk management": ["risk assessment", "risk mitigation"],
        "milestone tracking": ["milestones", "project tracking"],
        "dependency management": ["dependencies"],
    },
    "leadership": {
        "team building": ["team development", "building teams"],
        "mentoring": ["mentorship", "coaching", "mentor"],
        "coaching": ["coach", "personal development"],
        "performance management": ["performance reviews", "appraisals", "pdp"],
        "hiring": ["recruitment", "talent acquisition", "interviewing"],
        "talent development": ["learning and development", "l&d"],
        "culture": ["team culture", "company culture"],
        "succession planning": ["succession"],
        "org design": ["organizational design", "organisation design"],
        "people management": ["line management", "direct reports", "managing people"],
    },
    "strategy": {
        "digital transformation": ["transformation", "digitalization", "digital strategy"],
        "technology strategy": ["tech strategy", "it strategy", "technical strategy"],
        "roadmapping": ["roadmap", "product roadmap", "technical roadmap"],
        "vendor management": ["supplier management", "third party management"],
        "build vs buy": ["make vs buy", "build or buy"],
        "m&a due diligence": ["m&a", "mergers and acquisitions", "due diligence"],
        "p&l ownership": ["p&l", "profit and loss", "pnl"],
        "cost optimisation": ["cost optimization", "cost reduction", "cost savings"],
        "business case": ["business cases", "roi analysis"],
        "okrs": ["objectives and key results", "okr"],
    },
    "communication": {
        "executive presentations": ["exec presentations", "c-suite presentations", "board presentations"],
        "board reporting": ["board reports", "board level"],
        "technical writing": ["documentation", "technical documentation"],
        "public speaking": ["presenting", "presentations", "speaker"],
        "stakeholder updates": ["status updates", "progress reporting"],
        "change communication": ["change management", "comms"],
        "workshop facilitation": ["facilitating", "facilitation"],
    },
    "delivery": {
        "continuous improvement": ["ci", "kaizen", "improvement"],
        "lean": ["lean methodology", "lean principles"],
        "six sigma": ["6 sigma", "sixsigma"],
        "process optimisation": ["process optimization", "process improvement"],
        "operational excellence": ["opex", "operations excellence"],
        "kpis": ["key performance indicators", "metrics"],
        "slas": ["service level agreements", "service levels"],
    },
    "commercial": {
        "contract negotiation": ["negotiation", "contract management"],
        "licensing": ["software licensing", "license management"],
        "procurement": ["purchasing", "sourcing"],
        "supplier management": ["vendor relations", "supplier relations"],
        "commercial awareness": ["business acumen", "commercial acumen"],
        "revenue": ["revenue growth", "revenue generation"],
        "margins": ["profit margins", "gross margin"],
    },
}

# Keep TECH_SKILLS for backwards compatibility (deprecated)
TECH_SKILLS = [
    skill for category in SKILLS_TAXONOMY.values()
    for skill in category.keys()
]
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestSkillsTaxonomy -v`
Expected: PASS

**Step 5: Commit**

```bash
git add backend/app/services/matcher.py backend/tests/test_matcher.py
git commit -m ":sparkles: feat(matcher): add expanded skills taxonomy with 13 categories and synonyms"
```

---

## Task 3: Implement extract_skills_with_synonyms Function

**Files:**
- Modify: `backend/app/services/matcher.py`
- Modify: `backend/tests/test_matcher.py`

**Step 1: Write failing tests for extract_skills_with_synonyms**

Add to `TestSkillsTaxonomy` class:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestSkillsTaxonomy::test_extract_skills_finds_primary_skill -v`
Expected: FAIL with "cannot import name 'extract_skills_with_synonyms'"

**Step 3: Implement extract_skills_with_synonyms**

Add to `backend/app/services/matcher.py` after SKILLS_TAXONOMY:

```python
from typing import Dict

def extract_skills_with_synonyms(text: str) -> Dict[str, List[str]]:
    """
    Extract skills from text using taxonomy with synonym matching.

    Args:
        text: Text to search for skills (job description or CV)

    Returns:
        Dict mapping category names to lists of found skills.
        e.g., {"languages": ["python", "javascript"], "cloud": ["aws"]}
    """
    if not text:
        return {}

    text_lower = text.lower()
    found_skills: Dict[str, List[str]] = {}

    for category, skills in SKILLS_TAXONOMY.items():
        category_matches = []
        for skill, synonyms in skills.items():
            # Check primary skill name
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                category_matches.append(skill)
                continue

            # Check synonyms
            for synonym in synonyms:
                pattern = r'\b' + re.escape(synonym) + r'\b'
                if re.search(pattern, text_lower):
                    category_matches.append(skill)
                    break  # Found via synonym, move to next skill

        if category_matches:
            found_skills[category] = category_matches

    return found_skills
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestSkillsTaxonomy -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/app/services/matcher.py backend/tests/test_matcher.py
git commit -m ":sparkles: feat(matcher): add extract_skills_with_synonyms function"
```

---

## Task 4: Implement Salary Matching

**Files:**
- Modify: `backend/app/services/matcher.py`
- Modify: `backend/tests/test_matcher.py`

**Step 1: Write failing tests for match_salary**

Add to `backend/tests/test_matcher.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestSalaryMatching -v`
Expected: FAIL with "cannot import name 'match_salary'"

**Step 3: Implement match_salary function**

Add to `backend/app/services/matcher.py`:

```python
from typing import Optional, Tuple

def match_salary(
    job_min: Optional[int],
    job_max: Optional[int],
    profile_min: Optional[int],
    profile_target: Optional[int]
) -> Tuple[float, Optional[str]]:
    """
    Calculate salary match score with graduated penalties.

    Args:
        job_min: Job's minimum salary (can be None)
        job_max: Job's maximum salary (can be None)
        profile_min: User's minimum acceptable salary
        profile_target: User's target salary

    Returns:
        Tuple of (score 0-1, reason string or None)
    """
    # No job salary data
    if not job_min and not job_max:
        return 0.5, None

    # No profile preference
    if not profile_min and not profile_target:
        return 0.5, None

    # Calculate job midpoint
    if job_min and job_max:
        job_mid = (job_min + job_max) / 2
    else:
        job_mid = job_min or job_max

    # Determine target (use min if no target set)
    target = profile_target or profile_min

    # Meets or exceeds target
    if job_mid >= target:
        return 1.0, f"Salary: £{job_mid:,.0f} meets target"

    # Between minimum and target
    if profile_min and job_mid >= profile_min:
        if profile_target and profile_target > profile_min:
            range_size = profile_target - profile_min
            position = job_mid - profile_min
            score = 0.5 + (position / range_size) * 0.5
        else:
            score = 1.0  # At or above minimum with no target = good
        return score, f"Salary: £{job_mid:,.0f} above minimum"

    # Below minimum - graduated penalty
    if profile_min:
        shortfall_pct = (profile_min - job_mid) / profile_min
        score = max(0.0, 0.5 - shortfall_pct)
        return score, f"Salary: £{job_mid:,.0f} below minimum"

    return 0.5, None
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestSalaryMatching -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/app/services/matcher.py backend/tests/test_matcher.py
git commit -m ":sparkles: feat(matcher): add salary matching with graduated scoring"
```

---

## Task 5: Implement Exclusion Keywords

**Files:**
- Modify: `backend/app/services/matcher.py`
- Modify: `backend/tests/test_matcher.py`

**Step 1: Write failing tests for check_exclusions**

Add to `backend/tests/test_matcher.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestExclusionKeywords -v`
Expected: FAIL with "cannot import name 'check_exclusions'"

**Step 3: Implement check_exclusions function**

Add to `backend/app/services/matcher.py`:

```python
def check_exclusions(
    job_title: str,
    job_description: str,
    exclude_keywords: List[str]
) -> Tuple[bool, Optional[str]]:
    """
    Check if job contains any exclusion keywords.

    This is a hard filter - any match should result in score 0.

    Args:
        job_title: Job title to check
        job_description: Job description to check
        exclude_keywords: List of keywords to exclude

    Returns:
        Tuple of (should_exclude: bool, matched_keyword: str or None)
    """
    if not exclude_keywords:
        return False, None

    combined_text = f"{job_title} {job_description}".lower()

    for keyword in exclude_keywords:
        keyword_clean = keyword.lower().strip()
        if not keyword_clean:
            continue

        # Use word boundaries to avoid partial matches
        pattern = r'\b' + re.escape(keyword_clean) + r'\b'
        if re.search(pattern, combined_text):
            return True, keyword

    return False, None
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestExclusionKeywords -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/app/services/matcher.py backend/tests/test_matcher.py
git commit -m ":sparkles: feat(matcher): add exclusion keyword filtering"
```

---

## Task 6: Implement Graduated Location Matching

**Files:**
- Modify: `backend/app/services/matcher.py`
- Modify: `backend/tests/test_matcher.py`

**Step 1: Write failing tests for match_location_graduated**

Add to `backend/tests/test_matcher.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestGraduatedLocation -v`
Expected: FAIL with "cannot import name 'match_location_graduated'"

**Step 3: Implement UK_REGIONS and match_location_graduated**

Add to `backend/app/services/matcher.py`:

```python
# UK region hierarchy for graduated location matching
UK_REGIONS = {
    "london": ["central london", "greater london", "city of london", "east london", "west london", "north london", "south london"],
    "south_east": ["brighton", "reading", "oxford", "cambridge", "milton keynes", "southampton", "portsmouth", "guildford", "crawley"],
    "south_west": ["bristol", "bath", "exeter", "plymouth", "bournemouth", "swindon", "gloucester", "cheltenham"],
    "midlands": ["birmingham", "nottingham", "leicester", "coventry", "derby", "wolverhampton", "stoke"],
    "north_west": ["manchester", "liverpool", "chester", "warrington", "preston", "blackpool", "bolton"],
    "yorkshire": ["leeds", "sheffield", "york", "hull", "bradford", "harrogate", "doncaster"],
    "north_east": ["newcastle", "durham", "sunderland", "middlesbrough", "gateshead"],
    "scotland": ["edinburgh", "glasgow", "aberdeen", "dundee", "inverness", "stirling"],
    "wales": ["cardiff", "swansea", "newport", "bangor", "wrexham"],
    "northern_ireland": ["belfast", "derry", "lisburn"],
}


def _get_region(location: str) -> Optional[str]:
    """Get the UK region for a location string."""
    location_lower = location.lower()

    for region, cities in UK_REGIONS.items():
        # Check if region name is in location
        if region.replace("_", " ") in location_lower:
            return region
        # Check if any city in region matches
        for city in cities:
            if city in location_lower:
                return region

    return None


def match_location_graduated(
    job_location: str,
    preferred_locations: List[str]
) -> Tuple[float, Optional[str]]:
    """
    Calculate graduated location match with UK regional awareness.

    Scoring tiers:
        1.0 - Exact match or remote when preferred
        0.9 - Hybrid when remote/hybrid preferred
        0.8 - Same region or remote when not preferred
        0.6-0.7 - Commutable (adjacent regions)
        0.0 - No match

    Args:
        job_location: Job's location string
        preferred_locations: User's preferred locations

    Returns:
        Tuple of (score 0-1, reason string or None)
    """
    if not preferred_locations:
        return 1.0, None

    job_loc_lower = job_location.lower()

    # Check for remote
    is_remote = any(x in job_loc_lower for x in ["remote", "work from home", "wfh"])
    is_hybrid = "hybrid" in job_loc_lower

    # Check if user prefers remote
    prefers_remote = any("remote" in p.lower() for p in preferred_locations)
    prefers_hybrid = any("hybrid" in p.lower() for p in preferred_locations)

    if is_remote:
        if prefers_remote:
            return 1.0, "Remote work available"
        return 0.8, "Remote (flexible)"

    if is_hybrid:
        if prefers_remote or prefers_hybrid:
            return 0.9, "Hybrid work available"

    # Check exact location matches
    for pref in preferred_locations:
        pref_lower = pref.lower()
        if pref_lower in job_loc_lower or job_loc_lower in pref_lower:
            return 1.0, f"Location: {job_location}"

    # Check regional matches
    job_region = _get_region(job_location)

    if job_region:
        for pref in preferred_locations:
            pref_region = _get_region(pref)
            if pref_region and pref_region == job_region:
                return 0.8, f"Same region: {job_region.replace('_', ' ').title()}"

    return 0.0, None
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestGraduatedLocation -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/app/services/matcher.py backend/tests/test_matcher.py
git commit -m ":sparkles: feat(matcher): add graduated UK location matching"
```

---

## Task 7: Update calculate_match_score Function

**Files:**
- Modify: `backend/app/services/matcher.py`
- Modify: `backend/tests/test_matcher.py`

**Step 1: Write failing integration tests**

Add to `backend/tests/test_matcher.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestCalculateMatchScore -v`
Expected: FAIL (signature mismatch - new params not in current function)

**Step 3: Update calculate_match_score function signature and implementation**

Replace the entire `calculate_match_score` function in `backend/app/services/matcher.py`:

```python
def calculate_match_score(
    job_embedding: List[float],
    job_description: str,
    job_title: str,
    job_location: str,
    job_salary_min: Optional[int],
    job_salary_max: Optional[int],
    cv_embedding: List[float],
    cv_text: str,
    target_roles: List[str],
    preferred_locations: List[str],
    exclude_keywords: List[str],
    salary_min: Optional[int],
    salary_target: Optional[int],
    score_weights: dict,
) -> Tuple[float, List[str]]:
    """
    Calculate composite match score with 5 dimensions and exclusion filtering.

    Algorithm:
        0. Check exclusions (hard filter - returns 0 if match found)
        1. Semantic: cosine_similarity(cv_embedding, job_embedding)
        2. Skills: |cv_skills ∩ job_skills| / |job_skills| (using taxonomy)
        3. Seniority: 1.0 (exact), 0.5 (adjacent level), 0.0 (mismatch)
        4. Location: Graduated UK regional matching (0.0-1.0)
        5. Salary: Graduated scoring based on target/minimum

    Args:
        job_embedding: 1536-dim vector from OpenAI
        job_description: Full job posting text
        job_title: Job title for seniority detection
        job_location: Job location string
        job_salary_min: Job's minimum salary (can be None)
        job_salary_max: Job's maximum salary (can be None)
        cv_embedding: 1536-dim vector of candidate's CV
        cv_text: Full CV text for skill extraction
        target_roles: List of desired job titles
        preferred_locations: List of preferred locations
        exclude_keywords: Keywords that should disqualify jobs
        salary_min: User's minimum acceptable salary
        salary_target: User's target salary
        score_weights: Dict with keys: semantic, skills, seniority, location, salary

    Returns:
        Tuple of (score 0-100, list of up to 5 match reasons)
    """
    reasons = []

    # 0. Check exclusions first (hard filter)
    should_exclude, excluded_keyword = check_exclusions(
        job_title, job_description, exclude_keywords
    )
    if should_exclude:
        return 0.0, [f"Excluded: contains '{excluded_keyword}'"]

    # 1. Semantic similarity (embedding comparison)
    semantic_score = cosine_similarity(cv_embedding, job_embedding)
    semantic_score = max(0, min(1, semantic_score))  # Clamp to 0-1

    # 2. Skills match (using new taxonomy with synonyms)
    cv_skills = extract_skills_with_synonyms(cv_text)
    job_skills = extract_skills_with_synonyms(job_description)

    total_job_skills = sum(len(skills) for skills in job_skills.values())
    if cv_skills and job_skills:
        common_count = 0
        common_skills_list = []
        for category, job_cat_skills in job_skills.items():
            if category in cv_skills:
                overlap = set(cv_skills[category]) & set(job_cat_skills)
                common_count += len(overlap)
                common_skills_list.extend(list(overlap)[:2])

        skills_score = min(1.0, common_count / max(total_job_skills, 1))
        if common_skills_list:
            reasons.append(f"Skills: {', '.join(common_skills_list[:3])}")
    else:
        skills_score = 0.5  # Neutral if no skills detected

    # 3. Seniority match
    seniority_score = match_seniority(job_title, target_roles)
    if seniority_score == 1.0:
        reasons.append(f"Seniority: {extract_seniority(job_title).title()} level")

    # 4. Location match (graduated)
    location_score, location_reason = match_location_graduated(
        job_location, preferred_locations
    )
    if location_reason:
        reasons.append(location_reason)

    # 5. Salary match
    salary_score, salary_reason = match_salary(
        job_salary_min, job_salary_max, salary_min, salary_target
    )
    if salary_reason:
        reasons.append(salary_reason)

    # Calculate weighted composite score
    weights = score_weights
    composite = (
        semantic_score * weights.get("semantic", 0.25) +
        skills_score * weights.get("skills", 0.25) +
        seniority_score * weights.get("seniority", 0.20) +
        location_score * weights.get("location", 0.15) +
        salary_score * weights.get("salary", 0.15)
    )

    # Convert to 0-100 scale
    final_score = round(composite * 100, 1)

    # Add semantic match reason if high
    if semantic_score > 0.7:
        reasons.insert(0, "Strong CV match")
    elif semantic_score > 0.5:
        reasons.insert(0, "Good CV match")

    return final_score, reasons[:5]  # Limit to 5 reasons
```

**Step 4: Run tests to verify they pass**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py::TestCalculateMatchScore -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add backend/app/services/matcher.py backend/tests/test_matcher.py
git commit -m ":sparkles: feat(matcher): update calculate_match_score with 5 dimensions"
```

---

## Task 8: Update Scheduler to Pass New Parameters

**Files:**
- Modify: `backend/app/scheduler.py`

**Step 1: Update the calculate_match_score call in scheduler.py**

Find the call to `calculate_match_score` (around line 147) and update it:

```python
# Old call (lines 147-157):
score, reasons = calculate_match_score(
    job_embedding=embedding,
    job_description=job.description,
    job_title=job.title,
    job_location=job.location,
    cv_embedding=cv_embedding,
    cv_text=profile.cv_text,
    target_roles=profile.target_roles or [],
    preferred_locations=profile.locations or [],
    score_weights=profile.score_weights or {},
)

# New call with additional parameters:
score, reasons = calculate_match_score(
    job_embedding=embedding,
    job_description=job.description,
    job_title=job.title,
    job_location=job.location,
    job_salary_min=job.salary_min,
    job_salary_max=job.salary_max,
    cv_embedding=cv_embedding,
    cv_text=profile.cv_text,
    target_roles=profile.target_roles or [],
    preferred_locations=profile.locations or [],
    exclude_keywords=profile.exclude_keywords or [],
    salary_min=profile.salary_min,
    salary_target=profile.salary_target,
    score_weights=profile.score_weights or {},
)
```

**Step 2: Verify the scheduler module still imports correctly**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -c "from app.scheduler import fetch_and_process_jobs; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add backend/app/scheduler.py
git commit -m ":zap: refactor(scheduler): pass salary and exclusion params to matcher"
```

---

## Task 9: Update Profile Model Default Weights

**Files:**
- Modify: `backend/app/models/profile.py`

**Step 1: Update default score_weights to include salary**

In `backend/app/models/profile.py`, update the `score_weights` column default:

```python
# Old default (lines 47-55):
score_weights = Column(
    JSON,
    nullable=False,
    default=lambda: {
        "semantic": 0.30,
        "skills": 0.30,
        "seniority": 0.25,
        "location": 0.15,
    },
)

# New default with salary:
score_weights = Column(
    JSON,
    nullable=False,
    default=lambda: {
        "semantic": 0.25,
        "skills": 0.25,
        "seniority": 0.20,
        "location": 0.15,
        "salary": 0.15,
    },
)
```

**Step 2: Verify imports still work**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -c "from app.models.profile import Profile; print('OK')"`
Expected: OK

**Step 3: Commit**

```bash
git add backend/app/models/profile.py
git commit -m ":wrench: config(profile): update default score_weights to include salary (5 dimensions)"
```

---

## Task 10: Update Frontend Types

**Files:**
- Modify: `frontend/src/types/index.ts`

**Step 1: Update score_weights type to include salary**

In `frontend/src/types/index.ts`, update the `score_weights` interface in Profile:

```typescript
// Old (lines 72-77):
score_weights: {
  semantic: number; // CV-job embedding similarity
  skills: number; // Keyword overlap
  seniority: number; // Job level alignment
  location: number; // Geographic match
};

// New with salary:
/** Match score component weights (must sum to 1.0) */
score_weights: {
  semantic: number; // CV-job embedding similarity (default 0.25)
  skills: number; // Keyword overlap (default 0.25)
  seniority: number; // Job level alignment (default 0.20)
  location: number; // Geographic match (default 0.15)
  salary: number; // Salary match (default 0.15)
};
```

**Step 2: Verify TypeScript compiles**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/frontend && npm run build`
Expected: Build succeeds (or typecheck passes)

**Step 3: Commit**

```bash
git add frontend/src/types/index.ts
git commit -m ":label: types(frontend): add salary to score_weights type"
```

---

## Task 11: Run Full Test Suite

**Step 1: Run all matcher tests**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -m pytest tests/test_matcher.py -v`
Expected: All tests PASS

**Step 2: Run backend import verification**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/backend && python -c "from app.main import app; print('Backend OK')"`
Expected: Backend OK

**Step 3: Run frontend build**

Run: `cd /Users/nicksolly/Dev/jobRepos/claude_jobhunt/frontend && npm run build`
Expected: Build succeeds

**Step 4: Final commit with test results**

```bash
git add -A
git commit -m ":white_check_mark: tests: verify all Phase 1 enhancements pass"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Test infrastructure | Setup |
| 2 | SKILLS_TAXONOMY | 2 tests |
| 3 | extract_skills_with_synonyms | 6 tests |
| 4 | match_salary | 7 tests |
| 5 | check_exclusions | 7 tests |
| 6 | match_location_graduated | 8 tests |
| 7 | calculate_match_score update | 4 tests |
| 8 | scheduler.py update | Import check |
| 9 | Profile model update | Import check |
| 10 | Frontend types update | Build check |
| 11 | Full test suite | All tests |

**Total: 34 tests covering all new functionality**

After completion, use `superpowers:finishing-a-development-branch` to merge or create PR.
