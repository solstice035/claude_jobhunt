"""
Job Matching Service - AI-Powered Job Relevance Scoring

This module implements a multi-factor matching algorithm that calculates
how well a job posting matches a candidate's profile.

Match Score Composition (default weights):
    - Semantic Similarity (30%): OpenAI embedding cosine similarity
    - Skills Match (30%): Keyword extraction and overlap analysis
    - Seniority Match (25%): Job level alignment (junior → executive)
    - Location Match (15%): Geographic preference matching

Score Range: 0-100 where higher = better match

Complexity Analysis:
    - extract_skills_from_text: O(n*m) where n=text length, m=skills count
    - calculate_match_score: O(n) where n=text length (dominated by skill extraction)
"""

import re
from typing import Dict, List, Optional, Tuple
from app.services.embeddings import cosine_similarity

# Seniority keywords for level detection (ordered: most senior → least)
SENIORITY_LEVELS = {
    "executive": ["ceo", "cto", "cfo", "cio", "chief", "president", "vp", "vice president"],
    "director": ["director", "head of", "vp of"],
    "senior": ["senior", "lead", "principal", "staff", "architect"],
    "mid": ["manager", "specialist", "analyst", "engineer", "developer", "consultant"],
    "junior": ["junior", "associate", "assistant", "trainee", "graduate", "entry", "intern"],
}

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


def extract_seniority(title: str) -> str:
    """Extract seniority level from job title"""
    title_lower = title.lower()

    for level, keywords in SENIORITY_LEVELS.items():
        for keyword in keywords:
            if keyword in title_lower:
                return level

    return "mid"  # Default to mid-level


def extract_skills_from_text(text: str) -> List[str]:
    """Extract tech skills from text"""
    text_lower = text.lower()
    found_skills = []

    for skill in TECH_SKILLS:
        # Use word boundaries for accurate matching
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill)

    return found_skills


def match_location(job_location: str, preferred_locations: List[str]) -> float:
    """Calculate location match score (0-1)"""
    if not preferred_locations:
        return 1.0  # No preference means all locations match

    job_loc_lower = job_location.lower()

    for pref in preferred_locations:
        pref_lower = pref.lower()
        if pref_lower in job_loc_lower or job_loc_lower in pref_lower:
            return 1.0
        if "remote" in pref_lower and "remote" in job_loc_lower:
            return 1.0

    return 0.0


def match_seniority(job_title: str, target_roles: List[str]) -> float:
    """Calculate seniority match score (0-1)"""
    if not target_roles:
        return 0.5  # Neutral if no preference

    job_seniority = extract_seniority(job_title)

    # Extract target seniority levels
    target_levels = set()
    for role in target_roles:
        target_levels.add(extract_seniority(role))

    if job_seniority in target_levels:
        return 1.0

    # Partial match for adjacent levels
    level_order = ["junior", "mid", "senior", "director", "executive"]
    job_idx = level_order.index(job_seniority) if job_seniority in level_order else 2

    for target in target_levels:
        if target in level_order:
            target_idx = level_order.index(target)
            diff = abs(job_idx - target_idx)
            if diff == 1:
                return 0.5

    return 0.0


def calculate_match_score(
    job_embedding: List[float],
    job_description: str,
    job_title: str,
    job_location: str,
    cv_embedding: List[float],
    cv_text: str,
    target_roles: List[str],
    preferred_locations: List[str],
    score_weights: dict,
) -> Tuple[float, List[str]]:
    """
    Calculate composite match score and generate human-readable match reasons.

    Algorithm:
        1. Semantic: cosine_similarity(cv_embedding, job_embedding)
        2. Skills: |cv_skills ∩ job_skills| / |job_skills|
        3. Seniority: 1.0 (exact), 0.5 (adjacent level), 0.0 (mismatch)
        4. Location: 1.0 (match), 0.0 (no match)

    Args:
        job_embedding: 1536-dim vector from OpenAI text-embedding-3-small
        job_description: Full job posting text for skill extraction
        job_title: Job title for seniority detection
        job_location: Job location string (e.g., "London, UK")
        cv_embedding: 1536-dim vector of candidate's CV
        cv_text: Full CV text for skill extraction
        target_roles: List of desired job titles (e.g., ["CTO", "VP Engineering"])
        preferred_locations: List of preferred locations (e.g., ["London", "Remote"])
        score_weights: Dict with keys: semantic, skills, seniority, location

    Returns:
        Tuple of (score 0-100, list of up to 5 match reasons)

    Example:
        >>> score, reasons = calculate_match_score(...)
        >>> print(score)  # 75.5
        >>> print(reasons)  # ["Good CV match", "Skills: python, react", "Location: London"]
    """
    reasons = []

    # 1. Semantic similarity (embedding comparison)
    semantic_score = cosine_similarity(cv_embedding, job_embedding)
    semantic_score = max(0, min(1, semantic_score))  # Clamp to 0-1

    # 2. Skills match
    cv_skills = set(extract_skills_from_text(cv_text))
    job_skills = set(extract_skills_from_text(job_description))

    if cv_skills and job_skills:
        common_skills = cv_skills & job_skills
        skills_score = len(common_skills) / max(len(job_skills), 1)
        skills_score = min(1.0, skills_score)  # Cap at 1.0

        if common_skills:
            top_skills = list(common_skills)[:3]
            reasons.append(f"Skills: {', '.join(top_skills)}")
    else:
        skills_score = 0.5  # Neutral if no skills detected

    # 3. Seniority match
    seniority_score = match_seniority(job_title, target_roles)
    if seniority_score == 1.0:
        reasons.append(f"Seniority: {extract_seniority(job_title).title()} level match")

    # 4. Location match
    location_score = match_location(job_location, preferred_locations)
    if location_score == 1.0 and preferred_locations:
        reasons.append(f"Location: {job_location}")

    # Calculate weighted composite score
    weights = score_weights
    composite = (
        semantic_score * weights.get("semantic", 0.30) +
        skills_score * weights.get("skills", 0.30) +
        seniority_score * weights.get("seniority", 0.25) +
        location_score * weights.get("location", 0.15)
    )

    # Convert to 0-100 scale
    final_score = round(composite * 100, 1)

    # Add semantic match reason if high
    if semantic_score > 0.7:
        reasons.insert(0, "Strong CV match")
    elif semantic_score > 0.5:
        reasons.insert(0, "Good CV match")

    return final_score, reasons[:5]  # Limit to 5 reasons
