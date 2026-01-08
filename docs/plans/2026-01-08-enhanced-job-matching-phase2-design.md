# Enhanced Job Matching - Phase 2 Design: Skills Intelligence

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace static skills taxonomy with intelligent skill extraction using ESCO ontology and LLM-powered analysis, enabling skill gap identification and learning recommendations.

**Architecture:** ESCO skills database integration, Claude/GPT-based skill extraction API, skill relationship graph for inference.

**Tech Stack:** Python, FastAPI, ESCO API, OpenAI/Anthropic API, NetworkX for skill graphs

**Prerequisites:** Phase 1 complete (expanded taxonomy provides fallback)

---

## 1. ESCO Integration

### What is ESCO?

European Skills, Competences, Qualifications and Occupations - a multilingual classification providing:
- 13,890 skills with definitions
- 3,008 occupations
- Hierarchical relationships between skills
- Skill-to-occupation mappings

### Data Structure

```python
@dataclass
class ESCOSkill:
    uri: str                    # Unique identifier
    preferred_label: str        # Primary name
    alt_labels: List[str]       # Synonyms (avg 5-10 per skill)
    description: str            # Definition
    skill_type: str             # "skill" or "knowledge"
    broader_skills: List[str]   # Parent skills
    narrower_skills: List[str]  # Child skills
    related_skills: List[str]   # Horizontal relationships
```

### Integration Approach

| Option | Description | Trade-offs |
|--------|-------------|------------|
| **A. Full local copy** | Download ESCO CSV/JSON, store in PostgreSQL | Fast queries, offline capable, 50MB storage |
| **B. API integration** | Query ESCO REST API on demand | Always current, network dependency |
| **C. Hybrid** | Local cache with periodic sync | Best of both, more complexity |

**Recommendation:** Option A (Full local copy) - performance critical for matching.

### Database Schema

```sql
CREATE TABLE esco_skills (
    uri VARCHAR(255) PRIMARY KEY,
    preferred_label VARCHAR(255) NOT NULL,
    alt_labels JSONB,
    description TEXT,
    skill_type VARCHAR(50),
    broader_skills JSONB,
    narrower_skills JSONB,
    related_skills JSONB,
    embedding VECTOR(1536)  -- For semantic search within skills
);

CREATE INDEX idx_esco_label ON esco_skills USING gin(preferred_label gin_trgm_ops);
CREATE INDEX idx_esco_alt ON esco_skills USING gin(alt_labels);
```

---

## 2. LLM Skill Extraction

### Why LLM over Regex?

| Approach | Precision | Recall | Handles Context |
|----------|-----------|--------|-----------------|
| Regex + taxonomy | ~75% | ~60% | No |
| NER models (BERT) | ~92% | ~85% | Partially |
| LLM extraction | ~95% | ~90% | Yes |

LLMs understand context: "Python environment" (not the language) vs "Python developer" (the language).

### Extraction Prompt

```python
SKILL_EXTRACTION_PROMPT = """
Extract all skills, technologies, and competencies from this job description.

For each skill found, provide:
1. The exact skill name (normalized)
2. The category (technical, soft, domain, tool)
3. Whether it's required or preferred
4. Confidence (high, medium, low)

Job Description:
{job_description}

Return as JSON:
{
  "skills": [
    {"name": "Python", "category": "technical", "required": true, "confidence": "high"},
    {"name": "stakeholder management", "category": "soft", "required": true, "confidence": "high"},
    ...
  ]
}
"""
```

### Caching Strategy

```python
# Cache extracted skills per job (description hash)
# TTL: indefinite (job descriptions don't change)
cache_key = f"skills:{hash(job_description)}"

async def extract_skills(job_description: str) -> List[Skill]:
    cached = await redis.get(cache_key)
    if cached:
        return json.loads(cached)

    skills = await llm_extract(job_description)
    await redis.set(cache_key, json.dumps(skills))
    return skills
```

### Cost Management

| Model | Cost per 1K jobs | Quality |
|-------|------------------|---------|
| GPT-4o | ~$15 | Excellent |
| GPT-4o-mini | ~$1.50 | Very good |
| Claude Haiku | ~$0.75 | Very good |

**Recommendation:** GPT-4o-mini or Claude Haiku for extraction (quality sufficient, 10x cheaper).

---

## 3. Skill Relationship Graph

### Purpose

Enable skill inference:
- "Kubernetes" implies "Docker" knowledge (narrower → broader)
- "React" related to "JavaScript" (related skills)
- Missing "AWS" but has "GCP" = cloud skills present

### Graph Structure

```python
import networkx as nx

skill_graph = nx.DiGraph()

# Add nodes
skill_graph.add_node("kubernetes", type="technical")
skill_graph.add_node("docker", type="technical")
skill_graph.add_node("containerization", type="technical")

# Add edges with relationship types
skill_graph.add_edge("kubernetes", "docker", relation="requires")
skill_graph.add_edge("kubernetes", "containerization", relation="broader")
skill_graph.add_edge("docker", "containerization", relation="broader")
```

### Inference Rules

```python
def infer_skills(explicit_skills: Set[str]) -> Set[str]:
    """Expand skill set with inferred skills."""
    inferred = set()

    for skill in explicit_skills:
        # Add broader skills (if you know K8s, you know containers)
        broader = skill_graph.successors(skill, relation="broader")
        inferred.update(broader)

        # Add required skills (if you know K8s, you know Docker)
        required = skill_graph.successors(skill, relation="requires")
        inferred.update(required)

    return explicit_skills | inferred
```

### Scoring with Inference

```python
def calculate_skills_score(cv_skills: Set, job_skills: Set) -> float:
    # Expand CV skills with inferences
    cv_expanded = infer_skills(cv_skills)

    # Direct matches
    direct_matches = cv_skills & job_skills

    # Inferred matches (weighted less)
    inferred_matches = (cv_expanded - cv_skills) & job_skills

    score = (
        len(direct_matches) * 1.0 +
        len(inferred_matches) * 0.5
    ) / len(job_skills)

    return min(1.0, score)
```

---

## 4. Skill Gap Analysis

### Purpose

Show users what skills they're missing for their target roles, enabling:
- Better job targeting
- Learning recommendations
- CV improvement suggestions

### Data Model

```python
@dataclass
class SkillGap:
    skill: str
    frequency: float      # How often required in target jobs (0-1)
    importance: str       # "critical", "important", "nice-to-have"
    category: str         # "technical", "soft", "domain"
    learning_resources: List[str]  # Optional links
```

### Gap Calculation

```python
async def calculate_skill_gaps(
    profile: Profile,
    target_jobs: List[Job]
) -> List[SkillGap]:
    # Extract CV skills
    cv_skills = await extract_skills(profile.cv_text)
    cv_skill_names = {s.name.lower() for s in cv_skills}

    # Aggregate skills from target jobs
    job_skill_counts = Counter()
    for job in target_jobs:
        job_skills = await extract_skills(job.description)
        for skill in job_skills:
            if skill.required:
                job_skill_counts[skill.name.lower()] += 2
            else:
                job_skill_counts[skill.name.lower()] += 1

    # Find gaps
    gaps = []
    for skill, count in job_skill_counts.most_common():
        if skill not in cv_skill_names:
            frequency = count / len(target_jobs)
            gaps.append(SkillGap(
                skill=skill,
                frequency=frequency,
                importance="critical" if frequency > 0.7 else "important" if frequency > 0.4 else "nice-to-have",
                category=categorize_skill(skill),
                learning_resources=[]
            ))

    return gaps[:20]  # Top 20 gaps
```

### API Endpoint

```python
@router.get("/profile/skill-gaps")
async def get_skill_gaps(
    db: AsyncSession = Depends(get_db),
    profile: Profile = Depends(get_current_profile)
) -> List[SkillGap]:
    # Get recent jobs matching user's criteria
    target_jobs = await get_jobs_for_profile(db, profile, limit=50)
    return await calculate_skill_gaps(profile, target_jobs)
```

---

## 5. Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/app/services/esco.py` | ESCO database queries, skill lookup |
| `backend/app/services/skill_extractor.py` | LLM-based skill extraction |
| `backend/app/services/skill_graph.py` | Skill relationship graph and inference |
| `backend/app/services/skill_gaps.py` | Gap analysis logic |
| `backend/app/api/skills.py` | API endpoints for skill-related features |
| `backend/app/models/esco.py` | ESCO skill SQLAlchemy model |
| `backend/scripts/import_esco.py` | One-time ESCO data import script |

---

## 6. Success Metrics

| Metric | Target |
|--------|--------|
| Skill extraction accuracy | >90% (sample validation) |
| Skills matched per job | 15-30 (vs current 5-10) |
| Inference coverage | 20% additional matches via graph |
| Gap analysis usefulness | User feedback survey |

---

## 7. Dependencies

```
Phase 1 (taxonomy) ──► Phase 2 (intelligence)
                              │
                              ├── ESCO data import
                              ├── LLM extraction service
                              ├── Skill graph construction
                              └── Gap analysis API
```

Phase 1's taxonomy serves as fallback when LLM extraction is unavailable or for cost optimization on bulk processing.
