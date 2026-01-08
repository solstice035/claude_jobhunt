# Enhanced Job Matching - Phase 1 Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform job matching from 4 dimensions to 5, expand skills coverage from 41 to 150+ with synonyms, and utilize unused profile fields (salary, exclusions).

**Architecture:** Multi-factor weighted scoring with hard exclusion filter, categorized skills taxonomy with synonym support, and graduated location matching for UK regions.

**Tech Stack:** Python, FastAPI, OpenAI embeddings, regex for skill extraction

---

## 1. Skills Taxonomy

### Structure

13 categories organized into Technical (6) and Professional (7):

**Technical Categories:**
- languages
- frontend
- backend
- cloud
- data
- ai_ml

**Professional Categories:**
- consulting
- project_management
- leadership
- strategy
- communication
- delivery
- commercial

### Skills Detail

#### Technical

| Category | Primary Skills |
|----------|---------------|
| languages | python, javascript, typescript, java, csharp, go, rust, ruby, php, scala, kotlin, swift |
| frontend | react, angular, vue, svelte, html, css |
| backend | django, flask, fastapi, spring, express, rails |
| cloud | aws, azure, gcp, docker, kubernetes, terraform |
| data | postgresql, mysql, mongodb, redis, elasticsearch, kafka, spark |
| ai_ml | machine learning, deep learning, nlp, computer vision, pytorch, tensorflow, scikit-learn, langchain |

#### Professional

| Category | Primary Skills |
|----------|---------------|
| consulting | client management, stakeholder engagement, business development, proposal writing, solution design, discovery workshops, requirements gathering, RFP response, pre-sales, account management |
| project_management | agile delivery, scrum master, product owner, programme management, waterfall, PRINCE2, PMP, budgeting, resource planning, risk management, milestone tracking, dependency management |
| leadership | team building, mentoring, coaching, performance management, hiring, talent development, culture, succession planning, org design, people management, line management |
| strategy | digital transformation, technology strategy, roadmapping, vendor management, build vs buy, M&A due diligence, P&L ownership, cost optimisation, business case, OKRs |
| communication | executive presentations, board reporting, technical writing, public speaking, stakeholder updates, change communication, workshop facilitation |
| delivery | continuous improvement, lean, six sigma, kaizen, process optimisation, operational excellence, KPIs, SLAs |
| commercial | contract negotiation, licensing, procurement, supplier management, commercial awareness, revenue, margins |

### Synonym Examples

```python
"python": ["py", "python3", "cpython"]
"react": ["reactjs", "react.js", "jsx", "next.js", "nextjs"]
"kubernetes": ["k8s", "helm", "kubectl"]
"client management": ["client-facing", "client engagement"]
"agile delivery": ["agile", "scrum", "sprint planning"]
"P&L ownership": ["P&L", "profit and loss", "commercial acumen"]
```

### Matching Algorithm

1. Check if primary skill matches (word boundary regex)
2. If not, check each synonym
3. Return matched skills grouped by category
4. Skills score = common skills / job's required skills

---

## 2. Salary Matching

### Scoring Logic

| Job Salary vs Profile | Score | Reason |
|-----------------------|-------|--------|
| Meets/exceeds target | 1.0 | "Salary: £120k meets target" |
| Above minimum | 0.5-1.0 | "Salary: £95k above minimum" |
| Below minimum | 0.0-0.5 | "Salary: £70k below minimum" |
| No salary data | 0.5 | (neutral - no reason shown) |

### Calculation

1. Job midpoint = (salary_min + salary_max) / 2
2. Compare against profile's salary_target and salary_min
3. Graduated scoring based on position in range
4. Below minimum: progressively penalised

### Edge Cases

- Job has no salary: neutral 0.5
- Profile has no preference: neutral 0.5
- Job only has min or max: use available value

---

## 3. Exclusion Keywords

### Behaviour

Hard filter - any match returns score 0.0 immediately.

### Algorithm

1. Combine job title + description
2. For each keyword in profile's exclude_keywords:
   - Apply word-boundary regex
3. If any keyword found: return (0.0, ["Excluded: contains 'keyword'"])
4. If no matches: proceed with normal scoring

### Examples

| Exclude Keyword | Would Filter Out | Would NOT Filter Out |
|-----------------|------------------|----------------------|
| "junior" | Junior Developer | Juniper Networks |
| "contract" | 6-month contract | contractor management |
| "PHP" | PHP Developer | |
| "recruitment" | Recruitment Manager | |

---

## 4. Graduated Location Matching

### Scoring Tiers

| Match Type | Score | Example |
|------------|-------|---------|
| Exact match | 1.0 | Prefers "London", job is "London" |
| Remote (preferred) | 1.0 | Prefers "Remote", job is "Fully Remote" |
| Hybrid (flexible) | 0.9 | Prefers "Remote", job is "Hybrid - London" |
| Same region | 0.8 | Prefers "Brighton", job is "Reading" |
| Remote (not preferred) | 0.8 | Prefers "London", job is "Remote" |
| Commutable | 0.6-0.7 | Prefers "London", job is "Cambridge" |
| No match | 0.0 | Prefers "London", job is "Edinburgh" |

### UK Regions

```python
UK_REGIONS = {
    "london": ["central london", "greater london", "city of london"],
    "south_east": ["brighton", "reading", "oxford", "cambridge", "milton keynes"],
    "south_west": ["bristol", "bath", "exeter", "plymouth"],
    "midlands": ["birmingham", "nottingham", "leicester", "coventry"],
    "north_west": ["manchester", "liverpool", "chester"],
    "yorkshire": ["leeds", "sheffield", "york", "hull"],
    "north_east": ["newcastle", "durham", "sunderland"],
    "scotland": ["edinburgh", "glasgow", "aberdeen"],
    "wales": ["cardiff", "swansea", "newport"],
}
```

### Remote/Hybrid Detection

Scans for: "remote", "hybrid", "work from home", "WFH"

---

## 5. Final Scoring Structure

### Dimensions and Weights

| Dimension | Weight | What it measures |
|-----------|--------|------------------|
| Semantic (CV) | 25% | Embedding similarity CV to Job |
| Skills | 25% | Taxonomy overlap with categories |
| Seniority | 20% | Job level vs target roles |
| Location | 15% | Graduated UK region matching |
| Salary | 15% | Compensation vs expectations |

### Processing Flow

```
Job Input
    |
    v
Check Exclusions ──── Match found ───> Return (0.0, "Excluded: ...")
    |
    | No match
    v
Calculate 5 dimension scores
    |
    v
Apply weights, sum to 0-100
    |
    v
Generate up to 5 match reasons
    |
    v
Return (score, reasons)
```

### Profile Customisation

Users can override default weights via score_weights field in Profile.

---

## 6. Error Handling

| Scenario | Behaviour |
|----------|-----------|
| Missing CV embedding | Return neutral score (50) with reason "No CV for matching" |
| Empty job description | Skills score defaults to 0.5 (neutral) |
| Invalid salary data | Treat as missing, score 0.5 |
| No profile preferences | All preference-based scores default to 0.5 |
| Skill extraction finds nothing | Skills score 0.5, no skills reason shown |

---

## 7. Testing Strategy

| Test Category | Coverage |
|---------------|----------|
| Skills extraction | Synonym matching, category grouping, word boundary protection |
| Salary matching | Above target, between min/target, below min, missing data |
| Exclusions | Single keyword, multiple keywords, partial word protection |
| Location | Exact match, regional match, remote detection, hybrid handling |
| Integration | Full score calculation, weight customisation |
| Edge cases | Empty inputs, missing profile, zero vectors |

Test file: `backend/tests/test_matcher.py`
Target coverage: >90% on matcher.py

---

## 8. Files to Modify

| File | Changes |
|------|---------|
| `backend/app/services/matcher.py` | New taxonomy, salary/exclusion/location functions, updated calculate_match_score |
| `backend/app/models/profile.py` | Add salary weight to score_weights default |
| `backend/app/scheduler.py` | Pass new parameters to calculate_match_score |
| `backend/tests/test_matcher.py` | Comprehensive test suite |
| `frontend/src/types/index.ts` | Update score_weights type |

---

## 9. Success Metrics

| Metric | Before | After |
|--------|--------|-------|
| Skills coverage | 41 keywords | 150+ primary, 400+ with synonyms |
| Match dimensions | 4 | 5 |
| Location granularity | Binary | 5-level graduated |
| Profile field usage | ~50% | ~90% |
