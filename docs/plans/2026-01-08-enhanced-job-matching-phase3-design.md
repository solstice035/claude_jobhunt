# Enhanced Job Matching - Phase 3 Design: Advanced ML

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Achieve state-of-the-art matching accuracy (85-92%) through hybrid search, cross-encoder re-ranking, and optimized embeddings.

**Architecture:** Two-stage retrieval (embedding recall → re-ranker precision), hybrid semantic + keyword search, optional embedding model swap.

**Tech Stack:** Python, FastAPI, sentence-transformers, OpenAI/Cohere re-rank API, BM25

**Prerequisites:** Phase 1 complete, Phase 2 recommended

---

## 1. Hybrid Search Architecture

### Why Hybrid?

| Approach | Strengths | Weaknesses |
|----------|-----------|------------|
| Semantic only | Understands meaning, handles synonyms | Misses exact keywords, acronyms |
| Keyword only (BM25) | Exact matches, fast | No semantic understanding |
| **Hybrid** | Best of both | More complexity |

Research shows hybrid achieves 5-10% better recall than either alone.

### Architecture

```
Query (CV + preferences)
         │
         ├──────────────────┬────────────────────┐
         ▼                  ▼                    ▼
    Semantic Search    Keyword Search     Skill Filter
    (embeddings)       (BM25/TF-IDF)      (taxonomy)
         │                  │                    │
         └──────────────────┴────────────────────┘
                           │
                           ▼
                  Score Fusion (RRF)
                           │
                           ▼
                   Top-K Candidates
                           │
                           ▼
                  Cross-Encoder Re-rank
                           │
                           ▼
                   Final Ranked List
```

### Reciprocal Rank Fusion (RRF)

Combines rankings from multiple sources:

```python
def reciprocal_rank_fusion(
    rankings: List[List[str]],  # List of job ID rankings
    k: int = 60
) -> List[Tuple[str, float]]:
    """
    Combine multiple rankings using RRF.

    Formula: score(d) = Σ 1 / (k + rank_i(d))
    """
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, job_id in enumerate(ranking, start=1):
            scores[job_id] += 1.0 / (k + rank)

    # Sort by fused score
    return sorted(scores.items(), key=lambda x: -x[1])
```

### BM25 Implementation

```python
from rank_bm25 import BM25Okapi
import nltk

class JobBM25Index:
    def __init__(self, jobs: List[Job]):
        # Tokenize job descriptions
        self.job_ids = [job.id for job in jobs]
        tokenized = [
            nltk.word_tokenize(job.description.lower())
            for job in jobs
        ]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 100) -> List[str]:
        tokens = nltk.word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [self.job_ids[i] for i in top_indices]
```

---

## 2. Cross-Encoder Re-ranking

### Why Re-rank?

| Stage | Model Type | Speed | Accuracy |
|-------|-----------|-------|----------|
| Retrieval | Bi-encoder (embeddings) | Fast (ms) | Good (~87%) |
| Re-ranking | Cross-encoder | Slow (100ms) | Excellent (~92%) |

Cross-encoders see both texts together, enabling deeper comparison.

### Implementation Options

| Option | Cost | Quality | Latency |
|--------|------|---------|---------|
| **Cohere Rerank API** | $1/1K queries | Excellent | ~200ms |
| **OpenAI with prompt** | $0.50/1K | Very good | ~500ms |
| **Local cross-encoder** | Free (GPU needed) | Good | ~100ms |

**Recommendation:** Cohere Rerank for quality, local model for cost.

### Cohere Integration

```python
import cohere

co = cohere.Client(api_key=settings.cohere_api_key)

async def rerank_jobs(
    query: str,  # CV text or summary
    jobs: List[Job],
    top_k: int = 20
) -> List[Tuple[Job, float]]:
    """Re-rank jobs using Cohere's cross-encoder."""

    documents = [
        f"{job.title} at {job.company}\n{job.description[:1000]}"
        for job in jobs
    ]

    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_k
    )

    results = []
    for result in response.results:
        job = jobs[result.index]
        results.append((job, result.relevance_score))

    return results
```

### Local Cross-Encoder

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')

def rerank_local(query: str, jobs: List[Job], top_k: int = 20) -> List[Tuple[Job, float]]:
    pairs = [(query, job.description) for job in jobs]
    scores = model.predict(pairs)

    ranked = sorted(zip(jobs, scores), key=lambda x: -x[1])
    return ranked[:top_k]
```

---

## 3. Alternative Embedding Models

### Why Consider Alternatives?

Benchmarks show specialized models outperform OpenAI for retrieval:

| Model | MTEB Score | Dimensions | Cost |
|-------|------------|------------|------|
| OpenAI text-embedding-3-small | 62.3 | 1536 | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 64.6 | 3072 | $0.13/1M tokens |
| **nomic-embed-text-v1.5** | 69.1 | 768 | Free (local) |
| **BGE-large-en-v1.5** | 64.2 | 1024 | Free (local) |
| **E5-large-v2** | 62.4 | 1024 | Free (local) |
| Cohere embed-v3 | 64.5 | 1024 | $0.10/1M tokens |

### Migration Strategy

```python
# Abstract embedding interface
class EmbeddingProvider(Protocol):
    async def embed(self, text: str) -> List[float]: ...
    async def embed_batch(self, texts: List[str]) -> List[List[float]]: ...
    @property
    def dimensions(self) -> int: ...

# OpenAI implementation (current)
class OpenAIEmbeddings(EmbeddingProvider):
    dimensions = 1536
    # ... existing implementation

# Nomic implementation (alternative)
class NomicEmbeddings(EmbeddingProvider):
    dimensions = 768

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')

    async def embed(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()
```

### A/B Testing Framework

```python
@dataclass
class EmbeddingExperiment:
    name: str
    provider: EmbeddingProvider
    sample_percentage: float  # 0.0 to 1.0

experiments = [
    EmbeddingExperiment("openai", OpenAIEmbeddings(), 0.8),
    EmbeddingExperiment("nomic", NomicEmbeddings(), 0.2),
]

async def get_embedding_for_job(job: Job) -> Tuple[str, List[float]]:
    # Route based on job ID hash for consistency
    experiment = select_experiment(hash(job.id), experiments)
    embedding = await experiment.provider.embed(job.description)
    return experiment.name, embedding
```

---

## 4. Fine-Tuning (Advanced)

### When to Fine-Tune

- After collecting sufficient user interaction data (saves, applications, rejections)
- When generic models plateau in accuracy
- Domain-specific vocabulary not captured

### Data Requirements

| Data Type | Minimum | Ideal |
|-----------|---------|-------|
| Positive pairs (saved/applied jobs) | 1,000 | 10,000+ |
| Negative pairs (rejected/ignored) | 5,000 | 50,000+ |
| Total training samples | 10,000 | 100,000+ |

### Training Data Collection

```python
@dataclass
class TrainingPair:
    cv_text: str
    job_description: str
    label: float  # 1.0 = positive, 0.0 = negative

async def collect_training_data(db: AsyncSession) -> List[TrainingPair]:
    pairs = []

    # Positive: saved or applied jobs
    positive_jobs = await db.execute(
        select(Job, Profile)
        .where(Job.status.in_(["saved", "applied", "offered"]))
    )
    for job, profile in positive_jobs:
        pairs.append(TrainingPair(profile.cv_text, job.description, 1.0))

    # Negative: rejected or never interacted (random sample)
    negative_jobs = await db.execute(
        select(Job, Profile)
        .where(Job.status.in_(["rejected", "archived"]))
    )
    for job, profile in negative_jobs:
        pairs.append(TrainingPair(profile.cv_text, job.description, 0.0))

    return pairs
```

### Fine-Tuning Process

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

def fine_tune_embeddings(training_pairs: List[TrainingPair]):
    model = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5')

    # Prepare training examples
    train_examples = [
        InputExample(texts=[pair.cv_text, pair.job_description], label=pair.label)
        for pair in training_pairs
    ]

    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    # Fine-tune
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=3,
        warmup_steps=100,
        output_path="models/job-match-embeddings"
    )
```

---

## 5. Complete Pipeline

### Two-Stage Retrieval

```python
async def find_matching_jobs(
    profile: Profile,
    db: AsyncSession,
    top_k: int = 50
) -> List[Tuple[Job, float]]:
    """
    Two-stage retrieval: fast recall + precise re-ranking.
    """
    # Stage 1: Fast retrieval (get 200 candidates)
    # 1a. Semantic search
    semantic_results = await semantic_search(profile.cv_embedding, limit=150)

    # 1b. Keyword search (BM25)
    keyword_results = await keyword_search(profile.cv_text, limit=150)

    # 1c. Skill filter
    skill_results = await skill_filter(profile.target_roles, limit=100)

    # Combine with RRF
    candidate_ids = reciprocal_rank_fusion([
        semantic_results,
        keyword_results,
        skill_results
    ])[:200]

    # Stage 2: Re-rank top candidates
    candidates = await db.execute(
        select(Job).where(Job.id.in_(candidate_ids))
    )
    candidate_jobs = candidates.scalars().all()

    # Cross-encoder re-ranking
    reranked = await rerank_jobs(
        query=profile.cv_text[:2000],
        jobs=candidate_jobs,
        top_k=top_k
    )

    return reranked
```

---

## 6. Files to Create/Modify

| File | Purpose |
|------|---------|
| `backend/app/services/hybrid_search.py` | BM25 + semantic + RRF fusion |
| `backend/app/services/reranker.py` | Cross-encoder re-ranking |
| `backend/app/services/embedding_providers.py` | Abstract embedding interface |
| `backend/app/services/fine_tuning.py` | Training data collection, fine-tuning |
| `backend/app/api/search.py` | Enhanced search endpoints |
| `backend/app/config.py` | Add Cohere API key, model selection |

---

## 7. Success Metrics

| Metric | Phase 1 | Phase 3 Target |
|--------|---------|----------------|
| Precision@10 | ~70% | 85-90% |
| Recall@50 | ~75% | 90%+ |
| User save rate | baseline | +20% |
| Application rate | baseline | +15% |

### Measurement Approach

```python
# Track user interactions
@dataclass
class MatchEvent:
    job_id: str
    profile_id: str
    match_score: float
    rank_position: int
    action: str  # "viewed", "saved", "applied", "rejected", "ignored"
    timestamp: datetime

# Calculate precision@k
def precision_at_k(events: List[MatchEvent], k: int) -> float:
    top_k = sorted(events, key=lambda e: e.rank_position)[:k]
    positive_actions = {"saved", "applied"}
    relevant = sum(1 for e in top_k if e.action in positive_actions)
    return relevant / k
```

---

## 8. Dependencies

```
Phase 1 ──► Phase 2 ──► Phase 3
   │                       │
   │                       ├── Hybrid search (BM25 + semantic)
   │                       ├── Cross-encoder re-ranking
   │                       ├── Alternative embeddings
   │                       └── Fine-tuning pipeline
   │
   └── Can start hybrid search after Phase 1
       (Phase 2 enriches but not required)
```

Phase 3 can begin after Phase 1; Phase 2's skill extraction improves keyword matching but is not a hard dependency.
