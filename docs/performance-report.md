# Performance Report: ML Matching Endpoints

## Overview

This report documents the performance characteristics of the Enhanced Job Matching feature endpoints, including hybrid search, skills intelligence, and the improved jobs API.

## Tested Endpoints

| Endpoint | Method | Description | ML Components |
|----------|--------|-------------|---------------|
| `/api/search/hybrid` | POST | Hybrid BM25 + semantic search | BM25, Embeddings, Cross-encoder |
| `/api/search/rerank` | POST | Re-rank existing results | Cross-encoder model |
| `/api/search/status` | GET | Search service status | None |
| `/skills/search` | GET | ESCO skills database search | Text matching |
| `/skills/extract` | POST | Extract skills from text | LLM (OpenAI) |
| `/skills/gaps` | GET | Skill gap analysis | LLM + DB queries |
| `/skills/infer` | GET | Skill inference | Graph traversal |
| `/jobs` | GET | List jobs with match scores | DB queries |

## Performance Test Tools

Three testing approaches are provided:

### 1. Async Load Test Script
```bash
cd backend
python -m tests.performance.load_test_ml_endpoints \
    --base-url http://localhost:8000 \
    --concurrent 10 \
    --requests 100 \
    --export results.json
```

### 2. Locust Load Testing
```bash
cd backend
pip install locust

# Web UI mode
locust -f tests/performance/locustfile.py --host http://localhost:8000

# Headless mode with HTML report
locust -f tests/performance/locustfile.py \
    --headless -u 10 -r 2 -t 60s \
    --host http://localhost:8000 \
    --html=locust_report.html
```

### 3. Pytest Benchmarks
```bash
cd backend
pytest tests/performance/test_benchmark_ml_endpoints.py -v --durations=0
```

## Expected Performance Baselines

### Hybrid Search (`POST /api/search/hybrid`)

| Metric | Without Reranker | With Reranker |
|--------|------------------|---------------|
| p50 | < 100ms | < 500ms |
| p95 | < 200ms | < 2000ms |
| p99 | < 400ms | < 5000ms |

**Bottleneck Analysis:**
- **Without reranker**: Performance is dominated by BM25 search and embedding similarity. BM25 is CPU-bound and scales linearly with corpus size.
- **With reranker**: The cross-encoder model (sentence-transformers) adds significant latency. First request is slower due to model loading.

### Skills Search (`GET /skills/search`)

| Metric | Value |
|--------|-------|
| p50 | < 50ms |
| p95 | < 100ms |
| p99 | < 200ms |

**Notes:**
- Performance depends on ESCO database size and index efficiency
- Consider adding database indexes on skill labels for improved performance

### Skills Extraction (`POST /skills/extract`)

| Metric | Value |
|--------|-------|
| p50 | < 2000ms |
| p95 | < 5000ms |
| p99 | < 10000ms |

**Bottleneck Analysis:**
- Dominated by OpenAI API latency
- Consider caching common extraction patterns
- Async batching could improve throughput for bulk operations

### Jobs Listing (`GET /jobs`)

| Metric | Value |
|--------|-------|
| p50 | < 50ms |
| p95 | < 100ms |
| p99 | < 200ms |

**Notes:**
- Performance scales with database size
- Index on `match_score` and `created_at` for efficient ordering
- Pagination prevents large result set overhead

## Component-Level Performance

### BM25 Search (rank-bm25)

| Operation | Expected Time |
|-----------|---------------|
| Index build (100 docs) | < 500ms |
| Index build (1000 docs) | < 2000ms |
| Search (100 docs) | < 20ms |
| Search (1000 docs) | < 50ms |

**Optimization Notes:**
- Index is built lazily on first search request
- Index is cached in memory for subsequent requests
- Consider pre-building index during server startup for production

### Semantic Search (Embeddings)

| Operation | Expected Time |
|-----------|---------------|
| Similarity search (100 docs) | < 50ms |
| Similarity search (1000 docs) | < 200ms |

**Notes:**
- Uses NumPy for vectorized cosine similarity
- Memory usage scales linearly with corpus size
- Consider using approximate nearest neighbor (ANN) for large corpora (>10k docs)

### Cross-Encoder Reranker

| Operation | Expected Time |
|-----------|---------------|
| Model load (first request) | 5-30 seconds |
| Rerank 20 documents | 500-2000ms |
| Rerank 50 documents | 1000-4000ms |

**Optimization Notes:**
- Model is loaded lazily to avoid startup delay
- Consider using Cohere API for faster inference with higher throughput
- Batch processing improves GPU utilization if available

## Throughput Estimates

### Single Instance (8GB RAM, 4 CPU)

| Endpoint | Estimated RPS |
|----------|---------------|
| `/api/search/hybrid` (no reranker) | 50-100 |
| `/api/search/hybrid` (with reranker) | 5-20 |
| `/skills/search` | 100-200 |
| `/skills/extract` | 10-30 (API limited) |
| `/jobs` | 200-500 |

### Scaling Recommendations

1. **Horizontal Scaling**: Use multiple backend instances behind a load balancer
2. **Caching**: Implement Redis caching for frequently accessed job embeddings
3. **Async Processing**: Move heavy ML operations to background workers (Celery)
4. **Model Optimization**: Use ONNX or TensorRT for faster inference

## Identified Bottlenecks

### 1. Cross-Encoder Reranking
- **Impact**: High latency on hybrid search with reranking
- **Mitigation**:
  - Use lazy loading to avoid startup delay
  - Consider GPU acceleration
  - Reduce candidate set before reranking
  - Use Cohere API for higher throughput

### 2. LLM-Based Skill Extraction
- **Impact**: High latency on skill extraction (2-10 seconds)
- **Mitigation**:
  - Cache extraction results by text hash
  - Use smaller models for simple extractions
  - Batch processing for bulk operations

### 3. BM25 Index Rebuild
- **Impact**: Latency spike after job updates
- **Mitigation**:
  - Incremental index updates
  - Background index rebuilding
  - Index pre-warming during deployment

### 4. Database Queries with Large Result Sets
- **Impact**: Slow skill gap analysis with many jobs
- **Mitigation**:
  - Add database indexes
  - Implement query result caching
  - Use pagination for all list operations

## Monitoring Recommendations

### Key Metrics to Track

1. **Response Time Percentiles**: p50, p95, p99 for each endpoint
2. **Throughput**: Requests per second by endpoint
3. **Error Rate**: 4xx and 5xx responses
4. **Model Loading Time**: Cross-encoder initialization
5. **Cache Hit Rate**: If caching is implemented

### Alerting Thresholds

| Metric | Warning | Critical |
|--------|---------|----------|
| p95 latency (hybrid search) | > 3s | > 10s |
| p95 latency (skills search) | > 200ms | > 1s |
| Error rate | > 1% | > 5% |
| Memory usage | > 70% | > 90% |

## Test Execution Instructions

### Prerequisites

```bash
cd backend
pip install -r requirements.txt
pip install locust  # For Locust tests
```

### Running Tests

```bash
# 1. Start the backend server (in a separate terminal)
uvicorn app.main:app --reload --port 8000

# 2. Run async load tests
python -m tests.performance.load_test_ml_endpoints \
    --base-url http://localhost:8000 \
    --concurrent 10 \
    --requests 100

# 3. Run Locust tests (opens web UI at http://localhost:8089)
locust -f tests/performance/locustfile.py --host http://localhost:8000

# 4. Run pytest benchmarks
pytest tests/performance/test_benchmark_ml_endpoints.py -v
```

### Interpreting Results

- **Success Rate**: Should be > 99% for all endpoints
- **p95 Latency**: Primary SLA metric; should meet baselines
- **p99 Latency**: Important for user experience; monitors tail latency
- **Throughput**: Should meet expected RPS for production load

## Future Improvements

1. **Vector Database Integration**: Use ChromaDB or Pinecone for faster semantic search
2. **Model Quantization**: Reduce reranker model size for faster inference
3. **Edge Caching**: Cache common search queries at CDN level
4. **Streaming Responses**: Return partial results for long-running operations
5. **Request Prioritization**: Implement priority queues for different user tiers

---

*Report generated for the Enhanced Job Matching feature. Update baselines as performance optimizations are implemented.*
