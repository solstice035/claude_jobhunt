# Monitoring Guide

This document describes the monitoring setup for the Job Hunt application, including Prometheus metrics collection and Grafana dashboards.

## Architecture

```
                              ┌─────────────────┐
                              │     Grafana     │
                              │    :3001        │
                              └────────┬────────┘
                                       │
                              ┌────────▼────────┐
                              │   Prometheus    │
                              │    :9090        │
                              └────────┬────────┘
                                       │
          ┌────────────────────────────┼────────────────────────────┐
          │                            │                            │
┌─────────▼──────────┐    ┌───────────▼──────────┐    ┌───────────▼──────────┐
│   FastAPI Backend  │    │    Redis Exporter    │    │   Prometheus Self    │
│   :8000/metrics    │    │       :9121          │    │       :9090          │
└────────────────────┘    └──────────────────────┘    └──────────────────────┘
```

## Quick Start

### Accessing the Dashboards

1. **Start the monitoring stack:**
   ```bash
   docker compose up -d
   ```

2. **Access Grafana:**
   - URL: http://localhost:3001
   - Username: `admin`
   - Password: `admin` (or value of `GRAFANA_PASSWORD` env var)

3. **Access Prometheus:**
   - URL: http://localhost:9090

### Default Credentials

| Service    | Username | Password                        |
|------------|----------|---------------------------------|
| Grafana    | admin    | admin (override with `GRAFANA_PASSWORD`) |
| Prometheus | N/A      | N/A (no auth by default)        |

## Available Dashboards

The following dashboards are automatically provisioned:

### 1. System Overview
**Location:** JobHunt folder > System Overview

A high-level view of all system metrics including:
- API health status
- Request rate and error rate
- P95 latency
- Cache hit rate
- Job processing throughput
- Queue depth

**Use for:** Quick health checks and identifying issues at a glance.

### 2. API Performance
**Location:** JobHunt folder > API Performance

Detailed API metrics including:
- Request rate by endpoint
- Request rate by status code
- Response time percentiles (P50, P95, P99)
- P95 latency by endpoint
- Active requests

**Use for:** Performance troubleshooting and capacity planning.

### 3. Job Matching Performance
**Location:** JobHunt folder > Job Matching Performance

ML/AI-specific metrics including:
- Total embeddings generated
- Match scores calculated
- Embedding generation latency
- Match score calculation latency
- Vector database size and query latency

**Use for:** Monitoring AI/ML pipeline performance and OpenAI API usage.

### 4. Cache Performance
**Location:** JobHunt folder > Cache Performance

Cache layer metrics including:
- Overall cache hit rate (gauge)
- Hit rate by layer (L1: response, L2: match_score, L3: embedding)
- Cache operations rate
- Redis memory usage
- Redis connected clients

**Use for:** Optimizing cache effectiveness and Redis capacity.

### 5. Background Tasks
**Location:** JobHunt folder > Background Tasks

Celery task metrics including:
- Task failure count and rate
- Task duration percentiles
- Duration by task type
- Failure rate by task
- Queue depth over time

**Use for:** Monitoring background job health and identifying bottlenecks.

## Key Metrics to Monitor

### Critical Alerts

Set up alerts for these metrics:

| Metric | Warning Threshold | Critical Threshold |
|--------|-------------------|-------------------|
| Error Rate | > 1% | > 5% |
| P95 Latency | > 500ms | > 1s |
| Cache Hit Rate | < 80% | < 50% |
| Queue Depth | > 100 | > 500 |
| Task Failures | > 1/min | > 5/min |

### Performance Baselines

Typical healthy values:
- **P50 Latency:** < 50ms
- **P95 Latency:** < 200ms
- **P99 Latency:** < 500ms
- **Cache Hit Rate:** > 90%
- **Error Rate:** < 0.1%
- **Queue Depth:** < 10

## Prometheus Metrics Reference

### HTTP Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `http_requests_total` | Counter | method, endpoint, status | Total HTTP requests |
| `http_request_duration_seconds` | Histogram | method, endpoint, status | Request latency |
| `http_requests_active` | Gauge | method, endpoint | Currently active requests |

### Cache Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `cache_hits_total` | Counter | layer | Total cache hits |
| `cache_misses_total` | Counter | layer | Total cache misses |

Cache layers:
- `L1` - Response cache (HTTP responses)
- `L2` - Match score cache
- `L3` - Embedding cache

### ML/Embedding Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `embeddings_generated_total` | Counter | - | Total embeddings created |
| `match_scores_calculated_total` | Counter | - | Total scores calculated |
| `embedding_generation_seconds` | Histogram | provider | Embedding API latency |
| `match_score_calculation_seconds` | Histogram | - | Scoring computation time |
| `vector_query_seconds` | Histogram | operation | Vector DB query latency |
| `vector_db_embeddings_total` | Gauge | - | Embeddings in vector DB |

### Celery Task Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `celery_task_duration_seconds` | Histogram | task_name | Task execution time |
| `celery_task_failures_total` | Counter | task_name | Task failure count |
| `celery_queue_depth` | Gauge | queue_name | Pending tasks in queue |

### Redis Metrics (from redis_exporter)

| Metric | Type | Description |
|--------|------|-------------|
| `redis_memory_used_bytes` | Gauge | Redis memory usage |
| `redis_memory_max_bytes` | Gauge | Redis max memory limit |
| `redis_connected_clients` | Gauge | Number of client connections |
| `redis_blocked_clients` | Gauge | Clients blocked on operations |

## Customizing Dashboards

### Adding New Panels

1. Open Grafana at http://localhost:3001
2. Navigate to the dashboard you want to modify
3. Click the "+" icon to add a new panel
4. Configure your query using PromQL
5. Save the dashboard

### Exporting Dashboard Changes

Dashboard JSON files are stored in `grafana/dashboards/`. To export changes:

1. Open the dashboard in Grafana
2. Click the share icon (arrow)
3. Select "Export"
4. Choose "Save to file"
5. Replace the file in `grafana/dashboards/`

### Creating Custom Dashboards

1. Create a new JSON file in `grafana/dashboards/`
2. Follow the existing dashboard structure
3. Use a unique `uid` (e.g., `jobhunt-custom-dashboard`)
4. Add the `jobhunt` tag for consistency

## Prometheus Configuration

The Prometheus configuration is in `prometheus.yml`. Key settings:

```yaml
global:
  scrape_interval: 15s      # How often to scrape targets
  evaluation_interval: 15s  # How often to evaluate rules

scrape_configs:
  - job_name: 'jobhunt-api'
    static_configs:
      - targets: ['backend:8000']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Adding New Scrape Targets

To add a new service to monitor:

1. Edit `prometheus.yml`
2. Add a new job under `scrape_configs`
3. Restart Prometheus: `docker compose restart prometheus`

## Troubleshooting

### Grafana Shows "No Data"

1. Check Prometheus is running: http://localhost:9090
2. Verify the target is UP: http://localhost:9090/targets
3. Test the query directly in Prometheus
4. Check datasource configuration in Grafana

### Metrics Not Appearing

1. Verify the `/metrics` endpoint: `curl http://localhost:8000/metrics`
2. Check the backend logs for errors
3. Ensure `setup_metrics(app)` is called in `main.py`

### Redis Metrics Missing

1. Check redis-exporter is running: `docker compose ps redis-exporter`
2. Verify connection: `curl http://localhost:9121/metrics`
3. Check Redis is accessible from the exporter

### High Cardinality Issues

If Prometheus uses too much memory:

1. Review endpoint patterns (avoid path parameters in labels)
2. Reduce retention: `--storage.tsdb.retention.time=7d`
3. Consider using recording rules for complex queries

## Production Recommendations

### Security

1. Change default Grafana password:
   ```bash
   export GRAFANA_PASSWORD=your-secure-password
   ```

2. Enable HTTPS for Grafana (via nginx proxy)

3. Add basic auth to Prometheus if exposed externally

### Storage

1. Configure persistent storage for Prometheus data:
   ```yaml
   volumes:
     - prometheus_data:/prometheus
   ```

2. Set appropriate retention:
   ```yaml
   command:
     - "--storage.tsdb.retention.time=30d"
   ```

### Alerting (Optional)

To enable alerting:

1. Deploy Alertmanager
2. Create alert rules in `alerts/` directory
3. Configure notification channels (Slack, email, PagerDuty)

Example alert rule (`alerts/api.yml`):
```yaml
groups:
  - name: api
    rules:
      - alert: HighErrorRate
        expr: |
          100 * sum(rate(http_requests_total{status=~"5.."}[5m]))
          / sum(rate(http_requests_total[5m])) > 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: High API error rate
          description: Error rate is {{ $value | printf "%.1f" }}%
```

## Useful PromQL Queries

### Request Rate
```promql
sum(rate(http_requests_total[5m]))
```

### Error Rate Percentage
```promql
100 * sum(rate(http_requests_total{status=~"5.."}[5m]))
    / sum(rate(http_requests_total[5m]))
```

### P95 Latency
```promql
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
```

### Cache Hit Rate
```promql
100 * sum(rate(cache_hits_total[5m]))
    / (sum(rate(cache_hits_total[5m])) + sum(rate(cache_misses_total[5m])))
```

### Slowest Endpoints (P95)
```promql
topk(10, histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint)))
```
