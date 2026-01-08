"""
Prometheus Metrics Middleware

Provides request/response metrics for monitoring:
- HTTP request latency (p50, p95, p99)
- Request count by endpoint and status
- Active request gauge
- Cache hit/miss rates

Usage:
    from app.middleware.metrics import PrometheusMiddleware, setup_metrics

    # In main.py
    app = FastAPI()
    setup_metrics(app)

Metrics Endpoint:
    GET /metrics - Prometheus-format metrics
"""

import time
import logging
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CONTENT_TYPE_LATEST,
    generate_latest,
    REGISTRY,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Match

logger = logging.getLogger(__name__)

# ==================== Prometheus Metrics ====================

# Request latency histogram with custom buckets for sub-second monitoring
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint", "status"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Request counter
REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"]
)

# Active requests gauge
ACTIVE_REQUESTS = Gauge(
    "http_requests_active",
    "Number of active HTTP requests",
    ["method", "endpoint"]
)

# Cache metrics
CACHE_HITS = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["layer"]  # L1=response, L2=match_score, L3=embedding
)

CACHE_MISSES = Counter(
    "cache_misses_total",
    "Total cache misses",
    ["layer"]
)

# ML/Embedding metrics
EMBEDDING_LATENCY = Histogram(
    "embedding_generation_seconds",
    "Time to generate embeddings",
    ["provider"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]
)

MATCH_SCORE_LATENCY = Histogram(
    "match_score_calculation_seconds",
    "Time to calculate match scores",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

# Vector database metrics
VECTOR_QUERY_LATENCY = Histogram(
    "vector_query_seconds",
    "Vector database query latency",
    ["operation"],  # query, upsert, delete
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
)

VECTOR_DB_SIZE = Gauge(
    "vector_db_embeddings_total",
    "Total embeddings in vector database"
)

# Celery queue metrics
QUEUE_DEPTH = Gauge(
    "celery_queue_depth",
    "Number of tasks in Celery queue",
    ["queue_name"]
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for Prometheus metrics collection.

    Records:
    - Request latency
    - Request count by status code
    - Active request count
    """

    def __init__(self, app: FastAPI, app_name: str = "jobhunt"):
        super().__init__(app)
        self.app_name = app_name

    async def dispatch(
        self,
        request: Request,
        call_next: Callable
    ) -> Response:
        """Process request and record metrics."""
        # Get endpoint path (use route pattern for consistency)
        endpoint = self._get_endpoint(request)
        method = request.method

        # Skip metrics endpoint
        if endpoint == "/metrics":
            return await call_next(request)

        # Track active requests
        ACTIVE_REQUESTS.labels(method=method, endpoint=endpoint).inc()

        # Start timing
        start_time = time.perf_counter()

        try:
            response = await call_next(request)
            status = str(response.status_code)
        except Exception as e:
            status = "500"
            logger.error(f"Request error: {e}")
            raise
        finally:
            # Calculate duration
            duration = time.perf_counter() - start_time

            # Record metrics
            REQUEST_LATENCY.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).observe(duration)

            REQUEST_COUNT.labels(
                method=method,
                endpoint=endpoint,
                status=status
            ).inc()

            ACTIVE_REQUESTS.labels(
                method=method,
                endpoint=endpoint
            ).dec()

        return response

    def _get_endpoint(self, request: Request) -> str:
        """
        Get normalized endpoint path from request.

        Uses route pattern (e.g., /api/jobs/{id}) instead of
        actual path to avoid high cardinality.
        """
        # Try to match against routes
        for route in request.app.routes:
            match, _ = route.matches(request.scope)
            if match == Match.FULL:
                return route.path

        # Fallback to path
        return request.url.path


def metrics_endpoint(request: Request) -> Response:
    """
    Endpoint handler for Prometheus metrics scraping.

    Returns metrics in Prometheus text format.
    """
    return PlainTextResponse(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST
    )


def setup_metrics(app: FastAPI) -> None:
    """
    Configure Prometheus metrics for FastAPI app.

    Args:
        app: FastAPI application instance
    """
    # Add middleware
    app.add_middleware(PrometheusMiddleware, app_name="jobhunt")

    # Add metrics endpoint
    app.add_route("/metrics", metrics_endpoint, methods=["GET"])

    logger.info("Prometheus metrics configured")


# ==================== Helper Functions ====================

def record_cache_hit(layer: str) -> None:
    """Record a cache hit for the specified layer."""
    CACHE_HITS.labels(layer=layer).inc()


def record_cache_miss(layer: str) -> None:
    """Record a cache miss for the specified layer."""
    CACHE_MISSES.labels(layer=layer).inc()


def record_embedding_latency(provider: str, duration: float) -> None:
    """Record embedding generation latency."""
    EMBEDDING_LATENCY.labels(provider=provider).observe(duration)


def record_match_score_latency(duration: float) -> None:
    """Record match score calculation latency."""
    MATCH_SCORE_LATENCY.observe(duration)


def record_vector_query_latency(operation: str, duration: float) -> None:
    """Record vector database operation latency."""
    VECTOR_QUERY_LATENCY.labels(operation=operation).observe(duration)


def update_vector_db_size(count: int) -> None:
    """Update vector database size gauge."""
    VECTOR_DB_SIZE.set(count)


def update_queue_depth(queue_name: str, depth: int) -> None:
    """Update Celery queue depth gauge."""
    QUEUE_DEPTH.labels(queue_name=queue_name).set(depth)
