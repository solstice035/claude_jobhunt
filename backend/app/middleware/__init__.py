"""
Middleware Package

Contains FastAPI middleware for:
- Prometheus metrics collection
- Request/response logging
- Performance monitoring
"""

from app.middleware.metrics import (
    PrometheusMiddleware,
    setup_metrics,
    REQUEST_LATENCY,
    REQUEST_COUNT,
    ACTIVE_REQUESTS,
    CACHE_HITS,
    CACHE_MISSES,
)

__all__ = [
    "PrometheusMiddleware",
    "setup_metrics",
    "REQUEST_LATENCY",
    "REQUEST_COUNT",
    "ACTIVE_REQUESTS",
    "CACHE_HITS",
    "CACHE_MISSES",
]
