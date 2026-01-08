"""
Performance/Load Tests for ML Matching Endpoints

This module provides comprehensive load testing for the enhanced job matching API:
- Hybrid search (POST /api/search/hybrid)
- Skills search (GET /skills/search)
- Skills extraction (POST /skills/extract)
- Jobs listing with match scoring (GET /jobs)

Usage:
    # Run all performance tests
    python -m tests.performance.load_test_ml_endpoints

    # Run with custom settings
    python -m tests.performance.load_test_ml_endpoints --base-url http://localhost:8000 --concurrent 10 --requests 100

Requirements:
    - httpx (async HTTP client)
    - numpy (for statistics)
    - Running backend server at specified base URL
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import httpx
import numpy as np


@dataclass
class TestResult:
    """Result of a single request."""
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class EndpointStats:
    """Aggregated statistics for an endpoint."""
    endpoint: str
    method: str
    total_requests: int
    successful_requests: int
    failed_requests: int
    response_times_ms: List[float] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def p50(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return float(np.percentile(self.response_times_ms, 50))

    @property
    def p95(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return float(np.percentile(self.response_times_ms, 95))

    @property
    def p99(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return float(np.percentile(self.response_times_ms, 99))

    @property
    def avg(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return statistics.mean(self.response_times_ms)

    @property
    def min_time(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return min(self.response_times_ms)

    @property
    def max_time(self) -> float:
        if not self.response_times_ms:
            return 0.0
        return max(self.response_times_ms)


class MLEndpointLoadTester:
    """Load tester for ML matching endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        auth_token: Optional[str] = None,
        timeout: float = 30.0
    ):
        self.base_url = base_url.rstrip("/")
        self.auth_token = auth_token
        self.timeout = timeout
        self.results: List[TestResult] = []

    def _get_headers(self) -> Dict[str, str]:
        """Get request headers including auth if available."""
        headers = {"Content-Type": "application/json"}
        if self.auth_token:
            headers["Authorization"] = f"Bearer {self.auth_token}"
        return headers

    async def _make_request(
        self,
        client: httpx.AsyncClient,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> TestResult:
        """Make a single HTTP request and record timing."""
        url = f"{self.base_url}{endpoint}"
        start_time = time.perf_counter()

        try:
            if method.upper() == "GET":
                response = await client.get(url, params=params, headers=self._get_headers())
            elif method.upper() == "POST":
                response = await client.post(url, json=json_data, headers=self._get_headers())
            else:
                raise ValueError(f"Unsupported method: {method}")

            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000

            success = 200 <= response.status_code < 300
            error = None if success else f"HTTP {response.status_code}: {response.text[:200]}"

            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                success=success,
                error=error
            )

        except Exception as e:
            end_time = time.perf_counter()
            response_time_ms = (end_time - start_time) * 1000
            return TestResult(
                endpoint=endpoint,
                method=method,
                status_code=0,
                response_time_ms=response_time_ms,
                success=False,
                error=str(e)
            )

    async def test_hybrid_search(
        self,
        client: httpx.AsyncClient,
        query_text: str = "Python developer with AWS experience",
        use_reranker: bool = True
    ) -> TestResult:
        """Test the hybrid search endpoint."""
        payload = {
            "query_text": query_text,
            "top_k": 20,
            "bm25_weight": 0.5,
            "semantic_weight": 0.5,
            "use_rrf": True,
            "use_reranker": use_reranker
        }
        return await self._make_request(client, "POST", "/api/search/hybrid", json_data=payload)

    async def test_rerank(
        self,
        client: httpx.AsyncClient,
        query: str = "Senior Python developer",
        job_ids: Optional[List[str]] = None
    ) -> TestResult:
        """Test the rerank endpoint."""
        if job_ids is None:
            job_ids = ["job-1", "job-2", "job-3"]  # Placeholder IDs
        payload = {
            "query": query,
            "job_ids": job_ids,
            "top_k": 10,
            "provider": "local"
        }
        return await self._make_request(client, "POST", "/api/search/rerank", json_data=payload)

    async def test_search_status(self, client: httpx.AsyncClient) -> TestResult:
        """Test the search status endpoint."""
        return await self._make_request(client, "GET", "/api/search/status")

    async def test_skills_search(
        self,
        client: httpx.AsyncClient,
        query: str = "python"
    ) -> TestResult:
        """Test the skills search endpoint."""
        return await self._make_request(
            client, "GET", "/skills/search", params={"q": query, "limit": 20}
        )

    async def test_skills_extract(
        self,
        client: httpx.AsyncClient,
        text: str = "Looking for a Python developer with 5 years of experience in AWS, Docker, and Kubernetes."
    ) -> TestResult:
        """Test the skills extraction endpoint."""
        payload = {"text": text}
        return await self._make_request(client, "POST", "/skills/extract", json_data=payload)

    async def test_skills_infer(
        self,
        client: httpx.AsyncClient,
        skills: str = "python,docker"
    ) -> TestResult:
        """Test the skills inference endpoint."""
        return await self._make_request(
            client, "GET", "/skills/infer", params={"skills": skills, "include_related": "true"}
        )

    async def test_jobs_list(
        self,
        client: httpx.AsyncClient,
        score_min: Optional[float] = None
    ) -> TestResult:
        """Test the jobs listing endpoint."""
        params = {"page": 1, "per_page": 20}
        if score_min is not None:
            params["score_min"] = score_min
        return await self._make_request(client, "GET", "/jobs", params=params)

    async def run_concurrent_tests(
        self,
        test_func: Callable,
        num_requests: int,
        concurrency: int,
        **kwargs
    ) -> List[TestResult]:
        """Run a test function concurrently."""
        results = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Create batches based on concurrency
            for batch_start in range(0, num_requests, concurrency):
                batch_size = min(concurrency, num_requests - batch_start)
                tasks = [test_func(client, **kwargs) for _ in range(batch_size)]
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)

        return results

    async def run_full_load_test(
        self,
        num_requests: int = 50,
        concurrency: int = 5
    ) -> Dict[str, EndpointStats]:
        """Run load tests on all ML endpoints."""
        stats: Dict[str, EndpointStats] = {}

        # Define test cases
        test_cases = [
            ("POST /api/search/hybrid", self.test_hybrid_search, {}),
            ("POST /api/search/hybrid (no reranker)", self.test_hybrid_search, {"use_reranker": False}),
            ("GET /api/search/status", self.test_search_status, {}),
            ("GET /skills/search?q=python", self.test_skills_search, {"query": "python"}),
            ("GET /skills/search?q=kubernetes", self.test_skills_search, {"query": "kubernetes"}),
            ("POST /skills/extract", self.test_skills_extract, {}),
            ("GET /skills/infer", self.test_skills_infer, {}),
            ("GET /jobs", self.test_jobs_list, {}),
            ("GET /jobs?score_min=50", self.test_jobs_list, {"score_min": 50}),
        ]

        print(f"\n{'='*80}")
        print(f"ML Endpoint Load Test - {num_requests} requests, {concurrency} concurrent")
        print(f"Target: {self.base_url}")
        print(f"{'='*80}\n")

        for name, test_func, kwargs in test_cases:
            print(f"Testing: {name}...")

            try:
                results = await self.run_concurrent_tests(
                    test_func, num_requests, concurrency, **kwargs
                )

                # Calculate stats
                endpoint_stats = EndpointStats(
                    endpoint=name.split()[1] if len(name.split()) > 1 else name,
                    method=name.split()[0],
                    total_requests=len(results),
                    successful_requests=sum(1 for r in results if r.success),
                    failed_requests=sum(1 for r in results if not r.success),
                    response_times_ms=[r.response_time_ms for r in results if r.success]
                )

                stats[name] = endpoint_stats
                self.results.extend(results)

                # Print quick summary
                print(f"  Success: {endpoint_stats.success_rate:.1f}% "
                      f"| Avg: {endpoint_stats.avg:.1f}ms "
                      f"| p95: {endpoint_stats.p95:.1f}ms "
                      f"| p99: {endpoint_stats.p99:.1f}ms")

            except Exception as e:
                print(f"  FAILED: {e}")
                stats[name] = EndpointStats(
                    endpoint=name.split()[1] if len(name.split()) > 1 else name,
                    method=name.split()[0],
                    total_requests=0,
                    successful_requests=0,
                    failed_requests=0
                )

        return stats

    def print_report(self, stats: Dict[str, EndpointStats]) -> None:
        """Print a detailed performance report."""
        print(f"\n{'='*100}")
        print("PERFORMANCE TEST REPORT")
        print(f"{'='*100}\n")

        # Table header
        print(f"{'Endpoint':<45} {'Reqs':>6} {'OK%':>6} {'Avg':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'Max':>8}")
        print("-" * 100)

        for name, s in stats.items():
            print(f"{name:<45} {s.total_requests:>6} {s.success_rate:>5.1f}% "
                  f"{s.avg:>7.1f}ms {s.p50:>7.1f}ms {s.p95:>7.1f}ms {s.p99:>7.1f}ms {s.max_time:>7.1f}ms")

        print("-" * 100)

        # Summary statistics
        total_requests = sum(s.total_requests for s in stats.values())
        total_successful = sum(s.successful_requests for s in stats.values())
        all_times = [r.response_time_ms for r in self.results if r.success]

        if all_times:
            print(f"\nOverall Statistics:")
            print(f"  Total Requests: {total_requests}")
            print(f"  Successful: {total_successful} ({(total_successful/total_requests)*100:.1f}%)")
            print(f"  Overall Avg: {statistics.mean(all_times):.1f}ms")
            print(f"  Overall p50: {np.percentile(all_times, 50):.1f}ms")
            print(f"  Overall p95: {np.percentile(all_times, 95):.1f}ms")
            print(f"  Overall p99: {np.percentile(all_times, 99):.1f}ms")

        # Identify bottlenecks
        print(f"\nBottleneck Analysis:")
        slow_endpoints = [(n, s) for n, s in stats.items() if s.p95 > 1000]
        if slow_endpoints:
            print("  Slow endpoints (p95 > 1000ms):")
            for name, s in sorted(slow_endpoints, key=lambda x: x[1].p95, reverse=True):
                print(f"    - {name}: p95={s.p95:.1f}ms, p99={s.p99:.1f}ms")
        else:
            print("  No slow endpoints detected (all p95 < 1000ms)")

        # Error analysis
        failed_endpoints = [(n, s) for n, s in stats.items() if s.failed_requests > 0]
        if failed_endpoints:
            print("\n  Endpoints with failures:")
            for name, s in failed_endpoints:
                print(f"    - {name}: {s.failed_requests} failures ({100-s.success_rate:.1f}%)")

    def export_json(self, stats: Dict[str, EndpointStats], filename: str = "performance_results.json") -> None:
        """Export results to JSON file."""
        export_data = {
            "summary": {
                "total_requests": sum(s.total_requests for s in stats.values()),
                "total_successful": sum(s.successful_requests for s in stats.values()),
                "base_url": self.base_url
            },
            "endpoints": {}
        }

        for name, s in stats.items():
            export_data["endpoints"][name] = {
                "total_requests": s.total_requests,
                "successful_requests": s.successful_requests,
                "failed_requests": s.failed_requests,
                "success_rate": s.success_rate,
                "avg_ms": s.avg,
                "p50_ms": s.p50,
                "p95_ms": s.p95,
                "p99_ms": s.p99,
                "min_ms": s.min_time,
                "max_ms": s.max_time
            }

        with open(filename, "w") as f:
            json.dump(export_data, f, indent=2)

        print(f"\nResults exported to: {filename}")


async def main():
    """Main entry point for the load test."""
    parser = argparse.ArgumentParser(description="ML Endpoint Load Tester")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL for the API server"
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=5,
        help="Number of concurrent requests"
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=50,
        help="Total number of requests per endpoint"
    )
    parser.add_argument(
        "--auth-token",
        default=None,
        help="JWT authentication token"
    )
    parser.add_argument(
        "--export",
        default=None,
        help="Export results to JSON file"
    )

    args = parser.parse_args()

    tester = MLEndpointLoadTester(
        base_url=args.base_url,
        auth_token=args.auth_token
    )

    stats = await tester.run_full_load_test(
        num_requests=args.requests,
        concurrency=args.concurrent
    )

    tester.print_report(stats)

    if args.export:
        tester.export_json(stats, args.export)


if __name__ == "__main__":
    asyncio.run(main())
