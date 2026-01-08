"""
Locust Load Testing for ML Matching Endpoints

This module provides Locust-based load testing with realistic user behavior simulation.
It tests the enhanced job matching API endpoints with various user scenarios.

Usage:
    # Start Locust web UI
    cd backend && locust -f tests/performance/locustfile.py

    # Run headless with parameters
    cd backend && locust -f tests/performance/locustfile.py \
        --headless -u 10 -r 2 -t 60s --host http://localhost:8000

    # Generate HTML report
    cd backend && locust -f tests/performance/locustfile.py \
        --headless -u 10 -r 2 -t 60s --host http://localhost:8000 \
        --html=performance_report.html

Requirements:
    pip install locust

Configuration:
    - u: Number of users (default 10)
    - r: Spawn rate per second (default 2)
    - t: Test duration (default 60s)
"""

import json
import random
from typing import List

from locust import HttpUser, TaskSet, between, task


# Sample data for realistic test payloads
SAMPLE_QUERIES = [
    "Python developer with AWS experience",
    "Senior software engineer machine learning",
    "DevOps engineer Kubernetes Docker",
    "Full stack developer React Node.js",
    "Data scientist Python SQL",
    "Backend engineer Go microservices",
    "Frontend developer TypeScript React",
    "Cloud architect AWS Azure",
    "Site reliability engineer Linux",
    "Mobile developer iOS Swift",
]

SAMPLE_SKILLS = [
    "python", "javascript", "typescript", "go", "rust",
    "aws", "azure", "gcp", "kubernetes", "docker",
    "react", "vue", "angular", "node", "fastapi",
    "postgresql", "mongodb", "redis", "elasticsearch",
    "machine learning", "deep learning", "nlp",
]

SAMPLE_JOB_DESCRIPTIONS = [
    "We are looking for a Python developer with 5+ years of experience in AWS, Docker, and Kubernetes. "
    "The ideal candidate has experience with microservices architecture and CI/CD pipelines.",

    "Join our team as a Senior Frontend Developer. You will work with React, TypeScript, and modern "
    "web technologies. Experience with state management and testing frameworks required.",

    "Data Engineer position requiring expertise in Python, SQL, and cloud data platforms. "
    "Experience with Apache Spark, Airflow, and data warehousing is a plus.",

    "DevOps Engineer needed for a fast-growing startup. Must have strong experience with "
    "Terraform, Ansible, and cloud infrastructure automation.",

    "Machine Learning Engineer to develop and deploy ML models at scale. "
    "Required: Python, TensorFlow/PyTorch, and experience with ML pipelines.",
]


class SearchTaskSet(TaskSet):
    """Task set for search-related endpoints."""

    @task(10)
    def hybrid_search(self):
        """Test hybrid search with random query."""
        query = random.choice(SAMPLE_QUERIES)
        payload = {
            "query_text": query,
            "top_k": random.choice([10, 20, 50]),
            "bm25_weight": 0.5,
            "semantic_weight": 0.5,
            "use_rrf": True,
            "use_reranker": random.choice([True, False]),
        }
        with self.client.post(
            "/api/search/hybrid",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "results" in data:
                    response.success()
                else:
                    response.failure("Missing results in response")
            elif response.status_code == 401:
                response.failure("Authentication required")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(5)
    def hybrid_search_with_skills_filter(self):
        """Test hybrid search with skills filter."""
        query = random.choice(SAMPLE_QUERIES)
        skills = random.sample(SAMPLE_SKILLS, k=random.randint(1, 3))
        payload = {
            "query_text": query,
            "top_k": 20,
            "bm25_weight": 0.5,
            "semantic_weight": 0.5,
            "use_rrf": True,
            "use_reranker": True,
            "required_skills": skills,
        }
        with self.client.post(
            "/api/search/hybrid",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(2)
    def search_status(self):
        """Test search status endpoint."""
        self.client.get("/api/search/status")


class SkillsTaskSet(TaskSet):
    """Task set for skills-related endpoints."""

    @task(10)
    def search_skills(self):
        """Test ESCO skills search."""
        query = random.choice(SAMPLE_SKILLS)
        with self.client.get(
            f"/skills/search?q={query}&limit=20",
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    response.success()
                else:
                    response.failure("Expected list response")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(5)
    def extract_skills(self):
        """Test skill extraction from text."""
        text = random.choice(SAMPLE_JOB_DESCRIPTIONS)
        payload = {"text": text}
        with self.client.post(
            "/skills/extract",
            json=payload,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "skills" in data and "count" in data:
                    response.success()
                else:
                    response.failure("Missing skills data")
            else:
                response.failure(f"HTTP {response.status_code}")

    @task(3)
    def infer_skills(self):
        """Test skill inference."""
        skills = ",".join(random.sample(SAMPLE_SKILLS, k=random.randint(2, 4)))
        self.client.get(f"/skills/infer?skills={skills}&include_related=true")


class JobsTaskSet(TaskSet):
    """Task set for jobs-related endpoints."""

    @task(10)
    def list_jobs(self):
        """Test job listing."""
        page = random.randint(1, 3)
        per_page = random.choice([10, 20, 50])
        self.client.get(f"/jobs?page={page}&per_page={per_page}")

    @task(5)
    def list_jobs_with_filter(self):
        """Test job listing with filters."""
        params = {
            "page": 1,
            "per_page": 20,
            "score_min": random.choice([40, 50, 60, 70]),
        }
        if random.random() > 0.5:
            params["status"] = random.choice(["new", "saved"])
        if random.random() > 0.7:
            params["search"] = random.choice(["Python", "AWS", "React"])

        self.client.get("/jobs", params=params)

    @task(3)
    def list_jobs_high_score(self):
        """Test high-score job listing (common use case)."""
        self.client.get("/jobs?page=1&per_page=20&score_min=70")


class MLEndpointUser(HttpUser):
    """
    Simulated user for ML endpoint testing.

    Combines all task sets with weighted distribution.
    Simulates realistic user behavior with think time between requests.
    """

    wait_time = between(1, 3)  # Wait 1-3 seconds between tasks

    tasks = {
        SearchTaskSet: 3,   # 3x weight for search tasks
        SkillsTaskSet: 2,   # 2x weight for skills tasks
        JobsTaskSet: 2,     # 2x weight for jobs tasks
    }

    def on_start(self):
        """Called when a user starts (login if needed)."""
        # Note: Add authentication here if required
        # Example:
        # response = self.client.post("/auth/login", json={"password": "test"})
        # if response.ok:
        #     self.token = response.json().get("access_token")
        #     self.client.headers.update({"Authorization": f"Bearer {self.token}"})
        pass


class SearchFocusedUser(HttpUser):
    """User focused primarily on search operations (power user)."""

    wait_time = between(0.5, 2)  # Faster operations

    @task(10)
    def rapid_hybrid_search(self):
        """Rapid hybrid search testing."""
        query = random.choice(SAMPLE_QUERIES)
        payload = {
            "query_text": query,
            "top_k": 20,
            "bm25_weight": 0.5,
            "semantic_weight": 0.5,
            "use_rrf": True,
            "use_reranker": False,  # Faster without reranker
        }
        self.client.post("/api/search/hybrid", json=payload)

    @task(5)
    def hybrid_search_with_reranker(self):
        """Hybrid search with reranker (slower but more accurate)."""
        query = random.choice(SAMPLE_QUERIES)
        payload = {
            "query_text": query,
            "top_k": 20,
            "bm25_weight": 0.5,
            "semantic_weight": 0.5,
            "use_rrf": True,
            "use_reranker": True,
        }
        self.client.post("/api/search/hybrid", json=payload)

    @task(2)
    def check_status(self):
        """Check search status."""
        self.client.get("/api/search/status")


class SkillAnalysisUser(HttpUser):
    """User focused on skill analysis (recruiter/analyst persona)."""

    wait_time = between(2, 5)  # Slower, more thoughtful operations

    @task(5)
    def analyze_job_description(self):
        """Extract skills from job description."""
        text = random.choice(SAMPLE_JOB_DESCRIPTIONS)
        self.client.post("/skills/extract", json={"text": text})

    @task(3)
    def search_specific_skill(self):
        """Search for specific skill."""
        skill = random.choice(SAMPLE_SKILLS)
        self.client.get(f"/skills/search?q={skill}")

    @task(2)
    def get_skill_gaps(self):
        """Get skill gaps (requires auth in real usage)."""
        self.client.get("/skills/gaps?limit=10")

    @task(2)
    def get_recommendations(self):
        """Get learning recommendations."""
        self.client.get("/skills/recommendations?max_skills=5")


# Stress test configuration
class StressTestUser(HttpUser):
    """
    High-load stress test user.

    Use with caution - designed to push system limits.
    Run with: locust -f locustfile.py --headless -u 50 -r 10 -t 30s StressTestUser
    """

    wait_time = between(0.1, 0.5)  # Very fast

    @task
    def stress_hybrid_search(self):
        """Stress test hybrid search."""
        payload = {
            "query_text": random.choice(SAMPLE_QUERIES),
            "top_k": 50,
            "use_reranker": False,
        }
        self.client.post("/api/search/hybrid", json=payload)

    @task
    def stress_jobs_list(self):
        """Stress test jobs listing."""
        self.client.get("/jobs?page=1&per_page=100")

    @task
    def stress_skills_search(self):
        """Stress test skills search."""
        self.client.get(f"/skills/search?q={random.choice(SAMPLE_SKILLS)}")
