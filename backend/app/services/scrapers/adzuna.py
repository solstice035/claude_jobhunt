import httpx
import hashlib
from datetime import datetime
from typing import List, Optional
from app.services.scrapers.base import BaseScraper
from app.schemas import JobCreate
from app.config import get_settings

settings = get_settings()


class AdzunaScraper(BaseScraper):
    source = "adzuna"
    base_url = "https://api.adzuna.com/v1/api/jobs/gb/search"

    def __init__(self):
        self.app_id = settings.adzuna_app_id
        self.api_key = settings.adzuna_api_key

    async def fetch_jobs(
        self,
        search_query: str,
        location: str = "uk",
        results_per_page: int = 50,
        max_pages: int = 3,
    ) -> List[JobCreate]:
        jobs = []

        async with httpx.AsyncClient() as client:
            for page in range(1, max_pages + 1):
                params = {
                    "app_id": self.app_id,
                    "app_key": self.api_key,
                    "results_per_page": results_per_page,
                    "what": search_query,
                    "where": location,
                    "sort_by": "date",
                    "max_days_old": 30,
                }

                try:
                    response = await client.get(
                        f"{self.base_url}/{page}",
                        params=params,
                        timeout=30.0,
                    )
                    response.raise_for_status()
                    data = response.json()

                    for result in data.get("results", []):
                        job = self._parse_job(result)
                        if job:
                            jobs.append(job)

                    # Stop if we got fewer results than requested
                    if len(data.get("results", [])) < results_per_page:
                        break

                except httpx.HTTPError as e:
                    print(f"Adzuna API error on page {page}: {e}")
                    break

        return jobs

    def _parse_job(self, data: dict) -> Optional[JobCreate]:
        try:
            # Parse salary
            salary_min = None
            salary_max = None
            if data.get("salary_min"):
                salary_min = int(data["salary_min"])
            if data.get("salary_max"):
                salary_max = int(data["salary_max"])

            # Parse posted date
            posted_at = None
            if data.get("created"):
                posted_at = datetime.fromisoformat(data["created"].replace("Z", "+00:00"))

            return JobCreate(
                title=data.get("title", "Unknown Title"),
                company=data.get("company", {}).get("display_name", "Unknown Company"),
                location=data.get("location", {}).get("display_name", "UK"),
                salary_min=salary_min,
                salary_max=salary_max,
                description=data.get("description", ""),
                url=data.get("redirect_url", ""),
                source=self.source,
                posted_at=posted_at,
            )
        except Exception as e:
            print(f"Error parsing Adzuna job: {e}")
            return None


def generate_url_hash(url: str) -> str:
    """Generate a hash of the URL for deduplication"""
    return hashlib.sha256(url.encode()).hexdigest()
