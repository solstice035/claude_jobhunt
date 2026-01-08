from abc import ABC, abstractmethod
from typing import List
from app.schemas import JobCreate


class BaseScraper(ABC):
    """Base class for job scrapers"""

    source: str = "unknown"

    @abstractmethod
    async def fetch_jobs(self, search_query: str, location: str = "uk") -> List[JobCreate]:
        """Fetch jobs from the source"""
        pass
