"""
Skill Extractor Service - LLM-Powered Skill Extraction

This service uses OpenAI's GPT models to extract skills from text
(job descriptions, CVs, etc.) with high accuracy and context awareness.

Key Features:
- Extracts skills with category classification (technical, soft, domain, tool)
- Distinguishes required vs preferred skills
- Provides confidence levels for each extraction
- Caches results to minimize API costs
- Falls back to regex-based extraction if LLM fails

Cost Optimization:
- Uses GPT-4o-mini by default (~$0.15 per 1M input tokens)
- Caches results using text hash as key
- Batch processing support for multiple texts

Usage:
    from app.services.skill_extractor import SkillExtractor
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    extractor = SkillExtractor(openai_client=client)
    skills = await extractor.extract_skills(job_description)
"""

import json
import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Any, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ExtractedSkill:
    """
    A skill extracted from text with metadata.

    Attributes:
        name: Normalized skill name
        category: Classification (technical, soft, domain, tool)
        required: Whether the skill is required (True) or preferred (False)
        confidence: Extraction confidence (high, medium, low)
    """
    name: str
    category: str
    required: bool
    confidence: str

    def __hash__(self) -> int:
        return hash((self.name.lower(), self.category))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExtractedSkill):
            return False
        return self.name.lower() == other.name.lower() and self.category == other.category


class CacheProtocol(Protocol):
    """Protocol for cache implementations."""

    def get(self, key: str) -> Optional[str]:
        """Get value from cache."""
        ...

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        ...


class InMemoryCache:
    """Simple in-memory cache for development/testing."""

    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    def get(self, key: str) -> Optional[str]:
        return self._cache.get(key)

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> None:
        self._cache[key] = value


# Extraction prompt optimized for GPT-4o-mini
SKILL_EXTRACTION_PROMPT = """Extract all skills, technologies, and competencies from this text.

For each skill found, provide:
1. The skill name (normalized to standard form, e.g., "Kubernetes" not "k8s")
2. Category: "technical" (programming, tools, platforms), "soft" (communication, leadership), "domain" (industry knowledge), or "tool" (specific software/tools)
3. Whether it's required (true) or preferred/nice-to-have (false)
4. Confidence: "high" (explicitly mentioned), "medium" (implied), "low" (uncertain)

Text:
{text}

Return ONLY valid JSON in this exact format:
{{
  "skills": [
    {{"name": "Python", "category": "technical", "required": true, "confidence": "high"}},
    {{"name": "team leadership", "category": "soft", "required": true, "confidence": "high"}}
  ]
}}"""


class SkillExtractor:
    """
    LLM-powered skill extraction service.

    Uses OpenAI's GPT models to intelligently extract skills from text,
    understanding context and distinguishing between similar concepts.

    Attributes:
        openai_client: Async OpenAI client for API calls
        model: GPT model to use (default: gpt-4o-mini)
        cache: Optional cache for storing extraction results
    """

    def __init__(
        self,
        openai_client: Any,
        model: str = "gpt-4o-mini",
        cache: Optional[CacheProtocol] = None
    ):
        """
        Initialize the skill extractor.

        Args:
            openai_client: Configured OpenAI async client
            model: Model to use for extraction (default: gpt-4o-mini)
            cache: Optional cache implementation for result caching
        """
        self.client = openai_client
        self.model = model
        self.cache = cache or InMemoryCache()

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key from text hash."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
        return f"skills:{text_hash}"

    async def extract_skills(
        self,
        text: str,
        use_cache: bool = True
    ) -> List[ExtractedSkill]:
        """
        Extract skills from text using LLM.

        Args:
            text: Text to extract skills from (job description, CV, etc.)
            use_cache: Whether to use cached results (default: True)

        Returns:
            List of ExtractedSkill objects

        Raises:
            No exceptions raised - returns empty list on failure

        Example:
            >>> skills = await extractor.extract_skills("Python developer needed")
            >>> skills[0].name
            'Python'
        """
        if not text or not text.strip():
            return []

        # Check cache first
        cache_key = self._get_cache_key(text)
        if use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                try:
                    return self._parse_cached_skills(cached)
                except Exception:
                    pass  # Cache corrupted, re-extract

        # Call LLM for extraction
        try:
            skills = await self._call_llm(text)

            # Cache the result
            if use_cache and skills:
                cache_value = json.dumps([s.__dict__ for s in skills])
                self.cache.set(cache_key, cache_value)

            return skills

        except Exception as e:
            logger.error(f"Skill extraction failed: {e}")
            return []

    async def _call_llm(self, text: str) -> List[ExtractedSkill]:
        """
        Call OpenAI API for skill extraction.

        Args:
            text: Text to analyze

        Returns:
            List of ExtractedSkill objects
        """
        prompt = SKILL_EXTRACTION_PROMPT.format(text=text[:4000])  # Limit text length

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a skill extraction specialist. Extract skills accurately and return only valid JSON."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=1000
            )

            content = response.choices[0].message.content
            return self._parse_llm_response(content)

        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            raise

    def _parse_llm_response(self, content: str) -> List[ExtractedSkill]:
        """
        Parse LLM response into ExtractedSkill objects.

        Args:
            content: Raw response content from LLM

        Returns:
            List of ExtractedSkill objects
        """
        if not content:
            return []

        try:
            # Try to extract JSON from response
            content = content.strip()

            # Handle markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])

            data = json.loads(content)
            skills_data = data.get("skills", [])

            skills = []
            for item in skills_data:
                try:
                    skill = ExtractedSkill(
                        name=item.get("name", "").strip(),
                        category=item.get("category", "technical"),
                        required=item.get("required", True),
                        confidence=item.get("confidence", "medium")
                    )
                    if skill.name:  # Only add if name is not empty
                        skills.append(skill)
                except Exception:
                    continue

            return skills

        except json.JSONDecodeError:
            logger.warning(f"Failed to parse LLM response as JSON: {content[:100]}")
            return []

    def _parse_cached_skills(self, cached: str) -> List[ExtractedSkill]:
        """Parse cached skill data."""
        data = json.loads(cached)
        return [ExtractedSkill(**item) for item in data]

    async def extract_skills_batch(
        self,
        texts: List[str],
        use_cache: bool = True
    ) -> List[List[ExtractedSkill]]:
        """
        Extract skills from multiple texts.

        Args:
            texts: List of texts to extract skills from
            use_cache: Whether to use cached results

        Returns:
            List of skill lists (one per input text)
        """
        results = []
        for text in texts:
            skills = await self.extract_skills(text, use_cache=use_cache)
            results.append(skills)
        return results


# Fallback regex-based extraction using the existing taxonomy
def extract_skills_fallback(text: str) -> List[ExtractedSkill]:
    """
    Fallback skill extraction using regex and taxonomy.

    Used when LLM extraction fails or for cost optimization
    on bulk processing.

    Args:
        text: Text to extract skills from

    Returns:
        List of ExtractedSkill objects
    """
    from app.services.matcher import extract_skills_with_synonyms

    found_skills = extract_skills_with_synonyms(text)
    skills = []

    for category, skill_names in found_skills.items():
        for skill_name in skill_names:
            skills.append(ExtractedSkill(
                name=skill_name,
                category="technical" if category in [
                    "languages", "frontend", "backend", "cloud", "data", "ai_ml"
                ] else "soft",
                required=True,  # Cannot determine from regex
                confidence="medium"
            ))

    return skills


# ==============================================================================
# Singleton Pattern for Dependency Injection
# ==============================================================================

_skill_extractor: Optional[SkillExtractor] = None


def get_skill_extractor() -> SkillExtractor:
    """
    Get shared SkillExtractor instance (singleton pattern).

    Creates a single instance of AsyncOpenAI client and SkillExtractor
    that is reused across all API requests. This ensures the InMemoryCache
    persists between requests, making caching effective.

    Returns:
        SkillExtractor: Shared instance with persistent cache
    """
    global _skill_extractor
    if _skill_extractor is None:
        from openai import AsyncOpenAI
        from app.config import get_settings

        settings = get_settings()
        client = AsyncOpenAI(api_key=settings.openai_api_key)
        _skill_extractor = SkillExtractor(openai_client=client)
        logger.info("Created singleton SkillExtractor instance with shared cache")
    return _skill_extractor
