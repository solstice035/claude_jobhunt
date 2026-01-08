"""
Enhanced Search API - Phase 3 Advanced ML Endpoints

This module provides API endpoints for the enhanced job matching system:
- Hybrid search (BM25 + semantic with RRF fusion)
- Two-stage retrieval (fast recall + precise re-ranking)
- Configurable search parameters

Endpoints:
    POST /api/search/hybrid - Hybrid search with configurable weights
    POST /api/search/rerank - Re-rank existing results
    GET /api/search/status - Search service health status
"""

import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.database import get_db
from app.models import Job, Profile
from app.config import get_settings
from app.services.hybrid_search import HybridSearchService
from app.services.reranker import get_reranker, rerank_jobs

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])
settings = get_settings()

# Global search service instance (lazily initialized)
_hybrid_search_service: Optional[HybridSearchService] = None


class HybridSearchRequest(BaseModel):
    """Request body for hybrid search."""

    query_text: str = Field(
        ...,
        description="Text query for BM25 keyword search",
        min_length=1,
        max_length=5000
    )
    query_embedding: Optional[List[float]] = Field(
        None,
        description="Pre-computed embedding for semantic search"
    )
    top_k: int = Field(
        50,
        ge=1,
        le=200,
        description="Maximum number of results"
    )
    bm25_weight: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Weight for BM25 keyword search"
    )
    semantic_weight: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Weight for semantic embedding search"
    )
    use_rrf: bool = Field(
        True,
        description="Use Reciprocal Rank Fusion for combining results"
    )
    use_reranker: bool = Field(
        True,
        description="Apply cross-encoder re-ranking to results"
    )
    required_skills: Optional[List[str]] = Field(
        None,
        description="Skills that must appear in job description"
    )


class HybridSearchResult(BaseModel):
    """Individual search result."""

    job_id: str
    title: str
    company: str
    location: str
    match_score: float
    hybrid_score: float
    rerank_score: Optional[float] = None
    description_preview: str


class HybridSearchResponse(BaseModel):
    """Response for hybrid search."""

    results: List[HybridSearchResult]
    total: int
    query_text: str
    search_config: dict


class RerankRequest(BaseModel):
    """Request for re-ranking existing results."""

    query: str = Field(
        ...,
        description="Query for re-ranking (e.g., CV text)",
        min_length=1,
        max_length=10000
    )
    job_ids: List[str] = Field(
        ...,
        description="Job IDs to re-rank",
        min_items=1,
        max_items=200
    )
    top_k: int = Field(
        20,
        ge=1,
        le=100,
        description="Maximum results after re-ranking"
    )
    provider: str = Field(
        "local",
        description="Re-ranker provider: 'local' or 'cohere'"
    )


class RerankResult(BaseModel):
    """Individual re-rank result."""

    job_id: str
    title: str
    company: str
    relevance_score: float


class RerankResponse(BaseModel):
    """Response for re-rank endpoint."""

    results: List[RerankResult]
    total: int
    provider_used: str


class SearchStatusResponse(BaseModel):
    """Response for search status endpoint."""

    status: str
    bm25_index_size: int
    embedding_count: int
    hybrid_search_available: bool
    reranker_available: bool
    config: dict


async def get_or_build_search_service(db: AsyncSession) -> HybridSearchService:
    """
    Get or build the hybrid search service.

    Lazily builds the BM25 index from database on first call.
    """
    global _hybrid_search_service

    if _hybrid_search_service is None:
        _hybrid_search_service = HybridSearchService()

        # Load all jobs from database
        result = await db.execute(
            select(Job).where(Job.status != "archived")
        )
        jobs = result.scalars().all()

        # Build index
        job_docs = []
        for job in jobs:
            job_docs.append({
                "id": job.id,
                "title": job.title,
                "description": job.description,
                "embedding": job.embedding,
            })

        _hybrid_search_service.build_index(job_docs)
        logger.info(f"Built hybrid search index with {len(job_docs)} jobs")

    return _hybrid_search_service


def invalidate_search_index():
    """Invalidate the search index (call after job updates)."""
    global _hybrid_search_service
    _hybrid_search_service = None
    logger.info("Search index invalidated")


@router.post("/hybrid", response_model=HybridSearchResponse)
async def hybrid_search(
    request: HybridSearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Perform hybrid search combining BM25 and semantic search.

    This endpoint implements two-stage retrieval:
    1. Fast hybrid search (BM25 + embeddings with RRF)
    2. Optional cross-encoder re-ranking for precision

    Returns jobs ranked by combined relevance score.
    """
    try:
        # Get search service
        search_service = await get_or_build_search_service(db)

        # Stage 1: Hybrid search
        if request.required_skills:
            candidates = search_service.search_with_skill_filter(
                query_text=request.query_text,
                query_embedding=request.query_embedding,
                required_skills=request.required_skills,
                top_k=request.top_k * 2 if request.use_reranker else request.top_k
            )
        else:
            candidates = search_service.search(
                query_text=request.query_text,
                query_embedding=request.query_embedding,
                top_k=request.top_k * 2 if request.use_reranker else request.top_k,
                bm25_weight=request.bm25_weight,
                semantic_weight=request.semantic_weight,
                use_rrf=request.use_rrf
            )

        if not candidates:
            return HybridSearchResponse(
                results=[],
                total=0,
                query_text=request.query_text,
                search_config={
                    "bm25_weight": request.bm25_weight,
                    "semantic_weight": request.semantic_weight,
                    "use_rrf": request.use_rrf,
                    "use_reranker": request.use_reranker
                }
            )

        # Get job IDs from candidates
        candidate_ids = [c[0] for c in candidates]
        hybrid_scores = {c[0]: c[1] for c in candidates}

        # Load job details from database
        result = await db.execute(
            select(Job).where(Job.id.in_(candidate_ids))
        )
        jobs_map = {job.id: job for job in result.scalars().all()}

        # Stage 2: Optional re-ranking
        rerank_scores = {}
        if request.use_reranker and len(candidate_ids) > 0:
            try:
                # Prepare documents for re-ranking
                job_docs = []
                for job_id in candidate_ids:
                    if job_id in jobs_map:
                        job = jobs_map[job_id]
                        job_docs.append({
                            "id": job_id,
                            "title": job.title,
                            "description": job.description,
                            "company": job.company
                        })

                reranker = get_reranker(
                    provider=settings.reranker_provider,
                    api_key=settings.cohere_api_key if settings.reranker_provider == "cohere" else None,
                    lazy_load=True
                )

                reranked = await reranker.rerank_async(
                    query=request.query_text[:2000],
                    documents=[
                        {"id": d["id"], "text": f"{d['title']} at {d['company']}\n{d['description'][:1000]}"}
                        for d in job_docs
                    ],
                    top_k=request.top_k
                )

                rerank_scores = {r[0]: r[1] for r in reranked}
                # Reorder based on rerank scores
                candidate_ids = [r[0] for r in reranked]

            except Exception as e:
                logger.warning(f"Re-ranking failed, using hybrid scores: {e}")

        # Build response
        results = []
        for job_id in candidate_ids[:request.top_k]:
            if job_id not in jobs_map:
                continue

            job = jobs_map[job_id]
            results.append(HybridSearchResult(
                job_id=job.id,
                title=job.title,
                company=job.company,
                location=job.location,
                match_score=job.match_score,
                hybrid_score=hybrid_scores.get(job_id, 0.0),
                rerank_score=rerank_scores.get(job_id),
                description_preview=job.description[:300] + "..." if len(job.description) > 300 else job.description
            ))

        return HybridSearchResponse(
            results=results,
            total=len(results),
            query_text=request.query_text,
            search_config={
                "bm25_weight": request.bm25_weight,
                "semantic_weight": request.semantic_weight,
                "use_rrf": request.use_rrf,
                "use_reranker": request.use_reranker
            }
        )

    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rerank", response_model=RerankResponse)
async def rerank_results(
    request: RerankRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Re-rank existing job results using cross-encoder.

    Use this endpoint to improve precision on already-retrieved results.
    Accepts job IDs from any source (hybrid search, filters, etc.).
    """
    try:
        # Load jobs from database
        result = await db.execute(
            select(Job).where(Job.id.in_(request.job_ids))
        )
        jobs = result.scalars().all()

        if not jobs:
            return RerankResponse(
                results=[],
                total=0,
                provider_used=request.provider
            )

        # Re-rank
        reranked = await rerank_jobs(
            query=request.query,
            jobs=[{
                "id": job.id,
                "title": job.title,
                "description": job.description,
                "company": job.company
            } for job in jobs],
            top_k=request.top_k,
            provider=request.provider,
            api_key=settings.cohere_api_key if request.provider == "cohere" else None
        )

        # Build response
        results = []
        for job_dict, score in reranked:
            results.append(RerankResult(
                job_id=job_dict["id"],
                title=job_dict["title"],
                company=job_dict["company"],
                relevance_score=score
            ))

        return RerankResponse(
            results=results,
            total=len(results),
            provider_used=request.provider
        )

    except Exception as e:
        logger.error(f"Re-ranking failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=SearchStatusResponse)
async def search_status(db: AsyncSession = Depends(get_db)):
    """
    Get search service status and configuration.

    Returns information about index size, available features,
    and current configuration.
    """
    global _hybrid_search_service

    # Check if index exists
    index_size = 0
    embedding_count = 0

    if _hybrid_search_service:
        index_size = len(_hybrid_search_service.job_data)
        embedding_count = len(_hybrid_search_service.job_embeddings)

    # Check reranker availability
    reranker_available = False
    try:
        reranker = get_reranker(provider="local", lazy_load=True)
        reranker_available = True
    except Exception:
        pass

    return SearchStatusResponse(
        status="healthy",
        bm25_index_size=index_size,
        embedding_count=embedding_count,
        hybrid_search_available=_hybrid_search_service is not None,
        reranker_available=reranker_available,
        config={
            "embedding_provider": settings.embedding_provider,
            "reranker_provider": settings.reranker_provider,
            "hybrid_bm25_weight": settings.hybrid_bm25_weight,
            "hybrid_semantic_weight": settings.hybrid_semantic_weight,
            "hybrid_use_rrf": settings.hybrid_use_rrf,
            "retrieval_candidates": settings.retrieval_candidates,
            "rerank_top_k": settings.rerank_top_k
        }
    )


@router.post("/rebuild-index")
async def rebuild_search_index(db: AsyncSession = Depends(get_db)):
    """
    Rebuild the hybrid search index.

    Call this after bulk job updates to refresh the BM25 index.
    """
    invalidate_search_index()
    await get_or_build_search_service(db)

    return {"status": "success", "message": "Search index rebuilt"}
