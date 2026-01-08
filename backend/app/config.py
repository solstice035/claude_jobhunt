from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    database_url: str = "sqlite:///./data/jobs.db"
    openai_api_key: str = ""
    adzuna_app_id: str = ""
    adzuna_api_key: str = ""
    app_password: str = "changeme"
    secret_key: str = "dev-secret-key-change-in-production"
    scrape_interval_hours: int = 6

    # Redis configuration
    redis_url: str = "redis://localhost:6379"

    # Celery configuration
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"

    # ChromaDB configuration
    chroma_persist_directory: str = "./data/chroma_db"

    # Phase 3: Advanced ML Settings
    # Cohere API for re-ranking (optional)
    cohere_api_key: str = ""

    # Embedding provider: "openai" or "local"
    embedding_provider: str = "openai"
    # Local embedding model (when provider=local)
    local_embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"

    # Re-ranker provider: "local" or "cohere"
    reranker_provider: str = "local"
    # Local cross-encoder model
    local_reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"

    # Hybrid search settings
    hybrid_bm25_weight: float = 0.5
    hybrid_semantic_weight: float = 0.5
    hybrid_use_rrf: bool = True  # Use Reciprocal Rank Fusion

    # Two-stage retrieval settings
    retrieval_candidates: int = 200  # Stage 1: retrieve this many
    rerank_top_k: int = 50  # Stage 2: rerank to this many

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
