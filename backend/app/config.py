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

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
