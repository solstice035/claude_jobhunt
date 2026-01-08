from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, sessionmaker, Session
from app.config import get_settings

settings = get_settings()

# Convert sqlite:/// to sqlite+aiosqlite:///
database_url = settings.database_url.replace("sqlite:///", "sqlite+aiosqlite:///")

engine = create_async_engine(database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Synchronous engine for Celery tasks
sync_database_url = settings.database_url
sync_engine = create_engine(sync_database_url, echo=False)
SyncSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        yield session


def get_db_session() -> Session:
    """
    Get synchronous database session for Celery tasks.

    Returns:
        SQLAlchemy Session (caller must close)
    """
    return SyncSessionLocal()


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
