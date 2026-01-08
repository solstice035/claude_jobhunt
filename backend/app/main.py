from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from app.database import init_db
from app.api import api_router
from app.scheduler import start_scheduler, stop_scheduler, fetch_and_process_jobs


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    start_scheduler()
    # Run initial fetch after startup (in background)
    asyncio.create_task(fetch_and_process_jobs())
    yield
    stop_scheduler()


app = FastAPI(
    title="Job Search Agent API",
    description="AI-powered job matching API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
