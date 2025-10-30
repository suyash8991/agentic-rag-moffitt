from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import logging

from app.core.config import settings
from app.core.security import get_api_key

# Import routers
from app.api.endpoints import researchers, query

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Moffitt Agentic RAG API",
    description="API for the Moffitt Cancer Center Researcher Assistant",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LangSmith tracing if enabled
if settings.LANGCHAIN_TRACING_V2:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY or ""
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    logger.info(f"✓ LangSmith tracing enabled for project: {settings.LANGCHAIN_PROJECT}")
else:
    logger.info("LangSmith tracing disabled (set LANGCHAIN_TRACING_V2=true to enable)")

@app.get("/api/health", tags=["health"])
async def health_check():
    """
    Health check endpoint - no authentication required
    """
    return {
        "status": "healthy",
        "version": "1.0.0",
        "provider": settings.LLM_PROVIDER
    }


@app.get("/api/settings", tags=["settings"])
async def get_settings(api_key: str = Depends(get_api_key)):
    """
    Get API settings - requires authentication
    """
    return {
        "provider": settings.LLM_PROVIDER,
        "model": getattr(settings, f"{settings.LLM_PROVIDER.upper()}_MODEL", settings.LLM_MODEL_NAME),
        "embedding_model": settings.EMBEDDING_MODEL_NAME,
    }

# Include routers
app.include_router(researchers.router, prefix="/api", tags=["researchers"])
app.include_router(query.router, prefix="/api", tags=["query"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)