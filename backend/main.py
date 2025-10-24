from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.api.endpoints import researchers, query, admin
from app.core.config import settings

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

# Include routers
app.include_router(researchers.router, prefix="/api", tags=["researchers"])
app.include_router(query.router, prefix="/api", tags=["query"])
app.include_router(admin.router, prefix="/api/admin", tags=["admin"])

@app.get("/api/health", tags=["health"])
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)