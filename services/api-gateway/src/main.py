from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
import logging
import os

from .routes import ai, chat, agents, docs
from .middleware.rate_limit import limiter

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Agent Zero API Gateway",
    version="1.0.0",
    description="Centralna brama API dla wszystkich serwisów Agent Zero"
)

# Rate limiter
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded. Try again later."}
    )

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(ai.router)
app.include_router(chat.router)
app.include_router(agents.router)
app.include_router(docs.router)

@app.get("/")
@limiter.limit("10/minute")
async def root(request: Request):
    """Root endpoint"""
    return {
        "message": "Agent Zero API Gateway v1.0.0",
        "services": {
            "ai": "/ai",
            "chat": "/chat",
            "agents": "/agents",
            "docs": "/docs-api"
        },
        "docs": "/docs"
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "gateway": "api-gateway"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
