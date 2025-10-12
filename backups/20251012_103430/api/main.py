"""
FastAPI Main Application
Agent Zero REST API
"""

import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'shared'))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import logging

from persistence import init_database, get_database, get_cache
from core import AgentZeroCore, ProjectManager
from api import dependencies

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create FastAPI application"""
    
    app = FastAPI(
        title="Agent Zero API",
        description="REST API for Agent Zero - Autonomous AI Agent Platform",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize on startup
    @app.on_event("startup")
    async def startup():
        logger.info("🚀 Starting Agent Zero API...")
        
        # Initialize database
        init_database(db_url='sqlite:///agent_zero.db', echo=False)
        logger.info("✓ Database initialized")
        
        # Initialize core engine
        core = AgentZeroCore()
        dependencies.set_core_engine(core)
        logger.info("✓ Core Engine initialized")
        
        # Initialize project manager
        pm = ProjectManager(core)
        dependencies.set_project_manager(pm)
        logger.info("✓ Project Manager initialized")
        
        logger.info("✅ Agent Zero API Ready!")
    
    @app.on_event("shutdown")
    async def shutdown():
        logger.info("👋 Shutting down Agent Zero API...")
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        db = get_database()
        cache = get_cache()
        
        return {
            "status": "ok",
            "database": db.health_check(),
            "cache": True,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "Agent Zero API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/health"
        }
    
    # Include routers
    from api.routes import projects, agents, tasks, protocols, system
    
    app.include_router(projects.router, prefix="/api/v1/projects", tags=["Projects"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["Agents"])
    app.include_router(tasks.router, prefix="/api/v1/tasks", tags=["Tasks"])
    app.include_router(protocols.router, prefix="/api/v1/protocols", tags=["Protocols"])
    app.include_router(system.router, prefix="/api/v1/system", tags=["System"])
    
    # Error handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": exc.detail,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    
    return app


# Create app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
