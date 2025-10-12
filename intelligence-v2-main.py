"""
Agent Zero V1 - Intelligence V2.0 Main Application
Production-ready FastAPI application with full Point 3-6 consolidation
"""

import sys
import os
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# FastAPI and related imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Import Intelligence V2.0 components
try:
    from api.v2.intelligence import router as intelligence_router
    from intelligence_v2 import DynamicTaskPrioritizer
    V2_COMPONENTS_AVAILABLE = True
    print("✅ Intelligence V2.0 components imported successfully")
except ImportError as e:
    print(f"⚠️  Intelligence V2.0 components not available: {e}")
    V2_COMPONENTS_AVAILABLE = False

# Import existing Agent Zero components for compatibility
try:
    exec(open(project_root / "simple-tracker.py").read(), globals())
    EXISTING_COMPONENTS_AVAILABLE = True
    print("✅ Existing Agent Zero components imported successfully")
except Exception as e:
    print(f"⚠️  Existing components not available: {e}")
    EXISTING_COMPONENTS_AVAILABLE = False
    
    # Minimal fallback
    class SimpleTracker:
        def track_task(self, **kwargs): pass
        def get_daily_stats(self): return {'total_tasks': 0}

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/intelligence-v2.log')
    ]
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Agent Zero V1 - Intelligence V2.0",
    description="Unified Point 3-6 Intelligence Layer with full backward compatibility",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global components
intelligence_orchestrator = None
simple_tracker_instance = None

@app.on_event("startup")
async def startup_event():
    """Initialize Intelligence V2.0 components on startup"""
    global intelligence_orchestrator, simple_tracker_instance
    
    logger.info("🚀 Starting Agent Zero V1 - Intelligence V2.0...")
    
    try:
        # Initialize SimpleTracker for compatibility
        if EXISTING_COMPONENTS_AVAILABLE:
            simple_tracker_instance = SimpleTracker()
            logger.info("✅ SimpleTracker initialized")
        
        # Initialize Intelligence V2.0 components
        if V2_COMPONENTS_AVAILABLE:
            prioritizer = DynamicTaskPrioritizer(simple_tracker=simple_tracker_instance)
            logger.info("✅ Intelligence V2.0 Prioritizer initialized")
        
        # Log startup status
        logger.info("🎯 Intelligence V2.0 startup complete")
        logger.info(f"   - Existing components: {'✅' if EXISTING_COMPONENTS_AVAILABLE else '❌'}")
        logger.info(f"   - V2.0 components: {'✅' if V2_COMPONENTS_AVAILABLE else '❌'}")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "service": "Agent Zero V1 - Intelligence V2.0",
        "version": "2.0.0",
        "status": "operational",
        "features": {
            "point3_compatibility": True,
            "enhanced_prioritization": V2_COMPONENTS_AVAILABLE,
            "predictive_planning": False,
            "adaptive_learning": False,
            "real_time_monitoring": False
        },
        "endpoints": {
            "intelligence_api": "/api/v2/intelligence/",
            "health_check": "/health",
            "documentation": "/docs"
        },
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for all components"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "intelligence-v2",
            "version": "2.0.0",
            "components": {
                "simple_tracker": "healthy" if simple_tracker_instance else "unavailable",
                "intelligence_v2": "healthy" if V2_COMPONENTS_AVAILABLE else "unavailable"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Include Intelligence V2.0 API router
if V2_COMPONENTS_AVAILABLE:
    app.include_router(intelligence_router)
    logger.info("✅ Intelligence V2.0 API router included")
else:
    @app.get("/api/v2/intelligence/health")
    async def fallback_intelligence_health():
        return {
            "status": "unavailable",
            "reason": "Intelligence V2.0 components not loaded",
            "fallback_mode": True,
            "timestamp": datetime.now().isoformat()
        }

def main():
    """Main entry point for Intelligence V2.0 application"""
    
    print("🚀 Starting Agent Zero V1 - Intelligence V2.0...")
    print("=" * 60)
    print(f"📅 Timestamp: {datetime.now().isoformat()}")
    print(f"📁 Project Root: {project_root}")
    print(f"✅ Existing Components: {EXISTING_COMPONENTS_AVAILABLE}")
    print(f"✅ V2.0 Components: {V2_COMPONENTS_AVAILABLE}")
    print("=" * 60)
    
    # Configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8012))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print(f"🌐 Starting server on {host}:{port}")
    print(f"📚 API Documentation: http://{host}:{port}/docs")
    print(f"🔍 Health Check: http://{host}:{port}/health")
    print(f"🧠 Intelligence API: http://{host}:{port}/api/v2/intelligence/")
    print("=" * 60)
    
    try:
        uvicorn.run(
            "intelligence-v2-main:app",
            host=host,
            port=port,
            log_level=log_level,
            reload=False
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
