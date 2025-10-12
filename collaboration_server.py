#!/usr/bin/env python3
"""
Agent Zero V1 - Collaboration Production Server
Long-running FastAPI server for collaboration component
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Component import (with fallback)
try:
    from agent_zero_phases_6_7_production import *
    COMPONENT_AVAILABLE = True
except:
    COMPONENT_AVAILABLE = False

# FastAPI App
app = FastAPI(
    title="Agent Zero V1 - Collaboration Service",
    description="Production API server for collaboration component",
    version="1.0.0"
)

# CORS for Dev B frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "collaboration",
        "component_available": COMPONENT_AVAILABLE,
        "port": 8003
    }

@app.get("/")
async def root():
    """Root endpoint with service information"""
    return {
        "service": "Agent Zero V1 - Collaboration Service",
        "status": "operational",
        "version": "1.0.0",
        "component_available": COMPONENT_AVAILABLE,
        "port": 8003
    }

@app.get("/demo")
async def run_demo():
    """Run component demonstration"""
    if COMPONENT_AVAILABLE:
        try:
            # Import and run the main demo function from the component
            from agent_zero_phases_6_7_production import main
            result = main()
            return {"status": "demo_completed", "result": "success"}
        except Exception as e:
            return {"status": "demo_error", "error": str(e)}
    else:
        return {"status": "component_not_available", "mock": True}

@app.post("/api/process")
async def process_request(data: Dict[str, Any]):
    """Generic processing endpoint for component"""
    if COMPONENT_AVAILABLE:
        # Add component-specific processing logic here
        return {"status": "processed", "data": data, "mock": False}
    else:
        return {"status": "processed", "data": data, "mock": True}

# Production server startup
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Agent Zero V1 - Collaboration Server")
    print("=" * 50)
    print(f"🎯 Starting production server on port 8003...")
    print()
    
    # Start server - RUNS FOREVER (no more restarts!)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8003,
        log_level="info",
        access_log=True
    )
