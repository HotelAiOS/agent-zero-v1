#!/usr/bin/env python3
"""
Agent Zero V1 Main Entry Point
"""
import asyncio
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from api.api_server import app
from database.neo4j_connector import Neo4jConnector
from agents.agent_manager import AgentManager
from intelligence.intelligence_layer import IntelligenceLayer

def setup_agent_zero():
    """Initialize Agent Zero V1 components"""
    print("🚀 Starting Agent Zero V1...")
    
    # Initialize components
    db = Neo4jConnector()
    agent_manager = AgentManager() 
    intelligence = IntelligenceLayer()
    
    print("✅ Agent Zero V1 initialized successfully!")
    return {"db": db, "agents": agent_manager, "intelligence": intelligence}

if __name__ == "__main__":
    import uvicorn
    components = setup_agent_zero()
    print("🌐 Starting API server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
