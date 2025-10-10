"""
Agent Orchestrator Service - Main Application
Coordinates multiple AI agents for different tasks
"""
import logging
import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from swarm.coordinator import AgentCoordinator
from models.schemas import TaskRequest, TaskResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Agent Orchestrator",
    description="AI Agent coordination and task execution service",
    version="1.0.2"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Agent Coordinator
coordinator = AgentCoordinator()

def get_db_connection():
    """Get PostgreSQL database connection."""
    db_url = os.getenv('DATABASE_URL', 'postgresql://a0:a0dev@postgresql:5432/agentzero')
    try:
        return psycopg2.connect(db_url)
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "agent-orchestrator",
        "version": "1.0.2",
        "agents": len(coordinator.agents)
    }

@app.post("/agents/task")
async def execute_task(task: TaskRequest):
    """Execute task using appropriate agent."""
    try:
        logger.info(f"Received task: {task.task_type} - {task.description[:50]}...")
        
        # Execute task through coordinator
        result = await coordinator.execute_task(
            task_type=task.task_type,
            description=task.description,
            context=task.context or {}
        )
        
        return TaskResponse(
            task_id=str(uuid.uuid4()),
            status="completed",
            result=result,
            agent_used=task.task_type
        )
    except Exception as e:
        logger.error(f"Task execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/custom")
async def list_custom_agents():
    """List all custom agents from database."""
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        
        cur.execute("""
            SELECT id, name, description, system_prompt, 
                   capabilities, model_preference, is_active, 
                   created_at, updated_at
            FROM custom_agents
            WHERE is_active = true
            ORDER BY created_at DESC
        """)
        
        agents = cur.fetchall()
        cur.close()
        conn.close()
        
        # Convert to list of dicts
        agents_list = [dict(agent) for agent in agents]
        
        logger.info(f"Retrieved {len(agents_list)} custom agents")
        return {"custom_agents": agents_list}
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/create")
async def create_custom_agent(request: Dict[str, Any] = Body(...)):
    """
    Create new custom agent.
    
    Expected request body:
    {
        "name": "AgentName",
        "description": "Agent description",
        "prompt_template": "System prompt template",
        "model": "llama3.2:3b"  (optional)
    }
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Generate unique ID
        agent_id = str(uuid.uuid4())
        
        # Extract fields from request
        name = request.get('name')
        description = request.get('description')
        
        # MAP: prompt_template -> system_prompt (FIX for KeyError)
        system_prompt = request.get('prompt_template', request.get('system_prompt', ''))
        
        # Optional fields
        model_preference = request.get('model', 'llama3.2:3b')
        capabilities = request.get('capabilities', ['custom'])
        
        # Validate required fields
        if not name or not description:
            raise HTTPException(
                status_code=400, 
                detail="Missing required fields: name, description"
            )
        
        if not system_prompt:
            raise HTTPException(
                status_code=400,
                detail="Missing required field: prompt_template or system_prompt"
            )
        
        # Insert into database
        cur.execute("""
            INSERT INTO custom_agents (
                id, name, description, system_prompt, 
                capabilities, model_preference, is_active
            )
            VALUES (%s, %s, %s, %s, %s, %s, true)
            RETURNING id, name
        """, (
            agent_id,
            name,
            description,
            system_prompt,
            capabilities,
            model_preference
        ))
        
        conn.commit()
        result = cur.fetchone()
        cur.close()
        conn.close()
        
        logger.info(f"Created custom agent: {name} (ID: {agent_id})")
        
        return {
            "agent_id": result[0],
            "name": result[1],
            "status": "created",
            "message": f"Agent '{result[1]}' created successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/list")
async def list_available_agents():
    """List all available agent types."""
    return {
        "agents": [
            {
                "type": "code",
                "name": "Code Agent",
                "description": "Generates and analyzes code"
            },
            {
                "type": "test",
                "name": "Test Agent",
                "description": "Creates test cases and validates code"
            },
            {
                "type": "docs",
                "name": "Documentation Agent",
                "description": "Generates documentation"
            }
        ]
    }

@app.get("/agents/status")
async def get_agents_status():
    """Get status of all agents."""
    return {
        "coordinator": "running",
        "agents": coordinator.agents,
        "total_agents": len(coordinator.agents)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
