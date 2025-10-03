from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from typing import Dict, Any, List
import psycopg2

from .swarm.coordinator import SwarmCoordinator
from .agents.base_agent import AgentTask, TaskStatus
from .models.schemas import TaskRequest, TaskResponse, WorkflowRequest
from .agents.dynamic_agent import DynamicAgent

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Agent Zero Orchestrator",
    version="1.0.0",
    description="Multi-Agent Swarm Orchestration System"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Coordinator
AI_ROUTER_URL = os.getenv("AI_ROUTER_URL", "http://ai-router-service:8000")
coordinator = SwarmCoordinator(ai_router_url=AI_ROUTER_URL)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://a0:a0password@postgresql:5432/agentzero")

def get_db_connection():
    """Uzyskaj połączenie do bazy danych"""
    return psycopg2.connect(DATABASE_URL)

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agent Zero Orchestrator v1.0.0"}

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "agents": len(coordinator.agents)
    }

@app.get("/agents/")
async def list_agents():
    """Lista wszystkich agentów"""
    agents = []
    for name, agent in coordinator.agents.items():
        agents.append({
            "name": name,
            "capabilities": [cap.value for cap in agent.capabilities]
        })
    return {"agents": agents}

@app.post("/task", response_model=TaskResponse)
async def create_task(request: TaskRequest):
    """Wykonaj zadanie przez agenta"""
    try:
        logger.info(f"Received task: {request.type}")
        
        task = AgentTask(
            type=request.type,
            description=request.description,
            input_data=request.input_data
        )
        
        result = await coordinator.execute_task(task)
        
        return TaskResponse(
            task_id=result.id,
            status=result.status.value,
            result=result.result,
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/{workflow_name}")
async def execute_workflow(workflow_name: str, description: str, language: str = "python"):
    """Wykonaj workflow"""
    try:
        logger.info(f"Executing workflow: {workflow_name}")
        
        if workflow_name == "code-to-production":
            result = await coordinator.code_to_production_workflow(
                description=description,
                language=language
            )
            return result
        else:
            raise HTTPException(status_code=404, detail="Workflow not found")
            
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agents/create")
async def create_custom_agent(agent_data: Dict[str, Any]):
    """
    Utwórz nowego custom agenta
    
    Body:
    {
        "name": "MyAgent",
        "description": "My custom agent",
        "system_prompt": "You are...",
        "capabilities": ["capability1"],
        "model_preference": "llama3.2:3b",
        "temperature": 0.7
    }
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO custom_agents 
            (name, description, system_prompt, capabilities, model_preference, temperature, created_by)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id, name
        """, (
            agent_data['name'],
            agent_data.get('description', ''),
            agent_data['system_prompt'],
            agent_data.get('capabilities', []),
            agent_data.get('model_preference', 'auto'),
            agent_data.get('temperature', 0.7),
            agent_data.get('created_by', 'api')
        ))
        
        agent_id, agent_name = cursor.fetchone()
        conn.commit()
        
        # Dodaj do coordinator jako dynamic agent
        dynamic_agent = DynamicAgent(
            name=agent_name,
            system_prompt=agent_data['system_prompt'],
            capabilities=agent_data.get('capabilities', []),
            ai_router_url=coordinator.ai_router_url,
            model_preference=agent_data.get('model_preference', 'auto'),
            temperature=agent_data.get('temperature', 0.7)
        )
        
        coordinator.agents[agent_name] = dynamic_agent
        
        cursor.close()
        conn.close()
        
        logger.info(f"Created custom agent: {agent_name}")
        
        return {
            "id": str(agent_id),
            "name": agent_name,
            "status": "created",
            "message": f"Agent {agent_name} successfully created and loaded"
        }
        
    except Exception as e:
        logger.error(f"Failed to create agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents/custom")
async def list_custom_agents():
    """Lista wszystkich custom agentów"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, name, description, capabilities, is_active, created_at
            FROM custom_agents
            WHERE is_active = true
            ORDER BY created_at DESC
        """)
        
        agents = []
        for row in cursor.fetchall():
            agents.append({
                "id": str(row[0]),
                "name": row[1],
                "description": row[2],
                "capabilities": row[3],
                "is_active": row[4],
                "created_at": str(row[5])
            })
        
        cursor.close()
        conn.close()
        
        return {"custom_agents": agents}
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/agents/custom/{agent_name}")
async def delete_custom_agent(agent_name: str):
    """Usuń custom agenta"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE custom_agents SET is_active = false
            WHERE name = %s
            RETURNING id
        """, (agent_name,))
        
        result = cursor.fetchone()
        conn.commit()
        cursor.close()
        conn.close()
        
        if result:
            # Usuń z coordinator
            if agent_name in coordinator.agents:
                del coordinator.agents[agent_name]
            
            return {"message": f"Agent {agent_name} deactivated"}
        else:
            raise HTTPException(status_code=404, detail="Agent not found")
        
    except Exception as e:
        logger.error(f"Failed to delete agent: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
