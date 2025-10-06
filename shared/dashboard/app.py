"""
Live Dashboard - Agent Zero v1
Real-time monitoring systemu agentów z Neo4j knowledge stats
"""

import sys
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
import asyncio
import logging
import json
from datetime import datetime

# Dodaj shared do path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import z agent_factory
from agent_factory.factory import AgentFactory
from agent_factory.lifecycle import AgentLifecycleManager, AgentState

# Import dla Neo4j (optional)
try:
    from knowledge import Neo4jClient
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicjalizacja FastAPI
app = FastAPI(
    title="Agent Zero Dashboard",
    description="Live monitoring systemu agentów AI z Neo4j knowledge integration",
    version="1.0.0"
)

# Static files i templates
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

# Globalne instancje (inicjalizowane w startup)
lifecycle_manager: Optional[AgentLifecycleManager] = None
factory: Optional[AgentFactory] = None
neo4j_client: Optional['Neo4jClient'] = None


class ConnectionManager:
    """Manager dla WebSocket connections"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def broadcast(self, message: Dict[str, Any]):
        """Wyślij wiadomość do wszystkich połączonych klientów"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")
                disconnected.append(connection)
        
        # Usuń zerwane połączenia
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


@app.on_event("startup")
async def startup_event():
    """Inicjalizacja przy starcie aplikacji"""
    global lifecycle_manager, factory, neo4j_client
    
    logger.info("🚀 Uruchamianie Agent Zero Dashboard...")
    
    try:
        # Inicjalizuj Neo4j (optional)
        if NEO4J_AVAILABLE:
            try:
                neo4j_client = Neo4jClient()
                logger.info("✅ Neo4j client zainicjalizowany")
            except Exception as e:
                logger.warning(f"Neo4j connection failed: {e}")
                neo4j_client = None
        else:
            logger.info("ℹ️  Neo4j not available - knowledge stats disabled")
        
        # Utwórz lifecycle manager z messaging i Neo4j
        lifecycle_manager = AgentLifecycleManager(
            enable_messaging=True,
            neo4j_client=neo4j_client
        )
        
        # Utwórz factory z lifecycle manager
        factory = AgentFactory(lifecycle_manager=lifecycle_manager)
        
        logger.info("✅ Factory i Lifecycle Manager zainicjalizowane")
        
        # Uruchom background task do wysyłania metryk
        asyncio.create_task(broadcast_metrics())
        
    except Exception as e:
        logger.error(f"❌ Błąd podczas inicjalizacji: {e}")
        import traceback
        logger.error(traceback.format_exc())


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup przy zamykaniu"""
    logger.info("🛑 Zamykanie Agent Zero Dashboard...")
    
    if neo4j_client:
        try:
            neo4j_client.close()
            logger.info("✅ Neo4j connection closed")
        except Exception as e:
            logger.error(f"Error closing Neo4j: {e}")


async def broadcast_metrics():
    """Background task - wysyła metryki co 2 sekundy"""
    while True:
        try:
            if lifecycle_manager:
                health = lifecycle_manager.get_system_health()
                
                # Dodaj timestamp
                health["timestamp"] = datetime.now().isoformat()
                
                # Broadcast do wszystkich klientów
                await manager.broadcast(health)
            
            await asyncio.sleep(2)  # Update co 2s
            
        except Exception as e:
            logger.error(f"Error in broadcast_metrics: {e}")
            await asyncio.sleep(5)


@app.get("/", response_class=HTMLResponse)
async def dashboard_home(request: Request):
    """Strona główna dashboardu"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Agent Zero Live Dashboard"
    })


@app.get("/api/health")
async def get_health():
    """REST endpoint - system health"""
    if lifecycle_manager is None:
        return {"status": "not_initialized", "error": "Lifecycle manager not ready"}
    
    health = lifecycle_manager.get_system_health()
    return health


@app.get("/api/agents")
async def get_agents():
    """REST endpoint - lista agentów"""
    if lifecycle_manager is None:
        return {"agents": []}
    
    agents_data = []
    for agent_id, agent in lifecycle_manager.agents.items():
        agents_data.append({
            "agent_id": agent_id,
            "agent_type": agent.agent_type,
            "state": agent.state.value,
            "tasks_completed": agent.metrics.tasks_completed,
            "tasks_failed": agent.metrics.tasks_failed,
            "messages_sent": agent.metrics.messages_sent,
            "messages_received": agent.metrics.messages_received,
            "uptime": agent.metrics.uptime_seconds,
            "error_count": agent.metrics.error_count
        })
    
    return {"agents": agents_data, "total": len(agents_data)}


@app.get("/api/agents/{agent_id}/metrics")
async def get_agent_metrics(agent_id: str):
    """REST endpoint - metryki konkretnego agenta"""
    if lifecycle_manager is None:
        return {"error": "Lifecycle manager not ready"}
    
    metrics = lifecycle_manager.get_agent_metrics(agent_id)
    
    if metrics is None:
        return {"error": f"Agent {agent_id} not found"}
    
    return {
        "agent_id": agent_id,
        "metrics": {
            "tasks_completed": metrics.tasks_completed,
            "tasks_failed": metrics.tasks_failed,
            "total_tokens_used": metrics.total_tokens_used,
            "average_response_time": metrics.average_response_time,
            "uptime_seconds": metrics.uptime_seconds,
            "error_count": metrics.error_count,
            "messages_sent": metrics.messages_sent,
            "messages_received": metrics.messages_received,
            "last_active": metrics.last_active.isoformat() if metrics.last_active else None
        }
    }


@app.get("/api/knowledge/stats")
async def get_knowledge_stats():
    """REST endpoint - Neo4j knowledge statistics"""
    if neo4j_client is None:
        return {
            "enabled": False,
            "error": "Neo4j not available"
        }
    
    try:
        # Pobierz statystyki dla wszystkich agentów
        all_stats = []
        if lifecycle_manager:
            for agent_id in lifecycle_manager.agents.keys():
                try:
                    stats = neo4j_client.get_agent_stats(agent_id)
                    if stats:
                        stats['agent_id'] = agent_id
                        all_stats.append(stats)
                except Exception as e:
                    logger.error(f"Error getting stats for {agent_id}: {e}")
        
        # Agreguj stats
        total_tasks = sum(s.get('total_tasks', 0) for s in all_stats)
        total_experiences = sum(s.get('experiences', 0) for s in all_stats)
        
        return {
            "enabled": True,
            "agents": all_stats,
            "summary": {
                "total_agents": len(all_stats),
                "total_tasks": total_tasks,
                "total_experiences": total_experiences
            }
        }
    
    except Exception as e:
        logger.error(f"Error getting knowledge stats: {e}")
        return {
            "enabled": True,
            "error": str(e)
        }


@app.get("/api/agents/{agent_id}/experiences")
async def get_agent_experiences(agent_id: str, keywords: str = "", limit: int = 10):
    """REST endpoint - experiences agenta z Neo4j"""
    if lifecycle_manager is None or neo4j_client is None:
        return {"error": "System not ready or Neo4j not available"}
    
    try:
        keyword_list = [k.strip() for k in keywords.split(",") if k.strip()] if keywords else []
        
        if not keyword_list:
            return {"experiences": [], "message": "No keywords provided"}
        
        experiences = lifecycle_manager.get_agent_experiences(
            agent_id=agent_id,
            keywords=keyword_list,
            limit=limit
        )
        
        return {
            "agent_id": agent_id,
            "keywords": keyword_list,
            "experiences": experiences,
            "count": len(experiences)
        }
    
    except Exception as e:
        logger.error(f"Error getting experiences: {e}")
        return {"error": str(e)}


@app.post("/api/test/create-agents")
async def create_test_agents():
    """Endpoint testowy - tworzy 3 agentów dla demo"""
    if factory is None or lifecycle_manager is None:
        return {"error": "System not initialized"}
    
    try:
        # Utwórz 3 agentów
        backend = factory.create_agent("backend")
        frontend = factory.create_agent("frontend")
        database = factory.create_agent("database")
        
        # Symuluj aktywność
        backend.metrics.tasks_completed = 5
        backend.metrics.messages_sent = 12
        backend.metrics.messages_received = 8
        backend.metrics.uptime_seconds = 120.5
        lifecycle_manager.transition_state(backend.agent_id, AgentState.INITIALIZING)
        lifecycle_manager.transition_state(backend.agent_id, AgentState.READY)
        lifecycle_manager.transition_state(backend.agent_id, AgentState.BUSY)
        
        frontend.metrics.tasks_completed = 3
        frontend.metrics.messages_sent = 7
        frontend.metrics.messages_received = 5
        lifecycle_manager.transition_state(frontend.agent_id, AgentState.INITIALIZING)
        lifecycle_manager.transition_state(frontend.agent_id, AgentState.READY)
        
        database.metrics.tasks_completed = 8
        database.metrics.messages_sent = 15
        database.metrics.messages_received = 12
        database.metrics.uptime_seconds = 200.0
        lifecycle_manager.transition_state(database.agent_id, AgentState.INITIALIZING)
        lifecycle_manager.transition_state(database.agent_id, AgentState.READY)
        lifecycle_manager.transition_state(database.agent_id, AgentState.IDLE)
        
        logger.info(f"✅ Utworzono testowych agentów: {backend.agent_id}, {frontend.agent_id}, {database.agent_id}")
        
        return {
            "status": "success",
            "message": "Created 3 test agents",
            "agents": [backend.agent_id, frontend.agent_id, database.agent_id]
        }
    
    except Exception as e:
        import traceback
        logger.error(f"Błąd tworzenia agentów: {e}")
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint - real-time metrics"""
    await manager.connect(websocket)
    
    try:
        # Wyślij natychmiast aktualne dane
        if lifecycle_manager:
            health = lifecycle_manager.get_system_health()
            health["timestamp"] = datetime.now().isoformat()
            await websocket.send_json(health)
        
        # Trzymaj połączenie otwarte (broadcast_metrics wysyła dane)
        while True:
            # Odbieraj ping/pong żeby utrzymać połączenie
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Uruchamianie Agent Zero Dashboard na http://localhost:5000")
    print("📊 WebSocket: ws://localhost:5000/ws/metrics")
    print("🔌 REST API: http://localhost:5000/api/health")
    print("💡 Knowledge: http://localhost:5000/api/knowledge/stats")
    print("🧪 Test API: POST http://localhost:5000/api/test/create-agents")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )
