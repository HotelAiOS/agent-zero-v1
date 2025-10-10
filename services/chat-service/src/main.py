#!/usr/bin/env python3
"""
Integrated WebSocket Service for Agent Zero V1
INTEGRATION: Uses existing FeedbackLoopEngine and SimpleTracker for real-time monitoring
Compatible with existing CLI system and Docker infrastructure
"""

import sys
import os
from pathlib import Path
import json
import asyncio
import logging
from datetime import datetime
from typing import Dict, Set
import uuid

# FIXED: Correct path for Docker container
project_root = Path("/app/project")
sys.path.insert(0, str(project_root))

# FastAPI and WebSocket imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import existing Agent Zero components - FIXED PATHS
try:
    exec(open(project_root / "simple-tracker.py").read(), globals())
    # Try to import feedback-loop-engine if it exists
    feedback_engine_path = project_root / "feedback-loop-engine.py"
    if feedback_engine_path.exists():
        exec(open(feedback_engine_path).read(), globals())
    components_available = True
    logger.info("✅ WebSocket: Successfully imported Agent Zero components")
except FileNotFoundError as e:
    logger.warning(f"Could not import components: {e}")
    components_available = False
    # Fallback class to prevent errors
    class SimpleTracker:
        def get_daily_stats(self): 
            return {"total_tasks": 0, "feedback_rate": 0, "avg_rating": 0}
        def get_model_comparison(self, days=7): 
            return {}

# FastAPI app
app = FastAPI(
    title="Agent Zero V1 - Integrated WebSocket Service",
    version="1.0.0",
    description="Real-time monitoring integrated with existing feedback systems"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        # Initialize tracker with proper error handling
        try:
            self.tracker = SimpleTracker()
            logger.info("✅ WebSocket: SimpleTracker initialized")
        except Exception as e:
            logger.error(f"Error initializing SimpleTracker: {e}")
            self.tracker = None
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")
        
        # Send current status immediately upon connection
        await self.send_current_status(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.discard(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_current_status(self, websocket: WebSocket):
        """Send current system status using SimpleTracker data"""
        try:
            if self.tracker and components_available:
                daily_stats = self.tracker.get_daily_stats()
                model_comparison = self.tracker.get_model_comparison(days=1)
                
                status_message = {
                    "type": "status_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "agents": {
                            "active_models": list(model_comparison.keys()),
                            "total_tasks": daily_stats.get("total_tasks", 0),
                            "feedback_rate": daily_stats.get("feedback_rate", 0),
                            "avg_rating": daily_stats.get("avg_rating", 0)
                        },
                        "performance": model_comparison,
                        "system_health": "operational"
                    },
                    "source": "SimpleTracker_realtime",
                    "components_available": components_available
                }
            else:
                # Fallback status when components not available
                status_message = {
                    "type": "status_update",
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        "agents": {
                            "active_models": [],
                            "total_tasks": 0,
                            "feedback_rate": 0,
                            "avg_rating": 0
                        },
                        "performance": {},
                        "system_health": "degraded"
                    },
                    "source": "fallback_mode",
                    "components_available": components_available
                }
            
            await websocket.send_text(json.dumps(status_message))
            
        except Exception as e:
            logger.error(f"Error sending status: {e}")
            # Send error status
            error_message = {
                "type": "error",
                "message": "Error retrieving system status",
                "timestamp": datetime.now().isoformat()
            }
            try:
                await websocket.send_text(json.dumps(error_message))
            except:
                pass

    async def broadcast_update(self, message: Dict):
        """Broadcast update to all connected clients"""
        if not self.active_connections:
            return
            
        message_str = json.dumps(message)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                logger.warning(f"Error sending to client: {e}")
                disconnected.add(connection)
        
        # Remove disconnected clients
        self.active_connections -= disconnected

# Global connection manager
manager = ConnectionManager()

@app.websocket("/ws/agents/live-monitor")
async def websocket_endpoint(websocket: WebSocket):
    """
    INTEGRATION: Real-time agent monitoring using existing SimpleTracker
    Developer B Frontend connects here for live updates
    """
    await manager.connect(websocket)
    
    try:
        # Start monitoring loop
        monitoring_task = asyncio.create_task(monitor_system_changes(websocket))
        
        # Listen for client messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({
                        "type": "pong", 
                        "timestamp": datetime.now().isoformat()
                    }))
                
                elif message.get("type") == "request_status":
                    await manager.send_current_status(websocket)
                
                elif message.get("type") == "subscribe_model":
                    # Future: Subscribe to specific model updates
                    model_name = message.get("model")
                    await websocket.send_text(json.dumps({
                        "type": "subscription_confirmed",
                        "model": model_name,
                        "timestamp": datetime.now().isoformat()
                    }))
                    
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({"error": "Invalid JSON"}))
                
    except WebSocketDisconnect:
        logger.info("Client disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)
        if 'monitoring_task' in locals():
            monitoring_task.cancel()

async def monitor_system_changes(websocket: WebSocket):
    """
    INTEGRATION: Monitor changes in SimpleTracker and broadcast updates
    This replaces mock data with real system monitoring
    """
    last_stats = None
    
    while True:
        try:
            if manager.tracker and components_available:
                # Get current stats from SimpleTracker
                current_stats = manager.tracker.get_daily_stats()
                
                # Check if stats changed significantly
                if last_stats is None or stats_changed(last_stats, current_stats):
                    update_message = {
                        "type": "system_update",
                        "timestamp": datetime.now().isoformat(),
                        "changes": {
                            "total_tasks": current_stats.get("total_tasks", 0),
                            "feedback_rate": current_stats.get("feedback_rate", 0),
                            "avg_rating": current_stats.get("avg_rating", 0)
                        },
                        "integration": "SimpleTracker_monitoring"
                    }
                    
                    await manager.broadcast_update(update_message)
                    last_stats = current_stats
            
            # Check every 5 seconds
            await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            await asyncio.sleep(10)

def stats_changed(old_stats: Dict, new_stats: Dict, threshold: float = 0.1) -> bool:
    """Check if stats changed significantly"""
    if not old_stats:
        return True
        
    # Check for changes in key metrics
    keys_to_check = ["total_tasks", "feedback_rate", "avg_rating"]
    
    for key in keys_to_check:
        old_val = old_stats.get(key, 0) or 0
        new_val = new_stats.get(key, 0) or 0
        
        # For total_tasks, any change is significant
        if key == "total_tasks" and old_val != new_val:
            return True
            
        # For other metrics, check percentage change
        if old_val > 0:
            change = abs(new_val - old_val) / old_val
            if change > threshold:
                return True
    
    return False

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Agent Zero V1 - Integrated WebSocket Service",
        "version": "1.0.0",
        "websocket_endpoint": "/ws/agents/live-monitor",
        "integration": "SimpleTracker + FeedbackLoopEngine",
        "connected_clients": len(manager.active_connections),
        "components_available": components_available,
        "tracker_initialized": manager.tracker is not None
    }

@app.get("/health") 
async def health_check():
    """Health check for Docker"""
    return {
        "status": "healthy",
        "service": "integrated-websocket",
        "connections": len(manager.active_connections),
        "integration": "agent_zero_v1",
        "components_available": components_available,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v1/connections")
async def get_connections():
    """Get connection status"""
    return {
        "active_connections": len(manager.active_connections),
        "components_available": components_available,
        "tracker_status": "initialized" if manager.tracker else "not_available"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
