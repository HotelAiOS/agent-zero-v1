#!/usr/bin/env python3
import sys
import os
from pathlib import Path

# FIXED: Correct path to project root  
project_root = Path("/app/project")
sys.path.insert(0, str(project_root))

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
from typing import List
from datetime import datetime

app = FastAPI(title="Agent Zero WebSocket Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import project components with correct path
try:
    # Import from /app/project/ (mounted volume)
    exec(open("/app/project/simple-tracker.py").read(), globals())
    components_available = True
    print("✅ WebSocket: Successfully imported SimpleTracker")
except Exception as e:
    print(f"WARNING: WebSocket: Could not import SimpleTracker: {e}")
    components_available = False
    class SimpleTracker:
        def track_event(self, event): pass

# Initialize tracker
try:
    tracker = SimpleTracker()
except:
    tracker = None

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "websocket-service",
        "timestamp": datetime.now().isoformat(),
        "components_available": components_available,
        "active_connections": len(manager.active_connections)
    }

@app.get("/")
async def root():
    return {
        "message": "Agent Zero WebSocket Service",
        "status": "running", 
        "integration": "SimpleTracker" if components_available else "standalone"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            
            # Track event if components available
            if tracker:
                try:
                    tracker.track_event({
                        "type": "websocket_message",
                        "data": data,
                        "timestamp": datetime.now().isoformat()
                    })
                except:
                    pass
            
            await websocket.send_text(f"Echo: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
WEBEOF
