"""WebSocket Live Monitor - Real-time token streaming"""
import asyncio
import json
from typing import Set
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from shared.monitoring.livemonitor import LiveMonitor, AgentUpdate

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Set[WebSocket] = set()
        
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.add(websocket)
        
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()
live_monitor = LiveMonitor()

@app.websocket("/ws/agent_monitor")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Czekaj na updates od agentów
            await asyncio.sleep(0.1)
            # TODO: Podłącz do LiveMonitor queue
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def dashboard():
    return """
    <html>
        <head><title>Agent Zero - Live Monitor</title></head>
        <body>
            <h1>🤖 Agent Zero - Live Token Stream</h1>
            <div id="output"></div>
            <script>
                const ws = new WebSocket("ws://localhost:8000/ws/agent_monitor");
                ws.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    document.getElementById('output').innerHTML += 
                        `<p>${data.agent_id}: ${data.message}</p>`;
                };
            </script>
        </body>
    </html>
    """
