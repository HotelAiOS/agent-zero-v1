# WebSocket Monitor - Complete Implementation
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import json
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Agent Zero V1 - WebSocket Monitor")

# Complete HTML template for WebSocket dashboard
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="pl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚀 Agent Zero V1 - Live Monitor</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .status-bar {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        
        .connection-status {
            font-size: 1.2em;
            font-weight: bold;
            text-align: center;
            padding: 10px;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .connected {
            background-color: #d4edda;
            color: #155724;
            border: 2px solid #c3e6cb;
        }
        
        .disconnected {
            background-color: #f8d7da;
            color: #721c24;
            border: 2px solid #f5c6cb;
        }
        
        .connecting {
            background-color: #fff3cd;
            color: #856404;
            border: 2px solid #ffeaa7;
        }
        
        .agents-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .agent-card {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            transition: transform 0.3s ease;
        }
        
        .agent-card:hover {
            transform: translateY(-5px);
        }
        
        .agent-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        
        .agent-name {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
        }
        
        .agent-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: bold;
        }
        
        .status-active {
            background-color: #d4edda;
            color: #155724;
        }
        
        .status-inactive {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .status-working {
            background-color: #cce5ff;
            color: #004085;
        }
        
        .agent-details {
            font-size: 0.95em;
            color: #666;
            line-height: 1.5;
        }
        
        .logs-section {
            background: rgba(255,255,255,0.95);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            max-height: 400px;
            overflow-y: auto;
        }
        
        .logs-header {
            font-size: 1.4em;
            font-weight: bold;
            margin-bottom: 15px;
            color: #333;
            border-bottom: 2px solid #e9ecef;
            padding-bottom: 10px;
        }
        
        .log-entry {
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            margin: 8px 0;
            padding: 12px 15px;
            border-radius: 5px;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            transition: background-color 0.3s ease;
        }
        
        .log-entry:hover {
            background: #e9ecef;
        }
        
        .log-timestamp {
            color: #666;
            font-weight: bold;
            margin-right: 10px;
        }
        
        .log-message {
            color: #333;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Agent Zero V1 - Live Monitor</h1>
            <p>Multi-Agent System Monitoring Dashboard</p>
        </div>
        
        <div class="status-bar">
            <div id="connection-status" class="connection-status connecting">
                <span class="spinner"></span> Łączenie z systemem...
            </div>
        </div>
        
        <div id="agents-container" class="agents-grid">
            <!-- Agent cards will be dynamically populated -->
        </div>
        
        <div class="logs-section">
            <div class="logs-header">📋 System Logs</div>
            <div id="logs-container">
                <!-- Logs will be dynamically populated -->
            </div>
        </div>
    </div>
    
    <script>
        class AgentZeroMonitor {
            constructor() {
                this.ws = null;
                this.reconnectAttempts = 0;
                this.maxReconnectAttempts = 5;
                this.reconnectDelay = 3000;
                this.agents = new Map();
                
                this.connectionStatus = document.getElementById('connection-status');
                this.agentsContainer = document.getElementById('agents-container');
                this.logsContainer = document.getElementById('logs-container');
                
                this.connect();
            }
            
            connect() {
                try {
                    this.ws = new WebSocket("ws://localhost:8000/ws");
                    
                    this.ws.onopen = (event) => {
                        console.log('WebSocket connection established');
                        this.onConnected();
                    };
                    
                    this.ws.onmessage = (event) => {
                        try {
                            const data = JSON.parse(event.data);
                            this.handleMessage(data);
                        } catch (error) {
                            this.addLog('RAW', event.data);
                        }
                    };
                    
                    this.ws.onclose = (event) => {
                        console.log('WebSocket connection closed');
                        this.onDisconnected();
                        this.attemptReconnect();
                    };
                    
                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                        this.addLog('ERROR', 'WebSocket connection error');
                    };
                    
                } catch (error) {
                    console.error('Failed to create WebSocket connection:', error);
                    this.onDisconnected();
                    this.attemptReconnect();
                }
            }
            
            onConnected() {
                this.reconnectAttempts = 0;
                this.connectionStatus.className = 'connection-status connected';
                this.connectionStatus.innerHTML = '✅ Połączono z Agent Zero V1';
                this.addLog('SYSTEM', 'WebSocket connection established successfully');
            }
            
            onDisconnected() {
                this.connectionStatus.className = 'connection-status disconnected';
                this.connectionStatus.innerHTML = '❌ Rozłączono z systemem';
                this.addLog('SYSTEM', 'Connection lost - attempting to reconnect...');
            }
            
            attemptReconnect() {
                if (this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.reconnectAttempts++;
                    this.connectionStatus.className = 'connection-status connecting';
                    this.connectionStatus.innerHTML = `<span class="spinner"></span> Próba ponownego połączenia... (${this.reconnectAttempts}/${this.maxReconnectAttempts})`;
                    
                    setTimeout(() => {
                        this.connect();
                    }, this.reconnectDelay);
                } else {
                    this.connectionStatus.className = 'connection-status disconnected';
                    this.connectionStatus.innerHTML = '💀 Nie udało się połączyć po ' + this.maxReconnectAttempts + ' próbach';
                }
            }
            
            handleMessage(data) {
                switch (data.type) {
                    case 'agent_update':
                        this.updateAgent(data);
                        break;
                    case 'system_status':
                        this.updateSystemStatus(data);
                        break;
                    case 'task_progress':
                        this.updateTaskProgress(data);
                        break;
                    default:
                        this.addLog('UNKNOWN', JSON.stringify(data));
                }
            }
            
            updateAgent(data) {
                const agentId = data.agent || 'unknown';
                this.agents.set(agentId, data);
                
                let agentCard = document.getElementById(`agent-${agentId}`);
                if (!agentCard) {
                    agentCard = this.createAgentCard(agentId);
                    this.agentsContainer.appendChild(agentCard);
                }
                
                this.renderAgentCard(agentCard, data);
                this.addLog('AGENT', `${agentId}: ${data.status}`);
            }
            
            createAgentCard(agentId) {
                const card = document.createElement('div');
                card.className = 'agent-card';
                card.id = `agent-${agentId}`;
                return card;
            }
            
            renderAgentCard(card, data) {
                const statusClass = this.getStatusClass(data.active, data.status);
                const statusText = data.active ? 'ACTIVE' : 'INACTIVE';
                
                card.innerHTML = `
                    <div class="agent-header">
                        <div class="agent-name">🤖 ${data.agent}</div>
                        <div class="agent-status ${statusClass}">${statusText}</div>
                    </div>
                    <div class="agent-details">
                        <strong>Status:</strong> ${data.status || 'Unknown'}<br>
                        <strong>Last Update:</strong> ${new Date().toLocaleTimeString()}<br>
                        <strong>Type:</strong> ${data.agent_type || 'Multi-Purpose'}
                    </div>
                `;
            }
            
            getStatusClass(active, status) {
                if (active === true) return 'status-active';
                if (active === false) return 'status-inactive';
                if (status && status.toLowerCase().includes('work')) return 'status-working';
                return 'status-inactive';
            }
            
            updateSystemStatus(data) {
                this.addLog('SYSTEM', `System status: ${data.status}`);
            }
            
            updateTaskProgress(data) {
                this.addLog('TASK', `Task ${data.task_id}: ${data.progress}%`);
            }
            
            addLog(type, message) {
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';
                
                const timestamp = new Date().toLocaleTimeString();
                logEntry.innerHTML = `
                    <span class="log-timestamp">[${timestamp}]</span>
                    <span class="log-type">[${type}]</span>
                    <span class="log-message">${message}</span>
                `;
                
                this.logsContainer.appendChild(logEntry);
                this.logsContainer.scrollTop = this.logsContainer.scrollHeight;
                
                // Keep only last 100 logs
                while (this.logsContainer.children.length > 100) {
                    this.logsContainer.removeChild(this.logsContainer.firstChild);
                }
            }
        }
        
        // Initialize monitor when page loads
        document.addEventListener('DOMContentLoaded', () => {
            window.agentZeroMonitor = new AgentZeroMonitor();
        });
    </script>
</body>
</html>
"""

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"New WebSocket connection established. Active connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"WebSocket connection closed. Active connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: dict, websocket: WebSocket):
        try:
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

# Global connection manager
manager = ConnectionManager()

@app.get("/", response_class=HTMLResponse)
async def get_dashboard():
    """Serve the WebSocket dashboard"""
    return HTMLResponse(content=HTML_TEMPLATE)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Agent Zero V1 WebSocket Monitor",
        "active_connections": len(manager.active_connections),
        "timestamp": asyncio.get_event_loop().time()
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket endpoint for real-time monitoring"""
    await manager.connect(websocket)
    
    # Send welcome message
    await manager.send_personal_message({
        "type": "system_status",
        "status": "Connected to Agent Zero V1 monitoring system",
        "timestamp": asyncio.get_event_loop().time()
    }, websocket)
    
    try:
        # Start monitoring loop
        await monitor_agents(websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def monitor_agents(websocket: WebSocket):
    """Continuous monitoring loop for agents"""
    agents = [
        {"name": "TaskDecomposer", "type": "Foundation"},
        {"name": "CodeGenerator", "type": "Execution"}, 
        {"name": "QualityAnalyzer", "type": "Quality"},
        {"name": "ProjectOrchestrator", "type": "Execution"},
        {"name": "BackendAgent", "type": "Development"},
        {"name": "FrontendAgent", "type": "Development"},
        {"name": "DevOpsAgent", "type": "Infrastructure"},
        {"name": "DatabaseAgent", "type": "Data"}
    ]
    
    counter = 0
    while True:
        try:
            # Simulate agent status updates
            for i, agent in enumerate(agents):
                is_active = (counter + i) % 3 != 0  # Simulate some agents being inactive
                status_messages = [
                    "Idle - waiting for tasks",
                    "Processing request",
                    "Generating code",
                    "Running tests",
                    "Analyzing quality",
                    "Optimizing performance"
                ]
                
                status = status_messages[(counter + i) % len(status_messages)]
                
                await manager.send_personal_message({
                    "type": "agent_update",
                    "agent": agent["name"],
                    "agent_type": agent["type"],
                    "status": status,
                    "active": is_active,
                    "timestamp": asyncio.get_event_loop().time()
                }, websocket)
                
                # Small delay between agent updates
                await asyncio.sleep(0.2)
            
            # Send system status
            await manager.send_personal_message({
                "type": "system_status", 
                "status": f"All systems operational - Cycle {counter}",
                "agents_count": len(agents),
                "timestamp": asyncio.get_event_loop().time()
            }, websocket)
            
            counter += 1
            
            # Wait before next monitoring cycle
            await asyncio.sleep(10)
            
        except WebSocketDisconnect:
            break
        except Exception as e:
            logger.error(f"Error in monitoring loop: {e}")
            await asyncio.sleep(5)

# Background task for system monitoring
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Agent Zero V1 WebSocket Monitor started successfully!")
    logger.info("Dashboard available at: http://localhost:8000")
    
@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("Agent Zero V1 WebSocket Monitor shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")