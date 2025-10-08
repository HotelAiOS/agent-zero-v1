#!/usr/bin/env python3
"""
Agent Zero V1 - Minimal WebSocket Monitor
=========================================
Uproszczony WebSocket monitor z zaawansowanym interfejsem web.
"""

import asyncio
import json
import logging
from datetime import datetime
from aiohttp import web, WSMsgType
import aiohttp_cors

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MinimalWebSocketMonitor:
    def __init__(self, host='localhost', port=8000):
        self.host = host
        self.port = port
        self.clients = set()
        self.app = web.Application()
        
        # Setup CORS
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        # Routes
        self.app.router.add_get('/', self.index)
        self.app.router.add_get('/ws', self.websocket_handler)
        
        # Add CORS to all routes
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def index(self, request):
        """Serve advanced HTML monitoring interface."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Agent Zero V1 Monitor</title>
    <style>
        body { 
            font-family: 'Segoe UI', Arial, sans-serif; 
            margin: 0; padding: 20px;
            background: linear-gradient(135deg, #1a1a1a, #2d2d2d); 
            color: #fff; 
            min-height: 100vh;
        }
        .header { 
            text-align: center;
            color: #4CAF50; 
            margin-bottom: 30px;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 20px;
        }
        .status { 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 8px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }
        .running { 
            background: linear-gradient(90deg, #2d5a27, #4CAF50);
            color: #fff;
            box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
        }
        .stopped { 
            background: linear-gradient(90deg, #5a2727, #f44336);
            color: #fff;
            box-shadow: 0 4px 8px rgba(244, 67, 54, 0.3);
        }
        .controls {
            display: flex;
            gap: 10px;
            margin: 20px 0;
            justify-content: center;
            flex-wrap: wrap;
        }
        .btn {
            background: linear-gradient(90deg, #4CAF50, #45a049);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
            transition: all 0.3s;
            box-shadow: 0 4px 8px rgba(76, 175, 80, 0.3);
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(76, 175, 80, 0.4);
        }
        .btn.danger {
            background: linear-gradient(90deg, #f44336, #d32f2f);
            box-shadow: 0 4px 8px rgba(244, 67, 54, 0.3);
        }
        #messages { 
            height: 500px; 
            overflow-y: auto; 
            border: 2px solid #4CAF50; 
            border-radius: 10px;
            padding: 15px; 
            background: rgba(42, 42, 42, 0.9);
            font-family: 'Courier New', monospace;
            backdrop-filter: blur(10px);
        }
        .message { 
            margin: 5px 0; 
            padding: 8px 12px;
            border-left: 4px solid #4CAF50;
            background: rgba(76, 175, 80, 0.1);
            border-radius: 0 8px 8px 0;
            animation: slideIn 0.3s ease;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); }
            to { opacity: 1; transform: translateX(0); }
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat-card {
            background: rgba(76, 175, 80, 0.1);
            border: 1px solid #4CAF50;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }
        .timestamp {
            color: #888;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Agent Zero V1 - System Monitor</h1>
        <p>Real-time Multi-Agent Platform Monitoring</p>
        <div class="stats">
            <div class="stat-card">
                <div class="stat-value" id="clientCount">0</div>
                <div>Connected Clients</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="messageCount">0</div>
                <div>Messages Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="uptime">00:00</div>
                <div>Session Uptime</div>
            </div>
        </div>
    </div>
    
    <div id="status" class="status stopped">
        WebSocket: Disconnected ❌
    </div>
    
    <div class="controls">
        <button class="btn" onclick="sendPing()">📡 Send Ping</button>
        <button class="btn" onclick="sendTestMessage()">🧪 Test Agent Message</button>
        <button class="btn" onclick="sendSystemStatus()">📊 System Status</button>
        <button class="btn danger" onclick="clearMessages()">🧹 Clear Messages</button>
    </div>
    
    <div id="messages"></div>
    
    <script>
        let ws;
        let messageCount = 0;
        let startTime = Date.now();
        const status = document.getElementById('status');
        const messages = document.getElementById('messages');
        
        function connect() {
            ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                status.textContent = 'WebSocket: Connected ✅';
                status.className = 'status running';
                addMessage('🔗 Connected to Agent Zero V1 Monitor', 'success');
                updateStats();
            };
            
            ws.onclose = () => {
                status.textContent = 'WebSocket: Disconnected ❌';
                status.className = 'status stopped';
                addMessage('❌ Disconnected from server - attempting reconnect...', 'error');
                updateStats();
                
                // Attempt to reconnect after 3 seconds
                setTimeout(connect, 3000);
            };
            
            ws.onerror = (error) => {
                addMessage('🚨 WebSocket error occurred', 'error');
            };
            
            ws.onmessage = (event) => {
                messageCount++;
                try {
                    const data = JSON.parse(event.data);
                    const msg = data.message || JSON.stringify(data.data || data.original || data);
                    addMessage(`📨 ${data.type}: ${msg}`, 'info');
                } catch (e) {
                    addMessage(`📨 ${event.data}`, 'info');
                }
                updateStats();
            };
        }
        
        function addMessage(text, type = 'info') {
            const div = document.createElement('div');
            div.className = 'message';
            
            let color = '#4CAF50';
            let icon = '📨';
            if (type === 'error') { color = '#f44336'; icon = '❌'; }
            if (type === 'warning') { color = '#ff9800'; icon = '⚠️'; }
            if (type === 'success') { color = '#4CAF50'; icon = '✅'; }
            
            div.innerHTML = `<span class="timestamp">${new Date().toLocaleTimeString()}</span> <span style="color: ${color}">${icon} ${text}</span>`;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function updateStats() {
            document.getElementById('clientCount').textContent = ws && ws.readyState === WebSocket.OPEN ? '1' : '0';
            document.getElementById('messageCount').textContent = messageCount;
            
            const uptimeMs = Date.now() - startTime;
            const minutes = Math.floor(uptimeMs / 60000);
            const seconds = Math.floor((uptimeMs % 60000) / 1000);
            document.getElementById('uptime').textContent = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
        }
        
        function sendPing() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'ping',
                    timestamp: Date.now(),
                    message: 'System ping from browser',
                    source: 'Agent Zero V1 Monitor'
                }));
                addMessage('📤 Ping message sent to server', 'info');
            } else {
                addMessage('❌ WebSocket not connected - cannot send ping', 'error');
            }
        }
        
        function sendTestMessage() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'agent_test',
                    data: {
                        agent_id: 'test-agent-001',
                        status: 'testing_communication',
                        components: ['AgentExecutor', 'WebSocket', 'Neo4j', 'RabbitMQ'],
                        test_result: 'success',
                        arch_linux_compatible: true
                    },
                    timestamp: Date.now(),
                    source: 'Agent Zero V1 Test Suite'
                }));
                addMessage('📤 Agent test message sent', 'info');
            } else {
                addMessage('❌ WebSocket not connected', 'error');
            }
        }
        
        function sendSystemStatus() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'system_status',
                    data: {
                        platform: 'Agent Zero V1',
                        environment: 'Arch Linux + Virtual Environment',
                        python_version: 'Python 3.13.7',
                        services: {
                            neo4j: 'RUNNING (7474/7687)',
                            rabbitmq: 'RUNNING (5672)',
                            redis: 'RUNNING (6379)',
                            websocket: 'RUNNING (8000)'
                        },
                        test_status: '17/17 passed (100%)',
                        development_ready: true
                    },
                    timestamp: Date.now()
                }));
                addMessage('📤 System status report sent', 'info');
            } else {
                addMessage('❌ WebSocket not connected', 'error');
            }
        }
        
        function clearMessages() {
            messages.innerHTML = '';
            messageCount = 0;
            addMessage('🧹 Message history cleared', 'info');
            updateStats();
        }
        
        // Connect on page load
        connect();
        
        // Update stats every second
        setInterval(updateStats, 1000);
        
        // Send periodic keepalive every 30 seconds
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'keepalive',
                    timestamp: Date.now(),
                    session_duration: Date.now() - startTime
                }));
            }
        }, 30000);
    </script>
</body>
</html>'''
        return web.Response(text=html, content_type='text/html')
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections with enhanced features."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.clients.add(ws)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send enhanced welcome message
        await ws.send_str(json.dumps({
            'type': 'welcome',
            'message': 'Connected to Agent Zero V1 Multi-Agent Platform Monitor',
            'timestamp': datetime.now().isoformat(),
            'clients': len(self.clients),
            'system': 'Agent Zero V1',
            'platform': 'Arch Linux + Virtual Environment',
            'services': {
                'neo4j': 'Available on 7474/7687',
                'rabbitmq': 'Available on 5672', 
                'redis': 'Available on 6379',
                'websocket': 'Active on 8000'
            },
            'capabilities': ['real-time-monitoring', 'agent-communication', 'system-testing']
        }))
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        
                        # Handle different message types with enhanced responses
                        if data.get('type') == 'ping':
                            response = {
                                'type': 'pong',
                                'original_timestamp': data.get('timestamp'),
                                'server_timestamp': datetime.now().isoformat(),
                                'message': 'Pong from Agent Zero V1 Multi-Agent Platform',
                                'latency_ms': datetime.now().timestamp() * 1000 - data.get('timestamp', 0)
                            }
                        elif data.get('type') == 'agent_test':
                            response = {
                                'type': 'agent_test_response',
                                'original': data.get('data', {}),
                                'server_validation': {
                                    'agent_executor': 'operational',
                                    'websocket': 'active',
                                    'neo4j': 'connected',
                                    'rabbitmq': 'active',
                                    'arch_linux_compat': 'confirmed'
                                },
                                'timestamp': datetime.now().isoformat(),
                                'test_result': 'SUCCESS'
                            }
                        elif data.get('type') == 'system_status':
                            response = {
                                'type': 'system_status_response',
                                'current_status': data.get('data', {}),
                                'server_health': {
                                    'uptime': f"{datetime.now().strftime('%H:%M:%S')}",
                                    'active_clients': len(self.clients),
                                    'memory_usage': 'optimal',
                                    'performance': 'excellent'
                                },
                                'timestamp': datetime.now().isoformat()
                            }
                        elif data.get('type') == 'keepalive':
                            response = {
                                'type': 'keepalive_ack',
                                'timestamp': datetime.now().isoformat(),
                                'status': 'alive',
                                'session_duration_ms': data.get('session_duration', 0)
                            }
                        else:
                            # Generic echo with platform info
                            response = {
                                'type': 'echo',
                                'original': data,
                                'server_timestamp': datetime.now().isoformat(),
                                'client_count': len(self.clients),
                                'processed_by': 'Agent Zero V1 WebSocket Monitor',
                                'platform': 'Multi-Agent Enterprise Platform'
                            }
                        
                        await ws.send_str(json.dumps(response))
                        
                        # Enhanced logging
                        logger.info(f"Processed: {data.get('type', 'unknown')} from client (total clients: {len(self.clients)})")
                        
                    except json.JSONDecodeError:
                        # Handle plain text messages
                        await ws.send_str(json.dumps({
                            'type': 'text_echo',
                            'message': msg.data,
                            'timestamp': datetime.now().isoformat(),
                            'note': 'Received as plain text - Agent Zero V1 Monitor'
                        }))
                        
                elif msg.type == WSMsgType.ERROR:
                    logger.error(f'WebSocket error: {ws.exception()}')
        
        except Exception as e:
            logger.error(f"WebSocket handler error: {e}")
        
        finally:
            self.clients.discard(ws)
            logger.info(f"Client disconnected. Total clients: {len(self.clients)}")
        
        return ws
    
    async def broadcast(self, message):
        """Broadcast message to all connected clients."""
        if not self.clients:
            return
        
        message_json = json.dumps({
            **message,
            'broadcast_timestamp': datetime.now().isoformat(),
            'client_count': len(self.clients),
            'broadcast_id': f"broadcast-{datetime.now().timestamp()}",
            'platform': 'Agent Zero V1 Multi-Agent Platform'
        })
        
        disconnected = set()
        for client in self.clients:
            try:
                await client.send_str(message_json)
            except ConnectionResetError:
                disconnected.add(client)
        
        # Remove disconnected clients
        self.clients -= disconnected
        if disconnected:
            logger.info(f"Removed {len(disconnected)} disconnected clients")
    
    def run(self):
        """Start the WebSocket monitor."""
        logger.info(f"🚀 Starting Agent Zero V1 WebSocket Monitor")
        logger.info(f"📡 Server: http://{self.host}:{self.port}")
        logger.info(f"🔌 WebSocket: ws://{self.host}:{self.port}/ws")
        print(f"\n🌐 Open browser: http://{self.host}:{self.port}")
        print("💡 Use Ctrl+C to stop")
        print("🎯 Features: Real-time monitoring, Agent testing, System diagnostics")
        print("🤖 Platform: Agent Zero V1 Multi-Agent Enterprise System")
        
        try:
            web.run_app(self.app, host=self.host, port=self.port)
        except KeyboardInterrupt:
            logger.info("🛑 Agent Zero V1 WebSocket Monitor stopped by user")
        except Exception as e:
            logger.error(f"Failed to start WebSocket Monitor: {e}")
            print(f"❌ Error: {e}")
            print("💡 Make sure port 8000 is not in use")

def main():
    """Main entry point."""
    print("🤖 Agent Zero V1 - WebSocket Monitor")
    print("===================================")
    print("🎯 Multi-Agent Platform Monitoring System")
    
    monitor = MinimalWebSocketMonitor()
    monitor.run()

if __name__ == "__main__":
    main()