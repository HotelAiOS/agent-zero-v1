#!/usr/bin/env python3
"""
Agent Zero V1 - Minimal WebSocket Monitor
=========================================
Uproszczony WebSocket monitor bez zbędnych dependencies.
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
        """Serve basic HTML page."""
        html = '''<!DOCTYPE html>
<html>
<head>
    <title>Agent Zero V1 Monitor</title>
    <style>
        body { 
            font-family: Arial, sans-serif; 
            margin: 20px; 
            background: #1a1a1a; 
            color: #fff; 
        }
        .status { 
            padding: 10px; 
            margin: 5px; 
            border-radius: 5px; 
        }
        .running { 
            background: #2d5a27; 
            color: #90ee90; 
        }
        .stopped { 
            background: #5a2727; 
            color: #ffcccb; 
        }
        .header { 
            color: #4CAF50; 
            margin-bottom: 20px; 
        }
        #messages { 
            height: 400px; 
            overflow-y: auto; 
            border: 1px solid #444; 
            padding: 10px; 
            background: #2a2a2a;
            font-family: monospace;
            border-radius: 5px;
        }
        .message { 
            margin: 2px 0; 
            padding: 2px 5px;
            border-left: 3px solid #4CAF50;
        }
        .controls {
            margin: 10px 0;
        }
        .btn {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 8px 16px;
            margin: 5px;
            border-radius: 4px;
            cursor: pointer;
        }
        .btn:hover {
            background: #45a049;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Agent Zero V1 - System Monitor</h1>
        <p>Real-time system monitoring and WebSocket testing</p>
    </div>
    
    <div id="status" class="status stopped">
        WebSocket: Disconnected
    </div>
    
    <div class="controls">
        <button class="btn" onclick="sendPing()">Send Ping</button>
        <button class="btn" onclick="clearMessages()">Clear Messages</button>
        <button class="btn" onclick="sendTestMessage()">Test Message</button>
    </div>
    
    <div id="messages"></div>
    
    <script>
        let ws;
        const status = document.getElementById('status');
        const messages = document.getElementById('messages');
        
        function connect() {
            ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onopen = () => {
                status.textContent = 'WebSocket: Connected ✅';
                status.className = 'status running';
                addMessage('🔗 Connected to Agent Zero V1 Monitor', 'success');
            };
            
            ws.onclose = () => {
                status.textContent = 'WebSocket: Disconnected ❌';
                status.className = 'status stopped';
                addMessage('❌ Disconnected from server', 'error');
                
                // Attempt to reconnect after 3 seconds
                setTimeout(connect, 3000);
            };
            
            ws.onerror = (error) => {
                addMessage('🚨 WebSocket error occurred', 'error');
            };
            
            ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    const msg = data.message || JSON.stringify(data.data || data);
                    addMessage(`📨 ${data.type}: ${msg}`, 'info');
                } catch (e) {
                    addMessage(`📨 ${event.data}`, 'info');
                }
            };
        }
        
        function addMessage(text, type = 'info') {
            const div = document.createElement('div');
            div.className = 'message';
            
            let color = '#4CAF50';
            if (type === 'error') color = '#f44336';
            if (type === 'warning') color = '#ff9800';
            
            div.innerHTML = `<span style="color: #888">${new Date().toLocaleTimeString()}</span> <span style="color: ${color}">${text}</span>`;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }
        
        function sendPing() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'ping',
                    timestamp: Date.now(),
                    message: 'Ping from browser'
                }));
                addMessage('📤 Sent ping message', 'info');
            } else {
                addMessage('❌ WebSocket not connected', 'error');
            }
        }
        
        function sendTestMessage() {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'test',
                    data: {
                        agent_id: 'test-agent-001',
                        status: 'testing',
                        components: ['AgentExecutor', 'WebSocket', 'Neo4j']
                    },
                    timestamp: Date.now()
                }));
                addMessage('📤 Sent test message', 'info');
            } else {
                addMessage('❌ WebSocket not connected', 'error');
            }
        }
        
        function clearMessages() {
            messages.innerHTML = '';
            addMessage('🧹 Messages cleared', 'info');
        }
        
        // Connect on page load
        connect();
        
        // Send periodic keepalive
        setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({
                    type: 'keepalive',
                    timestamp: Date.now()
                }));
            }
        }, 30000);
    </script>
</body>
</html>'''
        return web.Response(text=html, content_type='text/html')
    
    async def websocket_handler(self, request):
        """Handle WebSocket connections."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.clients.add(ws)
        logger.info(f"Client connected. Total clients: {len(self.clients)}")
        
        # Send welcome message
        await ws.send_str(json.dumps({
            'type': 'welcome',
            'message': 'Connected to Agent Zero V1 Monitor',
            'timestamp': datetime.now().isoformat(),
            'clients': len(self.clients),
            'system': 'Agent Zero V1 Multi-Agent Platform'
        }))
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                        
                        # Handle different message types
                        if data.get('type') == 'ping':
                            response = {
                                'type': 'pong',
                                'original_timestamp': data.get('timestamp'),
                                'server_timestamp': datetime.now().isoformat(),
                                'message': 'Pong from Agent Zero V1 Monitor'
                            }
                        elif data.get('type') == 'keepalive':
                            response = {
                                'type': 'keepalive_ack',
                                'timestamp': datetime.now().isoformat(),
                                'status': 'alive'
                            }
                        else:
                            # Echo back received data with server info
                            response = {
                                'type': 'echo',
                                'original': data,
                                'server_timestamp': datetime.now().isoformat(),
                                'client_count': len(self.clients),
                                'processed_by': 'Agent Zero V1 WebSocket Monitor'
                            }
                        
                        await ws.send_str(json.dumps(response))
                        
                        # Log received message
                        logger.info(f"Received: {data.get('type', 'unknown')} from client")
                        
                    except json.JSONDecodeError:
                        # Handle plain text messages
                        await ws.send_str(json.dumps({
                            'type': 'text_echo',
                            'message': msg.data,
                            'timestamp': datetime.now().isoformat(),
                            'note': 'Received as plain text'
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
            'broadcast_id': f"broadcast-{datetime.now().timestamp()}"
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
        print("🎯 Features: Real-time monitoring, WebSocket testing, System status")
        
        try:
            web.run_app(self.app, host=self.host, port=self.port)
        except KeyboardInterrupt:
            logger.info("🛑 WebSocket Monitor stopped by user")
        except Exception as e:
            logger.error(f"Failed to start WebSocket Monitor: {e}")
            print(f"❌ Error: {e}")
            print("💡 Make sure port 8000 is not in use")

def main():
    """Main entry point."""
    print("🤖 Agent Zero V1 - WebSocket Monitor")
    print("===================================")
    
    monitor = MinimalWebSocketMonitor()
    monitor.run()

if __name__ == "__main__":
    main()