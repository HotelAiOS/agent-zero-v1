#!/usr/bin/env python3
"""
Agent Zero V1 – WebSocket Monitor
"""

import asyncio
import json
import logging
import os
from datetime import datetime

from aiohttp import web
import aiohttp_cors

# ---------------------------------------------------
# Logger – tylko StreamHandler (stdout)
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("WSMon")


class Monitor:
    def __init__(self):
        self.conns = set()
        self.agents = {
            "task_decomposer": "active",
            "code_executor":   "active",
            "file_manager":    "active",
            "web_search":      "active",
            "data_analyst":    "active",
            "communication":   "active",
            "orchestrator":    "active",
            "monitor":         "active",
        }

    async def register(self, ws: web.WebSocketResponse):
        self.conns.add(ws)
        logger.info(f"Conn+ → total connections: {len(self.conns)}")

    async def unregister(self, ws: web.WebSocketResponse):
        self.conns.discard(ws)
        logger.info(f"Conn– → total connections: {len(self.conns)}")

    async def broadcast(self):
        if not self.conns:
            return
        payload = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "agents": self.agents,
            "active_connections": len(self.conns)
        })
        to_remove = set()
        for ws in self.conns:
            try:
                await ws.send_str(payload)
            except Exception as e:
                logger.warning(f"Send failed: {e}")
                to_remove.add(ws)
        for ws in to_remove:
            await self.unregister(ws)


monitor = Monitor()


# ---------------------------
# Handlery HTTP / WebSocket
# ---------------------------
async def ws_handler(request: web.Request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    await monitor.register(ws)

    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                data = json.loads(msg.data)
                if data.get("type") == "ping":
                    await ws.send_str(json.dumps({"type": "pong"}))
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await monitor.unregister(ws)

    return ws


async def index_handler(request: web.Request):
    # Prosta strona HTML z wbudowanym JS
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>Agent Zero V1 Monitor</title>
  <style>
    body { font-family: sans-serif; background: #fafafa; color: #333; padding: 20px; }
    pre { background: #222; color: #0f0; padding: 1em; height: 60vh; overflow: auto; }
  </style>
</head>
<body>
<h1>🚀 Agent Zero V1 WebSocket Monitor</h1>
<pre id="log"></pre>
<script>
  const log = document.getElementById("log");
  const ws  = new WebSocket("ws://" + location.host + "/ws");
  ws.onopen    = () => log.textContent += "[OPEN]\\n";
  ws.onmessage = e  => log.textContent += e.data + "\\n";
  ws.onclose   = () => log.textContent += "[CLOSED]\\n";
  ws.onerror   = e  => log.textContent += "[ERROR] " + e + "\\n";
  setInterval(() => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "ping" }));
    }
  }, 30000);
</script>
</body>
</html>"""
    return web.Response(text=html, content_type="text/html")


async def health_handler(request: web.Request):
    return web.json_response({
        "status":      "healthy",
        "timestamp":   datetime.now().isoformat(),
        "connections": len(monitor.conns),
        "agents":      monitor.agents
    })


# -------------------------
# Aplikacja i broadcast
# -------------------------
async def init_app():
    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })

    app.router.add_get("/",       index_handler)
    app.router.add_get("/ws",     ws_handler)
    app.router.add_get("/health", health_handler)

    for route in list(app.router.routes()):
        cors.add(route)
    return app


async def broadcaster():
    while True:
        await monitor.broadcast()
        await asyncio.sleep(5)


def main():
    port = int(os.getenv("WEBSOCKET_PORT", "8000"))

    # Tworzymy nowy event loop i ustawiamy go jako domyślny
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    app = loop.run_until_complete(init_app())
    runner = web.AppRunner(app)
    loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner, "0.0.0.0", port)
    loop.run_until_complete(site.start())

    logger.info(f"🚀 WS Monitor started on http://localhost:{port}")

    try:
        loop.run_until_complete(broadcaster())
    except KeyboardInterrupt:
        logger.info("🔌 WS Monitor shutting down")
    finally:
        loop.run_until_complete(runner.cleanup())
        loop.close()


if __name__ == "__main__":
    main()

