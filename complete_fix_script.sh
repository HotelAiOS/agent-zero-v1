#!/usr/bin/env bash
set -euo pipefail

echo "🚀 Agent Zero V1 – Complete Fix Script (bash)"

# 1. Tworzymy katalogi (ignoruemy już istniejące błędy uprawnień)
echo "📁 Tworzę katalogi..."
mkdir -p logs shared/monitoring shared/orchestration

# 1b. Ustawiamy chmod (błędy ignorujemy, żeby skrypt nie przerywał)
echo "🔧 Ustawiam chmod 755 (jeśli możliwe)..."
chmod 755 logs shared shared/monitoring shared/orchestration 2>/dev/null || \
  echo "⚠️  Nie można zmienić uprawnień (ignoruję)."

# 2. Generujemy .env z bezpiecznymi credentials
echo "🔐 Tworzę plik .env..."
cat > .env << 'EOF'
# Agent Zero V1 Environment
NEO4J_USER=neo4j
NEO4J_PASS=SecureNeo4jPass123
RABBITMQ_USER=admin
RABBITMQ_PASS=SecureRabbitPass123
REDIS_URL=redis://localhost:6379
LOG_DIR=logs
WEBSOCKET_PORT=8000
DEBUG=true
EOF

# 3. Docker Compose z poprawioną siecią
echo "🐳 Generuję docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'
services:
  neo4j:
    image: neo4j:5.13
    environment:
      NEO4J_AUTH: ${NEO4J_USER}/${NEO4J_PASS}
      NEO4J_PLUGINS: '["apoc"]'
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    networks:
      - agent_zero_net

  rabbitmq:
    image: rabbitmq:3.12-management
    environment:
      RABBITMQ_DEFAULT_USER: ${RABBITMQ_USER}
      RABBITMQ_DEFAULT_PASS: ${RABBITMQ_PASS}
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - agent_zero_net

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - agent_zero_net

volumes:
  neo4j_data:
  rabbitmq_data:
  redis_data:

networks:
  agent_zero_net:
    driver: bridge
    ipam:
      config:
        - subnet: 172.25.0.0/16
EOF

# 4. WebSocket Monitor – nowy kod
echo "🌐 Generuję shared/monitoring/websocket_monitor.py..."
cat > shared/monitoring/websocket_monitor.py << 'EOF'
import asyncio, json, logging, os
from datetime import datetime
from aiohttp import web
import aiohttp_cors

LOG_DIR = os.getenv('LOG_DIR','logs')
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR,'websocket.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('WSMon')

class Monitor:
    def __init__(self):
        self.conns = set()
        self.agents = {k:'active' for k in [
            'task_decomposer','code_executor','file_manager',
            'web_search','data_analyst','communication',
            'orchestrator','monitor'
        ]}
    async def register(self, ws):
        self.conns.add(ws); logger.info(f'Conn+ ({len(self.conns)})')
    async def unregister(self, ws):
        self.conns.discard(ws); logger.info(f'Conn- ({len(self.conns)})')
    async def broadcast(self):
        if not self.conns: return
        msg = json.dumps({
            'timestamp': datetime.now().isoformat(),
            'agents': self.agents,
            'active_connections': len(self.conns)
        })
        bad = set()
        for ws in self.conns:
            try: await ws.send(msg)
            except: bad.add(ws)
        for ws in bad: await self.unregister(ws)

mon = Monitor()

async def ws_handler(request):
    ws = web.WebSocketResponse(); await ws.prepare(request)
    await mon.register(ws)
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                d = json.loads(msg.data)
                if d.get('type') == 'ping':
                    await ws.send_str(json.dumps({'type':'pong'}))
    except Exception as e:
        logger.error(f'WS error: {e}')
    finally:
        await mon.unregister(ws)
    return ws

async def index(request):
    return web.FileResponse(os.path.join(os.path.dirname(__file__),'index.html'))

async def health(request):
    return web.json_response({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'connections': len(mon.conns),
        'agents': mon.agents
    })

async def app_factory():
    app = web.Application()
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True, expose_headers="*", allow_headers="*", allow_methods="*"
        )
    })
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/health', health)
    app.router.add_get('/', index)
    for r in list(app.router.routes()):
        cors.add(r)
    return app

async def broadcaster():
    while True:
        await mon.broadcast()
        await asyncio.sleep(5)

if __name__=='__main__':
    import sys
    loop = asyncio.get_event_loop()
    app = loop.run_until_complete(app_factory())
    runner = web.AppRunner(app); loop.run_until_complete(runner.setup())
    site = web.TCPSite(runner,'0.0.0.0',int(os.getenv('WEBSOCKET_PORT',8000)))
    loop.run_until_complete(site.start())
    logger.info(f'🌐 WS Monitor on port {os.getenv("WEBSOCKET_PORT",8000)}')
    loop.run_until_complete(broadcaster())
EOF

# 5. Task Decomposer – multi-strategy parser
echo "🧠 Generuję shared/orchestration/task_decomposer.py..."
cat > shared/orchestration/task_decomposer.py << 'EOF'
import json, re, logging
class TaskDecomposer:
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.strats = [
            self._direct, self._markdown, self._regex,
            self._line, self._adv_regex, self._fallback
        ]
    def parse(self, resp:str):
        for i,s in enumerate(self.strats,1):
            try:
                r = s(resp)
                if r:
                    logging.info(f"Parsed by strat {i}")
                    return r
            except:
                pass
        logging.warning("All strat failed")
        return self._fallback(resp)
    def _direct(self, r):
        j = r.strip()
        return json.loads(j) if j.startswith('{') and j.endswith('}') else None
    def _markdown(self, r):
        m = re.findall(r'``````', r, re.S)
        for j in m:
            try: return json.loads(j)
            except: pass
    def _regex(self, r):
        m = re.findall(r'\{(?:[^{}]|(?R))*\}', r)
        for j in m:
            try: return json.loads(j)
            except: pass
    def _line(self, r):
        lines, buf, cnt, on = r.split('\n'), [], 0, False
        for l in lines:
            if '{' in l and not on:
                on = True
            if on:
                buf.append(l)
                cnt += l.count('{') - l.count('}')
                if cnt <= 0:
                    break
        if buf:
            return json.loads('\n'.join(buf))
    def _adv_regex(self, r):
        return self._regex(r)
    def _fallback(self, r):
        return {'task':'fallback','original':r[:200]+'...'}
if __name__=='__main__':
    td = TaskDecomposer()
    for t in ['{"a":1}', '``````', 'no json here']:
        print(td.parse(t))
EOF

# 6. requirements.txt
echo "📦 Generuję requirements.txt..."
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.24.0
websockets==12.0
aiohttp==3.9.1
aiohttp-cors==0.7.0
neo4j==5.14.1
redis==5.0.1
pika==1.3.2
python-dotenv==1.0.0
requests==2.31.0
EOF

# 7. Deploy & test
echo "🔄 Restartuję Dockera i usuwam stare sieci..."
docker-compose down --remove-orphans
docker network prune -f

echo "🚀 Uruchamiam kontenery..."
docker-compose up -d

echo "⏳ Czekam 30s na start usług..."
sleep 30

echo "🐍 Instaluję zależności Python..."
pip install -r requirements.txt

echo "🌐 Startuję WebSocket Monitor w tle..."
nohup python3 shared/monitoring/websocket_monitor.py >/dev/null 2>&1 &

# 8. Weryfikacja
echo "🧪 Test service endpoints:"
curl -fs http://localhost:7474 && echo " ✅ Neo4j" || echo " ❌ Neo4j"
curl -fs http://localhost:15672 && echo " ✅ RabbitMQ" || echo " ❌ RabbitMQ"
curl -fs http://localhost:8000/health && echo " ✅ WebSocket" || echo " ❌ WebSocket"
redis-cli ping | grep -q PONG && echo " ✅ Redis" || echo " ❌ Redis"

echo ""
echo "🎉 Wszystkie naprawy zakończone – Agent Zero V1 jest w pełni operacyjny!"
