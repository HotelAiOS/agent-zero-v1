#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Agent Zero V1 – Fix compose ports and Python import paths (GitHub-aligned)"

# 0. Weryfikacja plików bazowych (zgodnie z repo)
test -f docker-compose.yml || { echo "❌ Brak docker-compose.yml w katalogu"; exit 1; }
test -f services/chat-service/src/main.py || { echo "❌ Brak services/chat-service/src/main.py"; exit 1; }
test -f services/agent-orchestrator/src/main.py || { echo "❌ Brak services/agent-orchestrator/src/main.py"; exit 1; }

# 1. Backup plików
ts=$(date +%Y%m%d_%H%M%S)
cp docker-compose.yml docker-compose.yml.bak_$ts
cp services/chat-service/src/main.py services/chat-service/src/main.py.bak_$ts
cp services/agent-orchestrator/src/main.py services/agent-orchestrator/src/main.py.bak_$ts
echo "🗂️ Backup wykonany: *_bak_$ts"

# 2. Poprawa stałej ścieżki project_root w main.py (websocket i orchestrator)
# Zgodnie z Twoim montowaniem: ./:/app/project – to jest źródło prawdy w kontenerze
sed -i 's|project_root = Path(__file__).parent.parent.parent|project_root = Path("/app/project")|' services/chat-service/src/main.py
sed -i 's|project_root = Path(__file__).parent.parent.parent|project_root = Path("/app/project")|' services/agent-orchestrator/src/main.py

# 3. Poprawne healthcheck porty w docker-compose (na podstawie aktualnych plików z GitHub):
# - websocket-service: kontener expose/serve 8080, healthcheck powinien uderzać w 8080
# - agent-orchestrator: słucha na 8002 w kontenerze, healthcheck zostaje 8002 (zgodnie z Twoim compose)
# - api-gateway: 8000->8000 w compose – zostawiamy
# Upewnijmy się, że websocket-service ma ports 8080:8080 oraz healthcheck 8080
# Uwaga: Operujemy delikatnie (tylko jeśli nie są już poprawne)

# Ustaw poprawny mapping dla websocket-service (jeśli zmieniałeś lokalnie wcześniej)
if grep -q 'websocket-service:' -n docker-compose.yml; then
  # healthcheck -> 8080
  sed -i '/websocket-service:/,/healthcheck:/ s|http://localhost:[0-9]*/health|http://localhost:8080/health|' docker-compose.yml
  # ports -> 8080:8080
  sed -i '/websocket-service:/,/\(networks\|depends_on\|healthcheck\)/ s|"[0-9]\{4\}:[0-9]\{4\}"|"8080:8080"|' docker-compose.yml
fi

# 4. Walidacja składni docker-compose
docker-compose config >/dev/null || { echo "❌ Błąd składni docker-compose.yml"; exit 1; }
echo "✅ docker-compose.yml OK"

# 5. Restart wyłącznie services aplikacyjnych
echo "🔄 Restart websocket-service i agent-orchestrator..."
docker-compose restart websocket-service agent-orchestrator || true

echo "⏳ Czekam 20s na start..."
sleep 20

# 6. Testy wewnątrz kontenerów (pewne porty)
echo "🧪 Testy health wewnątrz kontenerów:"
docker exec agent-zero-websocket sh -lc 'curl -sf http://localhost:8080/health || (echo "ws FAIL" && exit 1)' && echo "✅ WebSocket OK" || echo "❌ WebSocket FAIL"
docker exec agent-zero-orchestrator sh -lc 'curl -sf http://localhost:8002/health || (echo "orch FAIL" && exit 1)' && echo "✅ Orchestrator OK" || echo "❌ Orchestrator FAIL"

# 7. Testy z hosta (odzwierciedlenie Twojego compose)
echo "🧪 Testy health z hosta:"
curl -sf http://localhost:8080/health && echo "✅ WebSocket (host:8080) OK" || echo "❌ WebSocket (host:8080) FAIL"
curl -sf http://localhost:8002/health && echo "✅ Orchestrator (host:8002) OK" || echo "❌ Orchestrator (host:8002) FAIL"
curl -sf http://localhost:8000/health && echo "✅ API Gateway (host:8000) OK" || echo "❌ API Gateway (host:8000) FAIL"

echo "🎯 Fix complete. Jeśli nadal FAIL, uruchom poniższy skrypt diagnostyczny."
