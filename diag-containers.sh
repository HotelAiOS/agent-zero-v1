#!/usr/bin/env bash
set -euo pipefail

echo "🔍 Diagnoza Agent Zero V1"

echo "📦 docker-compose ps"
docker-compose ps

echo
echo "📜 Logi WebSocket (20 linii)"
docker-compose logs websocket-service --tail=20 || true

echo
echo "📜 Logi Orchestrator (20 linii)"
docker-compose logs agent-orchestrator --tail=20 || true

echo
echo "🗂️ Zawartość /app/project w WebSocket"
docker exec agent-zero-websocket sh -lc 'ls -la /app/project | head -50' || true

echo
echo "🔎 Sprawdzenie critical files w WebSocket"
docker exec agent-zero-websocket sh -lc 'test -f /app/project/simple-tracker.py && echo "✅ simple-tracker.py OK" || echo "❌ simple-tracker.py MISSING"' || true
docker exec agent-zero-websocket sh -lc 'test -f /app/project/feedback-loop-engine.py && echo "ℹ️ feedback-loop-engine.py found" || echo "ℹ️ feedback-loop-engine.py not present (optional)"' || true

echo
echo "🔎 Sprawdzenie critical files w Orchestrator"
docker exec agent-zero-orchestrator sh -lc 'test -f /app/project/simple-tracker.py && echo "✅ simple-tracker.py OK" || echo "❌ simple-tracker.py MISSING"' || true
docker exec agent-zero-orchestrator sh -lc 'for f in agent_executor.py task_decomposer.py neo4j_client.py; do test -f /app/project/$f && echo "✅ $f OK" || echo "ℹ️ $f not found (ok jeśli nieużywany w ścieżce startu)"; done' || true

echo
echo "🌐 Testy wewnętrzne HTTP"
docker exec agent-zero-websocket sh -lc 'curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/health || true'
docker exec agent-zero-orchestrator sh -lc 'curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/health || true'
