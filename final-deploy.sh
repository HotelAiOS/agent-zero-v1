#!/bin/bash
# final-deploy.sh - Kompletny deployment Agent Zero V1

echo "🚀 Agent Zero V1 - Final Deployment"
echo "===================================="

# Sprawdź czy jesteś w root directory projektu
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Błąd: Uruchom script w root directory projektu (gdzie jest docker-compose.yml)"
    exit 1
fi

# 1. Zatrzymaj wszystkie services i wyczyść
echo "🧹 Czyszczenie poprzedniego deployment..."
docker-compose down -v --remove-orphans
docker system prune -f

# 2. Sprawdź konfigurację
echo "🔍 Sprawdzanie konfiguracji docker-compose..."
if ! docker-compose config --quiet; then
    echo "❌ Błąd w docker-compose.yml! Fix syntax errors first."
    exit 1
fi
echo "✅ docker-compose.yml syntax OK"

# 3. Build all images
echo "🔨 Building Docker images..."
docker-compose build --no-cache

if [ $? -ne 0 ]; then
    echo "❌ Build failed! Check Dockerfiles."
    exit 1
fi
echo "✅ All images built successfully"

# 4. Start infrastructure services first
echo "🏗️ Starting infrastructure services..."
docker-compose up -d neo4j redis rabbitmq

# 5. Wait for infrastructure
echo "⏳ Waiting for infrastructure to be healthy..."
WAIT_TIME=0
MAX_WAIT=120

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    NEO4J_HEALTH=$(docker inspect agent-zero-neo4j --format='{{.State.Health.Status}}' 2>/dev/null)
    REDIS_HEALTH=$(docker inspect agent-zero-redis --format='{{.State.Health.Status}}' 2>/dev/null)
    RABBITMQ_HEALTH=$(docker inspect agent-zero-rabbitmq --format='{{.State.Health.Status}}' 2>/dev/null)
    
    if [ "$NEO4J_HEALTH" = "healthy" ] && [ "$REDIS_HEALTH" = "healthy" ] && [ "$RABBITMQ_HEALTH" = "healthy" ]; then
        echo "✅ Infrastructure healthy after ${WAIT_TIME}s"
        break
    fi
    
    echo "⏳ Infrastructure status - Neo4j: $NEO4J_HEALTH, Redis: $REDIS_HEALTH, RabbitMQ: $RABBITMQ_HEALTH"
    sleep 15
    WAIT_TIME=$((WAIT_TIME + 15))
done

if [ $WAIT_TIME -ge $MAX_WAIT ]; then
    echo "❌ Infrastructure failed to become healthy within ${MAX_WAIT}s"
    echo "🔍 Checking logs..."
    docker-compose logs neo4j --tail=10
    exit 1
fi

# 6. Start application services
echo "🚀 Starting application services..."
docker-compose up -d api-gateway websocket-service agent-orchestrator

# 7. Wait for application services
echo "⏳ Waiting for application services to be ready..."
sleep 60

# 8. Health check all services
echo "🏥 Running health checks..."

# Test all endpoints
API_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health 2>/dev/null)
WS_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8001/health 2>/dev/null)
ORCH_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8002/health 2>/dev/null)

echo ""
echo "🎯 Service Status Report:"
echo "========================="
docker-compose ps
echo ""
echo "📡 Endpoint Health:"
echo "API Gateway (8000):      HTTP $API_STATUS"
echo "WebSocket Service (8001): HTTP $WS_STATUS"
echo "Agent Orchestrator (8002): HTTP $ORCH_STATUS"

# 9. Final verification
if [ "$API_STATUS" = "200" ] && [ "$WS_STATUS" = "200" ] && [ "$ORCH_STATUS" = "200" ]; then
    echo ""
    echo "🎉 SUCCESS! Agent Zero V1 is fully deployed and healthy!"
    echo ""
    echo "🌐 Available Services:"
    echo "┌─────────────────────────────────────────────────────┐"
    echo "│ API Gateway:          http://localhost:8000         │"
    echo "│ WebSocket Service:    http://localhost:8001         │"
    echo "│ Agent Orchestrator:   http://localhost:8002         │"
    echo "│ Neo4j Browser:        http://localhost:7474         │"
    echo "│   (Login: neo4j / password123)                      │"
    echo "│ RabbitMQ Management:  http://localhost:15672        │"
    echo "│   (Login: admin / SecureRabbitPass123)              │"
    echo "└─────────────────────────────────────────────────────┘"
    echo ""
    echo "🔗 Test API endpoints:"
    echo "curl http://localhost:8000/api/v1/agents/status"
    echo "curl http://localhost:8000/health"
    echo ""
    echo "✅ Developer B can start frontend integration!"
else
    echo ""
    echo "⚠️ Some services are not responding correctly"
    echo "🔍 Check individual service logs:"
    echo "docker-compose logs api-gateway"
    echo "docker-compose logs websocket-service"
    echo "docker-compose logs agent-orchestrator"
    
    # Show logs for failing services
    if [ "$API_STATUS" != "200" ]; then
        echo ""
        echo "📋 API Gateway logs (last 10 lines):"
        docker-compose logs api-gateway --tail=10
    fi
    
    if [ "$WS_STATUS" != "200" ]; then
        echo ""
        echo "📋 WebSocket Service logs (last 10 lines):"
        docker-compose logs websocket-service --tail=10
    fi
    
    if [ "$ORCH_STATUS" != "200" ]; then
        echo ""
        echo "📋 Agent Orchestrator logs (last 10 lines):"
        docker-compose logs agent-orchestrator --tail=10
    fi
fi

echo ""
echo "📋 Useful commands:"
echo "docker-compose ps                    # Check service status"
echo "docker-compose logs [service-name]   # View service logs"
echo "docker-compose restart [service]     # Restart specific service"
echo "docker-compose down                  # Stop all services"
