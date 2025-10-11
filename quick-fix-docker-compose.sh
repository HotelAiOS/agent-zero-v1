# HOTFIX - Agent Zero V2.0 Docker Compose Quick Fix
# Saturday, October 11, 2025 @ 09:30 CEST

# Usuń problematyczny docker-compose.yml i zastąp poprawnym

rm -f docker-compose.yml

# Stwórz nowy, poprawny docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

networks:
  agent-zero-network:
    driver: bridge

services:
  neo4j:
    image: neo4j:5.13
    container_name: agent-zero-neo4j
    environment:
      - NEO4J_AUTH=neo4j/agent-pass
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "neo4j", "-p", "agent-pass", "RETURN 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: agent-zero-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - agent-zero-network

  rabbitmq:
    image: rabbitmq:3.12-management
    container_name: agent-zero-rabbitmq
    environment:
      - RABBITMQ_DEFAULT_USER=agent
      - RABBITMQ_DEFAULT_PASS=zero123
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - agent-zero-network

  ai-intelligence:
    build: ./services/ai-intelligence
    container_name: agent-zero-ai-intelligence-v2
    environment:
      - LOG_LEVEL=INFO
    ports:
      - "8010:8010"
    volumes:
      - ai_intelligence_data:/app/data
    networks:
      - agent-zero-network
    depends_on:
      - neo4j
      - redis

  api-gateway:
    build: ./services/api-gateway
    container_name: agent-zero-api-gateway
    environment:
      - LOG_LEVEL=INFO
      - AI_INTELLIGENCE_URL=http://ai-intelligence:8010
    ports:
      - "8000:8080"
    networks:
      - agent-zero-network
    depends_on:
      - ai-intelligence

  websocket-service:
    build: ./services/websocket-service
    container_name: agent-zero-websocket
    environment:
      - LOG_LEVEL=INFO
    ports:
      - "8001:8001"
    networks:
      - agent-zero-network

  agent-orchestrator:
    build: ./services/agent-orchestrator
    container_name: agent-zero-orchestrator
    environment:
      - LOG_LEVEL=INFO
    ports:
      - "8002:8002"
    networks:
      - agent-zero-network

# SINGLE VOLUMES SECTION - BUG FIXED
volumes:
  neo4j_data:
  redis_data:
  rabbitmq_data:
  ai_intelligence_data:
EOF

echo "✅ Docker Compose naprawiony - pojedynczy volumes section"

# Test składni
docker-compose config

echo "🚀 Teraz uruchom ponownie:"
echo "docker-compose down"  
echo "docker-compose up -d"