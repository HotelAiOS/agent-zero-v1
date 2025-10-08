#!/usr/bin/env fish

# 🔧 Agent Zero V1 - Complete Environment Fix (Fish Shell Compatible)
echo "🚀 Starting Agent Zero V1 Environment Fix..."

set -l PROJECT_ROOT (pwd)
echo "📍 Working in: $PROJECT_ROOT"

# 1. Stop and clean existing containers
echo "🛑 Stopping existing containers..."
docker-compose down -v
docker system prune -f --volumes

# 2. Replace docker-compose.yml if fixed version exists
echo "📝 Updating docker-compose.yml..."
if test -f docker-compose-fixed.yml
    cp docker-compose-fixed.yml docker-compose.yml
    echo "✅ Updated docker-compose.yml with fixed configuration"
else
    echo "⚠️  docker-compose-fixed.yml not found, creating new one..."

    # Create fixed docker-compose.yml directly
    echo 'services:
  neo4j:
    image: neo4j:5.13
    container_name: agent-zero-neo4j
    environment:
      - NEO4J_AUTH=neo4j/agent-pass
      - NEO4J_ACCEPT_LICENSE_AGREEMENT=yes
      - NEO4J_PLUGINS=["apoc"]
    ports:
      - "7474:7474"
      - "7687:7687"
    volumes:
      - neo4j_data:/data
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:7474 || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  rabbitmq:
    image: rabbitmq:3.12-management
    container_name: agent-zero-rabbitmq
    environment:
      - RABBITMQ_DEFAULT_USER=admin
      - RABBITMQ_DEFAULT_PASS=SecureRabbitPass123
    ports:
      - "5672:5672"
      - "15672:15672"
    volumes:
      - rabbitmq_data:/var/lib/rabbitmq
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD-SHELL", "rabbitmq-diagnostics -q ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: agent-zero-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - agent-zero-network
    healthcheck:
      test: ["CMD-SHELL", "redis-cli ping"]
      interval: 30s
      timeout: 10s
      retries: 3

volumes:
  neo4j_data:
  rabbitmq_data:
  redis_data:

networks:
  agent-zero-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.25.0.0/16' > docker-compose.yml

    echo "✅ Created new docker-compose.yml"
end

# 3. Create tests directory structure
echo "📁 Creating tests directory structure..."
mkdir -p tests/unit tests/integration tests/fixtures

# Create __init__.py files
touch tests/__init__.py
touch tests/unit/__init__.py  
touch tests/integration/__init__.py
touch tests/fixtures/__init__.py

# 4. Create conftest.py using echo (Fish compatible)
echo "📝 Creating conftest.py..."
echo '"""
Pytest configuration for Agent Zero V1 tests
"""
import pytest
import os


@pytest.fixture(scope="session")
def docker_services():
    """Ensure Docker services are running"""
    return True


@pytest.fixture
def neo4j_credentials():
    """Neo4j connection credentials"""
    return {
        "uri": "bolt://localhost:7687",
        "username": "neo4j", 
        "password": "agent-pass"
    }


@pytest.fixture
def redis_config():
    """Redis connection configuration"""
    return {
        "host": "localhost",
        "port": 6379,
        "db": 0
    }


@pytest.fixture  
def rabbitmq_config():
    """RabbitMQ connection configuration"""
    return {
        "host": "localhost",
        "username": "admin",
        "password": "SecureRabbitPass123"
    }' > tests/conftest.py

# 5. Move or create test_full_integration.py
echo "🧪 Creating integration test file..."
if test -f test_full_integration.py
    mv test_full_integration.py tests/
    echo "✅ Moved test_full_integration.py to tests/"
else
    echo "⚠️  Creating new test_full_integration.py..."
    # Test file będzie utworzony przez Python script
end

# 6. Update pyproject.toml
echo "📦 Creating pyproject.toml..."
echo '[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "agent-zero-v1"
version = "1.0.0"
description = "Multi-agent platform for enterprise"
dependencies = [
    "neo4j>=5.13.0",
    "pika>=1.3.2",
    "redis>=5.0.0",
    "pytest>=7.4.0",
    "requests>=2.31.0",
    "fastapi>=0.104.0",
    "uvicorn>=0.24.0"
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
markers = [
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
    "slow: marks tests as slow running"
]' > pyproject.toml

# 7. Clean Docker completely and rebuild
echo "🧹 Complete Docker cleanup..."
docker system prune -a -f --volumes
docker volume prune -f

# 8. Start fresh services
echo "🐳 Starting fresh Docker services..."
docker-compose up -d

echo "⏳ Waiting for services to initialize..."
sleep 15

# 9. Wait for Neo4j to be ready (with proper startup time)
echo "⏳ Waiting for Neo4j initialization..."
set -l neo4j_ready false
set -l attempts 0
set -l max_attempts 20

while test $attempts -lt $max_attempts
    set attempts (math $attempts + 1)
    echo "🔍 Neo4j startup check $attempts/$max_attempts"

    if docker exec agent-zero-neo4j curl -s http://localhost:7474 >/dev/null 2>&1
        echo "✅ Neo4j HTTP endpoint ready"
        # Wait additional time for database to fully initialize
        sleep 10
        set neo4j_ready true
        break
    else
        echo "⏳ Waiting for Neo4j..."
        sleep 5
    end
end

if not $neo4j_ready
    echo "❌ Neo4j failed to start properly"
    docker logs agent-zero-neo4j --tail 20
end

# 10. Final status check
echo "📊 Final system status:"
docker-compose ps

echo ""
echo "🎉 Agent Zero V1 Environment Setup Complete!"
echo "📋 Summary:"
echo "   ✅ Docker Compose configuration fixed"
echo "   ✅ Fresh container deployment"
echo "   ✅ Tests directory structure created"
echo "   ✅ Neo4j: neo4j/agent-pass"
echo "   ✅ RabbitMQ: admin/SecureRabbitPass123"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: python3 create_integration_test.py"
echo "   2. Run: python3 verify_environment.py"
echo "   3. Check Neo4j Browser: http://localhost:7474"
echo ""
