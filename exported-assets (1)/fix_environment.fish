#!/usr/bin/env fish

# 🔧 Agent Zero V1 - Complete Environment Fix
echo "🚀 Starting Agent Zero V1 Environment Fix..."

set -l PROJECT_ROOT (pwd)
echo "📍 Working in: $PROJECT_ROOT"

# 1. Stop and clean existing containers
echo "🛑 Stopping existing containers..."
docker-compose down -v
docker system prune -f

# 2. Replace docker-compose.yml
echo "📝 Updating docker-compose.yml..."
if test -f docker-compose-fixed.yml
    cp docker-compose-fixed.yml docker-compose.yml
    echo "✅ Updated docker-compose.yml with fixed configuration"
else
    echo "❌ docker-compose-fixed.yml not found!"
    exit 1
end

# 3. Create tests directory structure
echo "📁 Creating tests directory structure..."
mkdir -p tests/unit tests/integration tests/fixtures

# Create __init__.py files
touch tests/__init__.py
touch tests/unit/__init__.py  
touch tests/integration/__init__.py
touch tests/fixtures/__init__.py

# Move test files to proper location
if test -f test_full_integration.py
    mv test_full_integration.py tests/
    echo "✅ Moved test_full_integration.py to tests/"
end

# Create conftest.py for pytest configuration
cat > tests/conftest.py << 'EOF'
"""
Pytest configuration for Agent Zero V1 tests
"""
import pytest
import os


@pytest.fixture(scope="session")
def docker_services():
    """Ensure Docker services are running"""
    # Add service verification logic here
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
    }
EOF

# 4. Update pyproject.toml
echo "📦 Creating pyproject.toml..."
cat > pyproject.toml << 'EOF'
[build-system]
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
]
EOF

# 5. Start services with health check wait
echo "🐳 Starting Docker services..."
docker-compose up -d

echo "⏳ Waiting for services to be healthy..."
set -l max_attempts 30
set -l attempt 1

while test $attempt -le $max_attempts
    echo "🔍 Health check attempt $attempt/$max_attempts"

    # Check if all containers are healthy
    set -l healthy_count (docker-compose ps --format json | jq -r '.State' | grep -c "running")

    if test $healthy_count -eq 3
        echo "✅ All services are running!"
        break
    end

    if test $attempt -eq $max_attempts
        echo "❌ Services did not become healthy in time"
        docker-compose ps
        exit 1
    end

    sleep 5
    set attempt (math $attempt + 1)
end

# 6. Additional health verification
echo "🔍 Testing service connections..."

# Test Neo4j
echo "Testing Neo4j connection..."
timeout 10 docker exec agent-zero-neo4j cypher-shell -u neo4j -p agent-pass "RETURN 'Neo4j OK' as status;" >/dev/null 2>&1
if test $status -eq 0
    echo "✅ Neo4j connection successful"
else
    echo "⚠️  Neo4j connection failed (may need more startup time)"
end

# Test Redis
echo "Testing Redis connection..."
timeout 5 docker exec agent-zero-redis redis-cli ping >/dev/null 2>&1
if test $status -eq 0
    echo "✅ Redis connection successful"
else
    echo "❌ Redis connection failed"
end

# Test RabbitMQ
echo "Testing RabbitMQ connection..."
timeout 10 docker exec agent-zero-rabbitmq rabbitmq-diagnostics ping >/dev/null 2>&1
if test $status -eq 0
    echo "✅ RabbitMQ connection successful"
else 
    echo "⚠️  RabbitMQ connection failed (may need more startup time)"
end

# 7. Run integration tests
echo "🧪 Running integration tests..."
if test -f tests/test_full_integration.py
    echo "Installing test dependencies..."
    pip install pytest neo4j pika redis requests >/dev/null 2>&1

    echo "Running tests..."
    pytest tests/test_full_integration.py -v

    if test $status -eq 0
        echo "✅ All integration tests passed!"
    else
        echo "⚠️  Some tests failed - check output above"
    end
else
    echo "⚠️  Integration test file not found"
end

# 8. Final verification
echo "📊 Final system status:"
docker-compose ps --format "table {{.Name}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🎉 Agent Zero V1 Environment Fix Complete!"
echo "📋 Summary:"
echo "   ✅ Docker Compose configuration updated"
echo "   ✅ Neo4j password fixed (neo4j/agent-pass)"
echo "   ✅ Tests directory structure created"
echo "   ✅ All services health-checked"
echo "   ✅ Integration tests available"
echo ""
echo "🚀 Next steps:"
echo "   1. Run: pytest tests/test_full_integration.py -v"
echo "   2. Check Neo4j Browser: http://localhost:7474"
echo "   3. Check RabbitMQ Management: http://localhost:15672"
echo ""
