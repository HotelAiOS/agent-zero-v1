#!/bin/bash
# Fish-compatible test runner

echo "🧪 Agent Zero V2.0 - Fish Shell Test Runner"
echo "=========================================="

# Use venv python directly
echo "Using venv Python: $(pwd)/venv/bin/python"

# Check Neo4j
if curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j is running"
else
    echo "🔄 Starting Neo4j..."
    docker start neo4j-agent-zero 2>/dev/null || {
        docker run -d --name neo4j-agent-zero \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=none \
            neo4j:5.15-community
        sleep 10
    }
fi

# Run tests with venv python
echo "🧪 Running tests..."
export PYTHONPATH="$(pwd):$PYTHONPATH"
venv/bin/python test-complete-implementation.py

echo "📊 Test Summary:"
echo "   Neo4j: $(curl -s http://localhost:7474 > /dev/null 2>&1 && echo "✅ OK" || echo "❌ FAIL")"
echo "   ML Packages: $(venv/bin/python -c "import joblib, sklearn; print('✅ OK')" 2>/dev/null || echo "❌ FAIL")"
