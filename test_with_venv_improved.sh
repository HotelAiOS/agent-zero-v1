#!/bin/bash
# Improved test script with better error handling

echo "🧪 Agent Zero V2.0 - Improved Test Runner"
echo "========================================"

# Check and activate virtual environment  
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found"
    echo "   Fix: Run ./quick-arch-fix.sh first"
    exit 1
fi

# Check Neo4j before testing
if curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j is running"
else
    echo "🔄 Starting Neo4j for tests..."
    docker start neo4j-agent-zero 2>/dev/null || {
        docker run -d --name neo4j-agent-zero \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=none \
            neo4j:5.15-community
        sleep 10
    }
fi

# Run test fixes first
echo "🔧 Running test fixes..."
python test_fixes.py

echo ""
echo "🧪 Running main tests..."
python test-complete-implementation.py "$@"

# Show summary
echo ""
echo "📊 Test Summary:"
echo "   Neo4j: $(curl -s http://localhost:7474 > /dev/null 2>&1 && echo "✅ OK" || echo "❌ FAIL")"
echo "   ML Packages: $(python -c "import joblib, sklearn; print('✅ OK')" 2>/dev/null || echo "❌ FAIL")"
echo "   Components: See test results above"
