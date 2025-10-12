#!/bin/bash
echo "🏥 Agent Zero V2.0 - Health Check"
echo "================================="

# Check virtual environment
if [[ -f "venv/bin/activate" ]]; then
    echo "✅ Virtual environment: OK"
    source venv/bin/activate
    
    # Check Python packages
    python -c "import joblib, sklearn, pandas, neo4j; print('✅ ML packages: OK')" 2>/dev/null || echo "❌ ML packages: FAIL"
else
    echo "❌ Virtual environment: MISSING"
fi

# Check Neo4j
if curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j: OK (http://localhost:7474)"
else
    echo "❌ Neo4j: NOT RUNNING"
    echo "   Fix: docker start neo4j-agent-zero"
fi

# Check API ports
for port in 8000 8001 8002; do
    if netstat -ln 2>/dev/null | grep -q ":${port} "; then
        echo "⚠️  Port ${port}: BUSY"
    else
        echo "✅ Port ${port}: Available"
        break
    fi
done

# Check Docker
if docker ps | grep -q neo4j; then
    echo "✅ Docker Neo4j: RUNNING"
else
    echo "⚠️  Docker Neo4j: NOT RUNNING"
fi

echo ""
echo "🚀 Quick Actions:"
echo "   Start system: ./start_v2_improved.sh"
echo "   Run tests: ./test_with_venv.sh"
echo "   Fix tests: python test_fixes.py"
