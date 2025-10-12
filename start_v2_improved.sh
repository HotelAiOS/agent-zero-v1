#!/bin/bash
echo "🚀 Agent Zero V2.0 - Ulepszone Uruchomienie"

# Default port
PORT=${1:-8000}

# Kill existing processes
pkill -f uvicorn 2>/dev/null || true
pkill -f "api.main" 2>/dev/null || true
sleep 2

# Activate virtual environment
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ Virtual environment not found"
    exit 1
fi

# Check Neo4j
if curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j running on port 7474"
else
    echo "🔄 Starting Neo4j..."
    docker start neo4j-agent-zero 2>/dev/null || {
        docker run -d --name neo4j-agent-zero \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=none \
            neo4j:5.15-community
        sleep 15
    }
fi

# Check port availability
if netstat -ln | grep -q ":${PORT} "; then
    echo "⚠️  Port ${PORT} is busy, trying port $((PORT + 1))"
    PORT=$((PORT + 1))
fi

echo "🌐 Starting API on port ${PORT}..."
echo "📊 Access: http://localhost:${PORT}/docs"
echo "🔍 Neo4j: http://localhost:7474"
echo ""
echo "Press Ctrl+C to stop"

# Start API with error handling
python -m uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --reload
