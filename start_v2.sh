#!/bin/bash
echo "🚀 Uruchamianie Agent Zero V2.0..."

# Activate virtual environment
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    echo "✅ Aktywowano środowisko wirtualne"
else
    echo "❌ Brak środowiska wirtualnego"
    exit 1
fi

# Check Neo4j
if curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j działa"
else
    echo "🔄 Uruchamianie Neo4j..."
    docker start neo4j-agent-zero 2>/dev/null || \
    docker run -d --name neo4j-agent-zero \
        -p 7474:7474 -p 7687:7687 \
        -e NEO4J_AUTH=none \
        neo4j:5.15-community
    sleep 15
fi

# Start API
echo "🌐 Uruchamianie API serwera..."
echo "📊 Dostęp: http://localhost:8000/docs"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
