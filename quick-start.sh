#!/bin/bash

# 🔧 AGENT ZERO V1 - QUICK SYSTEM REPAIR & START
# ================================================
# Szybka naprawa i uruchomienie głównego systemu

echo "🚀 AGENT ZERO V1 - SYSTEM REPAIR & START"
echo "========================================"

# 1. Zatrzymaj wszystkie konflitkujące procesy
echo "🛑 Zatrzymywanie istniejących procesów..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
pkill -f "integrated-system" 2>/dev/null || true
pkill -f "python.*integrated" 2>/dev/null || true

# 2. Fix Python 3.13 + Pydantic issues
echo "🔧 Naprawianie Python 3.13 + Pydantic..."
pip install --upgrade pip --quiet
pip install --only-binary=all fastapi uvicorn pydantic --quiet
pip install neo4j==5.17.0 --no-deps --quiet
pip install pytz --quiet

# 3. Sprawdź czy plik główny istnieje
if [ ! -f "integrated-system.py" ]; then
    echo "❌ ERROR: integrated-system.py nie istnieje!"
    echo "Sprawdź czy jesteś w właściwym katalogu"
    exit 1
fi

echo "✅ Główny plik znaleziony: integrated-system.py"

# 4. Sprawdź dostępność portów
if netstat -tlnp 2>/dev/null | grep -q ":8000"; then
    echo "⚠️ Port 8000 nadal zajęty - próbuję wyczyścić..."
    sudo fuser -k 8000/tcp 2>/dev/null || true
    sleep 2
fi

# 5. Test basic dependencies
echo "🧪 Testowanie podstawowych zależności..."
python3 -c "import fastapi, uvicorn, pydantic; print('✅ FastAPI dependencies OK')" || {
    echo "❌ Missing dependencies - installing..."
    pip install fastapi uvicorn pydantic aiohttp neo4j redis pika websockets
}

# 6. Uruchom system w trybie production
echo ""
echo "🚀 Uruchamianie Agent Zero V1 System..."
echo "Mode: Production Server"
echo "Port: 8000"
echo "Endpoints: /api/v1/health, /docs"
echo ""

# Start with proper error handling
python3 integrated-system.py --mode production &
SYSTEM_PID=$!

# Wait a moment for startup
sleep 3

# Test if system started correctly
if ps -p $SYSTEM_PID > /dev/null; then
    echo "✅ System started successfully (PID: $SYSTEM_PID)"
    echo ""
    echo "📡 Testing endpoints..."
    
    # Test health endpoint
    if curl -s http://localhost:8000/api/v1/health > /dev/null; then
        echo "✅ Health endpoint: http://localhost:8000/api/v1/health"
        echo "✅ Documentation: http://localhost:8000/docs"
        echo "✅ Redoc: http://localhost:8000/redoc"
        echo ""
        echo "🎉 SYSTEM READY!"
        echo "Use: curl http://localhost:8000/api/v1/health"
    else
        echo "⚠️ Health endpoint not responding yet (still starting...)"
        echo "Try: curl http://localhost:8000/api/v1/health in a moment"
    fi
else
    echo "❌ System failed to start - checking logs..."
    echo "Last few lines of output:"
    tail -10 agent_zero_integrated.log 2>/dev/null || echo "No log file found"
fi

echo ""
echo "📋 QUICK REFERENCE:"
echo "- Health: curl http://localhost:8000/api/v1/health"
echo "- Docs: http://localhost:8000/docs"
echo "- Stop: kill $SYSTEM_PID"
echo "- Logs: tail -f agent_zero_integrated.log"