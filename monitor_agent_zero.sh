#!/bin/bash
echo "📊 AGENT ZERO V1 - SYSTEM MONITORING"
echo "=================================="

echo "🔍 Running processes:"
ps aux | grep -E "(agent|uvicorn|python.*8[0-9]{3})" | grep -v grep || echo "No Agent Zero processes found"

echo ""
echo "🌐 Port usage:"
for port in 8000 8002 8003 8005 8006 8007 8008 9001; do
    if lsof -i :$port &>/dev/null; then
        echo "✅ Port $port: In use"
    else
        echo "❌ Port $port: Available"
    fi
done

echo ""
echo "🏥 Health checks:"
for port in 8000 8002 8003 8005 8006 8007 8008 9001; do
    if curl -s --max-time 2 "http://localhost:$port/" > /dev/null 2>&1; then
        echo "✅ Service on port $port: Healthy"
    else
        echo "❌ Service on port $port: Not responding"
    fi
done
