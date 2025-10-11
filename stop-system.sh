#!/bin/bash

# 🛑 Agent Zero V1 - System Shutdown Script

echo "🛑 STOPPING AGENT ZERO V1 INTEGRATED SYSTEM"
echo "=========================================="

# Kill system processes
if [ -f "system.pid" ]; then
    PID=$(cat system.pid)
    echo "🔌 Stopping main system (PID: $PID)..."
    kill $PID 2>/dev/null || echo "⚠️  Process already stopped"
    rm system.pid
fi

if [ -f "dashboard.pid" ]; then
    PID=$(cat dashboard.pid)
    echo "📊 Stopping dashboard (PID: $PID)..."
    kill $PID 2>/dev/null || echo "⚠️  Process already stopped"
    rm dashboard.pid
fi

# Stop Docker services
echo "🐳 Stopping Docker services..."
docker-compose down

echo "✅ SYSTEM STOPPED SUCCESSFULLY!"
