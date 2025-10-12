#!/bin/bash
echo "🛑 AGENT ZERO V1 - SYSTEM SHUTDOWN"
echo "================================"

echo "Stopping all Agent Zero processes..."
pkill -f "python.*agent" 2>/dev/null && echo "✅ Agent Zero processes stopped" || echo "⚠️ No Agent Zero processes found"

echo "Checking for remaining processes..."
ps aux | grep -E "(agent|uvicorn|python.*8[0-9]{3})" | grep -v grep || echo "✅ No remaining processes"

echo "🏁 Agent Zero system shutdown complete"
