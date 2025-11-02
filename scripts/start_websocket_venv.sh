#!/bin/bash
# Agent Zero V1 - Start WebSocket Monitor in Virtual Environment

set -e

echo "🌐 Agent Zero V1 - Starting WebSocket Monitor"
echo "============================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: ./create_venv_and_install.sh first"
    exit 1
fi

# Check if websocket_monitor_minimal.py exists
if [ ! -f "websocket_monitor_minimal.py" ]; then
    echo "❌ websocket_monitor_minimal.py not found!"
    echo "Make sure all files are copied to project directory"
    exit 1
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Start WebSocket monitor
echo "🌐 Starting WebSocket monitor on http://localhost:8000"
echo "💡 Press Ctrl+C to stop"
echo ""
python websocket_monitor_minimal.py