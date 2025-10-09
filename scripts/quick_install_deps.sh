#!/bin/bash
# Agent Zero V1 - Quick Install Dependencies

echo "🚀 Agent Zero V1 - Quick Dependency Install"
echo "==========================================="

# Check if we're in correct directory
if [ ! -f "shared/execution/agent_executor.py" ]; then
    echo "❌ Error: Not in agent-zero-v1 root directory"
    echo "Please run from: /home/ianua/projects/agent-zero-v1"
    exit 1
fi

# Install from requirements if it exists, otherwise individual packages
if [ -f "venv_requirements.txt" ]; then
    echo "📦 Installing from venv_requirements.txt..."
    
    # Create venv if not exists
    if [ ! -d "venv" ]; then
        echo "🔧 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    echo "🚀 Activating virtual environment..."
    source venv/bin/activate
    
    echo "📦 Installing all dependencies..."
    pip install --upgrade pip
    pip install -r venv_requirements.txt
    
else
    echo "📦 Installing individual packages..."
    
    # Create venv if not exists
    if [ ! -d "venv" ]; then
        echo "🔧 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    echo "🚀 Activating virtual environment..."
    source venv/bin/activate
    
    echo "📦 Installing critical packages..."
    pip install --upgrade pip
    pip install docker==6.1.3 aiohttp-cors==0.7.0 pytest==7.4.3
    pip install fastapi==0.104.1 uvicorn==0.24.0 websockets==12.0
    pip install neo4j==5.14.1 aiohttp==3.9.1
fi

echo "✅ Dependencies installed!"
echo ""
echo "🎯 Next steps:"
echo "   • To activate venv: source venv/bin/activate"
echo "   • To run tests: python agent_zero_system_test_venv.py"
echo "   • To start WebSocket: python websocket_monitor_minimal.py"