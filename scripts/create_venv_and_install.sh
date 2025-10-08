#!/bin/bash
# Agent Zero V1 - Arch Linux Virtual Environment Setup
# Rozwiązuje problem externally-managed-environment w Arch Linux

set -e

echo "🐧 Agent Zero V1 - Arch Linux Setup"
echo "===================================="

# Check if we're in correct directory
if [ ! -f "shared/execution/agent_executor.py" ]; then
    echo "❌ Error: Not in agent-zero-v1 root directory"
    echo "Please run from: /home/ianua/projects/agent-zero-v1"
    exit 1
fi

# Create virtual environment
echo "🔧 Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created: venv/"
else
    echo "✅ Virtual environment already exists: venv/"
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install core dependencies
echo "📦 Installing Agent Zero V1 dependencies..."

# Core framework
pip install fastapi==0.104.1
pip install uvicorn==0.24.0
pip install websockets==12.0
pip install aiohttp==3.9.1
pip install aiohttp-cors==0.7.0

# Database & Message Queue
pip install neo4j==5.14.1
pip install redis==5.0.1
pip install pika==1.3.2

# Environment & Configuration
pip install python-dotenv==1.0.0
pip install requests==2.31.0

# Docker SDK
pip install docker==6.1.3

# Testing framework
pip install pytest==7.4.3
pip install pytest-asyncio==0.21.1
pip install pytest-cov==4.1.0

# Development tools
pip install psutil==5.9.6

echo "✅ All dependencies installed in virtual environment!"

# Verify installations
echo "🔍 Verifying critical modules..."
python -c "import docker; print('✅ docker module available')"
python -c "import aiohttp_cors; print('✅ aiohttp_cors module available')"
python -c "import neo4j; print('✅ neo4j module available')"
python -c "import pytest; print('✅ pytest module available')"

echo ""
echo "🎯 Virtual environment ready!"
echo "To activate: source venv/bin/activate"
echo "To run tests: ./scripts/run_tests_venv.sh"
echo "To start WebSocket: ./scripts/start_websocket_venv.sh"