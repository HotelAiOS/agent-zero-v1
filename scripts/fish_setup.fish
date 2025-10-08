#!/usr/bin/env fish
# Agent Zero V1 - Fish Shell Setup Script

echo "🐠 Agent Zero V1 - Fish Shell Setup"
echo "===================================="

# Check if we're in correct directory
if not test -f "shared/execution/agent_executor.py"
    echo "❌ Error: Not in agent-zero-v1 root directory"
    echo "Please run from project root directory"
    exit 1
end

# Create virtual environment
echo "🔧 Creating Python virtual environment..."
if not test -d "venv"
    python3 -m venv venv
    echo "✅ Virtual environment created: venv/"
else
    echo "✅ Virtual environment already exists: venv/"
end

# Activate virtual environment (Fish syntax)
echo "🚀 Activating virtual environment..."
set -gx PATH "$PWD/venv/bin" $PATH
set -gx VIRTUAL_ENV "$PWD/venv"
set -gx PYTHONPATH "$PWD:$PYTHONPATH"

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip

# Install from requirements if available
if test -f "scripts/venv_requirements.txt"
    pip install -r scripts/venv_requirements.txt
else
    # Install critical packages individually
    pip install docker==6.1.3
    pip install aiohttp-cors==0.7.0
    pip install pytest==7.4.3
    pip install fastapi==0.104.1
    pip install uvicorn==0.24.0
    pip install websockets==12.0
    pip install aiohttp==3.9.1
    pip install neo4j==5.14.1
    pip install redis==5.0.1
    pip install pika==1.3.2
    pip install python-dotenv==1.0.0
    pip install requests==2.31.0
    pip install pytest-asyncio==0.21.1
    pip install pytest-cov==4.1.0
    pip install psutil==5.9.6
end

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   • Virtual environment is now active in Fish"
echo "   • To run tests: python scripts/agent_zero_system_test_venv.py"
echo "   • To start WebSocket: python scripts/websocket_monitor_minimal.py"
echo "   • Environment variables set for this Fish session"
echo ""
echo "🐠 Fish Shell + Agent Zero V1 = Ready for development!"