#!/bin/bash
# Agent Zero V1 - One-Click Arch Linux Setup

echo "🚀 Agent Zero V1 - One-Click Arch Linux Setup"
echo "=============================================="
echo ""

# Check if we're in the right directory
if [ ! -f "shared/execution/agent_executor.py" ]; then
    echo "❌ Error: Not in agent-zero-v1 root directory"
    echo "Please run from project root directory"
    exit 1
fi

# Step 1: Create virtual environment and install dependencies
echo "📦 Step 1: Setting up virtual environment..."
chmod +x scripts/create_venv_and_install.sh
./scripts/create_venv_and_install.sh

if [ $? -ne 0 ]; then
    echo "❌ Failed to setup virtual environment"
    exit 1
fi

# Step 2: Make other scripts executable
echo ""
echo "🔧 Step 2: Setting permissions..."
chmod +x scripts/run_tests_venv.sh
chmod +x scripts/start_websocket_venv.sh

# Step 3: Run tests
echo ""
echo "🧪 Step 3: Running system tests..."
./scripts/run_tests_venv.sh

echo ""
echo "🎉 Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "   • To run tests again: ./scripts/run_tests_venv.sh"  
echo "   • To start WebSocket monitor: ./scripts/start_websocket_venv.sh"
echo "   • To activate venv manually: source venv/bin/activate"
echo ""
echo "🐧 Virtual environment protects your Arch Linux system packages!"