#!/bin/bash
# Agent Zero V1 - Run Tests in Virtual Environment

set -e

echo "🧪 Agent Zero V1 - Running Tests (Virtual Environment)"
echo "====================================================="

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: ./create_venv_and_install.sh first"
    exit 1
fi

# Activate virtual environment
echo "🚀 Activating virtual environment..."
source venv/bin/activate

# Run the system test
echo "🧪 Running system test..."
python agent_zero_system_test_venv.py

echo "✅ Tests completed!"