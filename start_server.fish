#!/usr/bin/env fish
# Agent Zero V2.0 - Development Server

echo "🚀 Starting Agent Zero V2.0 Development Server"
echo "📊 API Docs: http://localhost:8000/docs"
echo "🔍 Health: http://localhost:8000/health"

source venv/bin/activate.fish
python3 -m uvicorn agent-zero-missing-features-production-implementation:app --reload --host 127.0.0.1 --port 8000
