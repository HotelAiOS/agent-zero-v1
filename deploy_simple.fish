#!/usr/bin/env fish
# Agent Zero V2.0 - Simple Deployment Script

echo "⚡ Agent Zero V2.0 - Simple Deploy"

# Check files exist
if not test -f "agent-zero-missing-features-production-implementation.py"
    echo "❌ Missing main implementation file!"
    exit 1
end

# Setup if needed
if not test -d venv
    echo "🔧 Running setup..."
    chmod +x setup_fixed.fish
    ./setup_fixed.fish
end

# Activate environment
source venv/bin/activate.fish

# Run database migration
echo "🗄️ Running database migration..."
python3 migrate-agent-zero-database.py

# Quick test
echo "🧪 Quick system test..."
python3 -c "
try:
    from agent_zero_missing_features_production_implementation import create_agent_zero_app
    app = create_agent_zero_app()
    print('✅ App creation: SUCCESS')
except Exception as e:
    print(f'❌ Error: {e}')
"

echo "✅ Deployment completed!"
echo "🚀 Start development server with: python3 -m uvicorn agent-zero-missing-features-production-implementation:app --reload --port 8000"
