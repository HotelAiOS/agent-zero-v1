#!/usr/bin/env fish
# 🛠️ Agent Zero V2.0 - Fish Shell Syntax Fix Script
# 🐠 Naprawia wszystkie problemy z składnią Fish Shell

echo "🔧 Fixing Fish Shell syntax issues..."
echo "📅 " (date)

# Fix the original deploy script by replacing problematic heredoc syntax
echo "📝 Creating corrected Fish Shell scripts..."

# Krok 1: Poprawny skrypt setup
echo '#!/usr/bin/env fish
# Agent Zero V2.0 - Production Setup Script (Fish Shell Compatible)

echo "🚀 Agent Zero V2.0 - Production Setup rozpoczęty..."
echo "📅 " (date)

# Python version check
set python_version (python3 --version 2>/dev/null | cut -d" " -f2)
if test -z "$python_version"
    echo "❌ Python 3 nie jest zainstalowany!"
    exit 1
end

echo "✅ Python $python_version detected"

# Virtual environment setup
if not test -d venv
    echo "📦 Tworzenie virtual environment..."
    python3 -m venv venv
end

source venv/bin/activate.fish

# Install dependencies
echo "📚 Instalowanie dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn
pip install numpy pandas requests
pip install openpyxl python-docx weasyprint
pip install slack-sdk scikit-learn
pip install pytest black flake8

# Create directories
echo "📁 Tworzenie struktury katalogów..."
mkdir -p src/team/recommendation src/team/learning
mkdir -p src/analytics/connectors src/analytics/export
mkdir -p src/collab/adapters src/collab/calendar
mkdir -p src/predictive src/learning src/quantum/providers
mkdir -p templates data logs reports tests

# Create __init__.py files
touch src/__init__.py
touch src/team/__init__.py src/team/recommendation/__init__.py
touch src/analytics/__init__.py src/analytics/connectors/__init__.py
touch src/collab/__init__.py src/predictive/__init__.py
touch src/learning/__init__.py src/quantum/__init__.py

echo "✅ Setup completed successfully!"' > setup_fixed.fish

# Krok 2: Prosty skrypt deployment bez heredoc
echo '#!/usr/bin/env fish
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
    print('\''✅ App creation: SUCCESS'\'')
except Exception as e:
    print(f'\''❌ Error: {e}'\'')
"

echo "✅ Deployment completed!"
echo "🚀 Start development server with: python3 -m uvicorn agent-zero-missing-features-production-implementation:app --reload --port 8000"' > deploy_simple.fish

# Krok 3: Skrypt startowy serwera
echo '#!/usr/bin/env fish
# Agent Zero V2.0 - Development Server

echo "🚀 Starting Agent Zero V2.0 Development Server"
echo "📊 API Docs: http://localhost:8000/docs"
echo "🔍 Health: http://localhost:8000/health"

source venv/bin/activate.fish
python3 -m uvicorn agent-zero-missing-features-production-implementation:app --reload --host 127.0.0.1 --port 8000' > start_server.fish

# Krok 4: Skrypt testowy
echo '#!/usr/bin/env fish
# Agent Zero V2.0 - Testing Script

echo "🧪 Testing Agent Zero V2.0"
source venv/bin/activate.fish

echo "🗄️ Testing Database..."
python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('\''agent_zero.db'\'')
    cursor = conn.cursor()
    cursor.execute('\''SELECT COUNT(*) FROM sqlite_master WHERE type=\"table\"'\'')
    count = cursor.fetchone()[0]
    print(f'\''✅ Database: {count} tables'\'')
    conn.close()
except Exception as e:
    print(f'\''❌ Database error: {e}'\'')
"

echo "🤖 Testing Core Components..."
python3 -c "
try:
    from agent_zero_missing_features_production_implementation import ExperienceRepository, ReportExporter
    repo = ExperienceRepository()
    exporter = ReportExporter()
    print('\''✅ Core components: OK'\'')
except Exception as e:
    print(f'\''❌ Component error: {e}'\'')
"

echo "✅ Testing completed!"' > test_system.fish

# Make all scripts executable
chmod +x setup_fixed.fish deploy_simple.fish start_server.fish test_system.fish

echo ""
echo "🔧 Fish Shell Syntax Fix - COMPLETE!"
echo "=" (string repeat -n 40 "=")
echo ""
echo "📋 Fixed scripts created:"
echo "   ✅ setup_fixed.fish - Environment setup"
echo "   ✅ deploy_simple.fish - Simple deployment"
echo "   ✅ start_server.fish - Development server"
echo "   ✅ test_system.fish - System testing"
echo ""
echo "🚀 Quick deployment sequence:"
echo "   1. ./setup_fixed.fish"
echo "   2. ./deploy_simple.fish" 
echo "   3. ./start_server.fish"
echo ""
echo "📊 API będzie dostępne na: http://localhost:8000/docs"
echo "🎯 Fish Shell compatible - ready to go!"