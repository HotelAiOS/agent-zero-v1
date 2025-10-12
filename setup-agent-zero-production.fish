#!/usr/bin/env fish
# 🐠 Agent Zero V2.0 - Production Setup Script (Fish Shell)
# 📦 Instalacja wszystkich dependencies + konfiguracja środowiska
# 🎯 Status: PRODUCTION READY dla Agent Zero V1/V2.0

echo "🚀 Agent Zero V2.0 - Production Setup rozpoczęty..."
echo "📅 " (date)
echo "🐠 Fish Shell Environment Detected"

# Sprawdź czy jesteśmy w właściwym repo
if not test -d .git
    echo "❌ Błąd: Nie jesteś w repozytorium git!"
    echo "📂 Przejdź do katalogu agent-zero-v1 i uruchom ponownie"
    exit 1
end

# Sprawdź Python 3.11+
set python_version (python3 --version 2>/dev/null | cut -d' ' -f2)
if test -z "$python_version"
    echo "❌ Python 3 nie jest zainstalowany!"
    exit 1
end

echo "✅ Python $python_version detected"

# Setup virtual environment
if not test -d venv
    echo "📦 Tworzenie virtual environment..."
    python3 -m venv venv
end

# Activate venv (Fish syntax)
source venv/bin/activate.fish

echo "🔧 Aktualizacja pip..."
pip install --upgrade pip setuptools wheel

# Core dependencies
echo "📚 Instalowanie core dependencies..."
pip install fastapi[all] uvicorn[standard]
pip install sqlalchemy sqlite3  
pip install numpy pandas
pip install requests httpx
pip install python-multipart
pip install pydantic[email]

# Analytics & Export dependencies  
echo "📊 Instalowanie analytics dependencies..."
pip install openpyxl xlsxwriter
pip install python-docx
pip install weasyprint
pip install matplotlib seaborn plotly
pip install jinja2

# Collaboration dependencies
echo "🤝 Instalowanie collaboration dependencies..."
pip install slack-sdk
pip install microsoft-graph-api

# ML & AI dependencies
echo "🧠 Instalowanie AI/ML dependencies..."
pip install scikit-learn
pip install scipy
pip install joblib

# Quantum computing (optional)
echo "⚛️  Instalowanie quantum dependencies..."
pip install qiskit qiskit-aer

# Development & Testing
echo "🔧 Instalowanie dev dependencies..."
pip install pytest pytest-asyncio
pip install black flake8 mypy
pip install pytest-cov

# Production monitoring
echo "📈 Instalowanie monitoring dependencies..."
pip install prometheus-client
pip install structlog

echo "✅ Wszystkie dependencies zainstalowane!"

# Tworzenie struktury katalogów
echo "📁 Tworzenie struktury katalogów..."

# Main production directories
mkdir -p src/team/recommendation
mkdir -p src/team/learning
mkdir -p src/analytics/connectors
mkdir -p src/analytics/export
mkdir -p src/collab/adapters
mkdir -p src/collab/calendar
mkdir -p src/predictive
mkdir -p src/learning
mkdir -p src/quantum/providers
mkdir -p templates
mkdir -p data
mkdir -p logs
mkdir -p reports
mkdir -p tests

# Init files dla Python packages
touch src/__init__.py
touch src/team/__init__.py
touch src/team/recommendation/__init__.py
touch src/team/learning/__init__.py
touch src/analytics/__init__.py
touch src/analytics/connectors/__init__.py
touch src/analytics/export/__init__.py
touch src/collab/__init__.py
touch src/collab/adapters/__init__.py
touch src/collab/calendar/__init__.py
touch src/predictive/__init__.py
touch src/learning/__init__.py
touch src/quantum/__init__.py
touch src/quantum/providers/__init__.py

echo "✅ Struktura katalogów utworzona!"

# Environment configuration
echo "🔧 Tworzenie pliku konfiguracyjnego..."

cat > .env.example << 'EOF'
# Agent Zero V2.0 - Production Environment Configuration

# Database
DATABASE_URL=sqlite:///./agent_zero.db
DATABASE_ECHO=false

# FastAPI
DEBUG=false
HOST=0.0.0.0
PORT=8000
WORKERS=4

# External APIs
SLACK_BOT_TOKEN=xoxb-your-slack-token
HUBSPOT_API_TOKEN=your-hubspot-token

# Microsoft/PowerBI
MS_TENANT_ID=your-tenant-id
MS_CLIENT_ID=your-client-id
MS_CLIENT_SECRET=your-client-secret

# Google Calendar
GOOGLE_CALENDAR_CREDENTIALS_FILE=./credentials/google-calendar.json

# Quantum Computing (Optional)
IBMQ_TOKEN=your-ibmq-token

# Monitoring
PROMETHEUS_PORT=9090
LOG_LEVEL=INFO

# Security
SECRET_KEY=your-super-secret-key-change-this
CORS_ORIGINS=http://localhost:3000,http://localhost:8080

# File Storage
REPORTS_STORAGE_PATH=./reports
TEMP_FILE_CLEANUP_HOURS=24
EOF

if not test -f .env
    cp .env.example .env
    echo "⚙️  Plik .env utworzony - skonfiguruj swoje tokeny!"
else
    echo "⚙️  .env już istnieje - aktualizuj według potrzeb"
end

# Konfiguracja git hooks (opcjonalne)
echo "🔗 Konfigurowanie git hooks..."
mkdir -p .githooks

cat > .githooks/pre-commit << 'EOF'
#!/bin/bash
# Agent Zero Pre-commit Hook
echo "🔍 Running pre-commit checks..."

# Activate venv
source venv/bin/activate

# Code formatting
echo "🎨 Formatting code..."
black src/ tests/ --line-length 100

# Linting
echo "🧹 Linting..."
flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503

# Type checking
echo "🔍 Type checking..."
mypy src/ --ignore-missing-imports

# Tests
echo "🧪 Running tests..."
pytest tests/ -v

echo "✅ Pre-commit checks completed!"
EOF

chmod +x .githooks/pre-commit

# Database initialization
echo "🗄️  Inicjalizacja bazy danych..."
python3 -c "
import sqlite3
import os

# Tworzenie bazy danych z podstawowymi tabelami
conn = sqlite3.connect('agent_zero.db')

# Agent Zero V1 compatibility tables (jeśli nie istnieją)
conn.executescript('''
-- V1 Tables (maintained for compatibility)
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY,
    agent_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    skills TEXT, -- JSON
    status TEXT DEFAULT 'available',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- V2.0 Enhancement Tables
CREATE TABLE IF NOT EXISTS team_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id TEXT NOT NULL,
    team_composition TEXT NOT NULL,
    outcome_success REAL NOT NULL,
    budget_delta REAL DEFAULT 0.0,
    timeline_delta REAL DEFAULT 0.0,
    quality_score REAL DEFAULT 0.0,
    team_satisfaction REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id)
);

CREATE TABLE IF NOT EXISTS agent_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    role TEXT NOT NULL,
    individual_score REAL NOT NULL,
    collaboration_score REAL NOT NULL,
    skill_growth REAL DEFAULT 0.0,
    feedback_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics_dataset (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    data_json TEXT NOT NULL,
    metadata_json TEXT,
    sync_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS analytics_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    report_data_json TEXT NOT NULL,
    format TEXT NOT NULL,
    file_path TEXT,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_team_history_project ON team_history(project_id);
CREATE INDEX IF NOT EXISTS idx_agent_perf_agent ON agent_performance(agent_id);
CREATE INDEX IF NOT EXISTS idx_analytics_source ON analytics_dataset(source, category);
''')

conn.commit()
conn.close()
print('✅ Database initialized successfully!')
"

echo "✅ Baza danych zainicjalizowana!"

# Create startup script
echo "🚀 Tworzenie skryptów startowych..."

cat > start_dev.fish << 'EOF'
#!/usr/bin/env fish
# Development server startup

source venv/bin/activate.fish

echo "🚀 Starting Agent Zero V2.0 Development Server..."
echo "📊 Access API docs: http://localhost:8000/docs"
echo "🔍 Health check: http://localhost:8000/health"

uvicorn agent-zero-missing-features-production-implementation:app --reload --port 8000
EOF

cat > start_prod.fish << 'EOF'
#!/usr/bin/env fish  
# Production server startup

source venv/bin/activate.fish

echo "🏭 Starting Agent Zero V2.0 Production Server..."
echo "⚡ Workers: 4, Host: 0.0.0.0, Port: 8000"

uvicorn agent-zero-missing-features-production-implementation:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --access-log \
    --no-reload
EOF

chmod +x start_dev.fish start_prod.fish

# Test script
cat > test_system.fish << 'EOF'
#!/usr/bin/env fish
# System testing script

source venv/bin/activate.fish

echo "🧪 Testing Agent Zero V2.0 System..."

# Test database
echo "🗄️  Testing database connection..."
python3 -c "
import sqlite3
conn = sqlite3.connect('agent_zero.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM sqlite_master WHERE type=\"table\"')
table_count = cursor.fetchone()[0]
print(f'✅ Database: {table_count} tables found')
conn.close()
"

# Test dependencies
echo "📦 Testing key dependencies..."
python3 -c "
try:
    import fastapi, numpy, requests, sqlite3
    from openpyxl import Workbook
    from docx import Document
    print('✅ All core dependencies available')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
"

# Test API startup (quick check)
echo "🚀 Testing API startup..."
timeout 10s python3 -c "
from agent_zero_missing_features_production_implementation import create_agent_zero_app
app = create_agent_zero_app()
print('✅ FastAPI app creation successful')
" && echo "✅ API startup test passed"

echo "🎯 System test completed!"
EOF

chmod +x test_system.fish

# Quick system test
echo "🧪 Uruchamianie szybkiego testu systemu..."
./test_system.fish

echo ""
echo "🎉 Agent Zero V2.0 - Setup Complete!"
echo ""
echo "📋 Następne kroki:"
echo "   1. Skonfiguruj tokeny w pliku .env"
echo "   2. Uruchom development server: ./start_dev.fish"
echo "   3. Sprawdź dokumentację: http://localhost:8000/docs"
echo "   4. Uruchom testy: pytest tests/"
echo ""
echo "🔗 Główne endpointy:"
echo "   • POST /api/v4/team/recommendations - AI team formation"
echo "   • POST /api/v4/team/learn - Learning from outcomes"
echo "   • POST /api/v5/analytics/datasource/sync - Data sync"
echo "   • GET /api/v5/analytics/reports/generate - Report generation"
echo ""
echo "📚 Dostępne skrypty:"
echo "   • ./start_dev.fish - Development server"
echo "   • ./start_prod.fish - Production server"
echo "   • ./test_system.fish - System testing"
echo ""
echo "🚀 Agent Zero V2.0 jest gotowy do pracy!"