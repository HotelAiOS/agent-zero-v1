#!/usr/bin/env fish
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

echo "✅ Setup completed successfully!"
