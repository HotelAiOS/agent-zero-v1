#!/bin/bash
# Agent Zero V1 - Quick System Health Check & Setup
# Uruchamia podstawowe sprawdzenie systemu przed auditem

set -e

echo "🚀 Agent Zero V1 - Quick Setup & Health Check"
echo "============================================="

# Sprawdź czy jesteśmy w repozytorium Agent Zero
if [ ! -d ".git" ]; then
    echo "❌ Not in a git repository. Please run this from Agent Zero V1 root directory."
    exit 1
fi

# Utwórz katalog stabilization
mkdir -p stabilization
echo "📁 Created stabilization directory"

# Sprawdź Python i podstawowe zależności
echo "🐍 Checking Python environment..."
python3 --version || { echo "❌ Python 3 not found"; exit 1; }

# Sprawdź czy pip jest dostępny
pip3 --version || { echo "❌ pip3 not found"; exit 1; }

# Zainstaluj wymagane narzędzia do audytu (jeśli nie ma)
echo "📦 Installing audit tools..."

# Lista narzędzi do audytu
tools=(
    "vulture"      # Dead code detection
    "unimport"     # Unused import detection  
    "pip-audit"    # Security vulnerability scan
    "flake8"       # Code style and quality
)

for tool in "${tools[@]}"; do
    if ! pip3 show "$tool" > /dev/null 2>&1; then
        echo "Installing $tool..."
        pip3 install "$tool" --user --quiet || echo "Warning: Could not install $tool"
    else
        echo "✅ $tool already installed"
    fi
done

# Sprawdź Docker
echo "🐳 Checking Docker..."
if command -v docker &> /dev/null; then
    docker --version
    echo "✅ Docker is available"
    
    # Sprawdź uruchomione kontenery
    running_containers=$(docker ps -q | wc -l)
    echo "📊 Currently running containers: $running_containers"
else
    echo "⚠️  Docker not found - some features may not work"
fi

# Sprawdź Git status
echo "📝 Checking Git status..."
git status --porcelain | wc -l | xargs echo "Uncommitted changes:"

# Podstawowe statystyki systemu
echo "📊 Basic system statistics:"
echo "Python files: $(find . -name '*.py' | wc -l)"
echo "Docker files: $(find . -name 'Dockerfile*' -o -name 'docker-compose*' | wc -l)"
echo "Config files: $(find . -name '*.yaml' -o -name '*.yml' -o -name '*.json' | wc -l)"

# Sprawdź czy są running processes Agent Zero
echo "🔍 Checking for running Agent Zero processes..."
pgrep -f "agent.*zero" || echo "No Agent Zero processes found"

# Sprawdź dostępność portów (podstawowych)
echo "🌐 Checking common ports..."
common_ports=(8000 8080 3000 5000 7474 6379)
for port in "${common_ports[@]}"; do
    if netstat -ln 2>/dev/null | grep ":$port " > /dev/null; then
        echo "Port $port is in use"
    fi
done

echo ""
echo "✅ Health check completed!"
echo "📋 Ready to run system audit with:"
echo "   python3 agent-zero-system-auditor.py"
echo ""
echo "🎯 Next steps:"
echo "   1. Run the full system audit"
echo "   2. Review the generated reports in stabilization/"
echo "   3. Address critical issues before proceeding"