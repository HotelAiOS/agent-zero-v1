#!/bin/bash
# Skrypt uruchomienia Agent Zero

echo "🚀 Uruchamianie Agent Zero Platform..."
echo "======================================"

# Zainstaluj zależności
echo "📦 Instalowanie zależności..."
pip install -r requirements.txt

echo "🌐 Uruchamianie serwera FastAPI..."
echo "📡 Aplikacja dostępna na: http://localhost:8000"
echo "📋 Dokumentacja API: http://localhost:8000/docs"
echo ""

uvicorn src.endpointy_autoryzacji:app --reload --host 0.0.0.0 --port 8000
