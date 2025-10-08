#!/bin/bash
# AgentExecutor Fix - Quick Setup Script
# Ten skrypt automatycznie tworzy wszystkie potrzebne pliki

echo "🚀 Agent Zero V1 - AgentExecutor Method Signature Fix"
echo "=================================================="

# Sprawdź czy jesteśmy w katalogu agent-zero-v1
if [[ ! -d "shared" ]]; then
    echo "❌ Błąd: Nie znaleziono katalogu 'shared'. Uruchom skrypt w katalogu agent-zero-v1"
    exit 1
fi

# Utwórz katalog shared/execution jeśli nie istnieje
mkdir -p shared/execution

# Sprawdź czy istnieją pliki do naprawy
echo "📋 Sprawdzanie istniejących plików..."

if [[ -f "shared/execution/agent_executor.py" ]]; then
    echo "⚠️ Znaleziono istniejący agent_executor.py - tworzę backup"
    cp shared/execution/agent_executor.py shared/execution/agent_executor.py.backup.$(date +%Y%m%d_%H%M%S)
fi

echo "📁 Lista plików do pobrania z Perplexity:"
echo "1. agent_executor.py -> shared/execution/"
echo "2. test_full_integration.py -> ."
echo "3. deployment_script.py -> ."  
echo "4. README_DEPLOYMENT.md -> ."
echo ""
echo "🔍 Szukam pobranych plików..."

# Funkcja do znajdowania plików po zawartości
find_downloaded_files() {
    echo "Szukam plików zawierających kluczowe fragmenty kodu..."
    
    # Szukaj agent_executor.py
    for file in *.py; do
        if [[ -f "$file" ]] && grep -q "class AgentExecutor" "$file" 2>/dev/null; then
            echo "✅ Znaleziono AgentExecutor w: $file"
            echo "   Kopiuję do shared/execution/agent_executor.py"
            cp "$file" shared/execution/agent_executor.py
            break
        fi
    done
    
    # Szukaj test_full_integration.py
    for file in *.py; do
        if [[ -f "$file" ]] && grep -q "TestAgentExecutorMethodSignature" "$file" 2>/dev/null; then
            echo "✅ Znaleziono test suite w: $file"
            if [[ "$file" != "test_full_integration.py" ]]; then
                echo "   Kopiuję do test_full_integration.py"
                cp "$file" test_full_integration.py
            fi
            break
        fi
    done
    
    # Szukaj deployment_script.py
    for file in *.py; do
        if [[ -f "$file" ]] && grep -q "DeploymentManager" "$file" 2>/dev/null; then
            echo "✅ Znaleziono deployment script w: $file"
            if [[ "$file" != "deployment_script.py" ]]; then
                echo "   Kopiuję do deployment_script.py"
                cp "$file" deployment_script.py
            fi
            break
        fi
    done
    
    # Szukaj README
    for file in *.md; do
        if [[ -f "$file" ]] && grep -q "AgentExecutor Method Signature Fix" "$file" 2>/dev/null; then
            echo "✅ Znaleziono README w: $file"
            if [[ "$file" != "README_DEPLOYMENT.md" ]]; then
                echo "   Kopiuję do README_DEPLOYMENT.md"
                cp "$file" README_DEPLOYMENT.md
            fi
            break
        fi
    done
}

# Uruchom wyszukiwanie
find_downloaded_files

echo ""
echo "🧪 Sprawdzanie czy pliki zostały poprawnie zainstalowane..."

# Sprawdź każdy plik
files_ok=0

if [[ -f "shared/execution/agent_executor.py" ]] && grep -q "def execute_task.*output_dir" shared/execution/agent_executor.py; then
    echo "✅ agent_executor.py - OK"
    ((files_ok++))
else
    echo "❌ agent_executor.py - BRAK lub niepoprawny"
fi

if [[ -f "test_full_integration.py" ]] && grep -q "TestAgentExecutorMethodSignature" test_full_integration.py; then
    echo "✅ test_full_integration.py - OK"
    ((files_ok++))
else
    echo "❌ test_full_integration.py - BRAK lub niepoprawny"
fi

if [[ -f "deployment_script.py" ]] && grep -q "DeploymentManager" deployment_script.py; then
    echo "✅ deployment_script.py - OK"
    ((files_ok++))
else
    echo "❌ deployment_script.py - BRAK lub niepoprawny"
fi

if [[ -f "README_DEPLOYMENT.md" ]]; then
    echo "✅ README_DEPLOYMENT.md - OK"
    ((files_ok++))
else
    echo "❌ README_DEPLOYMENT.md - BRAK"
fi

echo ""
echo "📊 Status: $files_ok/4 plików zainstalowanych poprawnie"

if [[ $files_ok -eq 4 ]]; then
    echo "🎉 Wszystkie pliki zainstalowane pomyślnie!"
    echo ""
    echo "🔧 Następne kroki:"
    echo "1. Uruchom test: python test_full_integration.py"
    echo "2. Lub pełny deployment: python deployment_script.py ."
    echo "3. Sprawdź instrukcje w: README_DEPLOYMENT.md"
elif [[ $files_ok -gt 0 ]]; then
    echo "⚠️ Niektóre pliki zostały zainstalowane, ale nie wszystkie."
    echo "💡 Sprawdź pobrane pliki ręcznie i skopiuj brakujące."
else
    echo "❌ Nie znaleziono żadnych plików do instalacji."
    echo ""
    echo "🔧 Ręczne rozwiązanie:"
    echo "1. Sprawdź jakie pliki pobrałeś: ls -la *.py *.md"
    echo "2. Znajdź plik z 'class AgentExecutor' i skopiuj go do shared/execution/agent_executor.py"
    echo "3. Pozostałe pliki skopiuj według instrukcji w README"
fi

echo ""
echo "📁 Aktualna zawartość katalogu:"
ls -la | grep -E '\.(py|md)$'