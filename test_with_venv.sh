#!/bin/bash
if [[ -f "venv/bin/activate" ]]; then
    source venv/bin/activate
    python test-complete-implementation.py "$@"
else
    echo "❌ Brak środowiska wirtualnego. Uruchom ./quick_arch_fix.sh"
    exit 1
fi
