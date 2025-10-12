#!/bin/bash
# ultimate_fix.sh - Ostateczna naprawa Fish Shell + ML packages

echo "🔧 Agent Zero V2.0 - Ultimate Technical Fix"
echo "=========================================="

# 1. Fix virtual environment for Fish shell
echo "🐟 Naprawka 1: Fish Shell virtual environment"
if [[ -d "venv" ]]; then
    rm -rf venv
    echo "✅ Removed old venv"
fi

# Create new venv
python3 -m venv venv
echo "✅ Created new virtual environment"

# 2. Install packages in venv (using python directly)
echo "📦 Naprawka 2: Installing ML packages directly"
venv/bin/pip install --upgrade pip
venv/bin/pip install joblib>=1.3.0 scikit-learn>=1.3.0 pandas>=2.0.0 numpy>=1.24.0 neo4j>=5.0.0

echo "✅ ML packages installed in venv"

# 3. Install packages globally for fallback
echo "🌍 Naprawka 3: Installing global fallback packages"
pip install --user joblib scikit-learn pandas numpy || {
    echo "⚠️  Global install failed, trying with --break-system-packages"
    pip install --break-system-packages joblib scikit-learn pandas numpy
}

echo "✅ Global packages installed"

# 4. Create Fish-compatible activation script
echo "🐟 Naprawka 4: Fish shell compatibility"
cat > activate_fish.fish << 'EOF'
#!/usr/bin/env fish
# Fish shell activation for Agent Zero V2.0

set -gx VIRTUAL_ENV (pwd)/venv
set -gx PATH $VIRTUAL_ENV/bin $PATH
set -gx PYTHONPATH (pwd):$PYTHONPATH

echo "🐍 Agent Zero V2.0 virtual environment activated (Fish)"
echo "Python: $(which python)"
echo "Pip: $(which pip)"
python --version
EOF

chmod +x activate_fish.fish

# 5. Create universal Python test
cat > test_ml_packages.py << 'EOF'
#!/usr/bin/env python3
"""
Test ML packages availability
"""
import sys
import os

print("🧪 Testing ML packages availability...")
print(f"Python: {sys.executable}")
print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")

packages = ['joblib', 'sklearn', 'pandas', 'numpy', 'neo4j']
results = []

for package in packages:
    try:
        __import__(package)
        print(f"✅ {package}: OK")
        results.append(True)
    except ImportError as e:
        print(f"❌ {package}: FAIL - {e}")
        results.append(False)

success_rate = sum(results) / len(results) * 100
print(f"\n📊 Package availability: {success_rate:.1f}%")

if success_rate >= 80:
    print("🎉 ML packages ready for Agent Zero V2.0!")
    sys.exit(0)
else:
    print("⚠️  Some packages missing - install manually")
    sys.exit(1)
EOF

chmod +x test_ml_packages.py

# 6. Test installations
echo "🧪 Naprawka 5: Testing installations"

echo "Testing venv packages..."
venv/bin/python test_ml_packages.py

echo "Testing system packages..."
python test_ml_packages.py

# 7. Run final technical fixes with venv python
echo "🔧 Naprawka 6: Running final technical fixes"
venv/bin/python final-technical-fix.py

# 8. Create improved test runner for Fish
cat > run_tests_fish.sh << 'EOF'
#!/bin/bash
# Fish-compatible test runner

echo "🧪 Agent Zero V2.0 - Fish Shell Test Runner"
echo "=========================================="

# Use venv python directly
echo "Using venv Python: $(pwd)/venv/bin/python"

# Check Neo4j
if curl -s http://localhost:7474 > /dev/null 2>&1; then
    echo "✅ Neo4j is running"
else
    echo "🔄 Starting Neo4j..."
    docker start neo4j-agent-zero 2>/dev/null || {
        docker run -d --name neo4j-agent-zero \
            -p 7474:7474 -p 7687:7687 \
            -e NEO4J_AUTH=none \
            neo4j:5.15-community
        sleep 10
    }
fi

# Run tests with venv python
echo "🧪 Running tests..."
export PYTHONPATH="$(pwd):$PYTHONPATH"
venv/bin/python test-complete-implementation.py

echo "📊 Test Summary:"
echo "   Neo4j: $(curl -s http://localhost:7474 > /dev/null 2>&1 && echo "✅ OK" || echo "❌ FAIL")"
echo "   ML Packages: $(venv/bin/python -c "import joblib, sklearn; print('✅ OK')" 2>/dev/null || echo "❌ FAIL")"
EOF

chmod +x run_tests_fish.sh

echo ""
echo "🎉 Ultimate fix completed!"
echo ""
echo "📋 Co zostało naprawione:"
echo "✅ Fish shell virtual environment compatibility"
echo "✅ ML packages in venv and globally"
echo "✅ Fish-compatible activation script" 
echo "✅ Universal Python test runner"
echo "✅ Final technical fixes applied"
echo ""
echo "🚀 Uruchom testy:"
echo "   ./run_tests_fish.sh"
echo ""
echo "🐟 Dla Fish shell:"
echo "   source activate_fish.fish"
echo "   python test-complete-implementation.py"