#!/usr/bin/env fish

# 🔧 Agent Zero V2.0 - Quick Import Fix
# 📦 PAKIET 5 Phase 3: Fix Tuple import error
# 🎯 Replace file and restart system
#
# Status: QUICK FIX
# Created: 12 października 2025, 18:57 CEST

set_color blue; echo "🔧 Agent Zero V2.0 - Quick Import Fix"; set_color normal
echo

# Backup original file
if test -f "security_integration_layer.py"
    cp security_integration_layer.py security_integration_layer.py.backup
    set_color green; echo "✅ Backup created: security_integration_layer.py.backup"; set_color normal
end

# Replace with fixed version
if test -f "security_integration_layer_fixed.py"
    cp security_integration_layer_fixed.py security_integration_layer.py
    set_color green; echo "✅ Updated security_integration_layer.py with fix"; set_color normal
else
    set_color red; echo "❌ Fixed file not found: security_integration_layer_fixed.py"; set_color normal
    exit 1
end

# Test the fix
set_color yellow; echo "🧪 Testing import fix..."; set_color normal
python3 -c "
from security_integration_layer import SecureAgentZeroSystem
print('✅ Import fix successful!')
" 2>/dev/null

if test $status -eq 0
    set_color green; echo "✅ Import fix verified!"; set_color normal
else
    set_color red; echo "❌ Import fix failed"; set_color normal
    # Restore backup
    if test -f "security_integration_layer.py.backup"
        cp security_integration_layer.py.backup security_integration_layer.py
        echo "🔄 Restored from backup"
    end
    exit 1
end

echo
set_color cyan; echo "🚀 READY TO RESTART:"; set_color normal
echo "   ./start_complete_system.fish    # Complete system with fix"
echo "   ./start_secure_api.fish         # Secure API with fix"

echo
set_color green; echo "🎉 Import fix applied successfully!"; set_color normal