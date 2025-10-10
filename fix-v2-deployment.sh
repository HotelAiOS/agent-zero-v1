#!/bin/bash
"""
Agent Zero V1 - V2.0 Quick Fix Script
Naprawia problemy po deployment V2.0 Intelligence Layer

Author: Developer A (Backend Architect)
Date: 10 października 2025, 17:13 CEST
Linear Issue: A0-28
"""

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${CYAN}🔧 Agent Zero V1 - V2.0 Quick Fix${NC}"
echo -e "${CYAN}=================================${NC}"
echo ""

fix_cli_import() {
    echo -e "${BLUE}🔧 Fixing CLI import issue...${NC}"
    
    # Create proper __init__.py for cli module
    cat > "$REPO_DIR/cli/__init__.py" << 'EOF'
#!/usr/bin/env python3
"""
Agent Zero V1 - CLI Module
V2.0 Intelligence Layer - CLI Components
"""

from .main import AgentZeroCLI

__all__ = ['AgentZeroCLI']
EOF

    # Rename __main__.py to main.py and fix imports
    if [[ -f "$REPO_DIR/cli/__main__.py" ]]; then
        mv "$REPO_DIR/cli/__main__.py" "$REPO_DIR/cli/main.py"
        echo -e "${GREEN}  ✅ Moved __main__.py to main.py${NC}"
    fi
    
    # Create new __main__.py that properly imports
    cat > "$REPO_DIR/cli/__main__.py" << 'EOF'
#!/usr/bin/env python3
"""
Agent Zero V1 - CLI Entry Point
V2.0 Intelligence Layer - Module Entry
"""

from .main import AgentZeroCLI

def main():
    cli = AgentZeroCLI()
    cli.main()

if __name__ == "__main__":
    main()
EOF

    echo -e "${GREEN}  ✅ Fixed CLI import structure${NC}"
}

initialize_production_database() {
    echo -e "${BLUE}🗄️ Initializing production database...${NC}"
    
    # Create initialization script
    cat > "$REPO_DIR/init_v2_database.py" << 'EOF'
#!/usr/bin/env python3
"""
Initialize V2.0 database tables in production
"""

import sys
import sqlite3
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def init_database():
    try:
        # Import and initialize all V2.0 components
        from shared.kaizen.intelligent_selector import IntelligentModelSelector
        from shared.kaizen.success_evaluator import SuccessEvaluator
        from shared.kaizen.metrics_analyzer import ActiveMetricsAnalyzer
        
        print("🔧 Initializing V2.0 database components...")
        
        # Initialize components (this creates all tables)
        selector = IntelligentModelSelector("agent_zero.db")
        evaluator = SuccessEvaluator("agent_zero.db")
        analyzer = ActiveMetricsAnalyzer("agent_zero.db")
        
        # Verify tables exist
        with sqlite3.connect("agent_zero.db") as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            v2_tables = [t for t in tables if t.startswith('v2_')]
            
            print(f"✅ V2.0 Tables created: {len(v2_tables)}")
            for table in v2_tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  {table}: {count} records")
        
        print("🎉 Database initialization complete!")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

if __name__ == "__main__":
    success = init_database()
    sys.exit(0 if success else 1)
EOF

    # Run database initialization
    echo -e "${YELLOW}  📥 Running database initialization...${NC}"
    
    cd "$REPO_DIR"
    if python3 init_v2_database.py; then
        echo -e "${GREEN}  ✅ Production database initialized${NC}"
        return 0
    else
        echo -e "${RED}  ❌ Database initialization failed${NC}"
        return 1
    fi
}

create_quick_test_script() {
    echo -e "${BLUE}🧪 Creating quick test script...${NC}"
    
    cat > "$REPO_DIR/test_v2_quick.py" << 'EOF'
#!/usr/bin/env python3
"""
Quick V2.0 functionality test
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_cli_import():
    try:
        from cli import AgentZeroCLI
        print("✅ CLI import successful")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False

def test_database_tables():
    try:
        import sqlite3
        
        with sqlite3.connect("agent_zero.db") as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            v2_tables = [t for t in tables if t.startswith('v2_')]
            
            if len(v2_tables) >= 4:
                print(f"✅ V2.0 tables found: {len(v2_tables)}")
                return True
            else:
                print(f"❌ Missing V2.0 tables: {v2_tables}")
                return False
                
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_components():
    try:
        from shared.kaizen.intelligent_selector import IntelligentModelSelector, TaskType
        
        selector = IntelligentModelSelector("agent_zero.db")
        context = {'complexity': 1.0, 'urgency': 1.0}
        recommendation = selector.recommend_model(TaskType.CODE_GENERATION, context)
        
        if recommendation.model_name:
            print(f"✅ AI recommendation: {recommendation.model_name}")
            return True
        else:
            print("❌ No AI recommendation generated")
            return False
            
    except Exception as e:
        print(f"❌ Component test failed: {e}")
        return False

def main():
    print("🧪 V2.0 Quick Test")
    print("=" * 20)
    
    tests = [
        ("CLI Import", test_cli_import),
        ("Database Tables", test_database_tables),
        ("AI Components", test_components)
    ]
    
    passed = 0
    
    for name, test_func in tests:
        print(f"\n{name}:")
        if test_func():
            passed += 1
    
    print(f"\n📊 Result: {passed}/{len(tests)} tests passed")
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

    echo -e "${GREEN}  ✅ Quick test script created${NC}"
}

main() {
    echo -e "${CYAN}Starting V2.0 fixes...${NC}"
    
    fix_cli_import
    
    if initialize_production_database; then
        create_quick_test_script
        
        echo ""
        echo -e "${GREEN}🎉 V2.0 fixes completed!${NC}"
        echo ""
        echo -e "${CYAN}🧪 Run quick test:${NC}"
        echo -e "${YELLOW}  python3 test_v2_quick.py${NC}"
        echo ""
        echo -e "${CYAN}🚀 Try CLI commands:${NC}"
        echo -e "${YELLOW}  python3 -m cli status${NC}"
        echo -e "${YELLOW}  python3 -m cli kaizen-report --days 7${NC}"
        echo ""
    else
        echo -e "${RED}❌ Database initialization failed${NC}"
        exit 1
    fi
}

main "$@"