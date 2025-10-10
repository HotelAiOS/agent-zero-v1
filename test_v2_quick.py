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
