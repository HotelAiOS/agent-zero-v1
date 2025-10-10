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
