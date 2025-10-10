#!/usr/bin/env python3
"""
Agent Zero V2.0 Database Schema Fix
Naprawa schema V2.0 i resolve import conflicts
"""

import sqlite3
import sys
import os

def fix_v2_database_schema():
    """Fix V2.0 database schema issues"""
    
    print("🔧 Agent Zero V2.0 - Database Schema Fix")
    print("=" * 50)
    
    db_path = "agent_zero.db"
    
    try:
        with sqlite3.connect(db_path) as conn:
            print("📊 Checking current V2.0 schema...")
            
            # Check existing V2.0 tables
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v2_%'")
            existing_tables = [row[0] for row in cursor.fetchall()]
            print(f"✅ Found V2.0 tables: {len(existing_tables)}")
            for table in existing_tables:
                print(f"   - {table}")
            
            # Fix v2_success_evaluations table - add missing metadata column
            print("\n🔧 Fixing v2_success_evaluations table...")
            try:
                # Check if metadata column exists
                cursor = conn.execute("PRAGMA table_info(v2_success_evaluations)")
                columns = [row[1] for row in cursor.fetchall()]
                
                if 'metadata' not in columns:
                    print("   Adding missing 'metadata' column...")
                    conn.execute("ALTER TABLE v2_success_evaluations ADD COLUMN metadata TEXT")
                    print("   ✅ metadata column added")
                else:
                    print("   ✅ metadata column already exists")
                
            except sqlite3.OperationalError as e:
                if "no such table" in str(e):
                    print("   Creating v2_success_evaluations table...")
                    conn.execute("""
                        CREATE TABLE v2_success_evaluations (
                            id TEXT PRIMARY KEY,
                            task_id TEXT NOT NULL,
                            task_type TEXT NOT NULL,
                            model_used TEXT NOT NULL,
                            overall_score REAL NOT NULL,
                            correctness_score REAL,
                            efficiency_score REAL,  
                            cost_score REAL,
                            latency_score REAL,
                            success_level TEXT NOT NULL,
                            cost_usd REAL,
                            execution_time_ms INTEGER,
                            recommendation_followed BOOLEAN DEFAULT FALSE,
                            user_override BOOLEAN DEFAULT FALSE,
                            timestamp TEXT NOT NULL,
                            metadata TEXT
                        )
                    """)
                    print("   ✅ v2_success_evaluations table created")
            
            # Ensure all V2.0 tables exist with correct schema
            print("\n📋 Ensuring complete V2.0 schema...")
            
            # v2_enhanced_tracker table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_enhanced_tracker (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    success_score REAL NOT NULL,
                    cost_usd REAL,
                    latency_ms INTEGER,
                    timestamp TEXT NOT NULL,
                    context TEXT,
                    tracking_level TEXT DEFAULT 'basic',
                    success_level TEXT,
                    dimension_scores TEXT,
                    user_feedback TEXT,
                    lessons_learned TEXT,
                    optimization_applied BOOLEAN DEFAULT FALSE,
                    metadata TEXT,
                    experience_id TEXT,
                    pattern_ids TEXT,
                    recommendation_ids TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            
            # v2_system_alerts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_system_alerts (
                    id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source_component TEXT NOT NULL,
                    related_task_id TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution_notes TEXT
                )
            """)
            
            conn.commit()
            
            # Final verification
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v2_%'")
            final_tables = [row[0] for row in cursor.fetchall()]
            
            print(f"\n✅ V2.0 Schema Fix Complete!")
            print(f"📊 Total V2.0 tables: {len(final_tables)}")
            
            return True
            
    except Exception as e:
        print(f"❌ Database schema fix failed: {e}")
        return False

def test_enhanced_tracker():
    """Test Enhanced SimpleTracker after schema fix"""
    
    print("\n🧪 Testing Enhanced SimpleTracker after schema fix...")
    
    try:
        sys.path.append('.')
        from shared.utils.enhanced_simple_tracker import EnhancedSimpleTracker, TrackingLevel
        
        tracker = EnhancedSimpleTracker()
        print("✅ Enhanced SimpleTracker initialized")
        
        # Test V2.0 tracking
        task_id = tracker.track_event(
            task_id='schema_fix_test_001',
            task_type='schema_validation',
            model_used='test_model',
            success_score=0.95,
            cost_usd=0.01,
            latency_ms=1000,
            tracking_level=TrackingLevel.FULL,
            user_feedback='Schema fix validation test'
        )
        
        print(f"✅ V2.0 Enhanced Tracking successful: {task_id}")
        
        # Test enhanced summary
        summary = tracker.get_enhanced_summary()
        print(f"✅ Enhanced summary: {summary['v1_metrics']['total_tasks']} tasks")
        
        # Test system health
        health = tracker.get_v2_system_health()
        print(f"✅ V2.0 System Health: {health['overall_health']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced SimpleTracker test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Agent Zero V2.0 Schema Fix...")
    
    # Fix database schema
    schema_ok = fix_v2_database_schema()
    
    if schema_ok:
        # Test enhanced tracker
        tracker_ok = test_enhanced_tracker()
        
        if tracker_ok:
            print("\n🎉 V2.0 SYSTEM FULLY OPERATIONAL!")
            print("✅ Database schema fixed")
            print("✅ Enhanced SimpleTracker working") 
            print("✅ V2.0 Intelligence Layer ready")
            
            print("\n🚀 You can now use V2.0 features:")
            print("   python3 cli/advanced_commands.py v2-system status")
            print("   python3 test-v2-core.py")
            
        else:
            print("\n⚠️  Schema fixed but tracker needs more work")
    else:
        print("\n❌ Schema fix failed - manual intervention needed")