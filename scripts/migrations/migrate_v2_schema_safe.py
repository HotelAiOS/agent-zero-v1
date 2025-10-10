#!/usr/bin/env python3
"""
Agent Zero V2.0 - Safe Database Schema Migration
Fixes schema issues and database locks
"""

import sqlite3
import sys
import os
import time
from pathlib import Path

def migrate_v2_schema_safe():
    """Safe V2.0 database schema migration without locks"""
    
    print("🔄 Agent Zero V2.0 - Safe Schema Migration")
    print("=" * 50)
    
    db_path = "agent_zero.db"
    
    # Check if database exists
    if not os.path.exists(db_path):
        print(f"❌ Database {db_path} not found")
        return False
    
    try:
        # Configure connection for safe operations
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("PRAGMA busy_timeout = 30000")  # 30 second timeout
        conn.execute("PRAGMA journal_mode = WAL")     # Write-Ahead Logging
        conn.execute("PRAGMA synchronous = NORMAL")   # Faster but safe
        
        print("✅ Database connection established with safe settings")
        
        # Step 1: Check current V2.0 tables
        print("\n📊 Checking current V2.0 schema...")
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v2_%'")
        existing_tables = [row[0] for row in cursor.fetchall()]
        
        print(f"Found {len(existing_tables)} V2.0 tables:")
        for table in existing_tables:
            print(f"   - {table}")
        
        # Step 2: Add missing metadata column to v2_success_evaluations
        print("\n🔧 Fixing v2_success_evaluations schema...")
        try:
            # Check current columns
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
        
        # Step 3: Create missing V2.0 tables
        print("\n📋 Ensuring all V2.0 tables exist...")
        
        # Enhanced tracker table
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
        print("   ✅ v2_enhanced_tracker ensured")
        
        # System alerts table (unified)
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
        print("   ✅ v2_system_alerts ensured")
        
        # Model decisions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS v2_model_decisions (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                recommended_model TEXT NOT NULL,
                confidence REAL NOT NULL,
                reasoning TEXT,
                fallback_models TEXT,
                created_at TEXT NOT NULL,
                used_at TEXT,
                performance_score REAL,
                metadata TEXT
            )
        """)
        print("   ✅ v2_model_decisions ensured")
        
        # Active metrics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS v2_active_metrics (
                id TEXT PRIMARY KEY,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metric_type TEXT NOT NULL,
                category TEXT,
                timestamp TEXT NOT NULL,
                metadata TEXT
            )
        """)
        print("   ✅ v2_active_metrics ensured")
        
        # Step 4: Migrate v2_alerts to v2_system_alerts if needed
        print("\n🔄 Checking for v2_alerts migration...")
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM v2_alerts")
            alerts_count = cursor.fetchone()[0]
            
            cursor = conn.execute("SELECT COUNT(*) FROM v2_system_alerts")
            system_alerts_count = cursor.fetchone()[0]
            
            if alerts_count > 0 and system_alerts_count == 0:
                print(f"   Migrating {alerts_count} alerts from v2_alerts to v2_system_alerts...")
                conn.execute("""
                    INSERT INTO v2_system_alerts (
                        id, alert_type, severity, title, description, 
                        source_component, related_task_id, created_at, metadata
                    )
                    SELECT 
                        id, 
                        COALESCE(alert_type, 'general') as alert_type,
                        COALESCE(severity, 'medium') as severity,
                        COALESCE(message, title, 'Alert') as title,
                        COALESCE(description, message, 'No description') as description,
                        COALESCE(source, 'system') as source_component,
                        task_id as related_task_id,
                        COALESCE(timestamp, datetime('now')) as created_at,
                        metadata
                    FROM v2_alerts
                """)
                migrated = conn.rowcount
                print(f"   ✅ Migrated {migrated} alerts")
            else:
                print("   ✅ No migration needed")
                
        except sqlite3.OperationalError:
            print("   ✅ v2_alerts table not found (no migration needed)")
        
        # Step 5: Create useful indexes
        print("\n📈 Creating performance indexes...")
        indexes = [
            ("idx_v2_tracker_task_type", "v2_enhanced_tracker", "task_type"),
            ("idx_v2_tracker_timestamp", "v2_enhanced_tracker", "timestamp"),
            ("idx_v2_alerts_severity", "v2_system_alerts", "severity"),
            ("idx_v2_alerts_created", "v2_system_alerts", "created_at"),
            ("idx_v2_evaluations_task", "v2_success_evaluations", "task_id"),
            ("idx_v2_metrics_name", "v2_active_metrics", "metric_name")
        ]
        
        for idx_name, table, column in indexes:
            try:
                conn.execute(f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table}({column})")
                print(f"   ✅ Index {idx_name} created")
            except sqlite3.OperationalError:
                print(f"   ⚠️  Index {idx_name} skipped (table may not exist)")
        
        # Step 6: Commit all changes
        conn.commit()
        
        # Step 7: Optimize database
        print("\n🔧 Optimizing database...")
        conn.execute("PRAGMA optimize")
        conn.execute("VACUUM")
        print("   ✅ Database optimized")
        
        # Final verification
        print("\n📊 Final V2.0 Schema Verification:")
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v2_%'")
        final_tables = [row[0] for row in cursor.fetchall()]
        
        for table in final_tables:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            print(f"   ✅ {table}: {count} records")
        
        conn.close()
        
        print(f"\n🎉 V2.0 Schema Migration Complete!")
        print(f"📊 Total V2.0 tables: {len(final_tables)}")
        print(f"✅ All metadata columns added")
        print(f"✅ All indexes created")
        print(f"✅ Database optimized")
        
        return True
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            print(f"⚠️  Database is locked. Waiting...")
            time.sleep(2)
            return migrate_v2_schema_safe()  # Retry once
        else:
            print(f"❌ Database operation failed: {e}")
            return False
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        return False

def verify_v2_schema():
    """Quick verification of V2.0 schema"""
    try:
        with sqlite3.connect("agent_zero.db") as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'v2_%'")
            tables = [row[0] for row in cursor.fetchall()]
            
            print(f"📊 V2.0 Tables Found: {len(tables)}")
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"   {table}: {count} records")
                
            return len(tables) >= 4  # Expect at least 4 V2.0 tables
            
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Agent Zero V2.0 Schema Migration...")
    
    # Change to project root if needed
    if not os.path.exists("agent_zero.db") and os.path.exists("../agent_zero.db"):
        os.chdir("..")
        print("📁 Changed to parent directory")
    
    if not os.path.exists("agent_zero.db"):
        print("❌ agent_zero.db not found. Run from project root directory.")
        sys.exit(1)
    
    # Run migration
    success = migrate_v2_schema_safe()
    
    if success:
        print("\n🧪 Running verification...")
        if verify_v2_schema():
            print("✅ V2.0 Schema Migration: SUCCESS")
            print("\n🚀 You can now run:")
            print("   python3 v2-quickfix.py")
            print("   python3 test-v2-core.py")
            print("   python3 cli/advanced_commands.py v2-system status")
        else:
            print("⚠️  Migration completed but verification failed")
    else:
        print("❌ V2.0 Schema Migration: FAILED")
        sys.exit(1)
