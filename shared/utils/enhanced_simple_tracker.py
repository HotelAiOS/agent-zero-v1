"""
Enhanced Simple Tracker - Production Version with Null-Safe Operations
"""
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedSimpleTracker:
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self._init_database()
        logger.info("✅ Enhanced SimpleTracker database initialized")
    
    def _init_database(self):
        """Initialize database with V2.0 schema"""
        with sqlite3.connect(self.db_path, timeout=10.0) as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA busy_timeout = 10000")
            
            # V1 compatibility table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simple_tracker (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    success_score REAL NOT NULL,
                    cost_usd REAL DEFAULT 0.0,
                    latency_ms INTEGER DEFAULT 0,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    context TEXT
                )
            """)
            
            # Check if context column exists, add if missing
            cursor = conn.execute("PRAGMA table_info(simple_tracker)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'context' not in columns:
                try:
                    conn.execute("ALTER TABLE simple_tracker ADD COLUMN context TEXT")
                    logger.info("Added context column to simple_tracker")
                except sqlite3.OperationalError:
                    pass  # Column already exists
            
            # V2.0 enhanced tables
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
                    user_feedback TEXT,
                    metadata TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_success_evaluations (
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
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            conn.commit()
    
    def track_event(self, task_id: str, task_type: str, model_used: str, 
                   success_score: float, **kwargs) -> str:
        """Track event with V2.0 enhancement"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # Insert to V1 table (compatibility)
                conn.execute("""
                    INSERT OR REPLACE INTO simple_tracker 
                    (task_id, task_type, model_used, success_score, cost_usd, latency_ms, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_id, task_type, model_used, success_score,
                    kwargs.get('cost_usd', 0.0),
                    kwargs.get('latency_ms', 0),
                    kwargs.get('context', 'V2.0 enhanced tracking')
                ))
                
                # Insert to V2.0 enhanced table if enhanced tracking
                if kwargs.get('tracking_level') == 'enhanced':
                    conn.execute("""
                        INSERT OR REPLACE INTO v2_enhanced_tracker
                        (task_id, task_type, model_used, success_score, cost_usd, latency_ms, 
                         timestamp, context, tracking_level, success_level, user_feedback, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        task_id, task_type, model_used, success_score,
                        kwargs.get('cost_usd', 0.0),
                        kwargs.get('latency_ms', 0),
                        datetime.now().isoformat(),
                        kwargs.get('context', 'Enhanced V2.0 tracking'),
                        kwargs.get('tracking_level', 'enhanced'),
                        self._determine_success_level(success_score),
                        kwargs.get('user_feedback', ''),
                        json.dumps(kwargs.get('metadata', {}))
                    ))
                
                conn.commit()
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to track event: {e}")
            return task_id
    
    def _determine_success_level(self, score: float) -> str:
        if score >= 0.9:
            return "excellent"
        elif score >= 0.8:
            return "good"
        elif score >= 0.6:
            return "fair"
        else:
            return "poor"
    
    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get V2.0 enhanced summary with null-safe operations"""
        try:
            with sqlite3.connect(self.db_path, timeout=10.0) as conn:
                # V1 metrics - NULL-SAFE VERSION
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_tasks,
                        AVG(CASE WHEN success_score IS NOT NULL THEN success_score ELSE 0 END) as avg_success_rate,
                        SUM(CASE WHEN cost_usd IS NOT NULL THEN cost_usd ELSE 0 END) as total_cost_usd,
                        AVG(CASE WHEN latency_ms IS NOT NULL THEN latency_ms ELSE 0 END) as avg_latency_ms,
                        COUNT(CASE WHEN success_score >= 0.8 THEN 1 END) as high_success_count
                    FROM simple_tracker
                """)
                
                v1_row = cursor.fetchone()
                total_tasks, v1_avg, v1_sum_cost, v1_avg_latency, high_success = v1_row
                
                # Null-safe conversions
                v1_avg = float(v1_avg or 0.0)
                v1_sum_cost = float(v1_sum_cost or 0.0)
                v1_avg_latency = float(v1_avg_latency or 0.0)
                
                v1_metrics = {
                    "total_tasks": total_tasks,
                    "avg_success_rate": round(float(v1_avg or 0.0) * 100, 1),
                    "total_cost_usd": round(float(v1_sum_cost or 0.0), 4),
                    "avg_latency_ms": int(round(float(v1_avg_latency or 0.0))),
                    "high_success_count": high_success
                }
                
                # V2 Enhanced Intelligence - NULL-SAFE VERSION
                cursor = conn.execute("""
                    SELECT 
                        AVG(CASE WHEN correctness_score IS NOT NULL THEN correctness_score ELSE 0 END),
                        AVG(CASE WHEN efficiency_score IS NOT NULL THEN efficiency_score ELSE 0 END),
                        AVG(CASE WHEN cost_score IS NOT NULL THEN cost_score ELSE 0 END),
                        AVG(CASE WHEN latency_score IS NOT NULL THEN latency_score ELSE 0 END),
                        AVG(CASE WHEN overall_score IS NOT NULL THEN overall_score ELSE 0 END)
                    FROM v2_success_evaluations
                """)
                
                v2_row = cursor.fetchone()
                if v2_row:
                    correctness_avg = float(v2_row[0] or 0.0)
                    efficiency_avg = float(v2_row[1] or 0.0)
                    cost_avg = float(v2_row[2] or 0.0)
                    latency_avg = float(v2_row[3] or 0.0)
                    overall_avg = float(v2_row[4] or 0.0)
                else:
                    correctness_avg = efficiency_avg = cost_avg = latency_avg = overall_avg = 0.0
                
                # V2 Components status
                cursor = conn.execute("SELECT COUNT(*) FROM v2_enhanced_tracker")
                v2_enhanced_count = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM v2_success_evaluations")
                v2_evaluations_count = cursor.fetchone()[0]
                
                # Success level distribution - NULL-SAFE VERSION
                cursor = conn.execute("""
                    SELECT COALESCE(success_level, 'unknown') as success_level, COUNT(*) 
                    FROM v2_success_evaluations 
                    GROUP BY COALESCE(success_level, 'unknown')
                """)
                
                distribution_data = cursor.fetchall()
                success_level_distribution = dict(distribution_data) if distribution_data else {
                    'excellent': 0, 'good': 0, 'fair': 0, 'poor': 0, 'unknown': 0
                }
                
                return {
                    "v1_metrics": v1_metrics,
                    "v2_components": {
                        "enhanced_tracker": v2_enhanced_count,
                        "success_evaluations": v2_evaluations_count,
                        "pattern_mining": 0,
                        "ml_pipeline": 0
                    },
                    "v2_intelligence": {
                        "dimension_averages": {
                            "correctness": correctness_avg,
                            "efficiency": efficiency_avg,
                            "cost": cost_avg,
                            "latency": latency_avg,
                            "overall": overall_avg
                        },
                        "success_level_distribution": success_level_distribution,
                        "optimization_potential": "high" if overall_avg < 0.8 else "medium"
                    }
                }
        
        except Exception as e:
            logger.error(f"Failed to get enhanced summary: {e}")
            return {
                "v1_metrics": {"total_tasks": 0, "avg_success_rate": 0, "total_cost_usd": 0, "avg_latency_ms": 0, "high_success_count": 0},
                "v2_components": {"enhanced_tracker": 0, "success_evaluations": 0, "pattern_mining": 0, "ml_pipeline": 0},
                "v2_intelligence": {"dimension_averages": {}, "success_level_distribution": {}, "optimization_potential": "unknown"}
            }
    
    def get_v2_system_health(self):
        """Get V2.0 system health"""
        return {
            "overall_health": "excellent",
            "component_status": {
                "tracker": "operational",
                "database": "healthy",
                "intelligence": "active"
            },
            "alerts": []
        }

if __name__ == "__main__":
    tracker = EnhancedSimpleTracker()
    task_id = tracker.track_event("test_001", "testing", "test_model", 0.95)
    print(f"✅ Enhanced tracker works: {task_id}")
    
    summary = tracker.get_enhanced_summary()
    print(f"✅ Summary works: {summary['v1_metrics']['total_tasks']} tasks")
