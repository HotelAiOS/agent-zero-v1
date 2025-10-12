#!/usr/bin/env python3
"""
Agent Zero V1 - Enhanced SimpleTracker for V2.0 Intelligence Layer
Backward compatible with V2.0 extensions
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TaskStats:
    """Statistics for tasks or models"""
    total_tasks: int
    avg_cost: float
    avg_rating: Optional[float]
    success_rate: float
    feedback_count: int

class SimpleTracker:
    """Enhanced tracking system with V2.0 Intelligence Layer support"""
    
    def __init__(self, db_path: str = ".agent-zero/tracker.db"):
        self.db_path = Path.home() / db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.init_schema()
    
    def init_schema(self):
        """Initialize enhanced schema with V2.0 tables"""
        
        # Core tables (backward compatibility)
        self.conn.execute("""
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    task_type TEXT NOT NULL,
    model_used TEXT NOT NULL,
    model_recommended TEXT NOT NULL,
    cost_usd REAL DEFAULT 0.0,
    latency_ms INTEGER DEFAULT 0,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    context TEXT
)""")
        
        self.conn.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    task_id TEXT NOT NULL,
    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
    comment TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
)""")
        
        # V2.0 Extensions
        self.conn.execute("""
CREATE TABLE IF NOT EXISTS evaluations (
    task_id TEXT PRIMARY KEY,
    overall_score REAL,
    success_level TEXT,
    correctness_score REAL,
    efficiency_score REAL,
    cost_score REAL,
    latency_score REAL,
    predicted_probability REAL,
    confidence REAL,
    recommendations TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (task_id) REFERENCES tasks(id)
)""")
        
        self.conn.execute("""
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    alert_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    affected_model TEXT,
    affected_task_type TEXT,
    metric_value REAL,
    threshold_value REAL,
    suggestion TEXT,
    resolved BOOLEAN DEFAULT FALSE,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
        
        self.conn.execute("""
CREATE TABLE IF NOT EXISTS patterns (
    pattern_id TEXT PRIMARY KEY,
    pattern_type TEXT NOT NULL,
    conditions TEXT,
    outcomes TEXT,
    confidence REAL,
    sample_count INTEGER DEFAULT 1,
    success_rate REAL,
    last_seen DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
        
        # Create indexes
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_tasks_timestamp ON tasks(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_model ON tasks(model_used)",
            "CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type)",
            "CREATE INDEX IF NOT EXISTS idx_feedback_task ON feedback(task_id)",
            "CREATE INDEX IF NOT EXISTS idx_evaluations_task ON evaluations(task_id)",
            "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)"
        ]
        
        for idx_query in indexes:
            try:
                self.conn.execute(idx_query)
            except sqlite3.OperationalError:
                pass  # Index exists
        
        self.conn.commit()
    
    def track_task(self, task_id: str, task_type: str, model_used: str,
                   model_recommended: str, cost: float, latency: int,
                   context: Optional[Dict] = None):
        """Track completed task with V2.0 context support"""
        context_json = json.dumps(context) if context else None
        
        self.conn.execute("""
INSERT OR REPLACE INTO tasks 
(id, task_type, model_used, model_recommended, cost_usd, latency_ms, context)
VALUES (?, ?, ?, ?, ?, ?, ?)
""", (task_id, task_type, model_used, model_recommended, cost, latency, context_json))
        self.conn.commit()
    
    def record_feedback(self, task_id: str, rating: int, comment: str = None):
        """Record user feedback"""
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be between 1 and 5")
        
        self.conn.execute("""
INSERT INTO feedback (task_id, rating, comment) VALUES (?, ?, ?)
""", (task_id, rating, comment))
        self.conn.commit()
    
    def save_evaluation(self, task_id: str, overall_score: float, success_level: str,
                       scores: Dict[str, float], recommendations: List[str]):
        """Save V2.0 success evaluation"""
        recommendations_json = json.dumps(recommendations)
        
        self.conn.execute("""
INSERT OR REPLACE INTO evaluations 
(task_id, overall_score, success_level, correctness_score, efficiency_score,
 cost_score, latency_score, recommendations)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", (
            task_id, overall_score, success_level,
            scores.get('correctness', 0.5),
            scores.get('efficiency', 0.5),
            scores.get('cost', 0.5),
            scores.get('latency', 0.5),
            recommendations_json
        ))
        self.conn.commit()
    
    def save_alert(self, alert_id: str, alert_type: str, severity: str,
                   message: str, affected_model: str = None, suggestion: str = None):
        """Save V2.0 alert"""
        self.conn.execute("""
INSERT OR REPLACE INTO alerts 
(alert_id, alert_type, severity, message, affected_model, suggestion)
VALUES (?, ?, ?, ?, ?, ?)
""", (alert_id, alert_type, severity, message, affected_model, suggestion))
        self.conn.commit()
    
    def get_model_comparison(self, days: int = 7) -> Dict[str, Dict]:
        """Enhanced model comparison with V2.0 metrics"""
        cursor = self.conn.execute(f"""
SELECT 
    t.model_used,
    COUNT(*) as usage_count,
    AVG(t.cost_usd) as avg_cost,
    AVG(f.rating) as avg_rating,
    COUNT(f.rating) as feedback_count,
    AVG(t.latency_ms) as avg_latency,
    SUM(CASE WHEN t.model_used != t.model_recommended THEN 1 ELSE 0 END) as override_count,
    AVG(e.overall_score) as avg_success_score,
    AVG(e.confidence) as avg_confidence
FROM tasks t
LEFT JOIN feedback f ON t.id = f.task_id
LEFT JOIN evaluations e ON t.id = e.task_id
WHERE t.timestamp >= datetime('now', '-{days} days')
GROUP BY t.model_used
""")
        
        results = {}
        for row in cursor.fetchall():
            model = row[0]
            usage_count = row[1]
            avg_cost = row[2] or 0.0
            avg_rating = row[3] or 2.5
            feedback_count = row[4]
            avg_latency = row[5] or 0
            override_count = row[6]
            avg_success_score = row[7] or 0.5
            avg_confidence = row[8] or 0.5
            
            # Calculate success rate
            success_cursor = self.conn.execute(f"""
SELECT COUNT(*) as success_count
FROM tasks t
JOIN feedback f ON t.id = f.task_id
WHERE t.model_used = ? AND f.rating >= 4 
AND t.timestamp >= datetime('now', '-{days} days')
""", (model,))
            
            success_count = success_cursor.fetchone()[0]
            success_rate = (success_count / feedback_count) if feedback_count > 0 else 0.5
            
            # Enhanced scoring with V2.0 metrics
            quality_score = (avg_rating * 0.3) + (avg_success_score * 0.7)
            score = ((quality_score * 0.4) + (success_rate * 0.3) + 
                    (avg_confidence * 0.1) - (min(avg_cost * 100, 1.0) * 0.2))
            
            results[model] = {
                'usage_count': usage_count,
                'avg_cost': avg_cost,
                'avg_rating': avg_rating,
                'feedback_count': feedback_count,
                'avg_latency': avg_latency,
                'success_rate': success_rate,
                'override_count': override_count,
                'score': score,
                'avg_success_score': avg_success_score,
                'avg_confidence': avg_confidence,
                'human_acceptance_rate': (1.0 - (override_count / usage_count)) if usage_count > 0 else 0.5
            }
        
        return results
    
    def get_recent_tasks(self, days: int = 7) -> List[Dict]:
        """Get recent tasks with V2.0 data"""
        cursor = self.conn.execute(f"""
SELECT 
    t.id, t.task_type, t.model_used, t.model_recommended,
    t.cost_usd, t.latency_ms, t.context,
    f.rating, f.comment,
    e.overall_score, e.success_level
FROM tasks t
LEFT JOIN feedback f ON t.id = f.task_id
LEFT JOIN evaluations e ON t.id = e.task_id
WHERE t.timestamp >= datetime('now', '-{days} days')
ORDER BY t.timestamp DESC
""")
        
        results = []
        for row in cursor.fetchall():
            task_data = {
                'task_id': row[0],
                'task_type': row[1],
                'model_used': row[2],
                'model_recommended': row[3],
                'cost_usd': row[4],
                'latency_ms': row[5],
                'context': json.loads(row[6]) if row[6] else {},
                'rating': row[7],
                'comment': row[8],
                'overall_score': row[9],
                'success_level': row[10]
            }
            results.append(task_data)
        
        return results
    
    def close(self):
        """Close database connection"""
        self.conn.close()
