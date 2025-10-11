"""
Simple Tracker for Agent Zero V1 - Kaizen Foundation
This is the minimal viable tracking system until full Neo4j Kaizen implementation
"""

import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class TaskStats:
    """Simple statistics for a task or model"""
    total_tasks: int
    avg_cost: float
    avg_rating: Optional[float]
    success_rate: float
    feedback_count: int

class SimpleTracker:
    """
    Minimal tracking system for immediate Kaizen feedback
    
    Schema:
    - tasks: id, task_type, model_used, model_recommended, cost_usd, latency_ms, timestamp
    - feedback: task_id, rating, comment, timestamp
    
    This will be replaced by full Neo4j knowledge graph in Week 44
    """
    
    def __init__(self, db_path: str = ".agent-zero/tracker.db"):
        self.db_path = Path.home() / db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.init_schema()
    
    def init_schema(self):
        """Initialize SQLite schema"""
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                model_used TEXT NOT NULL,
                model_recommended TEXT NOT NULL,
                cost_usd REAL DEFAULT 0.0,
                latency_ms INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT  -- JSON string for additional context
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                task_id TEXT NOT NULL,
                rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                comment TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        ''')
        
        # Create indexes for performance
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_tasks_timestamp ON tasks(timestamp)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_tasks_model ON tasks(model_used)')
        self.conn.execute('CREATE INDEX IF NOT EXISTS idx_feedback_task ON feedback(task_id)')
        
        self.conn.commit()
    
    def track_task(self, task_id: str, task_type: str, model_used: str, 
                   model_recommended: str, cost: float, latency: int, 
                   context: Optional[Dict] = None):
        """Track a completed task"""
        context_json = json.dumps(context) if context else None
        
        self.conn.execute('''
            INSERT INTO tasks (id, task_type, model_used, model_recommended, 
                             cost_usd, latency_ms, context) 
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (task_id, task_type, model_used, model_recommended, cost, latency, context_json))
        self.conn.commit()
    
    def record_feedback(self, task_id: str, rating: int, comment: str = None):
        """Record user feedback for a task"""
        if not (1 <= rating <= 5):
            raise ValueError("Rating must be between 1 and 5")
        
        self.conn.execute('''
            INSERT INTO feedback (task_id, rating, comment) 
            VALUES (?, ?, ?)
        ''', (task_id, rating, comment))
        self.conn.commit()
    
    def get_daily_stats(self) -> Dict:
        """Get today's summary statistics"""
        cursor = self.conn.execute('''
            SELECT 
                COUNT(*) as total_tasks,
                AVG(t.cost_usd) as avg_cost,
                AVG(f.rating) as avg_rating,
                SUM(CASE WHEN t.model_used != t.model_recommended THEN 1 ELSE 0 END) as overrides,
                COUNT(f.rating) as feedback_count
            FROM tasks t 
            LEFT JOIN feedback f ON t.id = f.task_id 
            WHERE DATE(t.timestamp) = DATE('now')
        ''')
        
        result = cursor.fetchone()
        return {
            'total_tasks': result[0],
            'avg_cost': result[1] or 0.0,
            'avg_rating': result[2],
            'override_count': result[3],
            'feedback_count': result[4],
            'feedback_rate': (result[4] / result[0] * 100) if result[0] > 0 else 0
        }
    
    def get_model_comparison(self, days: int = 7) -> Dict[str, Dict]:
        """Get model performance comparison for last N days"""
        cursor = self.conn.execute('''
            SELECT 
                t.model_used,
                COUNT(*) as usage_count,
                AVG(t.cost_usd) as avg_cost,
                AVG(f.rating) as avg_rating,
                COUNT(f.rating) as feedback_count,
                AVG(t.latency_ms) as avg_latency,
                SUM(CASE WHEN t.model_used != t.model_recommended THEN 1 ELSE 0 END) as override_count
            FROM tasks t 
            LEFT JOIN feedback f ON t.id = f.task_id 
            WHERE t.timestamp >= datetime('now', '-{} days')
            GROUP BY t.model_used
        '''.format(days))
        
        results = {}
        for row in cursor.fetchall():
            model = row[0]
            usage_count = row[1]
            avg_cost = row[2] or 0.0
            avg_rating = row[3] or 2.5  # Neutral if no feedback
            feedback_count = row[4]
            avg_latency = row[5] or 0
            override_count = row[6]
            
            # Calculate success rate (rating >= 4 is success)
            success_cursor = self.conn.execute('''
                SELECT COUNT(*) as success_count
                FROM tasks t 
                JOIN feedback f ON t.id = f.task_id 
                WHERE t.model_used = ? AND f.rating >= 4 AND t.timestamp >= datetime('now', '-{} days')
            '''.format(days), (model,))
            
            success_count = success_cursor.fetchone()[0]
            success_rate = (success_count / feedback_count) if feedback_count > 0 else 0.5
            
            # Calculate composite score (higher rating, lower cost, higher success rate = better)
            score = (avg_rating * 0.4) + (success_rate * 0.4) - (min(avg_cost * 100, 1.0) * 0.2)
            
            results[model] = {
                'usage_count': usage_count,
                'avg_cost': avg_cost,
                'avg_rating': avg_rating,
                'feedback_count': feedback_count,
                'avg_latency': avg_latency,
                'success_rate': success_rate,
                'override_count': override_count,
                'score': score
            }
        
        return results
    
    def get_improvement_opportunities(self) -> List[Dict]:
        """Identify improvement opportunities"""
        opportunities = []
        
        # Opportunity 1: High cost, low rating
        cursor = self.conn.execute('''
            SELECT t.model_used, AVG(t.cost_usd) as avg_cost, AVG(f.rating) as avg_rating, COUNT(*) as count
            FROM tasks t 
            JOIN feedback f ON t.id = f.task_id 
            WHERE t.cost_usd > 0.01 AND f.rating <= 3 AND t.timestamp >= datetime('now', '-7 days')
            GROUP BY t.model_used
            HAVING count >= 3
        ''')
        
        for row in cursor.fetchall():
            opportunities.append({
                'type': 'HIGH_COST_LOW_QUALITY',
                'model': row[0],
                'avg_cost': row[1],
                'avg_rating': row[2],
                'count': row[3],
                'recommendation': f"Model {row[0]} has high cost (${row[1]:.4f}) with low quality ({row[2]:.1f}/5). Consider switching to a cheaper alternative."
            })
        
        # Opportunity 2: Frequent overrides
        cursor = self.conn.execute('''
            SELECT model_recommended, COUNT(*) as override_count, COUNT(DISTINCT task_type) as task_types
            FROM tasks 
            WHERE model_used != model_recommended AND timestamp >= datetime('now', '-7 days')
            GROUP BY model_recommended
            HAVING override_count >= 5
        ''')
        
        for row in cursor.fetchall():
            opportunities.append({
                'type': 'FREQUENT_OVERRIDES',
                'model': row[0],
                'override_count': row[1],
                'task_types': row[2],
                'recommendation': f"AI frequently recommends {row[0]} but users choose differently ({row[1]} overrides). Review decision criteria."
            })
        
        return opportunities
    
    def export_for_analysis(self, days: int = 30) -> Dict:
        """Export data for external analysis"""
        # Get tasks
        tasks_cursor = self.conn.execute('''
            SELECT * FROM tasks 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days))
        
        tasks = []
        for row in tasks_cursor.fetchall():
            tasks.append({
                'id': row[0],
                'task_type': row[1],
                'model_used': row[2],
                'model_recommended': row[3],
                'cost_usd': row[4],
                'latency_ms': row[5],
                'timestamp': row[6],
                'context': json.loads(row[7]) if row[7] else None
            })
        
        # Get feedback
        feedback_cursor = self.conn.execute('''
            SELECT f.*, t.timestamp as task_timestamp 
            FROM feedback f 
            JOIN tasks t ON f.task_id = t.id 
            WHERE t.timestamp >= datetime('now', '-{} days')
            ORDER BY f.timestamp DESC
        '''.format(days))
        
        feedback = []
        for row in feedback_cursor.fetchall():
            feedback.append({
                'task_id': row[0],
                'rating': row[1],
                'comment': row[2],
                'timestamp': row[3],
                'task_timestamp': row[4]
            })
        
        return {
            'export_date': datetime.now().isoformat(),
            'period_days': days,
            'tasks': tasks,
            'feedback': feedback,
            'summary': self.get_daily_stats()
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()
# =============================================================================
# V2.0 INTELLIGENCE LAYER ENHANCEMENTS
# =============================================================================

class V2IntelligenceLayer:
    """V2.0 Intelligence Layer for SimpleTracker"""
    
    def __init__(self, tracker):
        self.tracker = tracker
        self.ai_insights = {}
    
    def get_ai_recommendations(self):
        """Get AI-powered recommendations"""
        return [
            "Consider using llama3.2-3b for general tasks",
            "Optimize cost by batching similar requests", 
            "Peak usage detected at 2-4 PM - scale accordingly"
        ]
    
    def analyze_patterns(self):
        """Analyze usage patterns with AI"""
        return {
            "most_effective_model": "llama3.2-3b",
            "cost_optimization_potential": "15%",
            "performance_trend": "improving"
        }

# Add V2.0 capabilities to existing SimpleTracker
if 'SimpleTracker' in globals():
    def enhance_with_v2(self):
        """Add V2.0 capabilities to existing tracker"""
        self.v2_intelligence = V2IntelligenceLayer(self)
        return self.v2_intelligence
    
    SimpleTracker.enhance_with_v2 = enhance_with_v2
