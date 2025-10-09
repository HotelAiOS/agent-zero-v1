"""
Human-AI Feedback Loop Engine for Agent Zero V1
Core Kaizen Engine - learning from every human feedback

This is the HEART of AI-first methodology:
- System proposes, human decides, system learns
"""

from enum import Enum
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
import sqlite3
from pathlib import Path

class FeedbackType(Enum):
    """Types of feedback from user"""
    THUMBS_UP = "thumbs_up"        # Success
    THUMBS_DOWN = "thumbs_down"    # Failure
    CORRECTION = "correction"       # Human corrected output
    ALTERNATIVE = "alternative"     # Human chose different model
    QUALITY_RATING = "quality_rating"  # 1-5 stars

@dataclass
class FeedbackEvent:
    """Single feedback event"""
    task_id: str
    feedback_type: FeedbackType
    quality_rating: Optional[int] = None  # 1-5
    model_used: str = ""
    model_recommended: str = ""
    human_choice: Optional[str] = None  # When user chose different model
    cost_actual: float = 0.0
    latency_actual: float = 0.0
    human_comment: Optional[str] = None
    timestamp: datetime = None
    context: Dict = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.context is None:
            self.context = {}

@dataclass
class Pattern:
    """Detected pattern from feedback analysis"""
    type: str  # LOW_SUCCESS_RATE, FREQUENT_OVERRIDE, POOR_COST_EFFICIENCY
    model: str
    task_type: Optional[str] = None
    metric: float = 0.0
    recommendation: str = ""
    confidence: float = 0.0
    occurrences: int = 0

@dataclass
class FeedbackProcessingResult:
    """Result of processing feedback"""
    feedback_id: str
    patterns_detected: List[Pattern]
    improvements_suggested: List[str]
    learning_applied: bool
    alerts_generated: List[str] = None

class FeedbackLoopEngine:
    """
    Core Kaizen Engine - learning from every feedback
    
    Philosophy: Every interaction is a learning opportunity
    Flow:
    1. Collect feedback
    2. Analyze patterns
    3. Update model performance stats
    4. Adjust decision weights
    5. Generate improvement suggestions
    """
    
    def __init__(self, db_path: str = ".agent-zero/kaizen.db"):
        self.db_path = Path.home() / db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.learning_rate = 0.1  # How fast we adapt
        self.init_schema()
    
    def init_schema(self):
        """Initialize feedback tracking schema"""
        
        # Extended tasks table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT NOT NULL,
                model_used TEXT NOT NULL,
                model_recommended TEXT NOT NULL,
                cost_usd REAL DEFAULT 0.0,
                latency_ms INTEGER DEFAULT 0,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                context TEXT,  -- JSON
                success_score REAL DEFAULT 0.5  -- Multi-dimensional success score
            )
        ''')
        
        # Enhanced feedback table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                rating INTEGER CHECK(rating >= 1 AND rating <= 5),
                comment TEXT,
                human_choice TEXT,  -- Alternative model chosen by human
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        ''')
        
        # Pattern recognition table
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT NOT NULL,
                model TEXT NOT NULL,
                task_type TEXT,
                metric_value REAL,
                confidence REAL,
                occurrences INTEGER,
                recommendation TEXT,
                first_detected DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Model performance tracking
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                success_rate REAL DEFAULT 0.5,
                avg_cost REAL DEFAULT 0.0,
                avg_latency REAL DEFAULT 0.0,
                avg_rating REAL DEFAULT 2.5,
                usage_count INTEGER DEFAULT 0,
                override_count INTEGER DEFAULT 0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model, task_type)
            )
        ''')
        
        # Decision weights for adaptive learning
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS decision_weights (
                model TEXT NOT NULL,
                task_type TEXT NOT NULL,
                quality_weight REAL DEFAULT 0.5,
                cost_weight REAL DEFAULT 0.3,
                latency_weight REAL DEFAULT 0.2,
                human_preference_boost REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (model, task_type)
            )
        ''')
        
        self.conn.commit()
    
    async def process_feedback(self, feedback: FeedbackEvent) -> FeedbackProcessingResult:
        """
        Process feedback and LEARN from it
        
        Flow:
        1. Store feedback
        2. Update model statistics  
        3. Detect patterns
        4. Adjust decision weights
        5. Generate alerts and suggestions
        """
        
        # 1. Store feedback
        feedback_id = await self.store_feedback(feedback)
        
        # 2. Update model performance statistics
        await self.update_model_stats(feedback)
        
        # 3. Pattern detection - this is where AI learns!
        patterns = await self.detect_patterns(feedback)
        
        # 4. Weight adjustment based on patterns
        if patterns:
            await self.adjust_decision_weights(patterns)
        
        # 5. Generate improvement suggestions
        suggestions = await self.generate_improvement_suggestions(feedback, patterns)
        
        # 6. Check for alerts (anomalies, critical issues)
        alerts = await self.check_for_alerts(feedback)
        
        return FeedbackProcessingResult(
            feedback_id=feedback_id,
            patterns_detected=patterns,
            improvements_suggested=suggestions,
            learning_applied=True,
            alerts_generated=alerts
        )
    
    async def store_feedback(self, feedback: FeedbackEvent) -> str:
        """Store feedback event"""
        feedback_id = f"fb_{int(datetime.now().timestamp())}"
        
        self.conn.execute('''
            INSERT INTO feedback (id, task_id, feedback_type, rating, comment, human_choice)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            feedback_id,
            feedback.task_id,
            feedback.feedback_type.value,
            feedback.quality_rating,
            feedback.human_comment,
            feedback.human_choice
        ))
        self.conn.commit()
        
        return feedback_id
    
    async def update_model_stats(self, feedback: FeedbackEvent):
        """Update model performance statistics"""
        
        # Get or create model performance record
        self.conn.execute('''
            INSERT OR IGNORE INTO model_performance (model, task_type, usage_count)
            VALUES (?, ?, 0)
        ''', (feedback.model_used, feedback.context.get('task_type', 'unknown')))
        
        # Update statistics
        if feedback.quality_rating:
            # Update with new rating
            self.conn.execute('''
                UPDATE model_performance 
                SET avg_rating = COALESCE(
                    (avg_rating * usage_count + ?) / (usage_count + 1), 
                    ?
                ),
                usage_count = usage_count + 1,
                success_rate = CASE 
                    WHEN ? >= 4 THEN COALESCE((success_rate * (usage_count - 1) + 1.0) / usage_count, 1.0)
                    ELSE COALESCE((success_rate * (usage_count - 1) + 0.0) / usage_count, 0.0)
                END,
                last_updated = CURRENT_TIMESTAMP
                WHERE model = ? AND task_type = ?
            ''', (
                feedback.quality_rating,
                feedback.quality_rating,
                feedback.quality_rating,
                feedback.model_used,
                feedback.context.get('task_type', 'unknown')
            ))
        
        # Track overrides (human chose different model)
        if feedback.human_choice and feedback.human_choice != feedback.model_recommended:
            self.conn.execute('''
                UPDATE model_performance 
                SET override_count = override_count + 1
                WHERE model = ? AND task_type = ?
            ''', (feedback.model_recommended, feedback.context.get('task_type', 'unknown')))
        
        self.conn.commit()
    
    async def detect_patterns(self, feedback: FeedbackEvent) -> List[Pattern]:
        """
        Pattern Recognition - the heart of Kaizen
        
        Detects patterns like:
        - Model X fails for complex analysis tasks
        - Claude is overkill for simple CRUD
        - User always overrides AI on critical tasks
        """
        patterns = []
        
        task_type = feedback.context.get('task_type', 'unknown')
        
        # Pattern 1: Low success rate
        cursor = self.conn.execute('''
            SELECT AVG(CASE WHEN f.rating >= 4 THEN 1.0 ELSE 0.0 END) as success_rate,
                   COUNT(*) as total_feedback
            FROM feedback f
            JOIN tasks t ON f.task_id = t.id
            WHERE t.model_used = ? AND t.task_type = ? AND f.rating IS NOT NULL
        ''', (feedback.model_used, task_type))
        
        result = cursor.fetchone()
        if result and result[1] >= 5:  # Need at least 5 samples
            success_rate = result[0]
            if success_rate < 0.7:  # Below 70% success
                patterns.append(Pattern(
                    type="LOW_SUCCESS_RATE",
                    model=feedback.model_used,
                    task_type=task_type,
                    metric=success_rate,
                    recommendation=f"Model {feedback.model_used} has low success rate ({success_rate:.1%}) for {task_type}. Consider switching to alternative.",
                    confidence=min(result[1] / 10.0, 1.0),  # Confidence grows with sample size
                    occurrences=result[1]
                ))
        
        # Store detected patterns
        for pattern in patterns:
            self.conn.execute('''
                INSERT OR REPLACE INTO patterns 
                (id, pattern_type, model, task_type, metric_value, confidence, occurrences, recommendation)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"{pattern.type}_{pattern.model}_{pattern.task_type}",
                pattern.type,
                pattern.model,
                pattern.task_type,
                pattern.metric,
                pattern.confidence,
                pattern.occurrences,
                pattern.recommendation
            ))
        
        self.conn.commit()
        return patterns
    
    async def adjust_decision_weights(self, patterns: List[Pattern]):
        """ADAPTIVE LEARNING: Adjust decision weights based on patterns"""
        
        for pattern in patterns:
            if pattern.type == "LOW_SUCCESS_RATE":
                # Decrease this model's preference for this task type
                self.conn.execute('''
                    INSERT OR IGNORE INTO decision_weights (model, task_type)
                    VALUES (?, ?)
                ''', (pattern.model, pattern.task_type))
                
                self.conn.execute('''
                    UPDATE decision_weights 
                    SET human_preference_boost = human_preference_boost - ?
                    WHERE model = ? AND task_type = ?
                ''', (self.learning_rate * pattern.confidence, pattern.model, pattern.task_type))
        
        self.conn.commit()
    
    async def generate_improvement_suggestions(self, feedback: FeedbackEvent, patterns: List[Pattern]) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        # Suggestion 1: Model selection improvement
        if feedback.human_choice and feedback.human_choice != feedback.model_recommended:
            suggestions.append(
                f"Human chose {feedback.human_choice} instead of {feedback.model_recommended}. "
                f"Review decision criteria for {feedback.context.get('task_type', 'this type of')} tasks."
            )
        
        # Suggestion 2: Quality improvement
        if feedback.quality_rating and feedback.quality_rating <= 2:
            suggestions.append(
                f"Low quality rating ({feedback.quality_rating}/5). Consider: "
                f"1) Using higher-quality model, 2) Improving prompt clarity, 3) Adding examples"
            )
        
        return suggestions
    
    async def check_for_alerts(self, feedback: FeedbackEvent) -> List[str]:
        """Check for conditions that need immediate attention"""
        alerts = []
        
        # Alert 1: Consistently poor feedback
        if feedback.quality_rating and feedback.quality_rating <= 2:
            # Check if this is a pattern
            cursor = self.conn.execute('''
                SELECT AVG(f.rating) as avg_rating, COUNT(*) as count
                FROM feedback f
                JOIN tasks t ON f.task_id = t.id
                WHERE t.model_used = ? AND f.timestamp >= datetime('now', '-1 day')
            ''', (feedback.model_used,))
            
            result = cursor.fetchone()
            if result and result[1] >= 3 and result[0] <= 2.5:
                alerts.append(f"🚨 ALERT: {feedback.model_used} consistently poor performance (avg {result[0]:.1f}/5 over {result[1]} tasks)")
        
        return alerts
    
    def get_learning_summary(self, days: int = 7) -> Dict:
        """Get summary of learning progress"""
        
        # Get pattern counts
        cursor = self.conn.execute('''
            SELECT pattern_type, COUNT(*) as count
            FROM patterns
            WHERE last_updated >= datetime('now', '-{} days')
            GROUP BY pattern_type
        '''.format(days))
        
        pattern_counts = dict(cursor.fetchall())
        
        # Get improvement metrics
        cursor = self.conn.execute('''
            SELECT 
                COUNT(DISTINCT f.task_id) as total_feedback,
                AVG(f.rating) as avg_rating,
                SUM(CASE WHEN t.model_used != t.model_recommended THEN 1 ELSE 0 END) * 1.0 / COUNT(*) as override_rate
            FROM feedback f
            JOIN tasks t ON f.task_id = t.id
            WHERE f.timestamp >= datetime('now', '-{} days')
        '''.format(days))
        
        metrics = cursor.fetchone()
        
        return {
            'period_days': days,
            'patterns_detected': pattern_counts,
            'total_feedback': metrics[0] if metrics else 0,
            'avg_rating': metrics[1] if metrics else None,
            'override_rate': metrics[2] if metrics else None,
            'learning_velocity': len(pattern_counts),  # Number of different pattern types
            'adaptation_score': 1.0 - (metrics[2] or 0.5)  # Lower override rate = better adaptation
        }
    
    def close(self):
        """Close database connection"""
        self.conn.close()