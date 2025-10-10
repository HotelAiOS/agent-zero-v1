#!/usr/bin/env python3
"""
Agent Zero V1 - Enhanced SimpleTracker V2.0 Upgrade
V2.0 Intelligence Layer - Week 44 Implementation

🎯 Week 44 Critical Task: Enhanced SimpleTracker V2.0 Schema Upgrade
Zadanie: Upgrade istniejącego SimpleTracker z V2.0 capabilities
Rezultat: Unified tracking system z V1 compatibility i V2.0 features
Impact: Seamless transition do V2.0 bez breaking changes

Author: Developer A (Backend Architect)  
Date: 10 października 2025
Linear Issue: A0-44 (Week 44 Implementation)
"""

import sqlite3
import json
import uuid
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackingLevel(Enum):
    BASIC = "basic"          # V1.0 compatible
    ENHANCED = "enhanced"    # V2.0 with additional metrics
    FULL = "full"           # Complete V2.0 intelligence

class SuccessLevel(Enum):
    EXCELLENT = "excellent"   # 0.9+
    GOOD = "good"            # 0.7-0.89
    FAIR = "fair"            # 0.5-0.69
    POOR = "poor"            # <0.5

@dataclass
class TrackingEntry:
    """Enhanced tracking entry with V2.0 capabilities"""
    # V1.0 compatible fields
    task_id: str
    task_type: str
    model_used: str
    success_score: float
    cost_usd: Optional[float]
    latency_ms: Optional[int]
    timestamp: datetime
    context: Optional[str]
    
    # V2.0 enhanced fields
    tracking_level: TrackingLevel = TrackingLevel.BASIC
    success_level: Optional[SuccessLevel] = None
    dimension_scores: Optional[Dict[str, float]] = None
    user_feedback: Optional[str] = None
    lessons_learned: Optional[List[str]] = None
    optimization_applied: bool = False
    metadata: Optional[Dict[str, Any]] = None

class EnhancedSimpleTracker:
    """
    Enhanced SimpleTracker with V2.0 Intelligence Layer Integration
    
    Maintains 100% backward compatibility with V1.0 while adding:
    - Multi-dimensional success evaluation
    - Enhanced metadata tracking  
    - V2.0 intelligence integration hooks
    - Advanced analytics capabilities
    - Pattern learning integration
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self._init_enhanced_database()
        self.components_available = self._check_v2_components()
    
    def _init_enhanced_database(self):
        """Initialize enhanced database schema with V1/V2 compatibility"""
        with sqlite3.connect(self.db_path) as conn:
            # Ensure V1.0 table exists (backward compatibility)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simple_tracker (
                    task_id TEXT PRIMARY KEY,
                    task_type TEXT,
                    model_used TEXT,
                    success_score REAL,
                    cost_usd REAL,
                    latency_ms INTEGER,
                    timestamp TEXT,
                    context TEXT
                )
            """)
            
            # V2.0 Enhanced tracking table
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
                    
                    -- V2.0 Enhanced Fields
                    tracking_level TEXT DEFAULT 'basic',
                    success_level TEXT,
                    dimension_scores TEXT,  -- JSON: {correctness, efficiency, cost, latency}
                    user_feedback TEXT,
                    lessons_learned TEXT,   -- JSON array
                    optimization_applied BOOLEAN DEFAULT FALSE,
                    metadata TEXT,          -- JSON
                    
                    -- V2.0 Intelligence References
                    experience_id TEXT,     -- Link to v2_experiences
                    pattern_ids TEXT,       -- JSON array of pattern IDs
                    recommendation_ids TEXT, -- JSON array of recommendation IDs
                    
                    -- Audit fields
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                )
            """)
            
            # V2.0 Success evaluations (existing table enhancement)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_success_evaluations (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    model_used TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    
                    -- Dimension breakdown
                    correctness_score REAL,
                    efficiency_score REAL,  
                    cost_score REAL,
                    latency_score REAL,
                    
                    -- V2.0 metadata
                    success_level TEXT NOT NULL,
                    cost_usd REAL,
                    execution_time_ms INTEGER,
                    recommendation_followed BOOLEAN DEFAULT FALSE,
                    user_override BOOLEAN DEFAULT FALSE,
                    
                    timestamp TEXT NOT NULL,
                    metadata TEXT  -- JSON
                )
            """)
            
            # V2.0 Model decisions tracking
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_model_decisions (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    recommended_model TEXT NOT NULL,
                    actual_model_used TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    user_override BOOLEAN DEFAULT FALSE,
                    override_reason TEXT,
                    decision_timestamp TEXT NOT NULL
                )
            """)
            
            # V2.0 Alerts and notifications
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_system_alerts (
                    id TEXT PRIMARY KEY,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,  -- 'low', 'medium', 'high', 'critical'
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    source_component TEXT NOT NULL,
                    related_task_id TEXT,
                    metadata TEXT,  -- JSON
                    created_at TEXT NOT NULL,
                    resolved_at TEXT,
                    resolution_notes TEXT
                )
            """)
            
            conn.commit()
            logger.info("✅ Enhanced SimpleTracker database initialized")
    
    def _check_v2_components(self) -> Dict[str, bool]:
        """Check availability of V2.0 components"""
        components = {}
        
        # Experience Manager
        try:
            from shared.experience_manager import ExperienceManager
            components['experience_manager'] = True
        except ImportError:
            components['experience_manager'] = False
        
        # Knowledge Graph
        try:
            from shared.knowledge.neo4j_knowledge_graph import KnowledgeGraphManager
            components['knowledge_graph'] = True
        except ImportError:
            components['knowledge_graph'] = False
        
        # Pattern Mining
        try:
            from shared.learning.pattern_mining_engine import PatternMiningEngine
            components['pattern_mining'] = True
        except ImportError:
            components['pattern_mining'] = False
        
        # ML Pipeline
        try:
            from shared.learning.ml_training_pipeline import MLModelTrainingPipeline
            components['ml_pipeline'] = True
        except ImportError:
            components['ml_pipeline'] = False
        
        return components
    
    def track_event(self, 
                   task_id: str,
                   task_type: str, 
                   model_used: str,
                   success_score: float,
                   cost_usd: Optional[float] = None,
                   latency_ms: Optional[int] = None,
                   context: Optional[Union[str, Dict]] = None,
                   tracking_level: TrackingLevel = TrackingLevel.BASIC,
                   user_feedback: Optional[str] = None,
                   lessons_learned: Optional[List[str]] = None) -> str:
        """
        Enhanced track_event with V2.0 capabilities
        Maintains 100% backward compatibility with V1.0 calls
        """
        
        timestamp = datetime.now()
        context_str = json.dumps(context) if isinstance(context, dict) else context
        
        # V1.0 backward compatibility - always insert to original table
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO simple_tracker
                (task_id, task_type, model_used, success_score, cost_usd, latency_ms, timestamp, context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (task_id, task_type, model_used, success_score, cost_usd, latency_ms, 
                  timestamp.isoformat(), context_str))
            
            # V2.0 Enhanced tracking if requested
            if tracking_level != TrackingLevel.BASIC:
                
                # Calculate success level
                success_level = self._calculate_success_level(success_score)
                
                # Calculate dimension scores if V2.0 evaluation available
                dimension_scores = None
                if tracking_level == TrackingLevel.FULL and self.components_available.get('experience_manager'):
                    dimension_scores = self._calculate_dimension_scores(
                        success_score, cost_usd, latency_ms
                    )
                
                # Enhanced tracking entry
                conn.execute("""
                    INSERT OR REPLACE INTO v2_enhanced_tracker
                    (task_id, task_type, model_used, success_score, cost_usd, latency_ms, 
                     timestamp, context, tracking_level, success_level, dimension_scores,
                     user_feedback, lessons_learned, metadata, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    task_id, task_type, model_used, success_score, cost_usd, latency_ms,
                    timestamp.isoformat(), context_str, tracking_level.value, success_level.value,
                    json.dumps(dimension_scores) if dimension_scores else None,
                    user_feedback, json.dumps(lessons_learned) if lessons_learned else None,
                    json.dumps({"v2_enhanced": True}), timestamp.isoformat()
                ))
                
                # V2.0 Success evaluation entry
                if tracking_level == TrackingLevel.FULL:
                    self._create_v2_success_evaluation(
                        task_id, task_type, model_used, success_score, 
                        cost_usd, latency_ms, success_level, dimension_scores
                    )
            
            conn.commit()
        
        # Trigger V2.0 intelligence hooks if available
        if tracking_level != TrackingLevel.BASIC and any(self.components_available.values()):
            self._trigger_v2_intelligence_hooks(task_id, task_type, model_used, success_score)
        
        logger.info(f"✅ Event tracked: {task_id} ({tracking_level.value} level)")
        return task_id
    
    def _calculate_success_level(self, success_score: float) -> SuccessLevel:
        """Calculate success level from score"""
        if success_score >= 0.9:
            return SuccessLevel.EXCELLENT
        elif success_score >= 0.7:
            return SuccessLevel.GOOD
        elif success_score >= 0.5:
            return SuccessLevel.FAIR
        else:
            return SuccessLevel.POOR
    
    def _calculate_dimension_scores(self, success_score: float, cost_usd: Optional[float], latency_ms: Optional[int]) -> Dict[str, float]:
        """Calculate V2.0 multi-dimensional scores"""
        dimensions = {
            'correctness_score': success_score,  # Based on task success
            'efficiency_score': success_score * 0.9,  # Slightly lower than correctness
            'cost_score': 0.8,  # Default good cost score
            'latency_score': 0.8   # Default good latency score
        }
        
        # Adjust cost score based on actual cost
        if cost_usd is not None:
            if cost_usd < 0.01:
                dimensions['cost_score'] = 0.95
            elif cost_usd < 0.02:
                dimensions['cost_score'] = 0.8
            elif cost_usd < 0.05:
                dimensions['cost_score'] = 0.6
            else:
                dimensions['cost_score'] = 0.4
        
        # Adjust latency score
        if latency_ms is not None:
            if latency_ms < 1000:
                dimensions['latency_score'] = 0.95
            elif latency_ms < 3000:
                dimensions['latency_score'] = 0.8
            elif latency_ms < 5000:
                dimensions['latency_score'] = 0.6
            else:
                dimensions['latency_score'] = 0.4
        
        return dimensions
    
    def _create_v2_success_evaluation(self, task_id: str, task_type: str, model_used: str, 
                                     success_score: float, cost_usd: Optional[float], 
                                     latency_ms: Optional[int], success_level: SuccessLevel,
                                     dimension_scores: Optional[Dict[str, float]]):
        """Create V2.0 success evaluation entry"""
        
        overall_score = success_score
        if dimension_scores:
            # Weighted overall score: correctness(50%) + efficiency(20%) + cost(15%) + latency(15%)
            overall_score = (
                dimension_scores['correctness_score'] * 0.5 +
                dimension_scores['efficiency_score'] * 0.2 +
                dimension_scores['cost_score'] * 0.15 +
                dimension_scores['latency_score'] * 0.15
            )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO v2_success_evaluations
                (id, task_id, task_type, model_used, overall_score,
                 correctness_score, efficiency_score, cost_score, latency_score,
                 success_level, cost_usd, execution_time_ms, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                str(uuid.uuid4()), task_id, task_type, model_used, overall_score,
                dimension_scores.get('correctness_score') if dimension_scores else None,
                dimension_scores.get('efficiency_score') if dimension_scores else None,
                dimension_scores.get('cost_score') if dimension_scores else None,
                dimension_scores.get('latency_score') if dimension_scores else None,
                success_level.value, cost_usd, latency_ms, 
                datetime.now().isoformat(), json.dumps({"v2_auto_generated": True})
            ))
            conn.commit()
    
    def _trigger_v2_intelligence_hooks(self, task_id: str, task_type: str, model_used: str, success_score: float):
        """Trigger V2.0 intelligence components when available"""
        
        # Experience Manager hook
        if self.components_available.get('experience_manager'):
            try:
                from shared.experience_manager import ExperienceManager
                from shared.experience_manager import TaskOutcome
                
                outcome = TaskOutcome.SUCCESS if success_score >= 0.8 else \
                         TaskOutcome.PARTIAL_SUCCESS if success_score >= 0.5 else \
                         TaskOutcome.FAILURE
                
                exp_manager = ExperienceManager()
                exp_manager.record_experience(
                    task_id=task_id,
                    task_type=task_type,
                    context={"source": "enhanced_tracker"},
                    outcome=outcome,
                    success_score=success_score,
                    cost_usd=0.01,  # Default cost
                    latency_ms=1000,  # Default latency
                    model_used=model_used,
                    parameters={},
                    lessons_learned=[f"Task completed with {success_score:.3f} success score"]
                )
                logger.debug(f"Experience recorded for task {task_id}")
                
            except Exception as e:
                logger.warning(f"Experience Manager hook failed: {e}")
        
        # Pattern Mining hook - trigger pattern analysis for high-value tasks
        if self.components_available.get('pattern_mining') and success_score >= 0.8:
            try:
                from shared.learning.pattern_mining_engine import PatternMiningEngine
                
                # Trigger async pattern analysis (in production would be background task)
                engine = PatternMiningEngine()
                patterns = engine.analyze_patterns(days_back=7)
                logger.debug(f"Pattern analysis triggered for high-success task {task_id}")
                
            except Exception as e:
                logger.warning(f"Pattern Mining hook failed: {e}")
    
    def get_enhanced_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Get enhanced summary with V2.0 intelligence insights"""
        
        cutoff_date = (datetime.now() - timedelta(days=days_back)).isoformat()
        summary = {
            'period_days': days_back,
            'timestamp': datetime.now().isoformat(),
            'v1_compatibility': True,
            'v2_enhancements': True
        }
        
        with sqlite3.connect(self.db_path) as conn:
            # V1.0 compatible metrics
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_tasks,
                    AVG(success_score) as avg_success,
                    SUM(cost_usd) as total_cost,
                    AVG(latency_ms) as avg_latency,
                    COUNT(DISTINCT model_used) as unique_models,
                    COUNT(DISTINCT task_type) as unique_task_types
                FROM simple_tracker
                WHERE timestamp >= ?
            """, [cutoff_date])
            
            v1_metrics = cursor.fetchone()
            if v1_metrics:
                summary['v1_metrics'] = {
                    'total_tasks': v1_metrics[0],
                    'avg_success_rate': round((v1_metrics[1] or 0) * 100, 1),
                    'total_cost_usd': round(v1_metrics[2] or 0, 4),
                    'avg_latency_ms': round(v1_metrics[3] or 0, 0),
                    'unique_models': v1_metrics[4],
                    'unique_task_types': v1_metrics[5]
                }
            
            # V2.0 Enhanced metrics
            try:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as v2_evaluations,
                        AVG(overall_score) as v2_avg_score,
                        SUM(CASE WHEN success_level = 'excellent' THEN 1 ELSE 0 END) as excellent_count,
                        SUM(CASE WHEN success_level = 'good' THEN 1 ELSE 0 END) as good_count,
                        SUM(CASE WHEN success_level = 'fair' THEN 1 ELSE 0 END) as fair_count,
                        SUM(CASE WHEN success_level = 'poor' THEN 1 ELSE 0 END) as poor_count,
                        AVG(correctness_score) as avg_correctness,
                        AVG(efficiency_score) as avg_efficiency,
                        AVG(cost_score) as avg_cost_score,
                        AVG(latency_score) as avg_latency_score
                    FROM v2_success_evaluations
                    WHERE timestamp >= ?
                """, [cutoff_date])
                
                v2_metrics = cursor.fetchone()
                if v2_metrics and v2_metrics[0] > 0:
                    summary['v2_intelligence'] = {
                        'total_evaluations': v2_metrics[0],
                        'avg_overall_score': round((v2_metrics[1] or 0), 3),
                        'success_level_distribution': {
                            'excellent': v2_metrics[2],
                            'good': v2_metrics[3],
                            'fair': v2_metrics[4],
                            'poor': v2_metrics[5]
                        },
                        'dimension_averages': {
                            'correctness': round(v2_metrics[6] or 0, 3),
                            'efficiency': round(v2_metrics[7] or 0, 3),
                            'cost': round(v2_metrics[8] or 0, 3),
                            'latency': round(v2_metrics[9] or 0, 3)
                        }
                    }
                else:
                    summary['v2_intelligence'] = {'status': 'no_data'}
            
            except sqlite3.OperationalError:
                summary['v2_intelligence'] = {'status': 'tables_not_ready'}
            
            # System alerts
            try:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_alerts,
                        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical_alerts,
                        SUM(CASE WHEN resolved_at IS NULL THEN 1 ELSE 0 END) as unresolved_alerts
                    FROM v2_system_alerts
                    WHERE created_at >= ?
                """, [cutoff_date])
                
                alert_data = cursor.fetchone()
                if alert_data:
                    summary['system_alerts'] = {
                        'total_alerts': alert_data[0],
                        'critical_alerts': alert_data[1],
                        'unresolved_alerts': alert_data[2],
                        'alert_level': 'critical' if alert_data[1] > 0 else 'normal'
                    }
            
            except sqlite3.OperationalError:
                summary['system_alerts'] = {'status': 'not_available'}
        
        # Component integration status
        summary['v2_components'] = self.components_available
        available_components = sum(self.components_available.values())
        total_components = len(self.components_available)
        summary['v2_integration_level'] = f"{available_components}/{total_components} components available"
        
        return summary
    
    def create_system_alert(self, alert_type: str, severity: str, title: str, 
                          description: str, source_component: str,
                          related_task_id: Optional[str] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create system alert for monitoring and notifications"""
        
        alert_id = str(uuid.uuid4())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO v2_system_alerts
                (id, alert_type, severity, title, description, source_component,
                 related_task_id, metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id, alert_type, severity, title, description, source_component,
                related_task_id, json.dumps(metadata) if metadata else None,
                datetime.now().isoformat()
            ))
            conn.commit()
        
        logger.info(f"🚨 System alert created: {severity.upper()} - {title}")
        return alert_id
    
    def get_recent_tasks(self, limit: int = 20, enhanced_data: bool = False) -> List[Dict[str, Any]]:
        """Get recent tasks with optional V2.0 enhanced data"""
        
        with sqlite3.connect(self.db_path) as conn:
            if enhanced_data:
                # Try to get V2.0 enhanced data
                try:
                    cursor = conn.execute("""
                        SELECT 
                            e.task_id, e.task_type, e.model_used, e.success_score,
                            e.cost_usd, e.latency_ms, e.timestamp, e.context,
                            e.tracking_level, e.success_level, e.dimension_scores,
                            e.user_feedback, e.lessons_learned
                        FROM v2_enhanced_tracker e
                        ORDER BY e.timestamp DESC
                        LIMIT ?
                    """, [limit])
                    
                    enhanced_tasks = cursor.fetchall()
                    
                    if enhanced_tasks:
                        return [
                            {
                                'task_id': row[0],
                                'task_type': row[1], 
                                'model_used': row[2],
                                'success_score': row[3],
                                'cost_usd': row[4],
                                'latency_ms': row[5],
                                'timestamp': row[6],
                                'context': row[7],
                                'tracking_level': row[8],
                                'success_level': row[9],
                                'dimension_scores': json.loads(row[10]) if row[10] else None,
                                'user_feedback': row[11],
                                'lessons_learned': json.loads(row[12]) if row[12] else None,
                                'enhanced': True
                            }
                            for row in enhanced_tasks
                        ]
                
                except sqlite3.OperationalError:
                    pass  # Fall back to V1.0 data
            
            # V1.0 fallback or standard request
            cursor = conn.execute("""
                SELECT task_id, task_type, model_used, success_score, 
                       cost_usd, latency_ms, timestamp, context
                FROM simple_tracker
                ORDER BY timestamp DESC
                LIMIT ?
            """, [limit])
            
            return [
                {
                    'task_id': row[0],
                    'task_type': row[1],
                    'model_used': row[2], 
                    'success_score': row[3],
                    'cost_usd': row[4],
                    'latency_ms': row[5],
                    'timestamp': row[6],
                    'context': row[7],
                    'enhanced': False
                }
                for row in cursor.fetchall()
            ]
    
    def get_v2_system_health(self) -> Dict[str, Any]:
        """Get comprehensive V2.0 system health status"""
        
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_health': 'healthy',
            'components': self.components_available,
            'database': {},
            'alerts': {},
            'performance': {}
        }
        
        # Database health
        try:
            with sqlite3.connect(self.db_path) as conn:
                # V1.0 table health
                cursor = conn.execute("SELECT COUNT(*) FROM simple_tracker")
                v1_count = cursor.fetchone()[0]
                
                # V2.0 tables health
                v2_counts = {}
                v2_tables = ['v2_enhanced_tracker', 'v2_success_evaluations', 
                           'v2_model_decisions', 'v2_system_alerts']
                
                for table in v2_tables:
                    try:
                        cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                        v2_counts[table] = cursor.fetchone()[0]
                    except sqlite3.OperationalError:
                        v2_counts[table] = 'not_created'
                
                health_status['database'] = {
                    'v1_records': v1_count,
                    'v2_records': v2_counts,
                    'status': 'healthy' if v1_count > 0 else 'empty'
                }
        
        except Exception as e:
            health_status['database'] = {'status': 'error', 'message': str(e)}
            health_status['overall_health'] = 'degraded'
        
        # Alert summary
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN severity = 'critical' THEN 1 ELSE 0 END) as critical,
                        SUM(CASE WHEN resolved_at IS NULL THEN 1 ELSE 0 END) as unresolved
                    FROM v2_system_alerts
                    WHERE created_at >= ?
                """, [(datetime.now() - timedelta(days=1)).isoformat()])
                
                alert_summary = cursor.fetchone()
                if alert_summary:
                    health_status['alerts'] = {
                        'total_24h': alert_summary[0],
                        'critical_24h': alert_summary[1], 
                        'unresolved': alert_summary[2],
                        'alert_level': 'critical' if alert_summary[1] > 0 else 'normal'
                    }
                    
                    if alert_summary[1] > 0:
                        health_status['overall_health'] = 'critical'
                    elif alert_summary[2] > 5:
                        health_status['overall_health'] = 'degraded'
        
        except sqlite3.OperationalError:
            health_status['alerts'] = {'status': 'not_available'}
        
        return health_status

# CLI Integration Functions (V1.0 backward compatibility)
def track_event(type: str, task: str, priority: str = "medium", **kwargs) -> str:
    """V1.0 backward compatible track_event"""
    tracker = EnhancedSimpleTracker()
    
    task_id = kwargs.get('task_id', f"task_{int(time.time())}")
    success_score = kwargs.get('success_score', 0.8)
    model_used = kwargs.get('model_used', 'llama3.2-3b')
    
    return tracker.track_event(
        task_id=task_id,
        task_type=type,
        model_used=model_used,
        success_score=success_score,
        tracking_level=TrackingLevel.BASIC
    )

def track_event_v2(task_id: str, task_type: str, model_used: str, success_score: float,
                   cost_usd: Optional[float] = None, latency_ms: Optional[int] = None,
                   context: Optional[Dict] = None, user_feedback: Optional[str] = None) -> str:
    """V2.0 enhanced track_event with full intelligence integration"""
    tracker = EnhancedSimpleTracker()
    
    return tracker.track_event(
        task_id=task_id,
        task_type=task_type,
        model_used=model_used,
        success_score=success_score,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        context=context,
        tracking_level=TrackingLevel.FULL,
        user_feedback=user_feedback
    )

def get_v2_system_summary() -> Dict[str, Any]:
    """Get comprehensive V2.0 system summary"""
    tracker = EnhancedSimpleTracker()
    return tracker.get_enhanced_summary()

def create_performance_alert(title: str, description: str, severity: str = "medium") -> str:
    """Create performance alert"""
    tracker = EnhancedSimpleTracker()
    return tracker.create_system_alert(
        alert_type="performance",
        severity=severity,
        title=title,
        description=description,
        source_component="enhanced_tracker"
    )

if __name__ == "__main__":
    # Test Enhanced SimpleTracker
    print("📊 Testing Enhanced SimpleTracker V2.0...")
    
    tracker = EnhancedSimpleTracker()
    
    # Test V1.0 compatibility
    task_id_v1 = track_event("test", "V1 compatibility test")
    print(f"✅ V1.0 Compatible: {task_id_v1}")
    
    # Test V2.0 enhanced
    task_id_v2 = track_event_v2(
        task_id="test_v2_001",
        task_type="code_generation",
        model_used="llama3.2-3b", 
        success_score=0.92,
        cost_usd=0.015,
        latency_ms=1200,
        context={"complexity": "medium", "language": "python"},
        user_feedback="Excellent results"
    )
    print(f"✅ V2.0 Enhanced: {task_id_v2}")
    
    # Test system summary
    summary = get_v2_system_summary()
    print(f"📈 System Summary: {summary['v1_metrics']['total_tasks']} tasks tracked")
    
    # Test alert creation
    alert_id = create_performance_alert("Test Alert", "Performance test alert")
    print(f"🚨 Alert Created: {alert_id}")
    
    print("\n🎉 Enhanced SimpleTracker V2.0 - OPERATIONAL!")