#!/usr/bin/env python3
"""
Agent Zero V1 - Experience Management System
V2.0 Intelligence Layer - Week 44 Implementation

🎯 Week 44 Critical Task: Experience Management System (8 SP)
Zadanie: Agregacja doświadczeń, normalizacja przebiegów, rekomendacje
Rezultat: Baza doświadczeń, API rekomendacji
Impact: System uczy się z każdego zadania i dostarcza actionable insights

Author: Developer A (Backend Architect)
Date: 10 października 2025
Linear Issue: A0-44 (Week 44 Implementation)
"""

import sqlite3
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TaskOutcome(Enum):
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"
    INCOMPLETE = "incomplete"

class RecommendationType(Enum):
    MODEL_SELECTION = "model_selection"
    PARAMETER_OPTIMIZATION = "parameter_optimization"
    COST_OPTIMIZATION = "cost_optimization"
    QUALITY_IMPROVEMENT = "quality_improvement"
    WORKFLOW_ENHANCEMENT = "workflow_enhancement"

@dataclass
class Experience:
    """Single experience record for Agent Zero tasks"""
    id: str
    task_id: str
    task_type: str
    context: Dict[str, Any]
    outcome: TaskOutcome
    success_score: float
    cost_usd: float
    latency_ms: int
    model_used: str
    parameters: Dict[str, Any]
    user_feedback: Optional[str]
    lessons_learned: List[str]
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class Pattern:
    """Pattern discovered from experiences"""
    id: str
    pattern_type: str
    description: str
    conditions: Dict[str, Any]
    success_rate: float
    sample_size: int
    confidence: float
    recommendations: List[str]
    created_at: datetime

@dataclass
class Recommendation:
    """Generated recommendation based on experience analysis"""
    id: str
    type: RecommendationType
    title: str
    description: str
    impact_score: float
    confidence: float
    supporting_evidence: List[str]
    suggested_action: str
    estimated_improvement: Dict[str, float]
    priority: str  # "high", "medium", "low"
    created_at: datetime

class ExperienceManager:
    """
    Core Experience Management System for Agent Zero V2.0
    
    Responsibilities:
    - Capture and store task experiences
    - Normalize and analyze execution patterns
    - Extract lessons learned and patterns
    - Generate actionable recommendations
    - Provide experience-based insights for optimization
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize V2.0 experience management tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Experiences table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_experiences (
                    id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    context TEXT NOT NULL,  -- JSON
                    outcome TEXT NOT NULL,
                    success_score REAL NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    model_used TEXT NOT NULL,
                    parameters TEXT NOT NULL,  -- JSON
                    user_feedback TEXT,
                    lessons_learned TEXT NOT NULL,  -- JSON array
                    timestamp TEXT NOT NULL,
                    metadata TEXT NOT NULL  -- JSON
                )
            """)
            
            # Patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    conditions TEXT NOT NULL,  -- JSON
                    success_rate REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    recommendations TEXT NOT NULL,  -- JSON array
                    created_at TEXT NOT NULL
                )
            """)
            
            # Recommendations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_recommendations (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    impact_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    supporting_evidence TEXT NOT NULL,  -- JSON array
                    suggested_action TEXT NOT NULL,
                    estimated_improvement TEXT NOT NULL,  -- JSON
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    applied BOOLEAN DEFAULT FALSE,
                    applied_at TEXT,
                    results TEXT  -- JSON, results after applying
                )
            """)
            
            conn.commit()
    
    def record_experience(self, 
                         task_id: str,
                         task_type: str, 
                         context: Dict[str, Any],
                         outcome: TaskOutcome,
                         success_score: float,
                         cost_usd: float,
                         latency_ms: int,
                         model_used: str,
                         parameters: Dict[str, Any],
                         user_feedback: Optional[str] = None,
                         lessons_learned: List[str] = None,
                         metadata: Dict[str, Any] = None) -> Experience:
        """Record a new experience from task execution"""
        
        experience = Experience(
            id=str(uuid.uuid4()),
            task_id=task_id,
            task_type=task_type,
            context=context or {},
            outcome=outcome,
            success_score=success_score,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
            model_used=model_used,
            parameters=parameters or {},
            user_feedback=user_feedback,
            lessons_learned=lessons_learned or [],
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO v2_experiences 
                (id, task_id, task_type, context, outcome, success_score, 
                 cost_usd, latency_ms, model_used, parameters, user_feedback,
                 lessons_learned, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experience.id, experience.task_id, experience.task_type,
                json.dumps(experience.context), experience.outcome.value,
                experience.success_score, experience.cost_usd, experience.latency_ms,
                experience.model_used, json.dumps(experience.parameters),
                experience.user_feedback, json.dumps(experience.lessons_learned),
                experience.timestamp.isoformat(), json.dumps(experience.metadata)
            ))
            conn.commit()
        
        logger.info(f"Recorded experience {experience.id} for task {task_id}")
        return experience
    
    def get_experiences(self, 
                       task_type: Optional[str] = None,
                       model_used: Optional[str] = None,
                       days_back: int = 30,
                       min_success_score: float = 0.0) -> List[Experience]:
        """Retrieve experiences with filtering"""
        
        query = """
            SELECT * FROM v2_experiences 
            WHERE timestamp >= ? AND success_score >= ?
        """
        params = [
            (datetime.now() - timedelta(days=days_back)).isoformat(),
            min_success_score
        ]
        
        if task_type:
            query += " AND task_type = ?"
            params.append(task_type)
        
        if model_used:
            query += " AND model_used = ?"
            params.append(model_used)
        
        query += " ORDER BY timestamp DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        experiences = []
        for row in rows:
            exp = Experience(
                id=row[0], task_id=row[1], task_type=row[2],
                context=json.loads(row[3]), 
                outcome=TaskOutcome(row[4]),
                success_score=row[5], cost_usd=row[6], latency_ms=row[7],
                model_used=row[8], parameters=json.loads(row[9]),
                user_feedback=row[10], lessons_learned=json.loads(row[11]),
                timestamp=datetime.fromisoformat(row[12]),
                metadata=json.loads(row[13])
            )
            experiences.append(exp)
        
        return experiences
    
    def analyze_patterns(self, days_back: int = 30) -> List[Pattern]:
        """Analyze patterns from recent experiences"""
        experiences = self.get_experiences(days_back=days_back)
        patterns = []
        
        # Pattern 1: Model Performance by Task Type
        model_task_performance = {}
        for exp in experiences:
            key = (exp.model_used, exp.task_type)
            if key not in model_task_performance:
                model_task_performance[key] = []
            model_task_performance[key].append(exp.success_score)
        
        for (model, task_type), scores in model_task_performance.items():
            if len(scores) >= 3:  # Minimum sample size
                avg_score = sum(scores) / len(scores)
                if avg_score > 0.8:  # High performance pattern
                    pattern = Pattern(
                        id=str(uuid.uuid4()),
                        pattern_type="model_task_performance",
                        description=f"Model {model} performs well on {task_type} tasks",
                        conditions={"model": model, "task_type": task_type},
                        success_rate=avg_score,
                        sample_size=len(scores),
                        confidence=min(len(scores) / 10.0, 1.0),
                        recommendations=[f"Prefer {model} for {task_type} tasks"],
                        created_at=datetime.now()
                    )
                    patterns.append(pattern)
        
        # Pattern 2: Cost Efficiency Patterns
        cost_efficiency = {}
        for exp in experiences:
            if exp.success_score > 0.7:  # Only successful tasks
                efficiency = exp.success_score / max(exp.cost_usd, 0.001)
                key = exp.model_used
                if key not in cost_efficiency:
                    cost_efficiency[key] = []
                cost_efficiency[key].append(efficiency)
        
        for model, efficiencies in cost_efficiency.items():
            if len(efficiencies) >= 3:
                avg_efficiency = sum(efficiencies) / len(efficiencies)
                best_efficiency = max(cost_efficiency.values(), 
                                    key=lambda x: sum(x)/len(x) if x else 0)
                
                if efficiencies == best_efficiency:
                    pattern = Pattern(
                        id=str(uuid.uuid4()),
                        pattern_type="cost_efficiency",
                        description=f"Model {model} shows highest cost efficiency",
                        conditions={"model": model, "metric": "cost_efficiency"},
                        success_rate=avg_efficiency,
                        sample_size=len(efficiencies),
                        confidence=min(len(efficiencies) / 10.0, 1.0),
                        recommendations=[f"Use {model} for cost-sensitive tasks"],
                        created_at=datetime.now()
                    )
                    patterns.append(pattern)
        
        # Store patterns
        for pattern in patterns:
            self._store_pattern(pattern)
        
        return patterns
    
    def _store_pattern(self, pattern: Pattern):
        """Store discovered pattern in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO v2_patterns
                (id, pattern_type, description, conditions, success_rate,
                 sample_size, confidence, recommendations, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.id, pattern.pattern_type, pattern.description,
                json.dumps(pattern.conditions), pattern.success_rate,
                pattern.sample_size, pattern.confidence,
                json.dumps(pattern.recommendations), pattern.created_at.isoformat()
            ))
            conn.commit()
    
    def generate_recommendations(self, context: Dict[str, Any] = None) -> List[Recommendation]:
        """Generate actionable recommendations based on experience analysis"""
        recommendations = []
        patterns = self.get_stored_patterns()
        experiences = self.get_experiences(days_back=7)
        
        # Recommendation 1: Model Selection Optimization
        if experiences:
            model_performance = {}
            for exp in experiences:
                if exp.model_used not in model_performance:
                    model_performance[exp.model_used] = []
                model_performance[exp.model_used].append(exp.success_score)
            
            best_model = max(model_performance.items(), 
                           key=lambda x: sum(x[1])/len(x[1]))
            
            if len(best_model[1]) >= 3:
                rec = Recommendation(
                    id=str(uuid.uuid4()),
                    type=RecommendationType.MODEL_SELECTION,
                    title=f"Optimize model selection: prefer {best_model[0]}",
                    description=f"Recent data shows {best_model[0]} has {sum(best_model[1])/len(best_model[1]):.3f} average success rate",
                    impact_score=0.8,
                    confidence=min(len(best_model[1]) / 10.0, 1.0),
                    supporting_evidence=[f"{len(best_model[1])} recent tasks analyzed"],
                    suggested_action=f"Configure {best_model[0]} as default for similar tasks",
                    estimated_improvement={
                        "success_rate": 0.15,
                        "cost_reduction": 0.10
                    },
                    priority="high",
                    created_at=datetime.now()
                )
                recommendations.append(rec)
        
        # Recommendation 2: Cost Optimization
        high_cost_experiences = [exp for exp in experiences if exp.cost_usd > 0.02]
        if len(high_cost_experiences) > 0:
            avg_high_cost = sum(exp.cost_usd for exp in high_cost_experiences) / len(high_cost_experiences)
            potential_savings = avg_high_cost * len(high_cost_experiences) * 0.3
            
            rec = Recommendation(
                id=str(uuid.uuid4()),
                type=RecommendationType.COST_OPTIMIZATION,
                title="Implement cost optimization strategies",
                description=f"Detected {len(high_cost_experiences)} high-cost tasks with potential savings of ${potential_savings:.4f}",
                impact_score=0.7,
                confidence=0.8,
                supporting_evidence=[f"{len(high_cost_experiences)} high-cost tasks identified"],
                suggested_action="Use local models for routine tasks, batch similar requests",
                estimated_improvement={
                    "cost_reduction": 0.30,
                    "efficiency_gain": 0.20
                },
                priority="medium",
                created_at=datetime.now()
            )
            recommendations.append(rec)
        
        # Store recommendations
        for rec in recommendations:
            self._store_recommendation(rec)
        
        return recommendations
    
    def _store_recommendation(self, rec: Recommendation):
        """Store recommendation in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO v2_recommendations
                (id, type, title, description, impact_score, confidence,
                 supporting_evidence, suggested_action, estimated_improvement,
                 priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rec.id, rec.type.value, rec.title, rec.description,
                rec.impact_score, rec.confidence,
                json.dumps(rec.supporting_evidence), rec.suggested_action,
                json.dumps(rec.estimated_improvement), rec.priority,
                rec.created_at.isoformat()
            ))
            conn.commit()
    
    def get_stored_patterns(self, pattern_type: Optional[str] = None) -> List[Pattern]:
        """Retrieve stored patterns"""
        query = "SELECT * FROM v2_patterns"
        params = []
        
        if pattern_type:
            query += " WHERE pattern_type = ?"
            params.append(pattern_type)
        
        query += " ORDER BY created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        patterns = []
        for row in rows:
            pattern = Pattern(
                id=row[0], pattern_type=row[1], description=row[2],
                conditions=json.loads(row[3]), success_rate=row[4],
                sample_size=row[5], confidence=row[6],
                recommendations=json.loads(row[7]),
                created_at=datetime.fromisoformat(row[8])
            )
            patterns.append(pattern)
        
        return patterns
    
    def get_recommendations(self, 
                          recommendation_type: Optional[RecommendationType] = None,
                          priority: Optional[str] = None,
                          applied: bool = False) -> List[Recommendation]:
        """Retrieve recommendations with filtering"""
        query = "SELECT * FROM v2_recommendations WHERE applied = ?"
        params = [applied]
        
        if recommendation_type:
            query += " AND type = ?"
            params.append(recommendation_type.value)
        
        if priority:
            query += " AND priority = ?"
            params.append(priority)
        
        query += " ORDER BY impact_score DESC, created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        recommendations = []
        for row in rows:
            rec = Recommendation(
                id=row[0], type=RecommendationType(row[1]), title=row[2],
                description=row[3], impact_score=row[4], confidence=row[5],
                supporting_evidence=json.loads(row[6]), suggested_action=row[7],
                estimated_improvement=json.loads(row[8]), priority=row[9],
                created_at=datetime.fromisoformat(row[10])
            )
            recommendations.append(rec)
        
        return recommendations
    
    def get_experience_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """Generate summary of recent experiences"""
        experiences = self.get_experiences(days_back=days_back)
        
        if not experiences:
            return {
                "total_experiences": 0,
                "avg_success_score": 0.0,
                "total_cost": 0.0,
                "avg_latency": 0.0,
                "most_used_model": None,
                "best_performing_task_type": None
            }
        
        total_cost = sum(exp.cost_usd for exp in experiences)
        avg_success = sum(exp.success_score for exp in experiences) / len(experiences)
        avg_latency = sum(exp.latency_ms for exp in experiences) / len(experiences)
        
        # Most used model
        model_counts = {}
        for exp in experiences:
            model_counts[exp.model_used] = model_counts.get(exp.model_used, 0) + 1
        most_used_model = max(model_counts.items(), key=lambda x: x[1])[0] if model_counts else None
        
        # Best performing task type
        task_performance = {}
        for exp in experiences:
            if exp.task_type not in task_performance:
                task_performance[exp.task_type] = []
            task_performance[exp.task_type].append(exp.success_score)
        
        best_task_type = None
        if task_performance:
            best_task_type = max(task_performance.items(), 
                               key=lambda x: sum(x[1])/len(x[1]))[0]
        
        return {
            "total_experiences": len(experiences),
            "avg_success_score": avg_success,
            "total_cost": total_cost,
            "avg_latency": avg_latency,
            "most_used_model": most_used_model,
            "best_performing_task_type": best_task_type,
            "outcome_distribution": {
                outcome.value: len([exp for exp in experiences if exp.outcome == outcome])
                for outcome in TaskOutcome
            }
        }

# CLI Integration Functions
def record_task_experience(task_id: str, task_type: str, success_score: float, 
                          cost_usd: float, latency_ms: int, model_used: str) -> str:
    """CLI function to record experience"""
    manager = ExperienceManager()
    
    outcome = TaskOutcome.SUCCESS if success_score >= 0.8 else \
              TaskOutcome.PARTIAL_SUCCESS if success_score >= 0.5 else \
              TaskOutcome.FAILURE
    
    exp = manager.record_experience(
        task_id=task_id,
        task_type=task_type,
        context={"source": "cli"},
        outcome=outcome,
        success_score=success_score,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
        model_used=model_used,
        parameters={},
        lessons_learned=[f"Task completed with {success_score:.3f} success score"]
    )
    
    return exp.id

def get_experience_based_recommendations() -> Dict[str, Any]:
    """CLI function to get recommendations"""
    manager = ExperienceManager()
    recommendations = manager.generate_recommendations()
    
    return {
        "total_recommendations": len(recommendations),
        "high_priority": len([r for r in recommendations if r.priority == "high"]),
        "recommendations": [
            {
                "title": r.title,
                "type": r.type.value,
                "priority": r.priority,
                "impact_score": r.impact_score,
                "suggested_action": r.suggested_action
            }
            for r in recommendations[:5]  # Top 5
        ]
    }

def analyze_experience_patterns() -> Dict[str, Any]:
    """CLI function to analyze patterns"""
    manager = ExperienceManager()
    patterns = manager.analyze_patterns()
    
    return {
        "patterns_discovered": len(patterns),
        "patterns": [
            {
                "type": p.pattern_type,
                "description": p.description,
                "success_rate": p.success_rate,
                "confidence": p.confidence,
                "sample_size": p.sample_size
            }
            for p in patterns
        ]
    }

if __name__ == "__main__":
    # Test Experience Management System
    manager = ExperienceManager()
    
    # Record sample experience
    exp_id = record_task_experience(
        task_id="test_001",
        task_type="code_generation", 
        success_score=0.85,
        cost_usd=0.015,
        latency_ms=1200,
        model_used="llama3.2-3b"
    )
    
    print(f"✅ Experience recorded: {exp_id}")
    
    # Analyze patterns
    patterns = analyze_experience_patterns()
    print(f"📊 Patterns discovered: {patterns['patterns_discovered']}")
    
    # Get recommendations
    recommendations = get_experience_based_recommendations()
    print(f"💡 Recommendations generated: {recommendations['total_recommendations']}")
    
    # Get summary
    summary = manager.get_experience_summary()
    print(f"📈 Experience summary: {summary['total_experiences']} experiences analyzed")
    
    print("\n🎉 Experience Management System - OPERATIONAL!")