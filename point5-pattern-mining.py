#!/usr/bin/env python3
"""
🔍 Agent Zero V1 - Point 5: Pattern Mining & Prediction Engine
============================================================
Finalna warstwa inteligencji: Advanced ML dla pattern discovery
Logika: Collect → Mine → Predict → Optimize → Evolve
"""

import logging
import json
import time
import uuid
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import os
import math
import statistics

# FastAPI components
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("PatternMiningEngine")

# ================================
# PATTERN MINING CORE
# ================================

class PatternType(Enum):
    """Typy wzorców w systemie"""
    SUCCESS_PATTERN = "SUCCESS_PATTERN"           # Wzorzec sukcesu
    PERFORMANCE_PATTERN = "PERFORMANCE_PATTERN"   # Wzorzec wydajności
    COLLABORATION_PATTERN = "COLLABORATION_PATTERN"  # Wzorzec współpracy
    EFFICIENCY_PATTERN = "EFFICIENCY_PATTERN"     # Wzorzec efektywności
    FAILURE_PATTERN = "FAILURE_PATTERN"           # Wzorzec niepowodzeń
    OPTIMIZATION_PATTERN = "OPTIMIZATION_PATTERN" # Wzorzec optymalizacji

class PredictionType(Enum):
    """Typy predykcji"""
    SUCCESS_PROBABILITY = "SUCCESS_PROBABILITY"   # Prawdopodobieństwo sukcesu
    COMPLETION_TIME = "COMPLETION_TIME"           # Czas ukończenia
    RESOURCE_USAGE = "RESOURCE_USAGE"             # Zużycie zasobów
    QUALITY_SCORE = "QUALITY_SCORE"               # Oczekiwana jakość
    COST_ESTIMATE = "COST_ESTIMATE"               # Szacowany koszt
    RISK_ASSESSMENT = "RISK_ASSESSMENT"           # Ocena ryzyka

@dataclass
class Pattern:
    """Wzorzec odkryty przez system"""
    id: str
    pattern_type: PatternType
    name: str
    description: str
    
    # Pattern characteristics
    conditions: List[str] = field(default_factory=list)
    outcomes: List[str] = field(default_factory=list)
    confidence: float = 0.8
    support: float = 0.5  # Częstość występowania
    
    # Supporting data
    supporting_experiences: List[str] = field(default_factory=list)
    statistical_significance: float = 0.0
    
    # Performance metrics
    success_rate: float = 0.8
    average_quality: float = 0.8
    average_duration: float = 2.0
    average_cost: float = 100.0
    
    # Applicability
    applicable_contexts: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    
    # Metadata
    discovered_at: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None
    usage_count: int = 0

@dataclass
class Prediction:
    """Predykcja wygenerowana przez system"""
    id: str
    prediction_type: PredictionType
    target_scenario: str
    
    # Prediction results
    predicted_value: float
    confidence_interval: Tuple[float, float]
    confidence_score: float = 0.8
    
    # Supporting patterns
    supporting_patterns: List[str] = field(default_factory=list)
    similar_cases: List[str] = field(default_factory=list)
    
    # Validation
    actual_value: Optional[float] = None
    accuracy_score: Optional[float] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    validated_at: Optional[datetime] = None

@dataclass
class OptimizationSuggestion:
    """Sugestia optymalizacji oparta na wzorcach"""
    id: str
    title: str
    description: str
    category: str
    
    # Impact assessment
    expected_improvement: float = 0.2  # Oczekiwana poprawa (20%)
    implementation_effort: float = 0.5  # Wysiłek implementacji (0-1)
    risk_level: float = 0.3  # Poziom ryzyka (0-1)
    
    # Supporting evidence
    supporting_patterns: List[str] = field(default_factory=list)
    evidence_strength: float = 0.8
    
    # Implementation
    recommended_steps: List[str] = field(default_factory=list)
    success_metrics: List[str] = field(default_factory=list)
    
    created_at: datetime = field(default_factory=datetime.now)

# ================================
# PATTERN MINING ENGINE
# ================================

class PatternMiningEngine:
    """
    Advanced Pattern Mining & Prediction Engine
    Ostatnia warstwa AI intelligence w Agent Zero V1
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Storage
        self.patterns: Dict[str, Pattern] = {}
        self.predictions: Dict[str, Prediction] = {}
        self.optimizations: Dict[str, OptimizationSuggestion] = {}
        
        # External connections
        self.experience_service = "http://localhost:8007"
        self.unified_service = "http://localhost:8006"
        
        # Mining statistics
        self.mining_stats = {
            "patterns_discovered": 0,
            "predictions_made": 0,
            "optimizations_suggested": 0,
            "average_prediction_accuracy": 0.0,
            "pattern_validation_rate": 0.0
        }
        
        # Initialize database and load existing patterns
        self._init_database()
        self._load_existing_patterns()
        
        self.logger.info("🔍 Pattern Mining Engine initialized!")
    
    def _init_database(self):
        """Initialize pattern mining database"""
        
        self.db_path = "pattern_mining.db"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Patterns table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS patterns (
                        id TEXT PRIMARY KEY,
                        pattern_type TEXT NOT NULL,
                        name TEXT,
                        description TEXT,
                        
                        conditions TEXT,
                        outcomes TEXT,
                        confidence REAL,
                        support REAL,
                        
                        supporting_experiences TEXT,
                        statistical_significance REAL,
                        
                        success_rate REAL,
                        average_quality REAL,
                        average_duration REAL,
                        average_cost REAL,
                        
                        applicable_contexts TEXT,
                        recommended_actions TEXT,
                        
                        discovered_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_validated DATETIME,
                        usage_count INTEGER DEFAULT 0
                    )
                """)
                
                # Predictions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS predictions (
                        id TEXT PRIMARY KEY,
                        prediction_type TEXT NOT NULL,
                        target_scenario TEXT,
                        
                        predicted_value REAL,
                        confidence_lower REAL,
                        confidence_upper REAL,
                        confidence_score REAL,
                        
                        supporting_patterns TEXT,
                        similar_cases TEXT,
                        
                        actual_value REAL,
                        accuracy_score REAL,
                        
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        validated_at DATETIME
                    )
                """)
                
                # Optimizations table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS optimizations (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        description TEXT,
                        category TEXT,
                        
                        expected_improvement REAL,
                        implementation_effort REAL,
                        risk_level REAL,
                        
                        supporting_patterns TEXT,
                        evidence_strength REAL,
                        
                        recommended_steps TEXT,
                        success_metrics TEXT,
                        
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("✅ Pattern mining database initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
    
    def _load_existing_patterns(self):
        """Load existing patterns from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM patterns")
                for row in cursor.fetchall():
                    pattern = Pattern(
                        id=row[0], pattern_type=PatternType(row[1]),
                        name=row[2], description=row[3],
                        confidence=row[6], support=row[7],
                        success_rate=row[10], usage_count=row[19]
                    )
                    self.patterns[pattern.id] = pattern
                    
            self.logger.info(f"📊 Loaded {len(self.patterns)} existing patterns")
            
        except Exception as e:
            self.logger.error(f"Pattern loading error: {e}")
    
    def discover_patterns_from_experiences(self, experiences_data: List[Dict]) -> List[Pattern]:
        """Discover new patterns from experience data"""
        
        discovered_patterns = []
        
        # Group experiences by similar characteristics
        grouped_experiences = self._group_similar_experiences(experiences_data)
        
        for group_key, group_experiences in grouped_experiences.items():
            if len(group_experiences) >= 3:  # Minimum support for pattern
                
                # Analyze group for patterns
                pattern = self._analyze_experience_group(group_key, group_experiences)
                
                if pattern and pattern.confidence > 0.7:
                    # Store pattern
                    self.patterns[pattern.id] = pattern
                    self._store_pattern(pattern)
                    discovered_patterns.append(pattern)
                    
                    self.mining_stats["patterns_discovered"] += 1
        
        self.logger.info(f"🔍 Discovered {len(discovered_patterns)} new patterns")
        return discovered_patterns
    
    def _group_similar_experiences(self, experiences: List[Dict]) -> Dict[str, List[Dict]]:
        """Group similar experiences for pattern analysis"""
        
        groups = {}
        
        for exp in experiences:
            # Create grouping key based on characteristics
            agents = sorted(exp.get("agents_involved", []))
            systems = sorted(exp.get("systems_used", []))
            success = exp.get("success", False)
            quality_range = int(exp.get("quality_score", 0.5) * 10) / 10  # Round to 0.1
            
            group_key = f"agents:{'-'.join(agents[:2])}_systems:{'-'.join(systems[:3])}_success:{success}_quality:{quality_range}"
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(exp)
        
        return groups
    
    def _analyze_experience_group(self, group_key: str, experiences: List[Dict]) -> Optional[Pattern]:
        """Analyze a group of experiences to extract pattern"""
        
        if len(experiences) < 3:
            return None
        
        # Calculate group statistics
        success_count = sum(1 for exp in experiences if exp.get("success", False))
        success_rate = success_count / len(experiences)
        
        quality_scores = [exp.get("quality_score", 0.5) for exp in experiences]
        avg_quality = statistics.mean(quality_scores)
        
        durations = [exp.get("duration_seconds", 2.0) for exp in experiences]
        avg_duration = statistics.mean(durations)
        
        # Determine pattern type and characteristics
        if success_rate > 0.8 and avg_quality > 0.8:
            pattern_type = PatternType.SUCCESS_PATTERN
            name = f"High-Success Pattern: {group_key[:30]}"
            description = f"Pattern achieving {success_rate:.1%} success rate with {avg_quality:.1%} quality"
        elif avg_duration < 1.0 and success_rate > 0.7:
            pattern_type = PatternType.EFFICIENCY_PATTERN
            name = f"Fast-Execution Pattern: {group_key[:30]}"
            description = f"Pattern completing tasks in {avg_duration:.2f}s with {success_rate:.1%} success"
        elif len(set(exp.get("systems_used", []) for exp in experiences)) == 1:
            # All experiences use same systems
            pattern_type = PatternType.COLLABORATION_PATTERN
            name = f"System-Coordination Pattern: {group_key[:30]}"
            description = f"Effective coordination pattern for multi-system tasks"
        else:
            pattern_type = PatternType.PERFORMANCE_PATTERN
            name = f"Performance Pattern: {group_key[:30]}"
            description = f"General performance pattern with {success_rate:.1%} success rate"
        
        # Extract conditions and outcomes
        conditions = []
        outcomes = []
        
        # Analyze common conditions
        all_agents = set()
        all_systems = set()
        for exp in experiences:
            all_agents.update(exp.get("agents_involved", []))
            all_systems.update(exp.get("systems_used", []))
        
        if len(all_agents) <= 3:
            conditions.append(f"Agents: {', '.join(sorted(all_agents))}")
        if len(all_systems) <= 4:
            conditions.append(f"Systems: {', '.join(sorted(all_systems))}")
        
        conditions.append(f"Task complexity: moderate")
        
        # Define outcomes
        outcomes.append(f"Success rate: {success_rate:.1%}")
        outcomes.append(f"Quality score: {avg_quality:.1%}")
        outcomes.append(f"Average duration: {avg_duration:.2f}s")
        
        # Calculate pattern confidence and support
        confidence = min(0.95, success_rate + (avg_quality - 0.5))
        support = len(experiences) / 100.0  # Normalize to reasonable support value
        
        # Create pattern
        pattern = Pattern(
            id=str(uuid.uuid4()),
            pattern_type=pattern_type,
            name=name,
            description=description,
            conditions=conditions,
            outcomes=outcomes,
            confidence=confidence,
            support=support,
            supporting_experiences=[exp.get("id", str(i)) for i, exp in enumerate(experiences)],
            success_rate=success_rate,
            average_quality=avg_quality,
            average_duration=avg_duration,
            average_cost=100.0,  # Default cost
            applicable_contexts=["general", "multi_system", "ai_coordination"],
            recommended_actions=[
                "Apply this pattern for similar task characteristics",
                "Monitor execution for pattern validation",
                "Collect additional data points for refinement"
            ]
        )
        
        return pattern
    
    def predict_task_outcome(
        self,
        task_description: str,
        proposed_setup: Dict[str, Any]
    ) -> Dict[str, Prediction]:
        """Predict multiple outcomes for a proposed task setup"""
        
        predictions = {}
        
        # Find relevant patterns
        relevant_patterns = self._find_relevant_patterns(proposed_setup)
        
        if not relevant_patterns:
            # Default predictions when no patterns match
            predictions["success_probability"] = Prediction(
                id=str(uuid.uuid4()),
                prediction_type=PredictionType.SUCCESS_PROBABILITY,
                target_scenario=task_description,
                predicted_value=0.75,
                confidence_interval=(0.6, 0.9),
                confidence_score=0.5
            )
            predictions["completion_time"] = Prediction(
                id=str(uuid.uuid4()),
                prediction_type=PredictionType.COMPLETION_TIME,
                target_scenario=task_description,
                predicted_value=3.0,
                confidence_interval=(2.0, 4.0),
                confidence_score=0.5
            )
        else:
            # Pattern-based predictions
            
            # Success probability prediction
            success_rates = [p.success_rate for p in relevant_patterns]
            avg_success = statistics.mean(success_rates)
            success_std = statistics.stdev(success_rates) if len(success_rates) > 1 else 0.1
            
            predictions["success_probability"] = Prediction(
                id=str(uuid.uuid4()),
                prediction_type=PredictionType.SUCCESS_PROBABILITY,
                target_scenario=task_description,
                predicted_value=avg_success,
                confidence_interval=(max(0.0, avg_success - success_std), min(1.0, avg_success + success_std)),
                confidence_score=min(0.95, sum(p.confidence for p in relevant_patterns) / len(relevant_patterns)),
                supporting_patterns=[p.id for p in relevant_patterns]
            )
            
            # Completion time prediction
            durations = [p.average_duration for p in relevant_patterns]
            avg_duration = statistics.mean(durations)
            duration_std = statistics.stdev(durations) if len(durations) > 1 else 0.5
            
            predictions["completion_time"] = Prediction(
                id=str(uuid.uuid4()),
                prediction_type=PredictionType.COMPLETION_TIME,
                target_scenario=task_description,
                predicted_value=avg_duration,
                confidence_interval=(max(0.1, avg_duration - duration_std), avg_duration + duration_std),
                confidence_score=min(0.95, sum(p.confidence for p in relevant_patterns) / len(relevant_patterns)),
                supporting_patterns=[p.id for p in relevant_patterns]
            )
            
            # Quality score prediction
            qualities = [p.average_quality for p in relevant_patterns]
            avg_quality = statistics.mean(qualities)
            quality_std = statistics.stdev(qualities) if len(qualities) > 1 else 0.1
            
            predictions["quality_score"] = Prediction(
                id=str(uuid.uuid4()),
                prediction_type=PredictionType.QUALITY_SCORE,
                target_scenario=task_description,
                predicted_value=avg_quality,
                confidence_interval=(max(0.0, avg_quality - quality_std), min(1.0, avg_quality + quality_std)),
                confidence_score=min(0.95, sum(p.confidence for p in relevant_patterns) / len(relevant_patterns)),
                supporting_patterns=[p.id for p in relevant_patterns]
            )
        
        # Store predictions
        for pred_type, prediction in predictions.items():
            self.predictions[prediction.id] = prediction
            self._store_prediction(prediction)
        
        self.mining_stats["predictions_made"] += len(predictions)
        
        return predictions
    
    def _find_relevant_patterns(self, setup: Dict[str, Any]) -> List[Pattern]:
        """Find patterns relevant to proposed setup"""
        
        relevant_patterns = []
        
        proposed_agents = set(setup.get("agents_involved", []))
        proposed_systems = set(setup.get("systems_used", []))
        
        for pattern in self.patterns.values():
            # Calculate relevance score
            relevance_score = 0.0
            
            # Check agent overlap (from supporting experiences)
            if pattern.supporting_experiences:
                # Simplified relevance check
                if len(proposed_agents) > 0:
                    relevance_score += 0.3
                if len(proposed_systems) > 0:
                    relevance_score += 0.4
            
            # Pattern type relevance
            if pattern.pattern_type in [PatternType.SUCCESS_PATTERN, PatternType.PERFORMANCE_PATTERN]:
                relevance_score += 0.3
            
            # Confidence threshold
            if pattern.confidence > 0.7:
                relevance_score += 0.2
            
            if relevance_score > 0.5:
                relevant_patterns.append(pattern)
        
        # Sort by confidence and usage
        relevant_patterns.sort(key=lambda p: (p.confidence, p.usage_count), reverse=True)
        
        return relevant_patterns[:5]  # Top 5 most relevant patterns
    
    def generate_optimizations(self, context: Dict[str, Any]) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on patterns"""
        
        optimizations = []
        
        # Analyze current performance against patterns
        current_success_rate = context.get("success_rate", 0.8)
        current_quality = context.get("quality_score", 0.8)
        current_duration = context.get("average_duration", 2.0)
        
        # Find high-performing patterns for comparison
        high_performance_patterns = [
            p for p in self.patterns.values()
            if p.success_rate > current_success_rate + 0.1 and p.confidence > 0.8
        ]
        
        for pattern in high_performance_patterns[:3]:  # Top 3 optimization opportunities
            
            improvement = pattern.success_rate - current_success_rate
            
            optimization = OptimizationSuggestion(
                id=str(uuid.uuid4()),
                title=f"Apply {pattern.name} for {improvement:.1%} improvement",
                description=f"Pattern shows potential for {improvement:.1%} success rate improvement",
                category="performance_optimization",
                expected_improvement=improvement,
                implementation_effort=0.6,  # Moderate effort
                risk_level=0.3,  # Low-moderate risk
                supporting_patterns=[pattern.id],
                evidence_strength=pattern.confidence,
                recommended_steps=[
                    f"Analyze current setup against pattern conditions",
                    f"Implement pattern recommendations: {pattern.recommended_actions[0] if pattern.recommended_actions else 'Follow pattern guidelines'}",
                    f"Monitor performance improvements",
                    f"Validate results and refine approach"
                ],
                success_metrics=[
                    f"Success rate improvement: {improvement:.1%}",
                    f"Quality score target: {pattern.average_quality:.1%}",
                    f"Duration target: {pattern.average_duration:.2f}s"
                ]
            )
            
            optimizations.append(optimization)
        
        # General optimizations based on pattern analysis
        if current_duration > 2.0:
            fast_patterns = [p for p in self.patterns.values() 
                           if p.pattern_type == PatternType.EFFICIENCY_PATTERN and p.average_duration < 1.5]
            
            if fast_patterns:
                best_fast_pattern = max(fast_patterns, key=lambda p: p.confidence)
                
                optimization = OptimizationSuggestion(
                    id=str(uuid.uuid4()),
                    title="Implement Fast Execution Pattern",
                    description=f"Reduce execution time from {current_duration:.2f}s to {best_fast_pattern.average_duration:.2f}s",
                    category="efficiency_optimization", 
                    expected_improvement=(current_duration - best_fast_pattern.average_duration) / current_duration,
                    implementation_effort=0.5,
                    risk_level=0.2,
                    supporting_patterns=[best_fast_pattern.id],
                    evidence_strength=best_fast_pattern.confidence,
                    recommended_steps=[
                        "Analyze fast execution patterns",
                        "Optimize system coordination",
                        "Implement parallel processing where possible",
                        "Monitor performance improvements"
                    ],
                    success_metrics=[
                        f"Target execution time: {best_fast_pattern.average_duration:.2f}s",
                        f"Maintain success rate: >{current_success_rate:.1%}"
                    ]
                )
                
                optimizations.append(optimization)
        
        # Store optimizations
        for opt in optimizations:
            self.optimizations[opt.id] = opt
            self._store_optimization(opt)
        
        self.mining_stats["optimizations_suggested"] += len(optimizations)
        
        return optimizations
    
    def get_pattern_analytics(self) -> Dict[str, Any]:
        """Get comprehensive pattern analytics"""
        
        # Pattern type distribution
        pattern_types = {}
        for pattern_type in PatternType:
            count = len([p for p in self.patterns.values() if p.pattern_type == pattern_type])
            pattern_types[pattern_type.value] = count
        
        # Top performing patterns
        top_patterns = sorted(self.patterns.values(), 
                            key=lambda p: (p.success_rate, p.confidence), reverse=True)[:10]
        
        # Prediction accuracy (if we have validations)
        validated_predictions = [p for p in self.predictions.values() if p.actual_value is not None]
        avg_accuracy = 0.0
        if validated_predictions:
            accuracies = [p.accuracy_score for p in validated_predictions if p.accuracy_score is not None]
            avg_accuracy = statistics.mean(accuracies) if accuracies else 0.0
        
        # Recent discoveries
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_patterns = [p for p in self.patterns.values() if p.discovered_at > recent_cutoff]
        
        return {
            "mining_statistics": self.mining_stats,
            "pattern_distribution": pattern_types,
            "top_patterns": [
                {
                    "id": p.id,
                    "name": p.name,
                    "type": p.pattern_type.value,
                    "success_rate": p.success_rate,
                    "confidence": p.confidence,
                    "usage_count": p.usage_count
                }
                for p in top_patterns
            ],
            "prediction_accuracy": {
                "average_accuracy": avg_accuracy,
                "total_predictions": len(self.predictions),
                "validated_predictions": len(validated_predictions)
            },
            "recent_discoveries": {
                "last_7_days": len(recent_patterns),
                "new_pattern_types": list(set(p.pattern_type.value for p in recent_patterns))
            },
            "optimization_opportunities": len(self.optimizations)
        }
    
    # Storage methods
    def _store_pattern(self, pattern: Pattern):
        """Store pattern in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO patterns 
                    (id, pattern_type, name, description, conditions, outcomes, confidence, support,
                     supporting_experiences, statistical_significance, success_rate, average_quality,
                     average_duration, average_cost, applicable_contexts, recommended_actions, usage_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pattern.id, pattern.pattern_type.value, pattern.name, pattern.description,
                    json.dumps(pattern.conditions), json.dumps(pattern.outcomes),
                    pattern.confidence, pattern.support, json.dumps(pattern.supporting_experiences),
                    pattern.statistical_significance, pattern.success_rate, pattern.average_quality,
                    pattern.average_duration, pattern.average_cost,
                    json.dumps(pattern.applicable_contexts), json.dumps(pattern.recommended_actions),
                    pattern.usage_count
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Pattern storage failed: {e}")
    
    def _store_prediction(self, prediction: Prediction):
        """Store prediction in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO predictions 
                    (id, prediction_type, target_scenario, predicted_value, confidence_lower,
                     confidence_upper, confidence_score, supporting_patterns, similar_cases)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction.id, prediction.prediction_type.value, prediction.target_scenario,
                    prediction.predicted_value, prediction.confidence_interval[0],
                    prediction.confidence_interval[1], prediction.confidence_score,
                    json.dumps(prediction.supporting_patterns), json.dumps(prediction.similar_cases)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Prediction storage failed: {e}")
    
    def _store_optimization(self, optimization: OptimizationSuggestion):
        """Store optimization in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO optimizations 
                    (id, title, description, category, expected_improvement, implementation_effort,
                     risk_level, supporting_patterns, evidence_strength, recommended_steps, success_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    optimization.id, optimization.title, optimization.description, optimization.category,
                    optimization.expected_improvement, optimization.implementation_effort,
                    optimization.risk_level, json.dumps(optimization.supporting_patterns),
                    optimization.evidence_strength, json.dumps(optimization.recommended_steps),
                    json.dumps(optimization.success_metrics)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Optimization storage failed: {e}")

# ================================
# FASTAPI APPLICATION
# ================================

app = FastAPI(
    title="Agent Zero V1 - Point 5: Pattern Mining & Prediction Engine",
    description="Finalna warstwa AI intelligence - Advanced pattern discovery i prediction",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pattern mining engine
pattern_engine = PatternMiningEngine()

@app.get("/")
async def pattern_system_root():
    """Pattern Mining & Prediction System Information"""
    
    analytics = pattern_engine.get_pattern_analytics()
    
    return {
        "system": "Agent Zero V1 - Point 5: Pattern Mining & Prediction Engine",
        "version": "1.0.0",
        "status": "OPERATIONAL",
        "description": "Finalna warstwa AI intelligence - Advanced pattern discovery, prediction i optimization",
        "architecture_position": "Experience Data → Pattern Mining → Predictions → Optimizations → System Evolution",
        "capabilities": [
            "Advanced pattern discovery from experience data",
            "Multi-outcome prediction modeling",
            "Optimization suggestion generation",
            "Success pattern identification",
            "Performance trend analysis",
            "Risk assessment and mitigation"
        ],
        "current_statistics": analytics["mining_statistics"],
        "pattern_insights": {
            "total_patterns": len(pattern_engine.patterns),
            "pattern_types": len(set(p.pattern_type for p in pattern_engine.patterns.values())),
            "prediction_accuracy": analytics["prediction_accuracy"]["average_accuracy"],
            "recent_discoveries": analytics["recent_discoveries"]["last_7_days"]
        },
        "endpoints": {
            "discover_patterns": "POST /api/v1/patterns/discover",
            "predict_outcome": "POST /api/v1/patterns/predict", 
            "get_optimizations": "POST /api/v1/patterns/optimize",
            "pattern_analytics": "GET /api/v1/patterns/analytics",
            "pattern_library": "GET /api/v1/patterns/library"
        }
    }

@app.post("/api/v1/patterns/discover")
async def discover_patterns_endpoint(discovery_request: dict):
    """Discover patterns from experience data"""
    
    try:
        experiences_data = discovery_request.get("experiences", [])
        
        if not experiences_data:
            return {
                "status": "error",
                "message": "No experience data provided for pattern discovery"
            }
        
        discovered_patterns = pattern_engine.discover_patterns_from_experiences(experiences_data)
        
        return {
            "status": "success",
            "patterns_discovered": len(discovered_patterns),
            "new_patterns": [
                {
                    "id": p.id,
                    "name": p.name,
                    "type": p.pattern_type.value,
                    "confidence": p.confidence,
                    "success_rate": p.success_rate,
                    "description": p.description
                }
                for p in discovered_patterns
            ],
            "message": f"🔍 Discovered {len(discovered_patterns)} new patterns from {len(experiences_data)} experiences"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/v1/patterns/predict")
async def predict_outcomes_endpoint(prediction_request: dict):
    """Predict outcomes for proposed task setup"""
    
    try:
        task_description = prediction_request.get("task_description", "")
        proposed_setup = prediction_request.get("setup", {})
        
        predictions = pattern_engine.predict_task_outcome(task_description, proposed_setup)
        
        predictions_data = {}
        for pred_type, prediction in predictions.items():
            predictions_data[pred_type] = {
                "predicted_value": prediction.predicted_value,
                "confidence_interval": prediction.confidence_interval,
                "confidence_score": prediction.confidence_score,
                "supporting_patterns": len(prediction.supporting_patterns)
            }
        
        return {
            "status": "success",
            "predictions": predictions_data,
            "prediction_summary": {
                "task": task_description,
                "total_predictions": len(predictions),
                "average_confidence": sum(p.confidence_score for p in predictions.values()) / len(predictions)
            },
            "message": f"🔮 Generated {len(predictions)} predictions with pattern-based modeling"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/v1/patterns/optimize")
async def generate_optimizations_endpoint(optimization_request: dict):
    """Generate optimization suggestions based on patterns"""
    
    try:
        context = optimization_request.get("context", {})
        
        optimizations = pattern_engine.generate_optimizations(context)
        
        optimizations_data = []
        for opt in optimizations:
            optimizations_data.append({
                "id": opt.id,
                "title": opt.title,
                "description": opt.description,
                "category": opt.category,
                "expected_improvement": f"{opt.expected_improvement:.1%}",
                "implementation_effort": opt.implementation_effort,
                "risk_level": opt.risk_level,
                "evidence_strength": opt.evidence_strength,
                "recommended_steps": opt.recommended_steps,
                "success_metrics": opt.success_metrics
            })
        
        return {
            "status": "success",
            "optimizations": optimizations_data,
            "optimization_summary": {
                "total_suggestions": len(optimizations),
                "high_impact_count": len([o for o in optimizations if o.expected_improvement > 0.2]),
                "low_risk_count": len([o for o in optimizations if o.risk_level < 0.3])
            },
            "message": f"⚡ Generated {len(optimizations)} optimization suggestions"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/patterns/analytics")
async def get_pattern_analytics_endpoint():
    """Get comprehensive pattern analytics"""
    
    try:
        analytics = pattern_engine.get_pattern_analytics()
        
        return {
            "status": "success",
            "analytics": analytics,
            "message": "📊 Pattern analytics generated successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/patterns/library")
async def get_pattern_library_endpoint(pattern_type: str = None, limit: int = 20):
    """Get pattern library with filtering"""
    
    try:
        patterns = list(pattern_engine.patterns.values())
        
        # Filter by type if specified
        if pattern_type:
            patterns = [p for p in patterns if p.pattern_type.value == pattern_type]
        
        # Sort by performance and confidence
        patterns.sort(key=lambda p: (p.success_rate, p.confidence), reverse=True)
        patterns = patterns[:limit]
        
        patterns_data = []
        for pattern in patterns:
            patterns_data.append({
                "id": pattern.id,
                "name": pattern.name,
                "type": pattern.pattern_type.value,
                "description": pattern.description,
                "success_rate": pattern.success_rate,
                "confidence": pattern.confidence,
                "support": pattern.support,
                "average_quality": pattern.average_quality,
                "average_duration": pattern.average_duration,
                "usage_count": pattern.usage_count,
                "conditions": pattern.conditions,
                "recommended_actions": pattern.recommended_actions,
                "discovered_at": pattern.discovered_at.isoformat()
            })
        
        return {
            "status": "success",
            "patterns": patterns_data,
            "library_stats": {
                "total_patterns": len(pattern_engine.patterns),
                "filtered_count": len(patterns_data),
                "pattern_types": list(set(p.pattern_type.value for p in pattern_engine.patterns.values()))
            },
            "message": f"📚 Pattern library retrieved: {len(patterns_data)} patterns"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    logger.info("🔍 Starting Point 5: Pattern Mining & Prediction Engine...")
    logger.info("🧠 Finalna warstwa AI intelligence dla Agent Zero V1")
    logger.info("📊 Advanced pattern discovery, prediction i optimization ready")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8008,
        workers=1,
        log_level="info",
        reload=False
    )