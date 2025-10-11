#!/usr/bin/env python3
"""
🎓 Agent Zero V1 - Point 4: Experience Management System
======================================================
Następny etap architektury: System uczenia się i optymalizacji
Logika: Capture → Learn → Optimize → Predict → Improve
"""

import asyncio
import logging
import json
import time
import uuid
import sqlite3
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pickle
import os

# FastAPI components
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Configure enterprise logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experience_management.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ExperienceManagement")

# ================================
# EXPERIENCE MANAGEMENT CORE
# ================================

class ExperienceType(Enum):
    """Typy doświadczeń w systemie"""
    TASK_EXECUTION = "TASK_EXECUTION"           # Wykonanie zadania
    COLLABORATION_SESSION = "COLLABORATION_SESSION"  # Sesja współpracy
    AGENT_PERFORMANCE = "AGENT_PERFORMANCE"     # Performance agenta
    SYSTEM_OPTIMIZATION = "SYSTEM_OPTIMIZATION" # Optymalizacja systemu
    ERROR_RESOLUTION = "ERROR_RESOLUTION"       # Rozwiązywanie błędów
    USER_INTERACTION = "USER_INTERACTION"       # Interakcje z użytkownikiem

class LearningOutcome(Enum):
    """Wyniki uczenia się"""
    SUCCESSFUL_PATTERN = "SUCCESSFUL_PATTERN"   # Wzorzec sukcesu
    FAILURE_PATTERN = "FAILURE_PATTERN"         # Wzorzec niepowodzenia
    OPTIMIZATION_OPPORTUNITY = "OPTIMIZATION_OPPORTUNITY"  # Możliwość optymalizacji
    BEST_PRACTICE = "BEST_PRACTICE"             # Najlepsza praktyka
    RISK_FACTOR = "RISK_FACTOR"                 # Czynnik ryzyka
    IMPROVEMENT_SUGGESTION = "IMPROVEMENT_SUGGESTION"  # Sugestia poprawy

@dataclass
class Experience:
    """Pojedyncze doświadczenie w systemie"""
    id: str
    experience_type: ExperienceType
    timestamp: datetime
    
    # Context data
    input_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Execution data
    agents_involved: List[str] = field(default_factory=list)
    systems_used: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    # Outcome data
    success: bool = True
    quality_score: float = 0.8
    cost: float = 0.0
    user_satisfaction: float = 0.8
    
    # Results and artifacts
    output_data: Dict[str, Any] = field(default_factory=dict)
    artifacts_created: List[str] = field(default_factory=list)
    
    # Learning extracted
    patterns_observed: List[str] = field(default_factory=list)
    insights_gained: List[str] = field(default_factory=list)
    improvements_suggested: List[str] = field(default_factory=list)
    
    # Metadata
    similar_experiences: List[str] = field(default_factory=list)
    replication_difficulty: float = 0.5  # 0.0 = easy to replicate, 1.0 = very difficult

@dataclass
class LearningInsight:
    """Wgląd nauczony z doświadczeń"""
    id: str
    insight_type: LearningOutcome
    confidence: float
    
    # Content
    title: str
    description: str
    conditions: List[str] = field(default_factory=list)
    
    # Supporting evidence
    supporting_experiences: List[str] = field(default_factory=list)
    statistical_significance: float = 0.0
    
    # Applicability
    applicable_contexts: List[str] = field(default_factory=list)
    expected_impact: float = 0.0
    
    # Implementation
    implementation_complexity: float = 0.5
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    last_validated: Optional[datetime] = None

@dataclass  
class PredictionModel:
    """Model predykcyjny oparty na doświadczeniach"""
    id: str
    name: str
    model_type: str
    
    # Training data
    training_experiences: List[str] = field(default_factory=list)
    features_used: List[str] = field(default_factory=list)
    
    # Model performance
    accuracy: float = 0.8
    precision: float = 0.8
    recall: float = 0.8
    confidence_threshold: float = 0.7
    
    # Model artifacts
    model_file_path: Optional[str] = None
    feature_importance: Dict[str, float] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=datetime.now)
    last_trained: Optional[datetime] = None

# ================================
# EXPERIENCE MANAGEMENT ENGINE
# ================================

class ExperienceManagementEngine:
    """
    Advanced Experience Management System
    Captures, analyzes, and learns from every system interaction
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Storage
        self.experiences: Dict[str, Experience] = {}
        self.insights: Dict[str, LearningInsight] = {}
        self.prediction_models: Dict[str, PredictionModel] = {}
        
        # System connections
        self.system_endpoints = {
            "nlu": "http://localhost:9001",
            "agent_selection": "http://localhost:8002",
            "priority": "http://localhost:8003", 
            "collaboration": "http://localhost:8005",
            "unified": "http://localhost:8006"
        }
        
        # Initialize database and models
        self._init_database()
        self._load_existing_models()
        
        # Statistics
        self.stats = {
            "total_experiences": 0,
            "successful_experiences": 0,
            "insights_generated": 0,
            "predictions_made": 0,
            "accuracy_improvement": 0.0
        }
        
        self.logger.info("🎓 Experience Management Engine initialized!")
    
    def _init_database(self):
        """Initialize experience management database"""
        
        self.db_path = "experience_management.db"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Experiences table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS experiences (
                        id TEXT PRIMARY KEY,
                        experience_type TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        
                        input_data TEXT,
                        context TEXT,
                        agents_involved TEXT,
                        systems_used TEXT,
                        duration_seconds REAL,
                        
                        success BOOLEAN,
                        quality_score REAL,
                        cost REAL,
                        user_satisfaction REAL,
                        
                        output_data TEXT,
                        artifacts_created TEXT,
                        
                        patterns_observed TEXT,
                        insights_gained TEXT,
                        improvements_suggested TEXT,
                        
                        similar_experiences TEXT,
                        replication_difficulty REAL
                    )
                """)
                
                # Learning insights table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_insights (
                        id TEXT PRIMARY KEY,
                        insight_type TEXT NOT NULL,
                        confidence REAL,
                        
                        title TEXT,
                        description TEXT,
                        conditions TEXT,
                        
                        supporting_experiences TEXT,
                        statistical_significance REAL,
                        
                        applicable_contexts TEXT,
                        expected_impact REAL,
                        implementation_complexity REAL,
                        resource_requirements TEXT,
                        
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_validated DATETIME
                    )
                """)
                
                # Prediction models table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS prediction_models (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        model_type TEXT,
                        
                        training_experiences TEXT,
                        features_used TEXT,
                        
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        confidence_threshold REAL,
                        
                        model_file_path TEXT,
                        feature_importance TEXT,
                        
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        last_trained DATETIME
                    )
                """)
                
                # Performance tracking table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS performance_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        experience_id TEXT,
                        system_name TEXT,
                        metric_name TEXT,
                        metric_value REAL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("✅ Experience management database initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
    
    def _load_existing_models(self):
        """Load existing prediction models"""
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                os.makedirs(models_dir)
                
            # Load models from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM prediction_models")
                for row in cursor.fetchall():
                    model = PredictionModel(
                        id=row[0], name=row[1], model_type=row[2],
                        accuracy=row[5], model_file_path=row[9]
                    )
                    self.prediction_models[model.id] = model
                    
            self.logger.info(f"📊 Loaded {len(self.prediction_models)} existing models")
            
        except Exception as e:
            self.logger.error(f"Model loading error: {e}")
    
    async def capture_experience(
        self,
        experience_type: ExperienceType,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        context: Dict[str, Any] = None,
        agents_involved: List[str] = None,
        systems_used: List[str] = None,
        success: bool = True,
        quality_score: float = 0.8,
        duration_seconds: float = 0.0
    ) -> str:
        """Capture a new experience for learning"""
        
        experience_id = str(uuid.uuid4())
        
        experience = Experience(
            id=experience_id,
            experience_type=experience_type,
            timestamp=datetime.now(),
            input_data=input_data,
            context=context or {},
            output_data=output_data,
            agents_involved=agents_involved or [],
            systems_used=systems_used or [],
            success=success,
            quality_score=quality_score,
            duration_seconds=duration_seconds
        )
        
        # Analyze experience for patterns
        await self._analyze_experience_patterns(experience)
        
        # Find similar experiences
        experience.similar_experiences = await self._find_similar_experiences(experience)
        
        # Extract insights
        insights = await self._extract_insights_from_experience(experience)
        experience.insights_gained = [insight.title for insight in insights]
        
        # Store experience
        self.experiences[experience_id] = experience
        await self._store_experience(experience)
        
        # Update statistics
        self.stats["total_experiences"] += 1
        if success:
            self.stats["successful_experiences"] += 1
        
        # Trigger learning update
        await self._update_learning_models(experience)
        
        self.logger.info(f"📚 Experience captured: {experience_id} ({experience_type.value})")
        
        return experience_id
    
    async def _analyze_experience_patterns(self, experience: Experience):
        """Analyze patterns in the experience"""
        
        patterns = []
        
        # Analyze input patterns
        if experience.input_data:
            if "complexity" in str(experience.input_data).lower():
                patterns.append("complex_task_handling")
            if "urgent" in str(experience.input_data).lower():
                patterns.append("urgent_request_processing")
        
        # Analyze agent patterns
        if len(experience.agents_involved) > 1:
            patterns.append("multi_agent_collaboration")
        
        # Analyze system patterns
        if len(experience.systems_used) >= 3:
            patterns.append("multi_system_integration")
        
        # Analyze performance patterns
        if experience.quality_score > 0.9:
            patterns.append("high_quality_outcome")
        if experience.duration_seconds < 1.0:
            patterns.append("fast_execution")
        
        experience.patterns_observed = patterns
    
    async def _find_similar_experiences(self, experience: Experience) -> List[str]:
        """Find experiences similar to current one"""
        
        similar = []
        
        for existing_id, existing_exp in self.experiences.items():
            if existing_exp.experience_type == experience.experience_type:
                # Calculate similarity based on multiple factors
                similarity_score = 0.0
                
                # Agent similarity
                common_agents = set(existing_exp.agents_involved) & set(experience.agents_involved)
                if experience.agents_involved:
                    similarity_score += len(common_agents) / len(experience.agents_involved) * 0.3
                
                # System similarity
                common_systems = set(existing_exp.systems_used) & set(experience.systems_used)
                if experience.systems_used:
                    similarity_score += len(common_systems) / len(experience.systems_used) * 0.3
                
                # Quality similarity
                quality_diff = abs(existing_exp.quality_score - experience.quality_score)
                similarity_score += (1.0 - quality_diff) * 0.4
                
                # Threshold for similarity
                if similarity_score > 0.7:
                    similar.append(existing_id)
        
        return similar[:5]  # Return top 5 similar experiences
    
    async def _extract_insights_from_experience(self, experience: Experience) -> List[LearningInsight]:
        """Extract learning insights from experience"""
        
        insights = []
        
        # Success pattern insights
        if experience.success and experience.quality_score > 0.9:
            insight = LearningInsight(
                id=str(uuid.uuid4()),
                insight_type=LearningOutcome.SUCCESSFUL_PATTERN,
                confidence=0.8,
                title=f"High-quality {experience.experience_type.value} execution",
                description=f"Pattern for achieving {experience.quality_score:.1%} quality",
                conditions=[f"Agents: {experience.agents_involved}", f"Systems: {experience.systems_used}"],
                supporting_experiences=[experience.id],
                expected_impact=0.8
            )
            insights.append(insight)
        
        # Speed optimization insights
        if experience.duration_seconds < 1.0 and experience.success:
            insight = LearningInsight(
                id=str(uuid.uuid4()),
                insight_type=LearningOutcome.OPTIMIZATION_OPPORTUNITY,
                confidence=0.7,
                title="Fast execution pattern",
                description=f"Execution completed in {experience.duration_seconds:.2f}s",
                conditions=experience.patterns_observed,
                supporting_experiences=[experience.id],
                expected_impact=0.6
            )
            insights.append(insight)
        
        # Multi-system coordination insights
        if len(experience.systems_used) >= 3 and experience.success:
            insight = LearningInsight(
                id=str(uuid.uuid4()),
                insight_type=LearningOutcome.BEST_PRACTICE,
                confidence=0.9,
                title="Multi-system coordination best practice",
                description="Successful coordination across multiple systems",
                conditions=[f"Systems: {experience.systems_used}"],
                supporting_experiences=[experience.id],
                expected_impact=0.9
            )
            insights.append(insight)
        
        # Store insights
        for insight in insights:
            self.insights[insight.id] = insight
            await self._store_insight(insight)
        
        self.stats["insights_generated"] += len(insights)
        
        return insights
    
    async def predict_task_outcome(
        self,
        task_description: str,
        context: Dict[str, Any],
        agents_proposed: List[str],
        systems_to_use: List[str]
    ) -> Dict[str, Any]:
        """Predict outcome of a proposed task based on experience"""
        
        # Create prediction input
        prediction_input = {
            "task_description": task_description,
            "context": context,
            "agents_proposed": agents_proposed,
            "systems_to_use": systems_to_use,
            "agent_count": len(agents_proposed),
            "system_count": len(systems_to_use)
        }
        
        # Find similar past experiences
        similar_experiences = await self._find_similar_prediction_cases(prediction_input)
        
        # Calculate predictions based on similar experiences
        if similar_experiences:
            success_rates = [exp.quality_score for exp in similar_experiences if exp.success]
            avg_success_rate = sum(success_rates) / len(success_rates) if success_rates else 0.5
            
            durations = [exp.duration_seconds for exp in similar_experiences]
            avg_duration = sum(durations) / len(durations) if durations else 2.0
            
            quality_scores = [exp.quality_score for exp in similar_experiences]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0.8
        else:
            # Default predictions when no similar experiences
            avg_success_rate = 0.7
            avg_duration = 3.0
            avg_quality = 0.8
        
        # Generate recommendations based on insights
        recommendations = await self._generate_recommendations(prediction_input, similar_experiences)
        
        prediction_result = {
            "predicted_success_rate": avg_success_rate,
            "predicted_duration_seconds": avg_duration,
            "predicted_quality_score": avg_quality,
            "confidence": len(similar_experiences) / 10.0,  # More similar experiences = higher confidence
            "similar_cases_found": len(similar_experiences),
            "recommendations": recommendations,
            "risk_factors": await self._identify_risk_factors(prediction_input),
            "optimization_suggestions": await self._suggest_optimizations(prediction_input)
        }
        
        self.stats["predictions_made"] += 1
        
        return prediction_result
    
    async def _find_similar_prediction_cases(self, prediction_input: Dict) -> List[Experience]:
        """Find experiences similar to prediction input"""
        
        similar_cases = []
        
        for experience in self.experiences.values():
            similarity_score = 0.0
            
            # Agent similarity
            if prediction_input["agents_proposed"]:
                common_agents = set(experience.agents_involved) & set(prediction_input["agents_proposed"])
                similarity_score += len(common_agents) / len(prediction_input["agents_proposed"]) * 0.4
            
            # System similarity  
            if prediction_input["systems_to_use"]:
                common_systems = set(experience.systems_used) & set(prediction_input["systems_to_use"])
                similarity_score += len(common_systems) / len(prediction_input["systems_to_use"]) * 0.4
            
            # Count similarity
            agent_count_diff = abs(experience.agents_involved.__len__() - prediction_input["agent_count"])
            similarity_score += max(0, 1.0 - agent_count_diff / 5.0) * 0.2
            
            if similarity_score > 0.5:
                similar_cases.append(experience)
        
        # Sort by most recent and highest quality
        similar_cases.sort(key=lambda x: (x.timestamp, x.quality_score), reverse=True)
        
        return similar_cases[:10]
    
    async def _generate_recommendations(self, prediction_input: Dict, similar_experiences: List[Experience]) -> List[str]:
        """Generate recommendations based on experience"""
        
        recommendations = []
        
        if similar_experiences:
            # Agent recommendations
            successful_agents = set()
            for exp in similar_experiences:
                if exp.success and exp.quality_score > 0.8:
                    successful_agents.update(exp.agents_involved)
            
            if successful_agents:
                recommendations.append(f"Consider using high-performing agents: {list(successful_agents)[:3]}")
            
            # System recommendations
            successful_systems = set()
            for exp in similar_experiences:
                if exp.success:
                    successful_systems.update(exp.systems_used)
            
            if successful_systems:
                recommendations.append(f"Recommended systems integration: {list(successful_systems)}")
        
        # General recommendations from insights
        relevant_insights = [insight for insight in self.insights.values() 
                           if insight.insight_type == LearningOutcome.BEST_PRACTICE]
        
        for insight in relevant_insights[:2]:
            recommendations.append(f"Best practice: {insight.title}")
        
        return recommendations
    
    async def _identify_risk_factors(self, prediction_input: Dict) -> List[str]:
        """Identify potential risk factors"""
        
        risk_factors = []
        
        # Complex multi-system risk
        if prediction_input["system_count"] > 3:
            risk_factors.append("High complexity due to multi-system integration")
        
        # Resource intensive risk
        if prediction_input["agent_count"] > 3:
            risk_factors.append("Coordination overhead with multiple agents")
        
        # Check for known failure patterns
        failure_insights = [insight for insight in self.insights.values() 
                          if insight.insight_type == LearningOutcome.FAILURE_PATTERN]
        
        for insight in failure_insights[:2]:
            risk_factors.append(f"Risk pattern: {insight.title}")
        
        return risk_factors
    
    async def _suggest_optimizations(self, prediction_input: Dict) -> List[str]:
        """Suggest optimizations based on experience"""
        
        optimizations = []
        
        # Optimization insights
        optimization_insights = [insight for insight in self.insights.values() 
                               if insight.insight_type == LearningOutcome.OPTIMIZATION_OPPORTUNITY]
        
        for insight in optimization_insights[:3]:
            optimizations.append(f"Optimization: {insight.title}")
        
        # General optimizations
        if prediction_input["system_count"] > 2:
            optimizations.append("Consider parallel system execution for better performance")
        
        if prediction_input["agent_count"] == 1:
            optimizations.append("Consider adding a QA agent for quality assurance")
        
        return optimizations
    
    async def get_experience_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics of experiences"""
        
        # Calculate success rates by type
        type_stats = {}
        for exp_type in ExperienceType:
            type_experiences = [exp for exp in self.experiences.values() if exp.experience_type == exp_type]
            if type_experiences:
                success_rate = sum(1 for exp in type_experiences if exp.success) / len(type_experiences)
                avg_quality = sum(exp.quality_score for exp in type_experiences) / len(type_experiences)
                type_stats[exp_type.value] = {
                    "count": len(type_experiences),
                    "success_rate": success_rate,
                    "average_quality": avg_quality
                }
        
        # Top performing agents
        agent_performance = {}
        for experience in self.experiences.values():
            for agent in experience.agents_involved:
                if agent not in agent_performance:
                    agent_performance[agent] = {"successes": 0, "total": 0, "quality_sum": 0.0}
                
                agent_performance[agent]["total"] += 1
                agent_performance[agent]["quality_sum"] += experience.quality_score
                if experience.success:
                    agent_performance[agent]["successes"] += 1
        
        # Calculate agent success rates
        for agent_data in agent_performance.values():
            agent_data["success_rate"] = agent_data["successes"] / agent_data["total"]
            agent_data["avg_quality"] = agent_data["quality_sum"] / agent_data["total"]
        
        # Recent trends
        recent_experiences = [exp for exp in self.experiences.values() 
                            if exp.timestamp > datetime.now() - timedelta(days=7)]
        
        return {
            "overall_statistics": self.stats,
            "experience_type_breakdown": type_stats,
            "agent_performance": dict(sorted(agent_performance.items(), 
                                           key=lambda x: x[1]["success_rate"], reverse=True)[:10]),
            "recent_trends": {
                "last_7_days_count": len(recent_experiences),
                "recent_success_rate": sum(1 for exp in recent_experiences if exp.success) / len(recent_experiences) if recent_experiences else 0,
                "recent_avg_quality": sum(exp.quality_score for exp in recent_experiences) / len(recent_experiences) if recent_experiences else 0
            },
            "learning_insights": {
                "total_insights": len(self.insights),
                "successful_patterns": len([i for i in self.insights.values() if i.insight_type == LearningOutcome.SUCCESSFUL_PATTERN]),
                "optimization_opportunities": len([i for i in self.insights.values() if i.insight_type == LearningOutcome.OPTIMIZATION_OPPORTUNITY]),
                "best_practices": len([i for i in self.insights.values() if i.insight_type == LearningOutcome.BEST_PRACTICE])
            }
        }
    
    # Storage methods
    async def _store_experience(self, experience: Experience):
        """Store experience in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO experiences 
                    (id, experience_type, input_data, context, agents_involved, systems_used,
                     duration_seconds, success, quality_score, output_data, patterns_observed,
                     insights_gained, similar_experiences, replication_difficulty)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experience.id, experience.experience_type.value,
                    json.dumps(experience.input_data), json.dumps(experience.context),
                    json.dumps(experience.agents_involved), json.dumps(experience.systems_used),
                    experience.duration_seconds, experience.success, experience.quality_score,
                    json.dumps(experience.output_data), json.dumps(experience.patterns_observed),
                    json.dumps(experience.insights_gained), json.dumps(experience.similar_experiences),
                    experience.replication_difficulty
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Experience storage failed: {e}")
    
    async def _store_insight(self, insight: LearningInsight):
        """Store learning insight in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO learning_insights 
                    (id, insight_type, confidence, title, description, conditions,
                     supporting_experiences, statistical_significance, applicable_contexts,
                     expected_impact, implementation_complexity, resource_requirements)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.id, insight.insight_type.value, insight.confidence,
                    insight.title, insight.description, json.dumps(insight.conditions),
                    json.dumps(insight.supporting_experiences), insight.statistical_significance,
                    json.dumps(insight.applicable_contexts), insight.expected_impact,
                    insight.implementation_complexity, json.dumps(insight.resource_requirements)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Insight storage failed: {e}")
    
    async def _update_learning_models(self, experience: Experience):
        """Update learning models with new experience"""
        # Placeholder for ML model updates
        # In a full implementation, this would retrain prediction models
        pass

# ================================
# FASTAPI APPLICATION
# ================================

app = FastAPI(
    title="Agent Zero V1 - Point 4: Experience Management System",
    description="Advanced learning system that captures, analyzes and optimizes from experience",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize experience management engine
experience_engine = ExperienceManagementEngine()

@app.get("/")
async def experience_system_root():
    """Experience Management System Information"""
    
    analytics = await experience_engine.get_experience_analytics()
    
    return {
        "system": "Agent Zero V1 - Point 4: Experience Management System",
        "version": "1.0.0",
        "status": "OPERATIONAL",
        "description": "Advanced learning system that captures and learns from every experience",
        "architecture_position": "Captures data from all other systems → Learns patterns → Predicts outcomes → Optimizes performance",
        "capabilities": [
            "Experience capture from all Agent Zero systems",
            "Pattern recognition and insight generation", 
            "Predictive modeling for task outcomes",
            "Performance optimization recommendations",
            "Continuous learning and improvement",
            "Success pattern identification"
        ],
        "current_statistics": analytics["overall_statistics"],
        "learning_status": {
            "experiences_captured": analytics["overall_statistics"]["total_experiences"],
            "insights_generated": analytics["overall_statistics"]["insights_generated"],
            "predictions_made": analytics["overall_statistics"]["predictions_made"],
            "success_rate": analytics["recent_trends"]["recent_success_rate"] if analytics["recent_trends"]["last_7_days_count"] > 0 else 0
        },
        "endpoints": {
            "capture_experience": "POST /api/v1/experience/capture",
            "predict_outcome": "POST /api/v1/experience/predict",
            "get_analytics": "GET /api/v1/experience/analytics",
            "get_insights": "GET /api/v1/experience/insights",
            "similar_experiences": "GET /api/v1/experience/similar"
        }
    }

@app.post("/api/v1/experience/capture")
async def capture_experience_endpoint(experience_data: dict):
    """Capture a new experience for learning"""
    
    try:
        experience_id = await experience_engine.capture_experience(
            experience_type=ExperienceType(experience_data.get("experience_type", "TASK_EXECUTION")),
            input_data=experience_data.get("input_data", {}),
            output_data=experience_data.get("output_data", {}),
            context=experience_data.get("context", {}),
            agents_involved=experience_data.get("agents_involved", []),
            systems_used=experience_data.get("systems_used", []),
            success=experience_data.get("success", True),
            quality_score=experience_data.get("quality_score", 0.8),
            duration_seconds=experience_data.get("duration_seconds", 0.0)
        )
        
        return {
            "status": "success",
            "experience_id": experience_id,
            "message": "Experience captured and analyzed successfully",
            "insights_generated": len(experience_engine.experiences[experience_id].insights_gained),
            "similar_experiences_found": len(experience_engine.experiences[experience_id].similar_experiences)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/v1/experience/predict")
async def predict_task_outcome_endpoint(prediction_request: dict):
    """Predict outcome of a proposed task"""
    
    try:
        prediction = await experience_engine.predict_task_outcome(
            task_description=prediction_request.get("task_description", ""),
            context=prediction_request.get("context", {}),
            agents_proposed=prediction_request.get("agents_proposed", []),
            systems_to_use=prediction_request.get("systems_to_use", [])
        )
        
        return {
            "status": "success",
            "prediction": prediction,
            "message": f"Prediction based on {prediction['similar_cases_found']} similar experiences"
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e)
        }

@app.get("/api/v1/experience/analytics")
async def get_experience_analytics_endpoint():
    """Get comprehensive experience analytics"""
    
    try:
        analytics = await experience_engine.get_experience_analytics()
        
        return {
            "status": "success",
            "analytics": analytics,
            "message": "Experience analytics generated successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/experience/insights")
async def get_learning_insights_endpoint(insight_type: str = None, limit: int = 10):
    """Get learning insights"""
    
    try:
        insights = list(experience_engine.insights.values())
        
        # Filter by type if specified
        if insight_type:
            insights = [i for i in insights if i.insight_type.value == insight_type]
        
        # Sort by confidence and impact
        insights.sort(key=lambda x: (x.confidence, x.expected_impact), reverse=True)
        
        # Limit results
        insights = insights[:limit]
        
        insights_data = []
        for insight in insights:
            insights_data.append({
                "id": insight.id,
                "type": insight.insight_type.value,
                "title": insight.title,
                "description": insight.description,
                "confidence": insight.confidence,
                "expected_impact": insight.expected_impact,
                "supporting_experiences": len(insight.supporting_experiences),
                "created_at": insight.created_at.isoformat()
            })
        
        return {
            "status": "success",
            "insights": insights_data,
            "total_insights": len(experience_engine.insights)
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    logger.info("🎓 Starting Point 4: Experience Management System...")
    logger.info("📚 Advanced learning and optimization engine")
    logger.info("🧠 Capturing patterns and generating insights from all experiences")
    
    uvicorn.run(
        "point4_experience_management:app",
        host="0.0.0.0",
        port=8007,
        workers=1,
        log_level="info",
        reload=False
    )