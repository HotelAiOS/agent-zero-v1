#!/usr/bin/env python3
"""
🎓 Agent Zero V1 - Point 4 FIXED: Experience Management System
============================================================
Fixed import issues - Production ready experience learning system
Logika: Capture → Learn → Optimize → Predict → Improve
"""

import logging
import json
import time
import uuid
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import os

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

logger = logging.getLogger("ExperienceManagementFixed")

# ================================
# EXPERIENCE MANAGEMENT CORE (FIXED)
# ================================

class ExperienceType(Enum):
    TASK_EXECUTION = "TASK_EXECUTION"
    COLLABORATION_SESSION = "COLLABORATION_SESSION"
    AGENT_PERFORMANCE = "AGENT_PERFORMANCE"
    SYSTEM_OPTIMIZATION = "SYSTEM_OPTIMIZATION"
    ERROR_RESOLUTION = "ERROR_RESOLUTION"
    USER_INTERACTION = "USER_INTERACTION"

class LearningOutcome(Enum):
    SUCCESSFUL_PATTERN = "SUCCESSFUL_PATTERN"
    FAILURE_PATTERN = "FAILURE_PATTERN"
    OPTIMIZATION_OPPORTUNITY = "OPTIMIZATION_OPPORTUNITY"
    BEST_PRACTICE = "BEST_PRACTICE"
    RISK_FACTOR = "RISK_FACTOR"
    IMPROVEMENT_SUGGESTION = "IMPROVEMENT_SUGGESTION"

@dataclass
class Experience:
    """Simplified Experience model"""
    id: str
    experience_type: ExperienceType
    timestamp: datetime
    
    # Core data
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    success: bool = True
    quality_score: float = 0.8
    duration_seconds: float = 0.0
    
    # System involvement
    agents_involved: List[str] = field(default_factory=list)
    systems_used: List[str] = field(default_factory=list)
    
    # Learning data
    patterns_observed: List[str] = field(default_factory=list)
    insights_gained: List[str] = field(default_factory=list)
    similar_experiences: List[str] = field(default_factory=list)

@dataclass
class LearningInsight:
    """Simplified Learning Insight model"""
    id: str
    insight_type: LearningOutcome
    title: str
    description: str
    confidence: float = 0.8
    expected_impact: float = 0.5
    supporting_experiences: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

# ================================
# FIXED EXPERIENCE MANAGEMENT ENGINE
# ================================

class FixedExperienceEngine:
    """Fixed Experience Management Engine - no import issues"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Storage
        self.experiences: Dict[str, Experience] = {}
        self.insights: Dict[str, LearningInsight] = {}
        
        # Statistics
        self.stats = {
            "total_experiences": 0,
            "successful_experiences": 0,
            "insights_generated": 0,
            "predictions_made": 0,
            "average_quality": 0.8
        }
        
        # Initialize database
        self._init_database()
        self.logger.info("🎓 Fixed Experience Management Engine initialized!")
    
    def _init_database(self):
        """Initialize experience database"""
        self.db_path = "experience_management_fixed.db"
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Experiences table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS experiences_fixed (
                        id TEXT PRIMARY KEY,
                        experience_type TEXT NOT NULL,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        
                        input_data TEXT,
                        output_data TEXT,
                        context TEXT,
                        
                        success BOOLEAN,
                        quality_score REAL,
                        duration_seconds REAL,
                        
                        agents_involved TEXT,
                        systems_used TEXT,
                        
                        patterns_observed TEXT,
                        insights_gained TEXT,
                        similar_experiences TEXT
                    )
                """)
                
                # Learning insights table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS learning_insights_fixed (
                        id TEXT PRIMARY KEY,
                        insight_type TEXT NOT NULL,
                        title TEXT,
                        description TEXT,
                        confidence REAL,
                        expected_impact REAL,
                        supporting_experiences TEXT,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("✅ Experience management database initialized")
                
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
    
    def capture_experience(
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
        """Capture a new experience - FIXED version"""
        
        experience_id = str(uuid.uuid4())
        
        experience = Experience(
            id=experience_id,
            experience_type=experience_type,
            timestamp=datetime.now(),
            input_data=input_data,
            output_data=output_data,
            context=context or {},
            agents_involved=agents_involved or [],
            systems_used=systems_used or [],
            success=success,
            quality_score=quality_score,
            duration_seconds=duration_seconds
        )
        
        # Simple pattern analysis
        experience.patterns_observed = self._analyze_patterns(experience)
        
        # Find similar experiences
        experience.similar_experiences = self._find_similar(experience)
        
        # Generate insights
        insights = self._generate_insights(experience)
        experience.insights_gained = [insight.title for insight in insights]
        
        # Store experience
        self.experiences[experience_id] = experience
        self._store_experience(experience)
        
        # Update statistics
        self.stats["total_experiences"] += 1
        if success:
            self.stats["successful_experiences"] += 1
        
        # Recalculate average quality
        if self.experiences:
            self.stats["average_quality"] = sum(exp.quality_score for exp in self.experiences.values()) / len(self.experiences)
        
        self.logger.info(f"📚 Experience captured: {experience_id}")
        return experience_id
    
    def _analyze_patterns(self, experience: Experience) -> List[str]:
        """Simple pattern analysis"""
        patterns = []
        
        if experience.success and experience.quality_score > 0.9:
            patterns.append("high_quality_success")
        
        if len(experience.systems_used) > 2:
            patterns.append("multi_system_coordination")
        
        if len(experience.agents_involved) > 1:
            patterns.append("collaborative_execution")
        
        if experience.duration_seconds < 1.0 and experience.success:
            patterns.append("fast_execution")
        
        return patterns
    
    def _find_similar(self, experience: Experience) -> List[str]:
        """Find similar experiences"""
        similar = []
        
        for existing_id, existing_exp in self.experiences.items():
            if existing_exp.experience_type == experience.experience_type:
                # Simple similarity check
                common_agents = set(existing_exp.agents_involved) & set(experience.agents_involved)
                common_systems = set(existing_exp.systems_used) & set(experience.systems_used)
                
                if len(common_agents) > 0 or len(common_systems) > 1:
                    similar.append(existing_id)
        
        return similar[:5]
    
    def _generate_insights(self, experience: Experience) -> List[LearningInsight]:
        """Generate insights from experience"""
        insights = []
        
        # Success pattern insight
        if experience.success and experience.quality_score > 0.9:
            insight = LearningInsight(
                id=str(uuid.uuid4()),
                insight_type=LearningOutcome.SUCCESSFUL_PATTERN,
                title="High-Quality Success Pattern",
                description=f"Achieved {experience.quality_score:.1%} quality with systems: {experience.systems_used}",
                confidence=0.8,
                expected_impact=0.7,
                supporting_experiences=[experience.id]
            )
            insights.append(insight)
        
        # Multi-system coordination insight
        if len(experience.systems_used) > 2 and experience.success:
            insight = LearningInsight(
                id=str(uuid.uuid4()),
                insight_type=LearningOutcome.BEST_PRACTICE,
                title="Multi-System Coordination Success",
                description="Successfully coordinated multiple systems for optimal results",
                confidence=0.9,
                expected_impact=0.8,
                supporting_experiences=[experience.id]
            )
            insights.append(insight)
        
        # Fast execution insight
        if experience.duration_seconds < 1.0 and experience.success:
            insight = LearningInsight(
                id=str(uuid.uuid4()),
                insight_type=LearningOutcome.OPTIMIZATION_OPPORTUNITY,
                title="Fast Execution Pattern",
                description=f"Completed task in {experience.duration_seconds:.3f}s",
                confidence=0.7,
                expected_impact=0.6,
                supporting_experiences=[experience.id]
            )
            insights.append(insight)
        
        # Store insights
        for insight in insights:
            self.insights[insight.id] = insight
            self._store_insight(insight)
        
        self.stats["insights_generated"] += len(insights)
        
        return insights
    
    def predict_outcome(
        self,
        task_description: str,
        context: Dict[str, Any],
        agents_proposed: List[str],
        systems_to_use: List[str]
    ) -> Dict[str, Any]:
        """Predict task outcome based on experience"""
        
        # Find similar experiences
        similar_experiences = []
        for experience in self.experiences.values():
            # Simple similarity matching
            common_agents = set(experience.agents_involved) & set(agents_proposed)
            common_systems = set(experience.systems_used) & set(systems_to_use)
            
            if len(common_agents) > 0 or len(common_systems) > 1:
                similar_experiences.append(experience)
        
        # Calculate predictions
        if similar_experiences:
            success_rate = sum(1 for exp in similar_experiences if exp.success) / len(similar_experiences)
            avg_quality = sum(exp.quality_score for exp in similar_experiences) / len(similar_experiences)
            avg_duration = sum(exp.duration_seconds for exp in similar_experiences) / len(similar_experiences)
        else:
            # Default predictions
            success_rate = 0.8
            avg_quality = 0.8
            avg_duration = 2.0
        
        # Generate recommendations
        recommendations = []
        if similar_experiences:
            top_performers = [exp for exp in similar_experiences if exp.quality_score > 0.9]
            if top_performers:
                recommendations.append("Based on high-performing similar cases, maintain current approach")
        
        recommendations.extend([
            "Monitor execution closely for optimization opportunities",
            "Consider adding quality checkpoints for complex tasks"
        ])
        
        # Risk factors
        risk_factors = []
        if len(systems_to_use) > 3:
            risk_factors.append("High complexity due to multiple system coordination")
        if not similar_experiences:
            risk_factors.append("Limited historical data for this task pattern")
        
        self.stats["predictions_made"] += 1
        
        return {
            "predicted_success_rate": success_rate,
            "predicted_quality_score": avg_quality,
            "predicted_duration_seconds": avg_duration,
            "confidence": min(1.0, len(similar_experiences) / 5.0),
            "similar_cases_found": len(similar_experiences),
            "recommendations": recommendations,
            "risk_factors": risk_factors
        }
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get experience analytics"""
        
        # Experience type breakdown
        type_breakdown = {}
        for exp_type in ExperienceType:
            type_experiences = [exp for exp in self.experiences.values() if exp.experience_type == exp_type]
            if type_experiences:
                success_rate = sum(1 for exp in type_experiences if exp.success) / len(type_experiences)
                avg_quality = sum(exp.quality_score for exp in type_experiences) / len(type_experiences)
                type_breakdown[exp_type.value] = {
                    "count": len(type_experiences),
                    "success_rate": success_rate,
                    "average_quality": avg_quality
                }
        
        # Agent performance
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
            if agent_data["total"] > 0:
                agent_data["success_rate"] = agent_data["successes"] / agent_data["total"]
                agent_data["avg_quality"] = agent_data["quality_sum"] / agent_data["total"]
        
        # Recent trends (last 24 hours for demo)
        recent_cutoff = datetime.now() - timedelta(hours=24)
        recent_experiences = [exp for exp in self.experiences.values() if exp.timestamp > recent_cutoff]
        
        return {
            "overall_statistics": self.stats,
            "experience_type_breakdown": type_breakdown,
            "agent_performance": dict(sorted(agent_performance.items(), 
                                           key=lambda x: x[1].get("success_rate", 0), reverse=True)[:10]),
            "recent_trends": {
                "last_24h_count": len(recent_experiences),
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
    
    def _store_experience(self, experience: Experience):
        """Store experience in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO experiences_fixed 
                    (id, experience_type, input_data, output_data, context, success, quality_score,
                     duration_seconds, agents_involved, systems_used, patterns_observed, insights_gained, similar_experiences)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experience.id, experience.experience_type.value,
                    json.dumps(experience.input_data), json.dumps(experience.output_data),
                    json.dumps(experience.context), experience.success, experience.quality_score,
                    experience.duration_seconds, json.dumps(experience.agents_involved),
                    json.dumps(experience.systems_used), json.dumps(experience.patterns_observed),
                    json.dumps(experience.insights_gained), json.dumps(experience.similar_experiences)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Experience storage failed: {e}")
    
    def _store_insight(self, insight: LearningInsight):
        """Store learning insight in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO learning_insights_fixed 
                    (id, insight_type, title, description, confidence, expected_impact, supporting_experiences)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    insight.id, insight.insight_type.value, insight.title, insight.description,
                    insight.confidence, insight.expected_impact, json.dumps(insight.supporting_experiences)
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Insight storage failed: {e}")

# ================================
# FIXED FASTAPI APPLICATION
# ================================

app = FastAPI(
    title="Agent Zero V1 - Point 4 FIXED: Experience Management",
    description="Fixed Experience Management System - Learning and optimization engine",
    version="1.0.1"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fixed experience engine
fixed_experience_engine = FixedExperienceEngine()

@app.get("/")
async def experience_system_root():
    """Fixed Experience Management System Information"""
    
    analytics = fixed_experience_engine.get_analytics()
    
    return {
        "system": "Agent Zero V1 - Point 4 FIXED: Experience Management System",
        "version": "1.0.1",
        "status": "OPERATIONAL",
        "description": "Fixed advanced learning system that captures and learns from every experience",
        "fixes_applied": [
            "Removed complex dependencies (numpy, pickle)",
            "Fixed import issues",
            "Simplified ML components", 
            "Streamlined database operations"
        ],
        "capabilities": [
            "Experience capture from all Agent Zero systems",
            "Pattern recognition and insight generation",
            "Predictive modeling for task outcomes", 
            "Performance optimization recommendations",
            "Success pattern identification"
        ],
        "current_statistics": analytics["overall_statistics"],
        "learning_status": {
            "experiences_captured": analytics["overall_statistics"]["total_experiences"],
            "insights_generated": analytics["overall_statistics"]["insights_generated"],
            "predictions_made": analytics["overall_statistics"]["predictions_made"],
            "success_rate": analytics["recent_trends"]["recent_success_rate"]
        },
        "endpoints": {
            "capture_experience": "POST /api/v1/experience/capture",
            "predict_outcome": "POST /api/v1/experience/predict",
            "get_analytics": "GET /api/v1/experience/analytics",
            "get_insights": "GET /api/v1/experience/insights"
        }
    }

@app.post("/api/v1/experience/capture")
async def capture_experience_endpoint(experience_data: dict):
    """Capture a new experience for learning"""
    
    try:
        experience_id = fixed_experience_engine.capture_experience(
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
        
        experience = fixed_experience_engine.experiences[experience_id]
        
        return {
            "status": "success",
            "experience_id": experience_id,
            "message": "✅ Experience captured and analyzed successfully!",
            "insights_generated": len(experience.insights_gained),
            "similar_experiences_found": len(experience.similar_experiences),
            "patterns_observed": experience.patterns_observed
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
        prediction = fixed_experience_engine.predict_outcome(
            task_description=prediction_request.get("task_description", ""),
            context=prediction_request.get("context", {}),
            agents_proposed=prediction_request.get("agents_proposed", []),
            systems_to_use=prediction_request.get("systems_to_use", [])
        )
        
        return {
            "status": "success", 
            "prediction": prediction,
            "message": f"✅ Prediction based on {prediction['similar_cases_found']} similar experiences"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/experience/analytics")
async def get_analytics_endpoint():
    """Get comprehensive experience analytics"""
    
    try:
        analytics = fixed_experience_engine.get_analytics()
        
        return {
            "status": "success",
            "analytics": analytics,
            "message": "📊 Experience analytics generated successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/experience/insights")
async def get_insights_endpoint(insight_type: str = None, limit: int = 10):
    """Get learning insights"""
    
    try:
        insights = list(fixed_experience_engine.insights.values())
        
        # Filter by type if specified
        if insight_type:
            insights = [i for i in insights if i.insight_type.value == insight_type]
        
        # Sort by confidence and impact
        insights.sort(key=lambda x: (x.confidence, x.expected_impact), reverse=True)
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
            "total_insights": len(fixed_experience_engine.insights),
            "message": "🧠 Learning insights retrieved successfully"
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    logger.info("🎓 Starting FIXED Point 4: Experience Management System...")
    logger.info("📚 Advanced learning engine - all import issues resolved")
    logger.info("🧠 Ready to capture patterns and generate insights")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8007,
        workers=1,
        log_level="info",
        reload=False
    )