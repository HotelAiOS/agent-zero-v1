#!/usr/bin/env python3
"""
Agent Zero V1 - Point 4: Predictive Resource Planning & Experience Management (FIXED)
Enhanced Intelligence V2.0 - Corrected dataclass structure
"""

import asyncio
import logging
import sqlite3
import json
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Try numpy import with fallback
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create numpy fallback
    class np:
        @staticmethod
        def sin(x): return 0.1  # Simple fallback

# Import existing Intelligence V2.0 components
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from intelligence_v2.interfaces import (
        Task, AgentProfile, TaskPriority, BusinessContext, 
        PredictiveOutcome, FeedbackItem, MonitoringSnapshot
    )
    from intelligence_v2.prioritization import DynamicTaskPrioritizer
    V2_INTEGRATION = True
except ImportError as e:
    logging.warning(f"V2.0 integration not available: {e}")
    V2_INTEGRATION = False

logger = logging.getLogger(__name__)

# === RESOURCE PLANNING ENUMS ===

class ResourceType(Enum):
    AGENT_HOURS = "agent_hours"
    COMPUTE_POWER = "compute_power" 
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK_BANDWIDTH = "network_bandwidth"
    AI_MODEL_CALLS = "ai_model_calls"

class PredictionAccuracy(Enum):
    HIGH = "high"        # >90% accuracy
    MEDIUM = "medium"    # 70-90% accuracy  
    LOW = "low"          # <70% accuracy

class ExperienceType(Enum):
    TASK_COMPLETION = "task_completion"
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE_METRIC = "performance_metric"
    FAILURE_ANALYSIS = "failure_analysis"
    SUCCESS_PATTERN = "success_pattern"

# === CORRECTED DATA STRUCTURES ===

@dataclass
class ResourcePrediction:
    """Prediction for future resource requirements"""
    # Required fields first
    resource_type: ResourceType
    time_horizon: timedelta
    predicted_usage: float
    confidence_score: float
    accuracy_level: PredictionAccuracy
    historical_baseline: float
    trend_factor: float
    seasonal_factor: float
    business_impact_factor: float
    model_version: str
    
    # Optional fields with defaults
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    
@dataclass 
class ExperienceRecord:
    """Captured experience from task execution - FIXED ORDER"""
    # Required fields first
    task_id: str
    experience_type: ExperienceType
    
    # Optional fields with defaults
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    success: bool = False
    quality_score: float = 0.0
    efficiency_score: float = 0.0
    cost_actual: Optional[float] = None
    resources_used: Dict[ResourceType, float] = field(default_factory=dict)
    business_contexts: List[BusinessContext] = field(default_factory=list)
    complexity_factors: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapacityRecommendation:
    """Recommendation for capacity adjustments"""
    # Required fields first
    resource_type: ResourceType
    current_capacity: float
    recommended_capacity: float
    confidence: float
    reasoning: str
    expected_benefit: str
    risk_assessment: str
    implementation_timeline: str
    cost_impact: float
    roi_estimate: float
    
    # Optional fields with defaults
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)

class PredictiveResourcePlanner:
    """
    Point 4: Predictive Resource Planning & Experience Management
    
    Integrates with Point 3 Dynamic Prioritization to provide:
    - Resource demand forecasting
    - Experience capture and analysis  
    - Capacity planning recommendations
    - Performance optimization insights
    """
    
    def __init__(self, db_path: str = "data/predictive_planner.db"):
        """Initialize with database for experience storage"""
        self.db_path = db_path
        self.experience_records: List[ExperienceRecord] = []
        self.resource_predictions: List[ResourcePrediction] = []
        self.capacity_recommendations: List[CapacityRecommendation] = []
        
        # Initialize database
        self._init_database()
        
        logger.info("PredictiveResourcePlanner initialized")
    
    def _init_database(self):
        """Initialize SQLite database for experience storage"""
        try:
            import os
            os.makedirs(Path(self.db_path).parent, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Experience records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS experience_records (
                    experience_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    experience_type TEXT NOT NULL,
                    agent_id TEXT,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration_seconds REAL,
                    success BOOLEAN NOT NULL,
                    quality_score REAL NOT NULL,
                    efficiency_score REAL NOT NULL,
                    cost_actual REAL,
                    resources_used TEXT,
                    business_contexts TEXT,
                    lessons_learned TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    async def capture_task_experience(self, task_id: str, agent_id: str, 
                                    outcome: Dict[str, Any]) -> Optional[ExperienceRecord]:
        """
        Capture experience from completed task execution
        """
        try:
            # Create experience record with required fields first
            experience = ExperienceRecord(
                task_id=task_id,
                experience_type=ExperienceType.TASK_COMPLETION
            )
            
            # Set optional fields
            experience.agent_id = agent_id
            experience.success = outcome.get('success', False)
            experience.quality_score = outcome.get('quality_score', 0.0)
            experience.efficiency_score = outcome.get('efficiency_score', 0.0)
            experience.cost_actual = outcome.get('cost_actual')
            experience.end_time = datetime.now()
            
            # Calculate duration if possible
            if 'start_time' in outcome:
                start = outcome['start_time']
                if isinstance(start, str):
                    start = datetime.fromisoformat(start)
                experience.start_time = start
                experience.duration_seconds = (experience.end_time - start).total_seconds()
            
            # Extract resource usage
            if 'resources_used' in outcome:
                for resource_name, usage in outcome['resources_used'].items():
                    try:
                        resource_type = ResourceType(resource_name)
                        experience.resources_used[resource_type] = usage
                    except ValueError:
                        logger.warning(f"Unknown resource type: {resource_name}")
            
            # Extract lessons learned
            if 'lessons_learned' in outcome:
                experience.lessons_learned = outcome['lessons_learned']
            
            # Store in database
            await self._store_experience_record(experience)
            
            # Add to memory
            self.experience_records.append(experience)
            
            logger.info(f"Captured experience for task {task_id} with agent {agent_id}")
            
            return experience
            
        except Exception as e:
            logger.error(f"Experience capture failed for task {task_id}: {e}")
            return None
    
    async def _store_experience_record(self, experience: ExperienceRecord):
        """Store experience record in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO experience_records 
                (experience_id, task_id, experience_type, agent_id, start_time, end_time,
                 duration_seconds, success, quality_score, efficiency_score, cost_actual,
                 resources_used, business_contexts, lessons_learned, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                experience.experience_id,
                experience.task_id,
                experience.experience_type.value,
                experience.agent_id,
                experience.start_time.isoformat(),
                experience.end_time.isoformat() if experience.end_time else None,
                experience.duration_seconds,
                experience.success,
                experience.quality_score,
                experience.efficiency_score,
                experience.cost_actual,
                json.dumps({rt.value: usage for rt, usage in experience.resources_used.items()}),
                json.dumps([bc.value for bc in experience.business_contexts]),
                json.dumps(experience.lessons_learned),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store experience record: {e}")
    
    async def predict_resource_requirements(self, task_count: int = 5, 
                                          time_horizon: timedelta = timedelta(hours=24)) -> List[ResourcePrediction]:
        """
        Predict future resource requirements based on task queue
        """
        predictions = []
        
        try:
            # Simple prediction model
            for resource_type in ResourceType:
                prediction = self._create_simple_prediction(
                    resource_type, task_count, time_horizon
                )
                if prediction:
                    predictions.append(prediction)
            
            self.resource_predictions = predictions
            logger.info(f"Generated {len(predictions)} resource predictions")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Resource prediction failed: {e}")
            return []
    
    def _create_simple_prediction(self, resource_type: ResourceType, 
                                task_count: int, time_horizon: timedelta) -> Optional[ResourcePrediction]:
        """Create a simple resource prediction"""
        try:
            # Base usage per task
            base_usage = {
                ResourceType.AGENT_HOURS: 2.0,
                ResourceType.AI_MODEL_CALLS: 10.0,
                ResourceType.COMPUTE_POWER: 1.0,
                ResourceType.MEMORY: 4.0,
                ResourceType.STORAGE: 10.0,
                ResourceType.NETWORK_BANDWIDTH: 5.0
            }
            
            # Calculate predicted usage
            predicted_usage = base_usage.get(resource_type, 1.0) * task_count
            
            # Add some variation
            trend_factor = 1.0 + (0.1 if task_count > 3 else 0.0)
            seasonal_factor = 1.0 + (0.05 if datetime.now().hour > 12 else 0.0)
            business_impact_factor = 1.2  # Assume some business impact
            
            final_usage = predicted_usage * trend_factor * seasonal_factor * business_impact_factor
            
            # Determine confidence and accuracy
            confidence_score = 0.8 if task_count > 0 else 0.5
            accuracy_level = PredictionAccuracy.MEDIUM
            
            return ResourcePrediction(
                resource_type=resource_type,
                time_horizon=time_horizon,
                predicted_usage=final_usage,
                confidence_score=confidence_score,
                accuracy_level=accuracy_level,
                historical_baseline=predicted_usage,
                trend_factor=trend_factor,
                seasonal_factor=seasonal_factor,
                business_impact_factor=business_impact_factor,
                model_version="v2.0.1"
            )
            
        except Exception as e:
            logger.error(f"Simple prediction failed for {resource_type}: {e}")
            return None
    
    async def generate_capacity_recommendations(self, current_capacity: Dict[str, float]) -> List[CapacityRecommendation]:
        """Generate capacity adjustment recommendations"""
        recommendations = []
        
        try:
            for prediction in self.resource_predictions:
                resource_name = prediction.resource_type.value
                current = current_capacity.get(resource_name, 1.0)
                predicted = prediction.predicted_usage
                
                # Check if adjustment needed
                utilization = predicted / current if current > 0 else 1.0
                
                if utilization > 0.9:  # Over 90% utilization
                    recommended = predicted * 1.2  # 20% buffer
                    reason = f"High utilization predicted ({utilization:.1%})"
                    benefit = "Prevent performance degradation"
                    risk = "Low risk - proactive scaling"
                    timeline = "Within 24 hours"
                    
                elif utilization < 0.3:  # Under 30% utilization
                    recommended = max(predicted * 1.1, current * 0.8)  # At least 10% buffer
                    reason = f"Low utilization predicted ({utilization:.1%})"
                    benefit = "Cost optimization opportunity"
                    risk = "Medium risk - ensure adequate capacity"
                    timeline = "Within 7 days"
                    
                else:
                    continue  # No adjustment needed
                
                recommendation = CapacityRecommendation(
                    resource_type=prediction.resource_type,
                    current_capacity=current,
                    recommended_capacity=recommended,
                    confidence=prediction.confidence_score,
                    reasoning=reason,
                    expected_benefit=benefit,
                    risk_assessment=risk,
                    implementation_timeline=timeline,
                    cost_impact=(recommended - current) * 10.0,
                    roi_estimate=max(0.1, abs(current - recommended) / current * 0.8)
                )
                
                recommendations.append(recommendation)
            
            self.capacity_recommendations = recommendations
            logger.info(f"Generated {len(recommendations)} capacity recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Capacity recommendation generation failed: {e}")
            return []
    
    def get_experience_insights(self) -> Dict[str, Any]:
        """Get insights from captured experience data"""
        try:
            if not self.experience_records:
                return {
                    'total_experiences': 0,
                    'success_rate': 0.0,
                    'average_quality': 0.0,
                    'lessons_learned_count': 0,
                    'insights': []
                }
            
            # Calculate metrics
            total_experiences = len(self.experience_records)
            successful_experiences = sum(1 for exp in self.experience_records if exp.success)
            success_rate = successful_experiences / total_experiences
            
            quality_scores = [exp.quality_score for exp in self.experience_records if exp.quality_score > 0]
            average_quality = statistics.mean(quality_scores) if quality_scores else 0.0
            
            lessons_learned_count = sum(len(exp.lessons_learned) for exp in self.experience_records)
            
            # Generate insights
            insights = []
            
            if success_rate > 0.9:
                insights.append("Excellent success rate - system performing very well")
            elif success_rate < 0.7:
                insights.append("Low success rate - investigate failure patterns")
            
            if average_quality > 0.8:
                insights.append("High quality outcomes - maintain current practices")
            elif average_quality < 0.6:
                insights.append("Quality concerns - review execution processes")
            
            return {
                'total_experiences': total_experiences,
                'success_rate': success_rate,
                'average_quality': average_quality,
                'lessons_learned_count': lessons_learned_count,
                'insights': insights
            }
            
        except Exception as e:
            logger.error(f"Experience insights generation failed: {e}")
            return {'error': str(e)}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive Point 4 system metrics"""
        return {
            'predictive_planner_status': 'operational',
            'experience_records': len(self.experience_records),
            'resource_predictions': len(self.resource_predictions),
            'capacity_recommendations': len(self.capacity_recommendations),
            'database_path': self.db_path,
            'v2_integration': V2_INTEGRATION,
            'numpy_available': NUMPY_AVAILABLE,
            'experience_insights': self.get_experience_insights()
        }

# === DEMO FUNCTION ===

async def demo_predictive_resource_planning():
    """Demonstrate Point 4 Predictive Resource Planning capabilities"""
    print("🔮 Point 4: Predictive Resource Planning & Experience Management Demo")
    print("=" * 80)
    print("📅 Agent Zero V2.0 Intelligence Layer - Point 4 of 6")
    print()
    
    # Initialize planner
    planner = PredictiveResourcePlanner()
    
    print("📋 Sample Task Queue Simulation:")
    print("  • Deploy ML Pipeline (high priority) - 8h")
    print("  • Security Audit (critical priority) - 4h") 
    print("  • Database Optimization (medium priority) - 6h")
    print("  • API Documentation (low priority) - 2h")
    print()
    
    # Generate resource predictions
    print("🔮 Generating Resource Predictions...")
    predictions = await planner.predict_resource_requirements(task_count=4, time_horizon=timedelta(hours=24))
    
    print(f"📊 Generated {len(predictions)} resource predictions:")
    for prediction in predictions:
        print(f"  • {prediction.resource_type.value}:")
        print(f"    - Predicted Usage: {prediction.predicted_usage:.2f}")
        print(f"    - Confidence: {prediction.confidence_score:.1%}")
        print(f"    - Accuracy Level: {prediction.accuracy_level.value}")
        print(f"    - Business Impact Factor: {prediction.business_impact_factor:.2f}")
    print()
    
    # Simulate task completion experience
    print("📝 Capturing Task Experience...")
    outcome = {
        'success': True,
        'quality_score': 0.85,
        'efficiency_score': 0.9,
        'cost_actual': 400.0,
        'resources_used': {
            'agent_hours': 2.1,
            'ai_model_calls': 15,
            'compute_power': 0.8
        },
        'lessons_learned': [
            'Automated testing saved significant time',
            'Better requirements gathering needed upfront'
        ],
        'start_time': datetime.now() - timedelta(hours=2)
    }
    
    experience = await planner.capture_task_experience("task_001", "agent_001", outcome)
    if experience:
        print(f"  ✅ Experience captured for task: task_001")
        print(f"     - Success: {experience.success}")
        print(f"     - Quality Score: {experience.quality_score}")
        print(f"     - Duration: {experience.duration_seconds:.1f}s")
        print(f"     - Lessons: {len(experience.lessons_learned)}")
    print()
    
    # Generate capacity recommendations
    print("💡 Generating Capacity Recommendations...")
    current_capacity = {
        'agent_hours': 40.0,
        'ai_model_calls': 1000.0,
        'compute_power': 4.0,
        'memory': 16.0,
        'storage': 500.0,
        'network_bandwidth': 100.0
    }
    
    recommendations = await planner.generate_capacity_recommendations(current_capacity)
    
    if recommendations:
        print(f"📋 Generated {len(recommendations)} capacity recommendations:")
        for rec in recommendations:
            print(f"  • {rec.resource_type.value}:")
            print(f"    - Current: {rec.current_capacity:.1f}")
            print(f"    - Recommended: {rec.recommended_capacity:.1f}")
            print(f"    - Confidence: {rec.confidence:.1%}")
            print(f"    - Reasoning: {rec.reasoning}")
            print(f"    - Timeline: {rec.implementation_timeline}")
    else:
        print("  ✅ No capacity adjustments needed - current levels adequate")
    print()
    
    # Show experience insights
    print("🧠 Experience Insights:")
    insights = planner.get_experience_insights()
    for key, value in insights.items():
        if key == 'insights':
            print(f"  • Key Insights:")
            for insight in value:
                print(f"    - {insight}")
        else:
            print(f"  • {key.replace('_', ' ').title()}: {value}")
    print()
    
    # Show system metrics
    print("📊 Point 4 System Metrics:")
    metrics = planner.get_system_metrics()
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"  • {key.replace('_', ' ').title()}:")
            for subkey, subvalue in value.items():
                print(f"    - {subkey.replace('_', ' ').title()}: {subvalue}")
        else:
            print(f"  • {key.replace('_', ' ').title()}: {value}")
    
    print()
    print("🎯 Point 4 Predictive Resource Planning Demo Completed!")
    print("   ✅ Resource demand forecasting operational")
    print("   ✅ Experience capture and analysis working") 
    print("   ✅ Capacity recommendations generated")
    print("   ✅ Ready for integration with Point 3 prioritization")
    print("   ✅ Database storage for historical analysis")

if __name__ == "__main__":
    asyncio.run(demo_predictive_resource_planning())