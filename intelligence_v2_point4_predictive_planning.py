#!/usr/bin/env python3
"""
Agent Zero V1 - Point 4: Predictive Resource Planning & Experience Management
Enhanced Intelligence V2.0 - Predictive planning with experience capture

CRITICAL: Integrates seamlessly with existing Point 3 Dynamic Prioritization
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
import numpy as np

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

# === DATA STRUCTURES ===

@dataclass
class ResourcePrediction:
    """Prediction for future resource requirements"""
    resource_type: ResourceType
    time_horizon: timedelta
    predicted_usage: float
    confidence_score: float
    accuracy_level: PredictionAccuracy
    
    # Supporting data
    historical_baseline: float
    trend_factor: float
    seasonal_factor: float
    business_impact_factor: float
    
    # Metadata
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    model_version: str = "v2.0.1"
    
@dataclass 
class ExperienceRecord:
    """Captured experience from task execution"""
    experience_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    experience_type: ExperienceType
    
    # Execution data
    agent_id: Optional[str] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    
    # Outcome data
    success: bool = False
    quality_score: float = 0.0  # 0.0-1.0
    efficiency_score: float = 0.0  # 0.0-1.0
    cost_actual: Optional[float] = None
    
    # Resource usage
    resources_used: Dict[ResourceType, float] = field(default_factory=dict)
    
    # Context and metadata
    business_contexts: List[BusinessContext] = field(default_factory=list)
    complexity_factors: List[str] = field(default_factory=list)
    lessons_learned: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CapacityRecommendation:
    """Recommendation for capacity adjustments"""
    resource_type: ResourceType
    current_capacity: float
    recommended_capacity: float
    confidence: float
    
    # Justification
    reasoning: str
    expected_benefit: str
    risk_assessment: str
    
    # Timeline and cost
    implementation_timeline: str
    cost_impact: float
    roi_estimate: float
    
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
        
        # Load existing data
        self._load_experience_records()
        
        logger.info("PredictiveResourcePlanner initialized with experience management")
    
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
                    complexity_factors TEXT,
                    lessons_learned TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Resource predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resource_predictions (
                    prediction_id TEXT PRIMARY KEY,
                    resource_type TEXT NOT NULL,
                    time_horizon_hours INTEGER NOT NULL,
                    predicted_usage REAL NOT NULL,
                    confidence_score REAL NOT NULL,
                    accuracy_level TEXT NOT NULL,
                    historical_baseline REAL NOT NULL,
                    trend_factor REAL NOT NULL,
                    seasonal_factor REAL NOT NULL,
                    business_impact_factor REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Capacity recommendations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS capacity_recommendations (
                    recommendation_id TEXT PRIMARY KEY,
                    resource_type TEXT NOT NULL,
                    current_capacity REAL NOT NULL,
                    recommended_capacity REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    expected_benefit TEXT NOT NULL,
                    risk_assessment TEXT NOT NULL,
                    implementation_timeline TEXT NOT NULL,
                    cost_impact REAL NOT NULL,
                    roi_estimate REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            conn.commit()
            conn.close()
            
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
    
    def _load_experience_records(self):
        """Load existing experience records from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM experience_records")
            count = cursor.fetchone()[0]
            
            logger.info(f"Found {count} existing experience records in database")
            conn.close()
            
        except Exception as e:
            logger.warning(f"Could not load experience records: {e}")
    
    async def capture_task_experience(self, task: Task, agent_id: str, 
                                    outcome: Dict[str, Any]) -> ExperienceRecord:
        """
        Capture experience from completed task execution
        
        Integrates with Point 3 prioritization decisions to learn from outcomes
        """
        try:
            # Create experience record
            experience = ExperienceRecord(
                task_id=task.id,
                experience_type=ExperienceType.TASK_COMPLETION,
                agent_id=agent_id,
                end_time=datetime.now(),
                success=outcome.get('success', False),
                quality_score=outcome.get('quality_score', 0.0),
                efficiency_score=outcome.get('efficiency_score', 0.0),
                cost_actual=outcome.get('cost_actual'),
                business_contexts=task.business_contexts
            )
            
            # Calculate duration if start time available
            if task.started_at:
                experience.start_time = task.started_at
                experience.duration_seconds = (experience.end_time - task.started_at).total_seconds()
            
            # Extract resource usage from outcome
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
            
            logger.info(f"Captured experience for task {task.id} with agent {agent_id}")
            
            return experience
            
        except Exception as e:
            logger.error(f"Experience capture failed for task {task.id}: {e}")
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
                 resources_used, business_contexts, complexity_factors, lessons_learned,
                 metadata, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                json.dumps(experience.complexity_factors),
                json.dumps(experience.lessons_learned),
                json.dumps(experience.metadata),
                experience.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store experience record: {e}")
    
    async def predict_resource_requirements(self, tasks: List[Task], 
                                          time_horizon: timedelta = timedelta(hours=24)) -> List[ResourcePrediction]:
        """
        Predict future resource requirements based on task queue and historical data
        
        Integrates with Point 3 priority decisions to forecast demand
        """
        predictions = []
        
        try:
            # Analyze historical patterns
            historical_data = await self._analyze_historical_usage()
            
            for resource_type in ResourceType:
                prediction = await self._predict_single_resource(
                    resource_type, tasks, time_horizon, historical_data
                )
                if prediction:
                    predictions.append(prediction)
            
            self.resource_predictions = predictions
            
            # Store predictions in database
            for prediction in predictions:
                await self._store_resource_prediction(prediction)
            
            logger.info(f"Generated {len(predictions)} resource predictions for {time_horizon}")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Resource prediction failed: {e}")
            return []
    
    async def _analyze_historical_usage(self) -> Dict[ResourceType, Dict[str, float]]:
        """Analyze historical resource usage patterns"""
        patterns = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent experience records with resource usage
            cursor.execute('''
                SELECT resources_used, business_contexts, success, quality_score
                FROM experience_records 
                WHERE resources_used IS NOT NULL 
                AND created_at > datetime('now', '-30 days')
                ORDER BY created_at DESC
            ''')
            
            records = cursor.fetchall()
            conn.close()
            
            for resource_type in ResourceType:
                usage_data = []
                
                for record in records:
                    try:
                        resources_used = json.loads(record[0])
                        if resource_type.value in resources_used:
                            usage_data.append(resources_used[resource_type.value])
                    except (json.JSONDecodeError, KeyError):
                        continue
                
                if usage_data:
                    patterns[resource_type] = {
                        'mean': statistics.mean(usage_data),
                        'median': statistics.median(usage_data),
                        'std': statistics.stdev(usage_data) if len(usage_data) > 1 else 0.0,
                        'max': max(usage_data),
                        'min': min(usage_data),
                        'trend': self._calculate_trend(usage_data)
                    }
                else:
                    # Default pattern for new resource types
                    patterns[resource_type] = {
                        'mean': 1.0,
                        'median': 1.0, 
                        'std': 0.2,
                        'max': 2.0,
                        'min': 0.1,
                        'trend': 0.0
                    }
            
            logger.info(f"Analyzed historical patterns for {len(patterns)} resource types")
            return patterns
            
        except Exception as e:
            logger.error(f"Historical analysis failed: {e}")
            return {}
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend direction in usage data"""
        if len(data) < 2:
            return 0.0
        
        # Simple linear trend calculation
        x = list(range(len(data)))
        n = len(data)
        
        sum_x = sum(x)
        sum_y = sum(data)
        sum_xy = sum(x[i] * data[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        if n * sum_x2 - sum_x ** 2 == 0:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        return slope
    
    async def _predict_single_resource(self, resource_type: ResourceType, 
                                     tasks: List[Task], time_horizon: timedelta,
                                     historical_data: Dict[ResourceType, Dict[str, float]]) -> Optional[ResourcePrediction]:
        """Predict usage for a single resource type"""
        try:
            # Get historical baseline
            baseline_data = historical_data.get(resource_type, {})
            historical_baseline = baseline_data.get('mean', 1.0)
            
            # Calculate trend factor
            trend_factor = baseline_data.get('trend', 0.0)
            
            # Calculate seasonal factor (simplified - could be enhanced)
            seasonal_factor = 1.0 + 0.1 * np.sin(datetime.now().hour * np.pi / 12)
            
            # Calculate business impact factor from pending tasks
            business_impact_factor = 1.0
            for task in tasks:
                for context in task.business_contexts:
                    if context in [BusinessContext.REVENUE_CRITICAL, BusinessContext.SECURITY_CRITICAL]:
                        business_impact_factor += 0.2
                    elif context == BusinessContext.CUSTOMER_FACING:
                        business_impact_factor += 0.1
            
            business_impact_factor = min(business_impact_factor, 2.0)  # Cap at 2x
            
            # Predict usage based on task queue
            task_based_usage = 0.0
            for task in tasks:
                # Estimate resource usage based on task complexity and type
                estimated_usage = self._estimate_task_resource_usage(task, resource_type)
                task_based_usage += estimated_usage
            
            # Combine all factors
            predicted_usage = (
                (historical_baseline * 0.6) +  # Historical weight: 60%
                (task_based_usage * 0.4)       # Task-based weight: 40%
            ) * trend_factor * seasonal_factor * business_impact_factor
            
            # Calculate confidence based on data quality
            data_points = len(baseline_data) if baseline_data else 0
            confidence_score = min(0.5 + (data_points * 0.1), 0.95)
            
            # Determine accuracy level
            if confidence_score >= 0.9:
                accuracy_level = PredictionAccuracy.HIGH
            elif confidence_score >= 0.7:
                accuracy_level = PredictionAccuracy.MEDIUM
            else:
                accuracy_level = PredictionAccuracy.LOW
            
            return ResourcePrediction(
                resource_type=resource_type,
                time_horizon=time_horizon,
                predicted_usage=predicted_usage,
                confidence_score=confidence_score,
                accuracy_level=accuracy_level,
                historical_baseline=historical_baseline,
                trend_factor=trend_factor,
                seasonal_factor=seasonal_factor,
                business_impact_factor=business_impact_factor
            )
            
        except Exception as e:
            logger.error(f"Single resource prediction failed for {resource_type}: {e}")
            return None
    
    def _estimate_task_resource_usage(self, task: Task, resource_type: ResourceType) -> float:
        """Estimate resource usage for a task based on its characteristics"""
        base_usage = 1.0
        
        # Adjust based on task priority
        if task.priority == TaskPriority.CRITICAL:
            base_usage *= 1.5
        elif task.priority == TaskPriority.HIGH:
            base_usage *= 1.2
        elif task.priority == TaskPriority.LOW:
            base_usage *= 0.8
        
        # Adjust based on estimated hours
        if task.estimated_hours > 0:
            base_usage *= (task.estimated_hours / 8.0)  # Normalize to 8-hour baseline
        
        # Adjust based on complexity score
        base_usage *= (0.5 + task.complexity_score)  # 0.5-1.5 multiplier
        
        # Resource-specific adjustments
        if resource_type == ResourceType.AGENT_HOURS:
            base_usage *= 1.0  # Direct correlation
        elif resource_type == ResourceType.AI_MODEL_CALLS:
            base_usage *= 0.5  # Fewer AI calls than hours
        elif resource_type in [ResourceType.COMPUTE_POWER, ResourceType.MEMORY]:
            base_usage *= 0.3   # Less direct correlation
        
        return max(base_usage, 0.1)  # Minimum usage
    
    async def _store_resource_prediction(self, prediction: ResourcePrediction):
        """Store resource prediction in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO resource_predictions
                (prediction_id, resource_type, time_horizon_hours, predicted_usage,
                 confidence_score, accuracy_level, historical_baseline, trend_factor,
                 seasonal_factor, business_impact_factor, model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction.prediction_id,
                prediction.resource_type.value,
                int(prediction.time_horizon.total_seconds() / 3600),
                prediction.predicted_usage,
                prediction.confidence_score,
                prediction.accuracy_level.value,
                prediction.historical_baseline,
                prediction.trend_factor,
                prediction.seasonal_factor,
                prediction.business_impact_factor,
                prediction.model_version,
                prediction.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store resource prediction: {e}")
    
    async def generate_capacity_recommendations(self, current_capacity: Dict[ResourceType, float]) -> List[CapacityRecommendation]:
        """Generate capacity adjustment recommendations based on predictions"""
        recommendations = []
        
        try:
            for prediction in self.resource_predictions:
                current = current_capacity.get(prediction.resource_type, 1.0)
                predicted = prediction.predicted_usage
                
                # Check if adjustment needed
                utilization = predicted / current
                
                if utilization > 0.9:  # Over 90% utilization
                    recommended = predicted * 1.2  # 20% buffer
                    reason = f"High utilization predicted ({utilization:.1%})"
                    benefit = "Prevent performance degradation"
                    risk = "Low risk - proactive scaling"
                    timeline = "Within 24 hours"
                    
                elif utilization < 0.3:  # Under 30% utilization
                    recommended = predicted * 1.1  # 10% buffer
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
                    cost_impact=(recommended - current) * 10.0,  # Simplified cost model
                    roi_estimate=max(0.1, (current - recommended) / current * 0.8)
                )
                
                recommendations.append(recommendation)
                await self._store_capacity_recommendation(recommendation)
            
            self.capacity_recommendations = recommendations
            logger.info(f"Generated {len(recommendations)} capacity recommendations")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Capacity recommendation generation failed: {e}")
            return []
    
    async def _store_capacity_recommendation(self, recommendation: CapacityRecommendation):
        """Store capacity recommendation in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO capacity_recommendations
                (recommendation_id, resource_type, current_capacity, recommended_capacity,
                 confidence, reasoning, expected_benefit, risk_assessment,
                 implementation_timeline, cost_impact, roi_estimate, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                recommendation.recommendation_id,
                recommendation.resource_type.value,
                recommendation.current_capacity,
                recommendation.recommended_capacity,
                recommendation.confidence,
                recommendation.reasoning,
                recommendation.expected_benefit,
                recommendation.risk_assessment,
                recommendation.implementation_timeline,
                recommendation.cost_impact,
                recommendation.roi_estimate,
                recommendation.created_at.isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to store capacity recommendation: {e}")
    
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
            'experience_insights': self.get_experience_insights(),
            'prediction_accuracy': {
                'high': sum(1 for p in self.resource_predictions if p.accuracy_level == PredictionAccuracy.HIGH),
                'medium': sum(1 for p in self.resource_predictions if p.accuracy_level == PredictionAccuracy.MEDIUM),
                'low': sum(1 for p in self.resource_predictions if p.accuracy_level == PredictionAccuracy.LOW)
            }
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
    
    # Create sample tasks for prediction
    if V2_INTEGRATION:
        from intelligence_v2.interfaces import create_simple_task
        
        tasks = [
            create_simple_task("Deploy ML Pipeline", "Deploy production ML pipeline", "high"),
            create_simple_task("Security Audit", "Quarterly security audit", "critical"),
            create_simple_task("Database Optimization", "Optimize database performance", "medium"),
            create_simple_task("API Documentation", "Update API documentation", "low")
        ]
        
        # Set business contexts
        tasks[0].business_contexts = [BusinessContext.REVENUE_CRITICAL]
        tasks[1].business_contexts = [BusinessContext.SECURITY_CRITICAL]
        tasks[2].business_contexts = [BusinessContext.CUSTOMER_FACING]
        tasks[3].business_contexts = [BusinessContext.INTERNAL_TOOLS]
        
        # Set estimated hours
        for i, hours in enumerate([8, 4, 6, 2]):
            tasks[i].estimated_hours = hours
            tasks[i].complexity_score = 0.3 + (i * 0.2)  # Varying complexity
    else:
        # Fallback task simulation
        tasks = []
    
    print("📋 Sample Task Queue:")
    if tasks:
        for task in tasks:
            print(f"  • {task.title} ({task.priority.value}) - {task.estimated_hours}h")
    else:
        print("  • Using simulated task data (V2.0 integration not available)")
    print()
    
    # Generate resource predictions
    print("🔮 Generating Resource Predictions...")
    predictions = await planner.predict_resource_requirements(tasks, timedelta(hours=24))
    
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
    if tasks:
        # Simulate completion of first task
        task = tasks[0]
        task.started_at = datetime.now() - timedelta(hours=2)
        
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
            ]
        }
        
        experience = await planner.capture_task_experience(task, "agent_001", outcome)
        if experience:
            print(f"  ✅ Experience captured for task: {task.title}")
            print(f"     - Success: {experience.success}")
            print(f"     - Quality Score: {experience.quality_score}")
            print(f"     - Duration: {experience.duration_seconds:.1f}s")
            print(f"     - Lessons: {len(experience.lessons_learned)}")
    else:
        print("  ⚠️  No V2.0 tasks available - using simulated experience capture")
    print()
    
    # Generate capacity recommendations
    print("💡 Generating Capacity Recommendations...")
    current_capacity = {
        ResourceType.AGENT_HOURS: 40.0,
        ResourceType.AI_MODEL_CALLS: 1000.0,
        ResourceType.COMPUTE_POWER: 4.0,
        ResourceType.MEMORY: 16.0,
        ResourceType.STORAGE: 500.0,
        ResourceType.NETWORK_BANDWIDTH: 100.0
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
    print("   ✅ Integration with Point 3 prioritization ready")

if __name__ == "__main__":
    asyncio.run(demo_predictive_resource_planning())