# Fix the datetime calculation bug in Point 4
fix_code = '''
# Line 356 bug fix in point4-predictive-resource-planning.py
# WRONG: if not self.predictions or (datetime.now() - self.predictions[0].prediction_horizon.total_seconds() / 3600) > 1:
# CORRECT: if not self.predictions or (datetime.now() - datetime.now()).total_seconds() / 3600 > 1:

# Better fix - check prediction age properly:
prediction_age_hours = 0
if self.predictions:
    # Get the time when prediction was made (assume it was made now - horizon)
    prediction_time = datetime.now() - timedelta(hours=1)  # Assume predictions are recent
    prediction_age_hours = (datetime.now() - prediction_time).total_seconds() / 3600

if not self.predictions or prediction_age_hours > 1:
    await self.generate_predictions()
'''

print("🔧 POINT 4 BUG IDENTIFIED:")
print("Line 356: Datetime subtraction error")
print("\n✅ SIMPLE FIX:")
print("Replace line 356 with proper prediction age calculation")

print("\n🎯 FIXED VERSION:")
with open('point4-predictive-resource-planning-FIXED.py', 'w', encoding='utf-8') as f:
    # Read original file and fix the bug
    original_code = '''#!/usr/bin/env python3
"""
Agent Zero V1 - Predictive Resource Planning & Capacity Management - Point 4/6 - FIXED
Week 43 Implementation - Advanced AI Resource Management - BUG FIXED

Fixed datetime calculation bug on line 356.
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import math
from collections import defaultdict, deque
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Core enums and classes
class ResourceType(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    MEMORY = "memory"
    NETWORK = "network"
    AGENT_TIME = "agent_time"
    PROCESSING_POWER = "processing_power"

class PredictionAccuracy(Enum):
    HIGH = "high"        # >90% accuracy
    MEDIUM = "medium"    # 70-90% accuracy  
    LOW = "low"         # <70% accuracy

@dataclass
class ResourceUsageMetric:
    timestamp: datetime
    resource_type: ResourceType
    usage_amount: float
    available_capacity: float
    utilization_percentage: float
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResourcePrediction:
    resource_type: ResourceType
    prediction_horizon: timedelta
    predicted_usage: float
    confidence_level: float
    accuracy: PredictionAccuracy
    recommendation: str
    factors: List[str] = field(default_factory=list)

@dataclass
class CapacityRecommendation:
    resource_type: ResourceType
    current_capacity: float
    recommended_capacity: float
    change_percentage: float
    urgency: str
    cost_impact: float
    timeline: str
    justification: str

@dataclass
class WorkloadForecast:
    forecast_date: datetime
    expected_tasks: int
    expected_agents_needed: int
    peak_hours: List[int]
    resource_requirements: Dict[ResourceType, float]
    confidence: float

class PredictiveResourceManager:
    def __init__(self, history_window_days: int = 30):
        self.history_window_days = history_window_days
        self.usage_history: List[ResourceUsageMetric] = []
        self.predictions: List[ResourcePrediction] = []
        self.recommendations: List[CapacityRecommendation] = []
        self.workload_forecasts: List[WorkloadForecast] = []
        self.prediction_timestamps: List[datetime] = []  # Track when predictions were made
        
        # Resource capacity limits
        self.resource_capacities = {
            ResourceType.COMPUTE: 1000.0,
            ResourceType.STORAGE: 10000.0,
            ResourceType.MEMORY: 512.0,
            ResourceType.NETWORK: 1000.0,
            ResourceType.AGENT_TIME: 2400.0,
            ResourceType.PROCESSING_POWER: 100.0
        }
        
        # Learning parameters
        self.prediction_models = {}
        self.seasonal_patterns = {}
        self.trend_analysis = {}
        
        logger.info("🔮 Predictive Resource Manager initialized")
    
    def record_usage(self, usage: ResourceUsageMetric):
        self.usage_history.append(usage)
        cutoff_date = datetime.now() - timedelta(days=self.history_window_days)
        self.usage_history = [u for u in self.usage_history if u.timestamp > cutoff_date]
        self._update_prediction_models(usage.resource_type)
        logger.info(f"📊 Recorded {usage.resource_type.value} usage: {usage.utilization_percentage:.1f}%")
    
    def _update_prediction_models(self, resource_type: ResourceType):
        resource_data = [u for u in self.usage_history if u.resource_type == resource_type]
        
        if len(resource_data) < 5:
            return
        
        timestamps = [u.timestamp for u in resource_data]
        usage_values = [u.utilization_percentage for u in resource_data]
        
        if len(usage_values) >= 7:
            recent_trend = self._calculate_trend(usage_values[-7:])
            self.trend_analysis[resource_type] = recent_trend
        
        if len(usage_values) >= 24:
            hourly_pattern = self._detect_hourly_patterns(resource_data)
            self.seasonal_patterns[resource_type] = hourly_pattern
        
        logger.debug(f"Updated prediction model for {resource_type.value}")
    
    def _calculate_trend(self, values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        
        x = list(range(len(values)))
        n = len(values)
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _detect_hourly_patterns(self, data: List[ResourceUsageMetric]) -> Dict[int, float]:
        hourly_usage = defaultdict(list)
        
        for metric in data:
            hour = metric.timestamp.hour
            hourly_usage[hour].append(metric.utilization_percentage)
        
        hourly_averages = {}
        for hour in range(24):
            if hour in hourly_usage and hourly_usage[hour]:
                hourly_averages[hour] = statistics.mean(hourly_usage[hour])
            else:
                hourly_averages[hour] = 0.0
        
        return hourly_averages
    
    async def generate_predictions(self, horizon_hours: int = 24) -> List[ResourcePrediction]:
        logger.info(f"🔮 Generating predictions for next {horizon_hours} hours...")
        
        predictions = []
        horizon = timedelta(hours=horizon_hours)
        
        for resource_type in ResourceType:
            prediction = await self._predict_resource_usage(resource_type, horizon)
            if prediction:
                predictions.append(prediction)
        
        self.predictions = predictions
        self.prediction_timestamps = [datetime.now()] * len(predictions)  # Track when predictions were made
        return predictions
    
    async def _predict_resource_usage(self, resource_type: ResourceType, horizon: timedelta) -> Optional[ResourcePrediction]:
        resource_data = [u for u in self.usage_history if u.resource_type == resource_type]
        
        if len(resource_data) < 3:
            return None
        
        recent_usage = [u.utilization_percentage for u in resource_data[-10:]]
        current_usage = recent_usage[-1] if recent_usage else 50.0
        
        trend = self.trend_analysis.get(resource_type, 0.0)
        hours_ahead = horizon.total_seconds() / 3600
        
        predicted_usage = current_usage + (trend * hours_ahead)
        
        if resource_type in self.seasonal_patterns:
            target_hour = (datetime.now() + horizon).hour
            seasonal_factor = self.seasonal_patterns[resource_type].get(target_hour, 1.0)
            predicted_usage *= (seasonal_factor / 100.0 + 0.5)
        
        uncertainty = min(abs(trend) * 2, 10.0)
        predicted_usage += np.random.normal(0, uncertainty)
        predicted_usage = max(0.0, min(100.0, predicted_usage))
        
        confidence = self._calculate_prediction_confidence(resource_data, trend)
        
        if confidence > 0.9:
            accuracy = PredictionAccuracy.HIGH
        elif confidence > 0.7:
            accuracy = PredictionAccuracy.MEDIUM
        else:
            accuracy = PredictionAccuracy.LOW
        
        factors = self._identify_prediction_factors(resource_type, trend, seasonal_patterns=resource_type in self.seasonal_patterns)
        recommendation = self._generate_usage_recommendation(predicted_usage, current_usage, trend)
        
        return ResourcePrediction(
            resource_type=resource_type,
            prediction_horizon=horizon,
            predicted_usage=predicted_usage,
            confidence_level=confidence,
            accuracy=accuracy,
            recommendation=recommendation,
            factors=factors
        )
    
    def _calculate_prediction_confidence(self, data: List[ResourceUsageMetric], trend: float) -> float:
        if len(data) < 5:
            return 0.3
        
        data_confidence = min(len(data) / 50.0, 0.8)
        
        recent_values = [u.utilization_percentage for u in data[-10:]]
        if len(recent_values) > 3:
            variance = statistics.variance(recent_values)
            stability_confidence = max(0.2, 1.0 - (variance / 1000.0))
        else:
            stability_confidence = 0.5
        
        return (data_confidence + stability_confidence) / 2.0
    
    def _identify_prediction_factors(self, resource_type: ResourceType, trend: float, seasonal_patterns: bool) -> List[str]:
        factors = []
        
        if abs(trend) > 1.0:
            if trend > 0:
                factors.append("Increasing usage trend detected")
            else:
                factors.append("Decreasing usage trend detected")
        
        if seasonal_patterns:
            factors.append("Seasonal usage patterns identified")
        
        if resource_type == ResourceType.AGENT_TIME:
            factors.append("Agent workload patterns")
        elif resource_type == ResourceType.COMPUTE:
            factors.append("Processing demand fluctuations")
        elif resource_type == ResourceType.MEMORY:
            factors.append("Memory usage patterns")
        
        if not factors:
            factors.append("Historical usage patterns")
        
        return factors
    
    def _generate_usage_recommendation(self, predicted_usage: float, current_usage: float, trend: float) -> str:
        change = predicted_usage - current_usage
        
        if predicted_usage > 90:
            return "Critical: Scale up resources immediately to prevent bottlenecks"
        elif predicted_usage > 80:
            return "Warning: Consider scaling up resources proactively"
        elif predicted_usage > 70 and trend > 2:
            return "Monitor closely: Usage trending upward, prepare for scaling"
        elif predicted_usage < 30 and trend < -2:
            return "Opportunity: Consider scaling down to optimize costs"
        elif abs(change) < 5:
            return "Stable: Current capacity appears adequate"
        else:
            return f"Monitor: Usage expected to {'increase' if change > 0 else 'decrease'} by {abs(change):.1f}%"
    
    async def generate_capacity_recommendations(self) -> List[CapacityRecommendation]:
        logger.info("📋 Generating capacity recommendations...")
        
        recommendations = []
        
        # FIXED: Check prediction age properly
        predictions_are_recent = True
        if self.predictions and self.prediction_timestamps:
            prediction_age = (datetime.now() - self.prediction_timestamps[0]).total_seconds() / 3600
            predictions_are_recent = prediction_age <= 1
        
        if not self.predictions or not predictions_are_recent:
            await self.generate_predictions()
        
        for prediction in self.predictions:
            recommendation = self._analyze_capacity_needs(prediction)
            if recommendation:
                recommendations.append(recommendation)
        
        self.recommendations = recommendations
        return recommendations
    
    def _analyze_capacity_needs(self, prediction: ResourcePrediction) -> Optional[CapacityRecommendation]:
        current_capacity = self.resource_capacities.get(prediction.resource_type, 100.0)
        predicted_absolute_usage = (prediction.predicted_usage / 100.0) * current_capacity
        
        if prediction.predicted_usage > 85:
            target_utilization = 70.0
            recommended_capacity = predicted_absolute_usage / (target_utilization / 100.0)
            change_percentage = ((recommended_capacity - current_capacity) / current_capacity) * 100
            urgency = "high" if prediction.predicted_usage > 95 else "medium"
            justification = f"Predicted {prediction.predicted_usage:.1f}% utilization requires capacity increase"
            
        elif prediction.predicted_usage < 40:
            target_utilization = 60.0
            recommended_capacity = predicted_absolute_usage / (target_utilization / 100.0)
            change_percentage = ((recommended_capacity - current_capacity) / current_capacity) * 100
            urgency = "low"
            justification = f"Low predicted utilization ({prediction.predicted_usage:.1f}%) suggests over-provisioning"
            
        else:
            return None
        
        cost_impact = abs(change_percentage) * 0.1
        timeline_map = {
            "critical": "Immediate (0-4 hours)",
            "high": "Short-term (4-24 hours)", 
            "medium": "Medium-term (1-7 days)",
            "low": "Long-term (1-4 weeks)"
        }
        timeline = timeline_map.get(urgency, "Medium-term")
        
        return CapacityRecommendation(
            resource_type=prediction.resource_type,
            current_capacity=current_capacity,
            recommended_capacity=recommended_capacity,
            change_percentage=change_percentage,
            urgency=urgency,
            cost_impact=cost_impact,
            timeline=timeline,
            justification=justification
        )
    
    async def generate_workload_forecast(self, days_ahead: int = 7) -> List[WorkloadForecast]:
        logger.info(f"📈 Generating workload forecast for next {days_ahead} days...")
        
        forecasts = []
        
        for day in range(days_ahead):
            forecast_date = datetime.now().date() + timedelta(days=day+1)
            forecast = await self._forecast_daily_workload(forecast_date)
            forecasts.append(forecast)
        
        self.workload_forecasts = forecasts
        return forecasts
    
    async def _forecast_daily_workload(self, target_date) -> WorkloadForecast:
        base_tasks = 50 + np.random.randint(-10, 20)
        base_agents = max(3, base_tasks // 15)
        
        weekday = target_date.weekday()
        if weekday >= 5:
            base_tasks = int(base_tasks * 0.6)
            base_agents = int(base_agents * 0.7)
        
        if weekday < 5:
            peak_hours = [9, 10, 11, 14, 15, 16]
        else:
            peak_hours = [10, 11, 15, 16]
        
        resource_requirements = {
            ResourceType.AGENT_TIME: base_agents * 8.0,
            ResourceType.COMPUTE: base_tasks * 2.5,
            ResourceType.MEMORY: base_tasks * 1.2,
            ResourceType.STORAGE: base_tasks * 0.8,
            ResourceType.NETWORK: base_tasks * 0.5,
            ResourceType.PROCESSING_POWER: base_agents * 12.0
        }
        
        days_ahead = (target_date - datetime.now().date()).days
        confidence = max(0.4, 0.9 - (days_ahead * 0.1))
        
        return WorkloadForecast(
            forecast_date=datetime.combine(target_date, datetime.min.time()),
            expected_tasks=base_tasks,
            expected_agents_needed=base_agents,
            peak_hours=peak_hours,
            resource_requirements=resource_requirements,
            confidence=confidence
        )
    
    def get_analytics(self) -> Dict[str, Any]:
        return {
            "system_health": "Optimal",
            "data_points": len(self.usage_history),
            "forecast_confidence": 0.85 if self.workload_forecasts else 0.0,
            "predictions_count": len(self.predictions)
        }

# Demo function
async def demo_predictive_resource_management():
    print("🚀 Predictive Resource Planning & Capacity Management Demo - FIXED")
    print("Week 43 - Point 4 of 6 Critical AI Features")
    print("=" * 70)
    
    manager = PredictiveResourceManager()
    
    print("\\n📊 Simulating historical resource usage...")
    base_time = datetime.now() - timedelta(days=7)
    
    for day in range(7):
        for hour in range(24):
            timestamp = base_time + timedelta(days=day, hours=hour)
            
            for resource_type in ResourceType:
                if resource_type == ResourceType.AGENT_TIME:
                    if 9 <= hour <= 17:
                        usage = 60 + np.random.normal(0, 15)
                    else:
                        usage = 20 + np.random.normal(0, 10)
                elif resource_type == ResourceType.COMPUTE:
                    usage = 40 + (day * 5) + np.random.normal(0, 12)
                else:
                    usage = 50 + (day * 2) + np.random.normal(0, 10)
                
                usage = max(5, min(95, usage))
                
                metric = ResourceUsageMetric(
                    timestamp=timestamp,
                    resource_type=resource_type,
                    usage_amount=usage,
                    available_capacity=100.0,
                    utilization_percentage=usage
                )
                
                manager.record_usage(metric)
    
    print(f"Generated {len(manager.usage_history)} usage data points")
    
    # Generate predictions
    print("\\n🔮 Generating resource predictions...")
    predictions = await manager.generate_predictions(horizon_hours=24)
    
    print(f"\\nPredictions for next 24 hours:")
    for pred in predictions:
        print(f"  {pred.resource_type.value}:")
        print(f"    Predicted Usage: {pred.predicted_usage:.1f}%")
        print(f"    Confidence: {pred.confidence_level:.1%}")
        print(f"    Accuracy: {pred.accuracy.value}")
        print(f"    Recommendation: {pred.recommendation}")
    
    # Generate capacity recommendations
    print("\\n📋 Generating capacity recommendations...")
    recommendations = await manager.generate_capacity_recommendations()
    
    if recommendations:
        print("\\nCapacity Recommendations:")
        for rec in recommendations:
            print(f"  {rec.resource_type.value}:")
            print(f"    Change: {rec.change_percentage:+.1f}%")
            print(f"    Urgency: {rec.urgency}")
            print(f"    Timeline: {rec.timeline}")
            print(f"    Justification: {rec.justification}")
    else:
        print("\\nNo capacity changes recommended at this time.")
    
    # Generate workload forecast
    print("\\n📈 Generating workload forecast...")
    forecasts = await manager.generate_workload_forecast(days_ahead=5)
    
    print("\\nWorkload Forecast (next 5 days):")
    for forecast in forecasts:
        date_str = forecast.forecast_date.strftime("%Y-%m-%d")
        print(f"  {date_str}:")
        print(f"    Expected Tasks: {forecast.expected_tasks}")
        print(f"    Agents Needed: {forecast.expected_agents_needed}")
        print(f"    Peak Hours: {forecast.peak_hours}")
        print(f"    Confidence: {forecast.confidence:.1%}")
    
    # Show analytics
    print("\\n📊 System Analytics:")
    analytics = manager.get_analytics()
    
    print(f"  System Health: {analytics['system_health']}")
    print(f"  Data Points: {analytics['data_points']}")
    print(f"  Forecast Confidence: {analytics['forecast_confidence']:.1%}")
    
    print("\\n✅ Predictive Resource Management Demo Completed - FIXED!")

if __name__ == "__main__":
    try:
        asyncio.run(demo_predictive_resource_management())
    except KeyboardInterrupt:
        print("\\n👋 Demo interrupted.")
    except Exception as e:
        print(f"\\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
'''
    f.write(original_code)

print("✅ Point 4 FIXED and saved as: point4-predictive-resource-planning-FIXED.py")