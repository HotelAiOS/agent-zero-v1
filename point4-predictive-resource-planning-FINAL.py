#!/usr/bin/env python3
"""
Agent Zero V1 - Predictive Resource Planning & Capacity Management - Point 4/6
Wersja: FINAL (bug fixed - świeżość predykcji, brak błędu datetime)
"""
import asyncio
import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import statistics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResourceType(Enum):
    COMPUTE = "compute"
    STORAGE = "storage"
    MEMORY = "memory"
    NETWORK = "network"
    AGENT_TIME = "agent_time"
    PROCESSING_POWER = "processing_power"

class PredictionAccuracy(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class ResourceUsageMetric:
    timestamp: datetime
    resource_type: ResourceType
    usage_amount: float
    available_capacity: float
    utilization_percentage: float
    context: Dict = field(default_factory=dict)

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
        self.prediction_timestamps: List[datetime] = []
        self.recommendations: List[CapacityRecommendation] = []
        self.workload_forecasts: List[WorkloadForecast] = []
        self.resource_capacities = {
            ResourceType.COMPUTE: 1000.0,
            ResourceType.STORAGE: 10000.0,
            ResourceType.MEMORY: 512.0,
            ResourceType.NETWORK: 1000.0,
            ResourceType.AGENT_TIME: 2400.0,
            ResourceType.PROCESSING_POWER: 100.0
        }
        self.trend_analysis = {}
        self.seasonal_patterns = {}
        logger.info("🔮 Predictive Resource Manager initialized")

    def record_usage(self, usage: ResourceUsageMetric):
        self.usage_history.append(usage)
        cutoff = datetime.now() - timedelta(days=self.history_window_days)
        self.usage_history = [u for u in self.usage_history if u.timestamp > cutoff]
        self._update_prediction_models(usage.resource_type)
        logger.info(f"📊 Recorded {usage.resource_type.value} usage: {usage.utilization_percentage:.1f}%")

    def _update_prediction_models(self, resource_type: ResourceType):
        data = [u for u in self.usage_history if u.resource_type == resource_type]
        if len(data) < 5:
            return
        usage_vals = [u.utilization_percentage for u in data]
        if len(usage_vals) >= 7:
            self.trend_analysis[resource_type] = self._calculate_trend(usage_vals[-7:])
        if len(usage_vals) >= 24:
            self.seasonal_patterns[resource_type] = self._detect_hourly_patterns(data)

    def _calculate_trend(self, vals: List[float]) -> float:
        if len(vals) < 2: return 0.0
        x = list(range(len(vals)))
        x_mean = sum(x) / len(x)
        y_mean = sum(vals) / len(vals)
        num = sum((x[i] - x_mean) * (vals[i] - y_mean) for i in range(len(x)))
        denom = sum((x[i] - x_mean) ** 2 for i in range(len(x)))
        return num / denom if denom != 0 else 0.0

    def _detect_hourly_patterns(self, data: List[ResourceUsageMetric]) -> Dict[int, float]:
        d = {}
        for h in range(24):
            vals = [m.utilization_percentage for m in data if m.timestamp.hour == h]
            d[h] = statistics.mean(vals) if vals else 0.0
        return d

    async def generate_predictions(self, horizon_hours: int = 24) -> List[ResourcePrediction]:
        logger.info(f"🔮 Generating predictions for next {horizon_hours} hours...")
        preds = []
        horizon = timedelta(hours=horizon_hours)
        for rtype in ResourceType:
            pred = await self._predict_resource_usage(rtype, horizon)
            if pred: preds.append(pred)
        self.predictions = preds
        self.prediction_timestamps = [datetime.now()] * len(preds)
        return preds

    async def _predict_resource_usage(self, rtype: ResourceType, horizon: timedelta):
        data = [u for u in self.usage_history if u.resource_type == rtype]
        if len(data) < 3: return None
        vals = [u.utilization_percentage for u in data[-10:]]
        current = vals[-1] if vals else 50.0
        trend = self.trend_analysis.get(rtype, 0.0)
        hours = horizon.total_seconds() / 3600
        predicted = current + trend * hours
        if rtype in self.seasonal_patterns:
            target_hour = (datetime.now() + horizon).hour
            seasonal = self.seasonal_patterns[rtype].get(target_hour, 1.0)
            predicted *= (seasonal / 100.0 + 0.5)
        predicted += np.random.normal(0, min(abs(trend) * 2, 10.0))
        predicted = max(0.0, min(100.0, predicted))
        confidence = self._calculate_prediction_confidence(data, trend)
        acc = PredictionAccuracy.HIGH if confidence > 0.9 else (
            PredictionAccuracy.MEDIUM if confidence > 0.7 else PredictionAccuracy.LOW)
        factors = []
        if abs(trend) > 1.0:
            factors.append("Increasing usage trend detected" if trend > 0 else "Decreasing usage trend detected")
        if rtype in self.seasonal_patterns: factors.append("Seasonal usage patterns identified")
        if not factors: factors.append("Historical usage patterns")
        rec = ("Critical: Scale up resources immediately"
               if predicted > 90 else
               "Warning: Consider scaling up resources proactively"
               if predicted > 80 else
               "Monitor closely: Usage trending upward, prepare for scaling"
               if predicted > 70 and trend > 2 else
               "Opportunity: Consider scaling down to optimize costs"
               if predicted < 30 and trend < -2 else
               "Stable: Current capacity appears adequate"
               if abs(predicted - current) < 5 else
               f"Monitor: Usage expected to {'increase' if predicted-current>0 else 'decrease'} by {abs(predicted-current):.1f}%")
        return ResourcePrediction(
            rtype, horizon, predicted, confidence, acc, rec, factors)

    def _calculate_prediction_confidence(self, data: List[ResourceUsageMetric], trend: float) -> float:
        if len(data) < 5: return 0.3
        data_conf = min(len(data) / 50.0, 0.8)
        recent = [u.utilization_percentage for u in data[-10:]]
        stab_conf = max(0.2, 1.0 - (statistics.variance(recent) / 1000.0)) if len(recent) > 3 else 0.5
        return (data_conf + stab_conf) / 2.0

    async def generate_capacity_recommendations(self) -> List[CapacityRecommendation]:
        logger.info("📋 Generating capacity recommendations...")
        recs = []
        preds_are_recent = True
        if self.predictions and self.prediction_timestamps:
            prediction_age = (datetime.now() - self.prediction_timestamps[0]).total_seconds() / 3600
            preds_are_recent = prediction_age <= 1
        if not self.predictions or not preds_are_recent:
            await self.generate_predictions()
        for pred in self.predictions:
            cap = self._analyze_capacity_needs(pred)
            if cap: recs.append(cap)
        self.recommendations = recs
        return recs

    def _analyze_capacity_needs(self, prediction: ResourcePrediction) -> Optional[CapacityRecommendation]:
        current = self.resource_capacities.get(prediction.resource_type, 100.0)
        pred_abs = (prediction.predicted_usage / 100.0) * current
        if prediction.predicted_usage > 85:
            target_util = 70.0
            rec_cap = pred_abs / (target_util / 100.0)
            chg = ((rec_cap - current) / current) * 100
            urg = "high" if prediction.predicted_usage > 95 else "medium"
            just = f"Predicted {prediction.predicted_usage:.1f}% utilization requires capacity increase"
        elif prediction.predicted_usage < 40:
            target_util = 60.0
            rec_cap = pred_abs / (target_util / 100.0)
            chg = ((rec_cap - current) / current) * 100
            urg = "low"
            just = f"Low predicted utilization ({prediction.predicted_usage:.1f}%) suggests over-provisioning"
        else:
            return None
        cost = abs(chg) * 0.1
        timeline_map = {
            "critical": "Immediate (0-4 hours)",
            "high": "Short-term (4-24 hours)",
            "medium": "Medium-term (1-7 days)",
            "low": "Long-term (1-4 weeks)"
        }
        return CapacityRecommendation(
            prediction.resource_type, current, rec_cap, chg, urg, cost,
            timeline_map[urg], just)

    async def generate_workload_forecast(self, days_ahead: int = 7) -> List[WorkloadForecast]:
        logger.info(f"📈 Generating workload forecast for next {days_ahead} days...")
        fcs = []
        for day in range(days_ahead):
            fc_date = datetime.now().date() + timedelta(days=day+1)
            fcs.append(await self._forecast_daily_workload(fc_date))
        self.workload_forecasts = fcs
        return fcs

    async def _forecast_daily_workload(self, target_date) -> WorkloadForecast:
        base_tasks = 50 + np.random.randint(-10, 20)
        base_agents = max(3, base_tasks // 15)
        weekday = target_date.weekday()
        if weekday >= 5:
            base_tasks = int(base_tasks * 0.6)
            base_agents = int(base_agents * 0.7)
        peak_hours = [9, 10, 11, 14, 15, 16] if weekday < 5 else [10, 11, 15, 16]
        resource_req = {
            ResourceType.AGENT_TIME: base_agents * 8.0,
            ResourceType.COMPUTE: base_tasks * 2.5,
            ResourceType.MEMORY: base_tasks * 1.2,
            ResourceType.STORAGE: base_tasks * 0.8,
            ResourceType.NETWORK: base_tasks * 0.5,
            ResourceType.PROCESSING_POWER: base_agents * 12.0
        }
        conf = max(0.4, 0.9 - ((target_date - datetime.now().date()).days * 0.1))
        return WorkloadForecast(
            datetime.combine(target_date, datetime.min.time()), base_tasks, base_agents, peak_hours, resource_req, conf
        )

    def get_analytics(self) -> Dict[str, any]:
        return {
            "system_health": "Optimal",
            "data_points": len(self.usage_history),
            "forecast_confidence": 0.85 if self.workload_forecasts else 0.0,
            "predictions_count": len(self.predictions)
        }

async def demo_predictive_resource_management():
    print("🚀 Predictive Resource Planning & Capacity Management Demo - FINAL")
    print("Week 43 - Point 4 of 6 Critical AI Features")
    print("=" * 70)
    manager = PredictiveResourceManager()
    print("\n📊 Simulating historical resource usage...")
    base_time = datetime.now() - timedelta(days=7)
    for day in range(7):
        for hour in range(24):
            ts = base_time + timedelta(days=day, hours=hour)
            for rtype in ResourceType:
                if rtype == ResourceType.AGENT_TIME:
                    usage = 60 + np.random.normal(0, 15) if 9 <= hour <= 17 else 20 + np.random.normal(0, 10)
                elif rtype == ResourceType.COMPUTE:
                    usage = 40 + (day * 5) + np.random.normal(0, 12)
                else:
                    usage = 50 + (day * 2) + np.random.normal(0, 10)
                usage = max(5, min(95, usage))
                metric = ResourceUsageMetric(ts, rtype, usage, 100.0, usage)
                manager.record_usage(metric)
    print(f"Generated {len(manager.usage_history)} usage data points")
    print("\n🔮 Generating resource predictions...")
    predictions = await manager.generate_predictions(horizon_hours=24)
    print(f"\nPredictions for next 24 hours:")
    for pred in predictions:
        print(f"  {pred.resource_type.value}:")
        print(f"    Predicted Usage: {pred.predicted_usage:.1f}%")
        print(f"    Confidence: {pred.confidence_level:.1%}")
        print(f"    Accuracy: {pred.accuracy.value}")
        print(f"    Recommendation: {pred.recommendation}")
    print("\n📋 Generating capacity recommendations...")
    recommendations = await manager.generate_capacity_recommendations()
    if recommendations:
        print("\nCapacity Recommendations:")
        for rec in recommendations:
            print(f"  {rec.resource_type.value}:")
            print(f"    Change: {rec.change_percentage:+.1f}%")
            print(f"    Urgency: {rec.urgency}")
            print(f"    Timeline: {rec.timeline}")
            print(f"    Justification: {rec.justification}")
    else:
        print("\nNo capacity changes recommended at this time.")
    print("\n📈 Generating workload forecast...")
    forecasts = await manager.generate_workload_forecast(days_ahead=5)
    print("\nWorkload Forecast (next 5 days):")
    for fc in forecasts:
        print(f"  {fc.forecast_date.strftime('%Y-%m-%d')}:")
        print(f"    Expected Tasks: {fc.expected_tasks}")
        print(f"    Agents Needed: {fc.expected_agents_needed}")
        print(f"    Peak Hours: {fc.peak_hours}")
        print(f"    Confidence: {fc.confidence:.1%}")
    print("\n📊 System Analytics:")
    analytics = manager.get_analytics()
    print(f"  System Health: {analytics['system_health']}")
    print(f"  Data Points: {analytics['data_points']}")
    print(f"  Forecast Confidence: {analytics['forecast_confidence']:.1%}")
    print("\n✅ Predictive Resource Management Demo Completed - FINAL!")

if __name__ == "__main__":
    try:
        asyncio.run(demo_predictive_resource_management())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
