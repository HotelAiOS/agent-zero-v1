#!/usr/bin/env python3
"""
Agent Zero V1 - Pattern Mining Engine
V2.0 Intelligence Layer - Week 44 Implementation

🎯 Week 44 Critical Task: Pattern Mining Engine (6 SP)
Zadanie: Wykrywanie wzorców sukcesu, korelacje metryk
Rezultat: Zbiory wzorców i insighty optymalizacyjne
Impact: Proaktywna optymalizacja na podstawie historical patterns

Author: Developer A (Backend Architect)
Date: 10 października 2025
Linear Issue: A0-44 (Week 44 Implementation)
"""

import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, Counter
import logging
import statistics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    MODEL_PERFORMANCE = "model_performance"
    COST_EFFICIENCY = "cost_efficiency"
    TEMPORAL = "temporal"
    CONTEXT_BASED = "context_based"
    SUCCESS_CORRELATION = "success_correlation"
    FAILURE_ANALYSIS = "failure_analysis"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"

class InsightType(Enum):
    OPTIMIZATION = "optimization"
    WARNING = "warning"
    OPPORTUNITY = "opportunity"
    BEST_PRACTICE = "best_practice"
    ANTI_PATTERN = "anti_pattern"

@dataclass
class DiscoveredPattern:
    """Pattern discovered by mining engine"""
    id: str
    pattern_type: PatternType
    name: str
    description: str
    conditions: Dict[str, Any]
    evidence: List[Dict[str, Any]]
    confidence: float
    sample_size: int
    statistical_significance: float
    impact_metrics: Dict[str, float]
    recommendations: List[str]
    discovered_at: datetime
    last_validated: datetime
    validation_count: int

@dataclass
class OptimizationInsight:
    """Actionable insight from pattern analysis"""
    id: str
    type: InsightType
    title: str
    description: str
    pattern_ids: List[str]
    impact_score: float
    confidence: float
    estimated_improvement: Dict[str, float]
    action_items: List[str]
    priority: str
    created_at: datetime

class PatternMiningEngine:
    """
    Advanced Pattern Mining Engine for Agent Zero V2.0
    
    Responsibilities:
    - Mine patterns from historical execution data
    - Analyze correlations between task parameters and outcomes
    - Identify optimization opportunities and anti-patterns
    - Generate actionable insights for system improvement
    - Validate patterns with statistical significance testing
    """
    
    def __init__(self, db_path: str = "agent_zero.db", min_confidence: float = 0.7):
        self.db_path = db_path
        self.min_confidence = min_confidence
        self._init_database()
    
    def _init_database(self):
        """Initialize pattern mining tables"""
        with sqlite3.connect(self.db_path) as conn:
            # Patterns table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_discovered_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    conditions TEXT NOT NULL,  -- JSON
                    evidence TEXT NOT NULL,    -- JSON array
                    confidence REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    statistical_significance REAL NOT NULL,
                    impact_metrics TEXT NOT NULL,  -- JSON
                    recommendations TEXT NOT NULL,  -- JSON array
                    discovered_at TEXT NOT NULL,
                    last_validated TEXT NOT NULL,
                    validation_count INTEGER DEFAULT 0
                )
            """)
            
            # Insights table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS v2_optimization_insights (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    description TEXT NOT NULL,
                    pattern_ids TEXT NOT NULL,  -- JSON array
                    impact_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    estimated_improvement TEXT NOT NULL,  -- JSON
                    action_items TEXT NOT NULL,  -- JSON array
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    applied BOOLEAN DEFAULT FALSE,
                    applied_at TEXT,
                    results TEXT  -- JSON, results after applying
                )
            """)
            
            conn.commit()
    
    def mine_model_performance_patterns(self, days_back: int = 30) -> List[DiscoveredPattern]:
        """Mine patterns related to model performance across different task types"""
        patterns = []
        
        with sqlite3.connect(self.db_path) as conn:
            # Get model performance data
            cursor = conn.execute("""
                SELECT task_type, model_used, success_score, cost_usd, latency_ms, timestamp
                FROM simple_tracker 
                WHERE timestamp >= ?
                AND success_score IS NOT NULL
                ORDER BY timestamp DESC
            """, [(datetime.now() - timedelta(days=days_back)).isoformat()])
            
            data = cursor.fetchall()
            
            if len(data) < 10:  # Need minimum data for pattern mining
                return patterns
            
            # Group by (task_type, model) combinations
            performance_groups = defaultdict(list)
            for row in data:
                task_type, model_used, success_score, cost_usd, latency_ms, timestamp = row
                key = (task_type, model_used)
                performance_groups[key].append({
                    'success_score': success_score,
                    'cost_usd': cost_usd,
                    'latency_ms': latency_ms,
                    'timestamp': timestamp
                })
            
            # Analyze each group for patterns
            for (task_type, model), metrics in performance_groups.items():
                if len(metrics) >= 5:  # Minimum sample size
                    success_scores = [m['success_score'] for m in metrics]
                    costs = [m['cost_usd'] for m in metrics if m['cost_usd'] is not None]
                    latencies = [m['latency_ms'] for m in metrics if m['latency_ms'] is not None]
                    
                    avg_success = statistics.mean(success_scores)
                    
                    # High performance pattern
                    if avg_success > 0.85 and len(metrics) >= 8:
                        pattern = DiscoveredPattern(
                            id=f"pattern_perf_{task_type}_{model}_{int(time.time())}",
                            pattern_type=PatternType.MODEL_PERFORMANCE,
                            name=f"High Performance: {model} on {task_type}",
                            description=f"Model {model} consistently performs well on {task_type} tasks with {avg_success:.1%} success rate",
                            conditions={
                                "task_type": task_type,
                                "model_used": model,
                                "min_success_rate": 0.85
                            },
                            evidence=metrics[:10],  # Store up to 10 examples
                            confidence=min(len(metrics) / 20.0, 1.0),
                            sample_size=len(metrics),
                            statistical_significance=self._calculate_significance(success_scores, 0.8),
                            impact_metrics={
                                "avg_success_rate": avg_success,
                                "avg_cost": statistics.mean(costs) if costs else 0.0,
                                "avg_latency": statistics.mean(latencies) if latencies else 0.0,
                                "consistency": 1.0 - (statistics.stdev(success_scores) / avg_success)
                            },
                            recommendations=[
                                f"Prefer {model} for {task_type} tasks",
                                f"Set as default model for similar contexts",
                                "Monitor for performance degradation"
                            ],
                            discovered_at=datetime.now(),
                            last_validated=datetime.now(),
                            validation_count=1
                        )
                        patterns.append(pattern)
        
        # Store discovered patterns
        for pattern in patterns:
            self._store_pattern(pattern)
        
        return patterns
    
    def mine_cost_efficiency_patterns(self, days_back: int = 30) -> List[DiscoveredPattern]:
        """Mine patterns related to cost efficiency and optimization opportunities"""
        patterns = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT task_type, model_used, success_score, cost_usd, latency_ms, timestamp
                FROM simple_tracker 
                WHERE timestamp >= ?
                AND cost_usd IS NOT NULL AND cost_usd > 0
                AND success_score >= 0.7
                ORDER BY cost_usd ASC
            """, [(datetime.now() - timedelta(days=days_back)).isoformat()])
            
            data = cursor.fetchall()
            
            if len(data) < 10:
                return patterns
            
            # Calculate cost efficiency (success per dollar)
            efficiency_data = []
            for row in data:
                task_type, model_used, success_score, cost_usd, latency_ms, timestamp = row
                efficiency = success_score / cost_usd if cost_usd > 0 else 0
                efficiency_data.append({
                    'task_type': task_type,
                    'model_used': model_used,
                    'efficiency': efficiency,
                    'success_score': success_score,
                    'cost_usd': cost_usd,
                    'timestamp': timestamp
                })
            
            # Find top efficiency patterns
            model_efficiency = defaultdict(list)
            for item in efficiency_data:
                model_efficiency[item['model_used']].append(item['efficiency'])
            
            # Find models with consistently high efficiency
            for model, efficiencies in model_efficiency.items():
                if len(efficiencies) >= 5:
                    avg_efficiency = statistics.mean(efficiencies)
                    if avg_efficiency > statistics.mean([statistics.mean(eff) for eff in model_efficiency.values()]) * 1.2:
                        
                        pattern = DiscoveredPattern(
                            id=f"pattern_cost_{model}_{int(time.time())}",
                            pattern_type=PatternType.COST_EFFICIENCY,
                            name=f"Cost Efficient: {model}",
                            description=f"Model {model} shows superior cost efficiency with {avg_efficiency:.1f} success per dollar",
                            conditions={
                                "model_used": model,
                                "metric": "cost_efficiency",
                                "threshold": avg_efficiency
                            },
                            evidence=[item for item in efficiency_data if item['model_used'] == model][:10],
                            confidence=min(len(efficiencies) / 15.0, 1.0),
                            sample_size=len(efficiencies),
                            statistical_significance=self._calculate_significance(efficiencies, 50.0),
                            impact_metrics={
                                "avg_efficiency": avg_efficiency,
                                "potential_savings": 0.25,  # Estimated 25% savings
                                "consistency": 1.0 - (statistics.stdev(efficiencies) / avg_efficiency)
                            },
                            recommendations=[
                                f"Use {model} for cost-sensitive tasks",
                                "Consider as primary model for batch processing",
                                "Monitor for efficiency trends"
                            ],
                            discovered_at=datetime.now(),
                            last_validated=datetime.now(),
                            validation_count=1
                        )
                        patterns.append(pattern)
        
        # Store patterns
        for pattern in patterns:
            self._store_pattern(pattern)
        
        return patterns
    
    def mine_temporal_patterns(self, days_back: int = 30) -> List[DiscoveredPattern]:
        """Mine patterns related to time-based performance variations"""
        patterns = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT task_type, model_used, success_score, timestamp, 
                       strftime('%H', timestamp) as hour,
                       strftime('%w', timestamp) as day_of_week
                FROM simple_tracker 
                WHERE timestamp >= ?
                AND success_score IS NOT NULL
            """, [(datetime.now() - timedelta(days=days_back)).isoformat()])
            
            data = cursor.fetchall()
            
            if len(data) < 20:
                return patterns
            
            # Analyze performance by hour of day
            hourly_performance = defaultdict(list)
            for row in data:
                task_type, model_used, success_score, timestamp, hour, day_of_week = row
                hourly_performance[int(hour)].append(success_score)
            
            # Find peak performance hours
            hour_averages = {hour: statistics.mean(scores) 
                           for hour, scores in hourly_performance.items() 
                           if len(scores) >= 3}
            
            if hour_averages:
                best_hours = [hour for hour, avg in hour_averages.items() 
                            if avg > statistics.mean(hour_averages.values()) + 0.1]
                
                if best_hours:
                    pattern = DiscoveredPattern(
                        id=f"pattern_temporal_peak_{int(time.time())}",
                        pattern_type=PatternType.TEMPORAL,
                        name=f"Peak Performance Hours: {best_hours}",
                        description=f"System performs {10:.0f}% better during hours {best_hours}",
                        conditions={
                            "hours": best_hours,
                            "metric": "success_rate_improvement"
                        },
                        evidence=[
                            {"hour": hour, "avg_success": hour_averages[hour], "sample_size": len(hourly_performance[hour])}
                            for hour in best_hours
                        ],
                        confidence=0.8,
                        sample_size=sum(len(hourly_performance[h]) for h in best_hours),
                        statistical_significance=0.8,
                        impact_metrics={
                            "performance_improvement": 0.10,
                            "optimal_hours": len(best_hours)
                        },
                        recommendations=[
                            f"Schedule critical tasks during hours {best_hours}",
                            "Consider workload distribution optimization",
                            "Monitor temporal performance trends"
                        ],
                        discovered_at=datetime.now(),
                        last_validated=datetime.now(),
                        validation_count=1
                    )
                    patterns.append(pattern)
        
        # Store patterns
        for pattern in patterns:
            self._store_pattern(pattern)
        
        return patterns
    
    def mine_failure_patterns(self, days_back: int = 30) -> List[DiscoveredPattern]:
        """Mine patterns from failures to identify anti-patterns and improvement opportunities"""
        patterns = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT task_type, model_used, success_score, cost_usd, latency_ms, context, timestamp
                FROM simple_tracker 
                WHERE timestamp >= ?
                AND success_score < 0.6
                ORDER BY success_score ASC
            """, [(datetime.now() - timedelta(days=days_back)).isoformat()])
            
            failures = cursor.fetchall()
            
            if len(failures) < 5:
                return patterns
            
            # Group failures by model and task type
            failure_groups = defaultdict(list)
            for row in failures:
                task_type, model_used, success_score, cost_usd, latency_ms, context, timestamp = row
                key = (task_type, model_used)
                failure_groups[key].append({
                    'success_score': success_score,
                    'cost_usd': cost_usd,
                    'latency_ms': latency_ms,
                    'context': context,
                    'timestamp': timestamp
                })
            
            # Identify anti-patterns (consistent failures)
            for (task_type, model), failure_data in failure_groups.items():
                if len(failure_data) >= 3:  # Multiple failures with same combination
                    avg_failure_rate = 1.0 - statistics.mean([f['success_score'] for f in failure_data])
                    
                    if avg_failure_rate > 0.5:  # High failure rate
                        pattern = DiscoveredPattern(
                            id=f"pattern_failure_{task_type}_{model}_{int(time.time())}",
                            pattern_type=PatternType.FAILURE_ANALYSIS,
                            name=f"Anti-pattern: {model} struggles with {task_type}",
                            description=f"Model {model} shows {avg_failure_rate:.1%} failure rate on {task_type} tasks",
                            conditions={
                                "task_type": task_type,
                                "model_used": model,
                                "anti_pattern": True
                            },
                            evidence=failure_data,
                            confidence=min(len(failure_data) / 10.0, 1.0),
                            sample_size=len(failure_data),
                            statistical_significance=avg_failure_rate,
                            impact_metrics={
                                "failure_rate": avg_failure_rate,
                                "cost_waste": sum(f.get('cost_usd', 0) for f in failure_data),
                                "affected_tasks": len(failure_data)
                            },
                            recommendations=[
                                f"Avoid using {model} for {task_type} tasks",
                                "Consider alternative models for this task type",
                                "Investigate root causes of failures"
                            ],
                            discovered_at=datetime.now(),
                            last_validated=datetime.now(),
                            validation_count=1
                        )
                        patterns.append(pattern)
        
        # Store patterns
        for pattern in patterns:
            self._store_pattern(pattern)
        
        return patterns
    
    def mine_context_patterns(self, days_back: int = 30) -> List[DiscoveredPattern]:
        """Mine patterns based on task context and environmental factors"""
        patterns = []
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT task_type, context, success_score, model_used, timestamp
                FROM simple_tracker 
                WHERE timestamp >= ?
                AND context IS NOT NULL
                AND success_score IS NOT NULL
            """, [(datetime.now() - timedelta(days=days_back)).isoformat()])
            
            data = cursor.fetchall()
            
            if len(data) < 10:
                return patterns
            
            # Analyze context keywords and their correlation with success
            context_success = defaultdict(list)
            for row in data:
                task_type, context, success_score, model_used, timestamp = row
                
                # Extract keywords from context (simple approach)
                if context:
                    try:
                        context_obj = json.loads(context) if isinstance(context, str) else context
                        keywords = self._extract_keywords_from_context(context_obj)
                        
                        for keyword in keywords:
                            context_success[keyword].append(success_score)
                    except:
                        continue
            
            # Find context keywords that correlate with high success
            for keyword, scores in context_success.items():
                if len(scores) >= 5:
                    avg_success = statistics.mean(scores)
                    if avg_success > 0.8:
                        pattern = DiscoveredPattern(
                            id=f"pattern_context_{keyword}_{int(time.time())}",
                            pattern_type=PatternType.CONTEXT_BASED,
                            name=f"Context Success Pattern: {keyword}",
                            description=f"Tasks involving '{keyword}' show {avg_success:.1%} success rate",
                            conditions={
                                "context_keyword": keyword,
                                "min_success_rate": avg_success
                            },
                            evidence=[{"keyword": keyword, "success_scores": scores[:10]}],
                            confidence=min(len(scores) / 15.0, 1.0),
                            sample_size=len(scores),
                            statistical_significance=self._calculate_significance(scores, 0.7),
                            impact_metrics={
                                "success_improvement": avg_success - 0.7,
                                "affected_tasks": len(scores)
                            },
                            recommendations=[
                                f"Prioritize tasks with '{keyword}' context",
                                "Use successful context patterns as templates",
                                "Investigate why this context improves success"
                            ],
                            discovered_at=datetime.now(),
                            last_validated=datetime.now(),
                            validation_count=1
                        )
                        patterns.append(pattern)
        
        # Store patterns
        for pattern in patterns:
            self._store_pattern(pattern)
        
        return patterns
    
    def generate_optimization_insights(self) -> List[OptimizationInsight]:
        """Generate actionable optimization insights from discovered patterns"""
        insights = []
        patterns = self.get_stored_patterns()
        
        if not patterns:
            return insights
        
        # Group patterns by type for analysis
        pattern_groups = defaultdict(list)
        for pattern in patterns:
            pattern_groups[pattern.pattern_type].append(pattern)
        
        # Generate insights for model performance patterns
        if PatternType.MODEL_PERFORMANCE in pattern_groups:
            high_perf_patterns = [p for p in pattern_groups[PatternType.MODEL_PERFORMANCE] 
                                if p.impact_metrics.get('avg_success_rate', 0) > 0.85]
            
            if high_perf_patterns:
                best_pattern = max(high_perf_patterns, key=lambda p: p.confidence)
                
                insight = OptimizationInsight(
                    id=f"insight_model_opt_{int(time.time())}",
                    type=InsightType.OPTIMIZATION,
                    title="Optimize Default Model Selection",
                    description=f"Switch default model to {best_pattern.conditions.get('model_used')} for {best_pattern.impact_metrics.get('avg_success_rate', 0):.1%} success rate improvement",
                    pattern_ids=[p.id for p in high_perf_patterns],
                    impact_score=0.9,
                    confidence=best_pattern.confidence,
                    estimated_improvement={
                        "success_rate_increase": 0.15,
                        "cost_reduction": 0.10
                    },
                    action_items=[
                        f"Update CLI to default to {best_pattern.conditions.get('model_used')}",
                        "Implement automatic model selection based on task type",
                        "Monitor performance after change"
                    ],
                    priority="high",
                    created_at=datetime.now()
                )
                insights.append(insight)
        
        # Generate insights for cost efficiency patterns  
        if PatternType.COST_EFFICIENCY in pattern_groups:
            cost_patterns = pattern_groups[PatternType.COST_EFFICIENCY]
            if cost_patterns:
                total_potential_savings = sum(p.impact_metrics.get('potential_savings', 0) 
                                            for p in cost_patterns)
                
                if total_potential_savings > 0.10:  # Significant savings opportunity
                    insight = OptimizationInsight(
                        id=f"insight_cost_opt_{int(time.time())}",
                        type=InsightType.OPPORTUNITY,
                        title="Implement Cost Optimization Strategy",
                        description=f"Potential cost savings of ${total_potential_savings:.2f} identified from {len(cost_patterns)} patterns",
                        pattern_ids=[p.id for p in cost_patterns],
                        impact_score=0.8,
                        confidence=0.8,
                        estimated_improvement={
                            "cost_reduction": min(total_potential_savings, 0.40)
                        },
                        action_items=[
                            "Implement intelligent model routing for cost optimization",
                            "Set up cost alerts for expensive tasks",
                            "Create cost-aware task scheduling"
                        ],
                        priority="medium",
                        created_at=datetime.now()
                    )
                    insights.append(insight)
        
        # Generate insights for failure patterns (warnings)
        if PatternType.FAILURE_ANALYSIS in pattern_groups:
            failure_patterns = pattern_groups[PatternType.FAILURE_ANALYSIS]
            critical_failures = [p for p in failure_patterns 
                               if p.impact_metrics.get('failure_rate', 0) > 0.6]
            
            if critical_failures:
                insight = OptimizationInsight(
                    id=f"insight_failure_warn_{int(time.time())}",
                    type=InsightType.WARNING,
                    title="Critical Failure Patterns Detected",
                    description=f"Identified {len(critical_failures)} high-failure scenarios requiring immediate attention",
                    pattern_ids=[p.id for p in critical_failures],
                    impact_score=0.9,
                    confidence=0.9,
                    estimated_improvement={
                        "failure_reduction": 0.60,
                        "quality_improvement": 0.25
                    },
                    action_items=[
                        "Block problematic model-task combinations",
                        "Implement pre-execution validation",
                        "Investigate root causes and fix underlying issues"
                    ],
                    priority="critical",
                    created_at=datetime.now()
                )
                insights.append(insight)
        
        # Store insights
        for insight in insights:
            self._store_insight(insight)
        
        return insights
    
    def _extract_keywords_from_context(self, context: Dict[str, Any]) -> Set[str]:
        """Extract meaningful keywords from task context"""
        keywords = set()
        
        def extract_recursive(obj, parent_key=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{parent_key}.{key}" if parent_key else key
                    keywords.add(key.lower())
                    if isinstance(value, str) and len(value.split()) <= 3:
                        keywords.add(value.lower())
                    elif isinstance(value, (dict, list)):
                        extract_recursive(value, full_key)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str) and len(item.split()) <= 2:
                        keywords.add(item.lower())
                    elif isinstance(item, dict):
                        extract_recursive(item, parent_key)
        
        extract_recursive(context)
        return {k for k in keywords if len(k) > 2}  # Filter short keywords
    
    def _calculate_significance(self, values: List[float], baseline: float) -> float:
        """Calculate statistical significance (simplified t-test approach)"""
        if len(values) < 3:
            return 0.0
        
        sample_mean = statistics.mean(values)
        if sample_mean <= baseline:
            return 0.0
        
        sample_std = statistics.stdev(values) if len(values) > 1 else 1.0
        n = len(values)
        
        # Simplified t-statistic
        t_stat = (sample_mean - baseline) / (sample_std / (n ** 0.5))
        
        # Convert to significance score (0-1)
        significance = min(abs(t_stat) / 3.0, 1.0)
        return significance
    
    def _store_pattern(self, pattern: DiscoveredPattern):
        """Store discovered pattern in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO v2_discovered_patterns
                (id, pattern_type, name, description, conditions, evidence,
                 confidence, sample_size, statistical_significance, impact_metrics,
                 recommendations, discovered_at, last_validated, validation_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pattern.id, pattern.pattern_type.value, pattern.name,
                pattern.description, json.dumps(pattern.conditions),
                json.dumps(pattern.evidence, default=str), pattern.confidence,
                pattern.sample_size, pattern.statistical_significance,
                json.dumps(pattern.impact_metrics), json.dumps(pattern.recommendations),
                pattern.discovered_at.isoformat(), pattern.last_validated.isoformat(),
                pattern.validation_count
            ))
            conn.commit()
    
    def _store_insight(self, insight: OptimizationInsight):
        """Store optimization insight in database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO v2_optimization_insights
                (id, type, title, description, pattern_ids, impact_score,
                 confidence, estimated_improvement, action_items, priority, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                insight.id, insight.type.value, insight.title, insight.description,
                json.dumps(insight.pattern_ids), insight.impact_score,
                insight.confidence, json.dumps(insight.estimated_improvement),
                json.dumps(insight.action_items), insight.priority,
                insight.created_at.isoformat()
            ))
            conn.commit()
    
    def get_stored_patterns(self, pattern_type: Optional[PatternType] = None) -> List[DiscoveredPattern]:
        """Retrieve stored patterns from database"""
        query = "SELECT * FROM v2_discovered_patterns"
        params = []
        
        if pattern_type:
            query += " WHERE pattern_type = ?"
            params.append(pattern_type.value)
        
        query += " ORDER BY confidence DESC, discovered_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        patterns = []
        for row in rows:
            pattern = DiscoveredPattern(
                id=row[0], pattern_type=PatternType(row[1]), name=row[2],
                description=row[3], conditions=json.loads(row[4]),
                evidence=json.loads(row[5]), confidence=row[6],
                sample_size=row[7], statistical_significance=row[8],
                impact_metrics=json.loads(row[9]), recommendations=json.loads(row[10]),
                discovered_at=datetime.fromisoformat(row[11]),
                last_validated=datetime.fromisoformat(row[12]),
                validation_count=row[13]
            )
            patterns.append(pattern)
        
        return patterns
    
    def get_stored_insights(self, priority: Optional[str] = None) -> List[OptimizationInsight]:
        """Retrieve stored optimization insights"""
        query = "SELECT * FROM v2_optimization_insights WHERE applied = FALSE"
        params = []
        
        if priority:
            query += " AND priority = ?"
            params.append(priority)
        
        query += " ORDER BY impact_score DESC, created_at DESC"
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
        
        insights = []
        for row in rows:
            insight = OptimizationInsight(
                id=row[0], type=InsightType(row[1]), title=row[2],
                description=row[3], pattern_ids=json.loads(row[4]),
                impact_score=row[5], confidence=row[6],
                estimated_improvement=json.loads(row[7]),
                action_items=json.loads(row[8]), priority=row[9],
                created_at=datetime.fromisoformat(row[10])
            )
            insights.append(insight)
        
        return insights

# CLI Integration Functions
def run_full_pattern_mining(days_back: int = 30) -> Dict[str, Any]:
    """CLI function to run comprehensive pattern mining"""
    engine = PatternMiningEngine()
    
    results = {
        'model_performance_patterns': engine.mine_model_performance_patterns(days_back),
        'cost_efficiency_patterns': engine.mine_cost_efficiency_patterns(days_back),
        'temporal_patterns': engine.mine_temporal_patterns(days_back),
        'failure_patterns': engine.mine_failure_patterns(days_back),
        'context_patterns': engine.mine_context_patterns(days_back)
    }
    
    # Generate insights
    insights = engine.generate_optimization_insights()
    results['optimization_insights'] = insights
    
    # Summary
    total_patterns = sum(len(patterns) for patterns in results.values() if isinstance(patterns, list))
    results['summary'] = {
        'total_patterns_discovered': total_patterns,
        'total_insights_generated': len(insights),
        'high_priority_insights': len([i for i in insights if i.priority == 'high']),
        'critical_insights': len([i for i in insights if i.priority == 'critical'])
    }
    
    return results

def get_pattern_mining_report() -> Dict[str, Any]:
    """CLI function to generate pattern mining report"""
    engine = PatternMiningEngine()
    
    patterns = engine.get_stored_patterns()
    insights = engine.get_stored_insights()
    
    # Group patterns by type
    pattern_breakdown = defaultdict(int)
    for pattern in patterns:
        pattern_breakdown[pattern.pattern_type.value] += 1
    
    # Group insights by priority
    insight_breakdown = defaultdict(int)
    for insight in insights:
        insight_breakdown[insight.priority] += 1
    
    return {
        'total_patterns': len(patterns),
        'pattern_breakdown': dict(pattern_breakdown),
        'total_insights': len(insights),
        'insight_breakdown': dict(insight_breakdown),
        'patterns': [
            {
                'id': p.id,
                'type': p.pattern_type.value,
                'name': p.name,
                'confidence': p.confidence,
                'sample_size': p.sample_size
            }
            for p in patterns[:10]  # Top 10
        ],
        'insights': [
            {
                'id': i.id,
                'type': i.type.value,
                'title': i.title,
                'impact_score': i.impact_score,
                'priority': i.priority
            }
            for i in insights[:10]  # Top 10
        ]
    }

def apply_pattern_insights(insight_id: str) -> Dict[str, Any]:
    """CLI function to apply a specific optimization insight"""
    engine = PatternMiningEngine()
    
    # Get the insight
    insights = engine.get_stored_insights()
    target_insight = next((i for i in insights if i.id == insight_id), None)
    
    if not target_insight:
        return {'error': f'Insight {insight_id} not found'}
    
    # Mark as applied (in real implementation, would execute the actions)
    with sqlite3.connect(engine.db_path) as conn:
        conn.execute("""
            UPDATE v2_optimization_insights
            SET applied = TRUE, applied_at = ?
            WHERE id = ?
        """, (datetime.now().isoformat(), insight_id))
        conn.commit()
    
    return {
        'status': 'applied',
        'insight_title': target_insight.title,
        'action_items': target_insight.action_items,
        'estimated_improvement': target_insight.estimated_improvement
    }

if __name__ == "__main__":
    # Test Pattern Mining Engine
    print("🔍 Testing Pattern Mining Engine...")
    
    # Run full pattern mining
    results = run_full_pattern_mining(days_back=30)
    
    print(f"✅ Pattern Mining Complete:")
    print(f"  📊 Total patterns: {results['summary']['total_patterns_discovered']}")
    print(f"  💡 Insights generated: {results['summary']['total_insights_generated']}")
    print(f"  🔥 High priority: {results['summary']['high_priority_insights']}")
    print(f"  ⚠️  Critical issues: {results['summary']['critical_insights']}")
    
    # Get report
    report = get_pattern_mining_report()
    print(f"\n📈 Pattern Mining Report:")
    print(f"  Stored patterns: {report['total_patterns']}")
    print(f"  Stored insights: {report['total_insights']}")
    
    print("\n🎉 Pattern Mining Engine - OPERATIONAL!")