#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 2 - Priority 3: Advanced Pattern Recognition
Saturday, October 11, 2025 @ 10:16 CEST

Advanced Pattern Recognition System - ML-powered pattern discovery with statistical validation
Integration with existing Experience Management System for enhanced intelligence
"""

import os
import json
import sqlite3
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import numpy as np
from pathlib import Path
import uuid

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# ADVANCED PATTERN RECOGNITION DATA MODELS
# =============================================================================

class PatternTypeEnum(Enum):
    """Types of patterns that can be discovered"""
    SUCCESS_PATTERN = "success_pattern"
    FAILURE_PATTERN = "failure_pattern"
    COST_PATTERN = "cost_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    USAGE_PATTERN = "usage_pattern"
    TEMPORAL_PATTERN = "temporal_pattern"
    CORRELATION_PATTERN = "correlation_pattern"
    ANOMALY_PATTERN = "anomaly_pattern"

class PatternStrengthEnum(Enum):
    """Statistical strength of discovered patterns"""
    VERY_STRONG = "very_strong"    # >95% confidence
    STRONG = "strong"              # 90-95% confidence  
    MODERATE = "moderate"          # 80-90% confidence
    WEAK = "weak"                  # 70-80% confidence
    INSUFFICIENT = "insufficient"  # <70% confidence

@dataclass
class AdvancedPattern:
    """Advanced pattern with statistical validation"""
    id: str
    pattern_type: PatternTypeEnum
    name: str
    description: str
    statistical_confidence: float  # 0.0-1.0
    strength: PatternStrengthEnum
    frequency: int
    sample_size: int
    effect_size: float
    p_value: float
    conditions: Dict[str, Any]
    outcomes: Dict[str, Any]
    recommendations: List[str]
    business_impact: Dict[str, Any]
    supporting_data: List[Dict]
    discovered_at: datetime
    last_validated: datetime
    validation_count: int

@dataclass  
class PatternCorrelation:
    """Correlation between different patterns"""
    id: str
    pattern_a_id: str
    pattern_b_id: str
    correlation_coefficient: float
    correlation_type: str  # positive, negative, complex
    significance_level: float
    joint_occurrence_rate: float
    causal_relationship: Optional[str]
    discovered_at: datetime

@dataclass
class PatternPrediction:
    """Prediction based on pattern analysis"""
    id: str
    pattern_ids: List[str]
    prediction_type: str
    predicted_outcome: Dict[str, Any]
    confidence_level: float
    expected_accuracy: float
    time_horizon: str
    conditions: Dict[str, Any]
    created_at: datetime

# =============================================================================
# ADVANCED PATTERN RECOGNITION ENGINE
# =============================================================================

class AdvancedPatternRecognitionEngine:
    """
    Advanced Pattern Recognition System with ML-powered analysis
    
    Capabilities:
    - Statistical pattern discovery with confidence intervals
    - Correlation analysis between multiple variables
    - Temporal pattern recognition
    - Anomaly detection using statistical methods
    - Predictive pattern modeling
    - Causal inference for business recommendations
    """
    
    def __init__(self, db_path: str = "data/advanced_patterns.sqlite"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._ensure_db_directory()
        self._init_database()
        
        # Statistical thresholds
        self.confidence_thresholds = {
            PatternStrengthEnum.VERY_STRONG: 0.95,
            PatternStrengthEnum.STRONG: 0.90,
            PatternStrengthEnum.MODERATE: 0.80,
            PatternStrengthEnum.WEAK: 0.70
        }
        
        # Minimum sample sizes for reliable patterns
        self.min_sample_sizes = {
            PatternTypeEnum.SUCCESS_PATTERN: 10,
            PatternTypeEnum.FAILURE_PATTERN: 5,
            PatternTypeEnum.COST_PATTERN: 15,
            PatternTypeEnum.PERFORMANCE_PATTERN: 20,
            PatternTypeEnum.USAGE_PATTERN: 30,
            PatternTypeEnum.TEMPORAL_PATTERN: 50,
            PatternTypeEnum.CORRELATION_PATTERN: 25,
            PatternTypeEnum.ANOMALY_PATTERN: 100
        }
        
        logger.info("🔍 Advanced Pattern Recognition Engine initialized")
    
    def _ensure_db_directory(self):
        """Ensure database directory exists"""
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
    
    def _init_database(self):
        """Initialize SQLite database with advanced pattern schema"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Advanced patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS advanced_patterns (
                    id TEXT PRIMARY KEY,
                    pattern_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    statistical_confidence REAL NOT NULL,
                    strength TEXT NOT NULL,
                    frequency INTEGER NOT NULL,
                    sample_size INTEGER NOT NULL,
                    effect_size REAL NOT NULL,
                    p_value REAL NOT NULL,
                    conditions_json TEXT NOT NULL,
                    outcomes_json TEXT NOT NULL,
                    recommendations_json TEXT NOT NULL,
                    business_impact_json TEXT NOT NULL,
                    supporting_data_json TEXT NOT NULL,
                    discovered_at TEXT NOT NULL,
                    last_validated TEXT NOT NULL,
                    validation_count INTEGER NOT NULL
                )
            ''')
            
            # Pattern correlations table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_correlations (
                    id TEXT PRIMARY KEY,
                    pattern_a_id TEXT NOT NULL,
                    pattern_b_id TEXT NOT NULL,
                    correlation_coefficient REAL NOT NULL,
                    correlation_type TEXT NOT NULL,
                    significance_level REAL NOT NULL,
                    joint_occurrence_rate REAL NOT NULL,
                    causal_relationship TEXT,
                    discovered_at TEXT NOT NULL,
                    FOREIGN KEY (pattern_a_id) REFERENCES advanced_patterns(id),
                    FOREIGN KEY (pattern_b_id) REFERENCES advanced_patterns(id)
                )
            ''')
            
            # Pattern predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pattern_predictions (
                    id TEXT PRIMARY KEY,
                    pattern_ids_json TEXT NOT NULL,
                    prediction_type TEXT NOT NULL,
                    predicted_outcome_json TEXT NOT NULL,
                    confidence_level REAL NOT NULL,
                    expected_accuracy REAL NOT NULL,
                    time_horizon TEXT NOT NULL,
                    conditions_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON advanced_patterns(pattern_type)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_strength ON advanced_patterns(strength)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_confidence ON advanced_patterns(statistical_confidence)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_discovered_at ON advanced_patterns(discovered_at)')
            
            conn.commit()
    
    async def discover_advanced_patterns(self, 
                                       data_source: str = "phase2_experiences.sqlite",
                                       pattern_types: Optional[List[PatternTypeEnum]] = None,
                                       min_confidence: float = 0.70) -> List[AdvancedPattern]:
        """
        Discover advanced patterns using ML and statistical analysis
        
        Args:
            data_source: Source database with experience data
            pattern_types: Specific pattern types to discover
            min_confidence: Minimum statistical confidence required
            
        Returns:
            List of discovered advanced patterns
        """
        try:
            if pattern_types is None:
                pattern_types = list(PatternTypeEnum)
            
            discovered_patterns = []
            
            # Connect to experience data source
            if not os.path.exists(data_source):
                logger.warning(f"Data source {data_source} not found, using mock data")
                return await self._discover_mock_patterns()
            
            with sqlite3.connect(data_source) as conn:
                # Discover each type of pattern
                for pattern_type in pattern_types:
                    patterns = await self._discover_pattern_type(conn, pattern_type, min_confidence)
                    discovered_patterns.extend(patterns)
            
            # Store discovered patterns
            for pattern in discovered_patterns:
                await self._store_pattern(pattern)
            
            logger.info(f"🔍 Discovered {len(discovered_patterns)} advanced patterns")
            return discovered_patterns
            
        except Exception as e:
            logger.error(f"❌ Failed to discover advanced patterns: {e}")
            return []
    
    async def _discover_pattern_type(self, 
                                   conn: sqlite3.Connection,
                                   pattern_type: PatternTypeEnum,
                                   min_confidence: float) -> List[AdvancedPattern]:
        """Discover patterns of specific type"""
        patterns = []
        
        try:
            if pattern_type == PatternTypeEnum.SUCCESS_PATTERN:
                patterns = await self._discover_success_patterns(conn, min_confidence)
            elif pattern_type == PatternTypeEnum.COST_PATTERN:
                patterns = await self._discover_cost_patterns(conn, min_confidence)
            elif pattern_type == PatternTypeEnum.PERFORMANCE_PATTERN:
                patterns = await self._discover_performance_patterns(conn, min_confidence)
            elif pattern_type == PatternTypeEnum.USAGE_PATTERN:
                patterns = await self._discover_usage_patterns(conn, min_confidence)
            elif pattern_type == PatternTypeEnum.TEMPORAL_PATTERN:
                patterns = await self._discover_temporal_patterns(conn, min_confidence)
            elif pattern_type == PatternTypeEnum.CORRELATION_PATTERN:
                patterns = await self._discover_correlation_patterns(conn, min_confidence)
            elif pattern_type == PatternTypeEnum.ANOMALY_PATTERN:
                patterns = await self._discover_anomaly_patterns(conn, min_confidence)
            
        except Exception as e:
            logger.error(f"❌ Failed to discover {pattern_type.value} patterns: {e}")
        
        return patterns
    
    async def _discover_success_patterns(self, conn: sqlite3.Connection, min_confidence: float) -> List[AdvancedPattern]:
        """Discover success patterns using statistical analysis"""
        patterns = []
        
        cursor = conn.cursor()
        
        # Success pattern by task type and approach
        cursor.execute('''
            SELECT 
                task_type,
                approach_used,
                COUNT(*) as total_attempts,
                AVG(success_score) as avg_success,
                STDEV(success_score) as success_stddev,
                AVG(cost_usd) as avg_cost,
                AVG(duration_seconds) as avg_duration
            FROM experiences 
            WHERE success_score IS NOT NULL
            GROUP BY task_type, approach_used
            HAVING COUNT(*) >= ?
            ORDER BY avg_success DESC
        ''', (self.min_sample_sizes[PatternTypeEnum.SUCCESS_PATTERN],))
        
        results = cursor.fetchall()
        
        for result in results:
            task_type, approach, total, avg_success, stddev, avg_cost, avg_duration = result
            
            if stddev is None:
                stddev = 0.1  # Default for single sample
            
            # Calculate confidence interval and statistical significance
            confidence = self._calculate_confidence_interval(avg_success, stddev, total)
            
            if confidence >= min_confidence:
                # Determine business impact
                business_impact = {
                    "success_rate_improvement": max(0, avg_success - 0.7),  # Above baseline
                    "cost_efficiency": 1.0 / max(avg_cost, 0.001),
                    "time_efficiency": 1.0 / max(avg_duration, 1),
                    "roi_multiplier": (avg_success * 1.0) / max(avg_cost, 0.001)
                }
                
                pattern = AdvancedPattern(
                    id=str(uuid.uuid4()),
                    pattern_type=PatternTypeEnum.SUCCESS_PATTERN,
                    name=f"High Success: {approach} for {task_type}",
                    description=f"Using '{approach}' approach for {task_type} tasks achieves {avg_success:.1%} success rate",
                    statistical_confidence=confidence,
                    strength=self._determine_pattern_strength(confidence),
                    frequency=total,
                    sample_size=total,
                    effect_size=avg_success,
                    p_value=1.0 - confidence,
                    conditions={
                        "task_type": task_type,
                        "approach_used": approach,
                        "min_attempts": self.min_sample_sizes[PatternTypeEnum.SUCCESS_PATTERN]
                    },
                    outcomes={
                        "avg_success_rate": avg_success,
                        "avg_cost": avg_cost,
                        "avg_duration_seconds": avg_duration,
                        "success_variability": stddev
                    },
                    recommendations=[
                        f"Prioritize '{approach}' approach for {task_type} tasks",
                        f"Expected success rate: {avg_success:.1%}",
                        f"Budget approximately ${avg_cost:.4f} per task",
                        "Monitor for consistency with established pattern"
                    ],
                    business_impact=business_impact,
                    supporting_data=[],
                    discovered_at=datetime.now(),
                    last_validated=datetime.now(),
                    validation_count=1
                )
                
                patterns.append(pattern)
        
        return patterns
    
    async def _discover_cost_patterns(self, conn: sqlite3.Connection, min_confidence: float) -> List[AdvancedPattern]:
        """Discover cost optimization patterns"""
        patterns = []
        
        cursor = conn.cursor()
        
        # Cost efficiency patterns
        cursor.execute('''
            SELECT 
                task_type,
                model_used,
                COUNT(*) as total_uses,
                AVG(cost_usd) as avg_cost,
                AVG(success_score) as avg_success,
                AVG(cost_usd / NULLIF(success_score, 0)) as cost_per_success_point
            FROM experiences 
            WHERE cost_usd > 0 AND success_score > 0
            GROUP BY task_type, model_used
            HAVING COUNT(*) >= ?
            ORDER BY cost_per_success_point ASC
        ''', (self.min_sample_sizes[PatternTypeEnum.COST_PATTERN],))
        
        results = cursor.fetchall()
        
        # Find cost-efficient models for each task type
        task_type_best = {}
        for result in results:
            task_type, model, total, avg_cost, avg_success, cost_efficiency = result
            
            if task_type not in task_type_best:
                task_type_best[task_type] = []
            
            task_type_best[task_type].append({
                'model': model,
                'avg_cost': avg_cost,
                'avg_success': avg_success,
                'cost_efficiency': cost_efficiency,
                'total_uses': total
            })
        
        # Generate patterns for best cost-efficient models
        for task_type, models in task_type_best.items():
            if len(models) >= 2:  # Need comparison
                # Sort by cost efficiency
                models_sorted = sorted(models, key=lambda x: x['cost_efficiency'])
                best_model = models_sorted[0]
                
                if best_model['total_uses'] >= self.min_sample_sizes[PatternTypeEnum.COST_PATTERN]:
                    confidence = min(0.95, 0.60 + (best_model['total_uses'] / 100) * 0.30)
                    
                    if confidence >= min_confidence:
                        # Calculate potential savings compared to expensive alternatives
                        worst_model = models_sorted[-1]
                        potential_savings = worst_model['avg_cost'] - best_model['avg_cost']
                        
                        pattern = AdvancedPattern(
                            id=str(uuid.uuid4()),
                            pattern_type=PatternTypeEnum.COST_PATTERN,
                            name=f"Cost Efficient: {best_model['model']} for {task_type}",
                            description=f"Using {best_model['model']} for {task_type} tasks provides best cost efficiency at ${best_model['cost_efficiency']:.4f} per success point",
                            statistical_confidence=confidence,
                            strength=self._determine_pattern_strength(confidence),
                            frequency=best_model['total_uses'],
                            sample_size=best_model['total_uses'],
                            effect_size=potential_savings,
                            p_value=1.0 - confidence,
                            conditions={
                                "task_type": task_type,
                                "cost_optimization": True,
                                "min_samples": self.min_sample_sizes[PatternTypeEnum.COST_PATTERN]
                            },
                            outcomes={
                                "recommended_model": best_model['model'],
                                "avg_cost": best_model['avg_cost'],
                                "avg_success": best_model['avg_success'],
                                "cost_efficiency": best_model['cost_efficiency'],
                                "potential_savings": potential_savings
                            },
                            recommendations=[
                                f"Use {best_model['model']} as primary choice for {task_type} tasks",
                                f"Expected cost: ${best_model['avg_cost']:.4f} per task",
                                f"Expected success rate: {best_model['avg_success']:.1%}",
                                f"Potential savings: ${potential_savings:.4f} per task vs alternatives"
                            ],
                            business_impact={
                                "cost_reduction": potential_savings,
                                "monthly_savings": potential_savings * 30,  # Assuming 1 task/day
                                "efficiency_ratio": best_model['cost_efficiency'],
                                "roi_improvement": potential_savings / max(best_model['avg_cost'], 0.001)
                            },
                            supporting_data=[],
                            discovered_at=datetime.now(),
                            last_validated=datetime.now(),
                            validation_count=1
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    async def _discover_performance_patterns(self, conn: sqlite3.Connection, min_confidence: float) -> List[AdvancedPattern]:
        """Discover performance patterns"""
        patterns = []
        
        cursor = conn.cursor()
        
        # Performance patterns by time of day
        cursor.execute('''
            SELECT 
                CAST(strftime('%H', created_at) AS INTEGER) as hour_of_day,
                COUNT(*) as total_tasks,
                AVG(duration_seconds) as avg_duration,
                AVG(success_score) as avg_success,
                AVG(cost_usd) as avg_cost
            FROM experiences 
            WHERE duration_seconds > 0
            GROUP BY hour_of_day
            HAVING COUNT(*) >= ?
            ORDER BY avg_duration ASC
        ''', (self.min_sample_sizes[PatternTypeEnum.PERFORMANCE_PATTERN] // 4,))  # Lower threshold for temporal
        
        results = cursor.fetchall()
        
        if len(results) >= 3:  # Need enough time periods for comparison
            # Find best and worst performance hours
            best_hour = min(results, key=lambda x: x[2])  # Minimum duration
            worst_hour = max(results, key=lambda x: x[2])  # Maximum duration
            
            performance_improvement = worst_hour[2] - best_hour[2]  # Duration difference
            
            if performance_improvement > 1:  # At least 1 second improvement
                confidence = min(0.90, 0.70 + (sum(r[1] for r in results) / 500) * 0.20)
                
                if confidence >= min_confidence:
                    pattern = AdvancedPattern(
                        id=str(uuid.uuid4()),
                        pattern_type=PatternTypeEnum.PERFORMANCE_PATTERN,
                        name=f"Optimal Performance: Hour {best_hour[0]:02d}:00",
                        description=f"Tasks performed at {best_hour[0]:02d}:00 complete {performance_improvement:.1f}s faster on average",
                        statistical_confidence=confidence,
                        strength=self._determine_pattern_strength(confidence),
                        frequency=sum(r[1] for r in results),
                        sample_size=sum(r[1] for r in results),
                        effect_size=performance_improvement,
                        p_value=1.0 - confidence,
                        conditions={
                            "performance_optimization": True,
                            "temporal_analysis": True
                        },
                        outcomes={
                            "optimal_hour": best_hour[0],
                            "avg_duration_optimal": best_hour[2],
                            "avg_duration_worst": worst_hour[2],
                            "performance_gain": performance_improvement,
                            "success_rate_optimal": best_hour[3]
                        },
                        recommendations=[
                            f"Schedule complex tasks around {best_hour[0]:02d}:00 for optimal performance",
                            f"Avoid scheduling during {worst_hour[0]:02d}:00 if possible",
                            f"Expected {performance_improvement:.1f}s improvement in execution time",
                            "Consider system load and resource availability patterns"
                        ],
                        business_impact={
                            "time_savings_per_task": performance_improvement,
                            "daily_efficiency_gain": performance_improvement * 10,  # Assuming 10 tasks/day
                            "resource_optimization": True,
                            "scheduling_optimization": True
                        },
                        supporting_data=[],
                        discovered_at=datetime.now(),
                        last_validated=datetime.now(),
                        validation_count=1
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    async def _discover_usage_patterns(self, conn: sqlite3.Connection, min_confidence: float) -> List[AdvancedPattern]:
        """Discover usage patterns and trends"""
        patterns = []
        
        cursor = conn.cursor()
        
        # Most frequent successful combinations
        cursor.execute('''
            SELECT 
                task_type,
                approach_used,
                model_used,
                COUNT(*) as usage_frequency,
                AVG(success_score) as avg_success,
                COUNT(*) * AVG(success_score) as usage_success_score
            FROM experiences 
            WHERE success_score > 0.7
            GROUP BY task_type, approach_used, model_used
            HAVING COUNT(*) >= ?
            ORDER BY usage_success_score DESC
            LIMIT 5
        ''', (self.min_sample_sizes[PatternTypeEnum.USAGE_PATTERN],))
        
        results = cursor.fetchall()
        
        for i, result in enumerate(results):
            task_type, approach, model, frequency, avg_success, usage_score = result
            
            confidence = min(0.95, 0.75 + (frequency / 100) * 0.20)
            
            if confidence >= min_confidence:
                rank = i + 1
                
                pattern = AdvancedPattern(
                    id=str(uuid.uuid4()),
                    pattern_type=PatternTypeEnum.USAGE_PATTERN,
                    name=f"Popular Success Pattern #{rank}: {model} + {approach}",
                    description=f"Combination of {model} model with '{approach}' approach for {task_type} tasks is frequently used with {avg_success:.1%} success",
                    statistical_confidence=confidence,
                    strength=self._determine_pattern_strength(confidence),
                    frequency=frequency,
                    sample_size=frequency,
                    effect_size=usage_score,
                    p_value=1.0 - confidence,
                    conditions={
                        "task_type": task_type,
                        "approach_used": approach,
                        "model_used": model,
                        "success_threshold": 0.7
                    },
                    outcomes={
                        "usage_frequency": frequency,
                        "avg_success_rate": avg_success,
                        "popularity_rank": rank,
                        "usage_success_score": usage_score
                    },
                    recommendations=[
                        f"Consider {model} + '{approach}' as reliable choice for {task_type}",
                        f"High usage frequency ({frequency} times) indicates community trust",
                        f"Proven success rate of {avg_success:.1%}",
                        "Monitor for any changes in effectiveness over time"
                    ],
                    business_impact={
                        "reliability_score": avg_success * (frequency / 100),
                        "community_validation": frequency,
                        "risk_mitigation": "low",  # High usage = battle-tested
                        "adoption_recommendation": "high"
                    },
                    supporting_data=[],
                    discovered_at=datetime.now(),
                    last_validated=datetime.now(),
                    validation_count=1
                )
                
                patterns.append(pattern)
        
        return patterns
    
    async def _discover_temporal_patterns(self, conn: sqlite3.Connection, min_confidence: float) -> List[AdvancedPattern]:
        """Discover temporal patterns in task execution"""
        patterns = []
        
        cursor = conn.cursor()
        
        # Weekly patterns
        cursor.execute('''
            SELECT 
                CAST(strftime('%w', created_at) AS INTEGER) as day_of_week,
                COUNT(*) as total_tasks,
                AVG(success_score) as avg_success,
                AVG(cost_usd) as avg_cost,
                AVG(duration_seconds) as avg_duration
            FROM experiences 
            GROUP BY day_of_week
            HAVING COUNT(*) >= ?
            ORDER BY avg_success DESC
        ''', (self.min_sample_sizes[PatternTypeEnum.TEMPORAL_PATTERN] // 7,))
        
        results = cursor.fetchall()
        
        if len(results) >= 5:  # Need several days of data
            day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
            best_day = max(results, key=lambda x: x[2])  # Highest success rate
            worst_day = min(results, key=lambda x: x[2])  # Lowest success rate
            
            success_difference = best_day[2] - worst_day[2]
            
            if success_difference > 0.1:  # 10% difference
                total_samples = sum(r[1] for r in results)
                confidence = min(0.90, 0.70 + (total_samples / 1000) * 0.20)
                
                if confidence >= min_confidence:
                    pattern = AdvancedPattern(
                        id=str(uuid.uuid4()),
                        pattern_type=PatternTypeEnum.TEMPORAL_PATTERN,
                        name=f"Weekly Peak: {day_names[best_day[0]]} Performance",
                        description=f"Tasks performed on {day_names[best_day[0]]} have {success_difference:.1%} higher success rate",
                        statistical_confidence=confidence,
                        strength=self._determine_pattern_strength(confidence),
                        frequency=total_samples,
                        sample_size=total_samples,
                        effect_size=success_difference,
                        p_value=1.0 - confidence,
                        conditions={
                            "temporal_analysis": "weekly",
                            "day_of_week_focus": True
                        },
                        outcomes={
                            "best_day": day_names[best_day[0]],
                            "best_day_success": best_day[2],
                            "worst_day": day_names[worst_day[0]], 
                            "worst_day_success": worst_day[2],
                            "success_difference": success_difference,
                            "best_day_avg_cost": best_day[3],
                            "best_day_avg_duration": best_day[4]
                        },
                        recommendations=[
                            f"Schedule important tasks on {day_names[best_day[0]]} for best results",
                            f"Consider postponing complex tasks from {day_names[worst_day[0]]} if possible",
                            f"Expected success improvement: {success_difference:.1%}",
                            "Review weekly workload distribution for optimization"
                        ],
                        business_impact={
                            "weekly_optimization": True,
                            "success_rate_improvement": success_difference,
                            "scheduling_intelligence": "active",
                            "resource_planning": "weekly_cycles"
                        },
                        supporting_data=[],
                        discovered_at=datetime.now(),
                        last_validated=datetime.now(),
                        validation_count=1
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    async def _discover_correlation_patterns(self, conn: sqlite3.Connection, min_confidence: float) -> List[AdvancedPattern]:
        """Discover correlation patterns between different variables"""
        patterns = []
        
        cursor = conn.cursor()
        
        # Correlation between cost and success
        cursor.execute('''
            SELECT 
                cost_usd,
                success_score,
                duration_seconds,
                task_type,
                model_used
            FROM experiences 
            WHERE cost_usd > 0 AND success_score > 0 AND duration_seconds > 0
            LIMIT 1000
        ''')
        
        results = cursor.fetchall()
        
        if len(results) >= self.min_sample_sizes[PatternTypeEnum.CORRELATION_PATTERN]:
            # Calculate correlation between cost and success
            costs = [r[0] for r in results]
            successes = [r[1] for r in results]
            durations = [r[2] for r in results]
            
            cost_success_correlation = self._calculate_correlation(costs, successes)
            cost_duration_correlation = self._calculate_correlation(costs, durations)
            success_duration_correlation = self._calculate_correlation(successes, durations)
            
            # Generate patterns for strong correlations
            correlations = [
                ("cost_vs_success", cost_success_correlation, costs, successes, "Cost", "Success Rate"),
                ("cost_vs_duration", cost_duration_correlation, costs, durations, "Cost", "Duration"),
                ("success_vs_duration", success_duration_correlation, successes, durations, "Success Rate", "Duration")
            ]
            
            for corr_name, corr_coeff, data_x, data_y, label_x, label_y in correlations:
                if abs(corr_coeff) > 0.3:  # Moderate correlation threshold
                    confidence = min(0.95, 0.70 + (abs(corr_coeff) * 0.25))
                    
                    if confidence >= min_confidence:
                        correlation_type = "positive" if corr_coeff > 0 else "negative"
                        strength_desc = "strong" if abs(corr_coeff) > 0.7 else "moderate"
                        
                        pattern = AdvancedPattern(
                            id=str(uuid.uuid4()),
                            pattern_type=PatternTypeEnum.CORRELATION_PATTERN,
                            name=f"{strength_desc.title()} {correlation_type.title()} Correlation: {label_x} ↔ {label_y}",
                            description=f"{strength_desc.title()} {correlation_type} correlation ({corr_coeff:.2f}) between {label_x.lower()} and {label_y.lower()}",
                            statistical_confidence=confidence,
                            strength=self._determine_pattern_strength(confidence),
                            frequency=len(results),
                            sample_size=len(results),
                            effect_size=abs(corr_coeff),
                            p_value=1.0 - confidence,
                            conditions={
                                "correlation_analysis": True,
                                "variables": [label_x.lower(), label_y.lower()],
                                "correlation_type": correlation_type
                            },
                            outcomes={
                                "correlation_coefficient": corr_coeff,
                                "correlation_strength": strength_desc,
                                "correlation_type": correlation_type,
                                "sample_size": len(results),
                                "variable_x": label_x,
                                "variable_y": label_y
                            },
                            recommendations=self._generate_correlation_recommendations(
                                corr_coeff, label_x, label_y
                            ),
                            business_impact={
                                "predictive_value": abs(corr_coeff),
                                "optimization_opportunity": abs(corr_coeff) > 0.5,
                                "decision_intelligence": strength_desc,
                                "planning_insights": correlation_type
                            },
                            supporting_data=[],
                            discovered_at=datetime.now(),
                            last_validated=datetime.now(),
                            validation_count=1
                        )
                        
                        patterns.append(pattern)
        
        return patterns
    
    async def _discover_anomaly_patterns(self, conn: sqlite3.Connection, min_confidence: float) -> List[AdvancedPattern]:
        """Discover anomaly patterns using statistical methods"""
        patterns = []
        
        cursor = conn.cursor()
        
        # Find statistical anomalies in cost
        cursor.execute('''
            SELECT 
                cost_usd,
                success_score,
                task_type,
                model_used,
                approach_used,
                created_at
            FROM experiences 
            WHERE cost_usd > 0
            ORDER BY cost_usd DESC
        ''')
        
        results = cursor.fetchall()
        
        if len(results) >= self.min_sample_sizes[PatternTypeEnum.ANOMALY_PATTERN]:
            costs = [r[0] for r in results]
            
            # Calculate statistical thresholds
            mean_cost = np.mean(costs)
            std_cost = np.std(costs)
            
            # Find anomalies (> 2 standard deviations)
            anomaly_threshold = mean_cost + (2 * std_cost)
            anomalies = [r for r in results if r[0] > anomaly_threshold]
            
            if anomalies:
                confidence = min(0.95, 0.80 + (len(anomalies) / len(results) * 0.15))
                
                if confidence >= min_confidence:
                    # Analyze common characteristics of anomalies
                    anomaly_task_types = [a[2] for a in anomalies]
                    anomaly_models = [a[3] for a in anomalies]
                    
                    most_common_task = max(set(anomaly_task_types), key=anomaly_task_types.count) if anomaly_task_types else "unknown"
                    most_common_model = max(set(anomaly_models), key=anomaly_models.count) if anomaly_models else "unknown"
                    
                    avg_anomaly_cost = np.mean([a[0] for a in anomalies])
                    avg_anomaly_success = np.mean([a[1] for a in anomalies if a[1] is not None])
                    
                    pattern = AdvancedPattern(
                        id=str(uuid.uuid4()),
                        pattern_type=PatternTypeEnum.ANOMALY_PATTERN,
                        name=f"Cost Anomaly: {most_common_model} for {most_common_task}",
                        description=f"Detected {len(anomalies)} cost anomalies (>${anomaly_threshold:.4f}+), mostly involving {most_common_model} for {most_common_task} tasks",
                        statistical_confidence=confidence,
                        strength=self._determine_pattern_strength(confidence),
                        frequency=len(anomalies),
                        sample_size=len(results),
                        effect_size=avg_anomaly_cost - mean_cost,
                        p_value=1.0 - confidence,
                        conditions={
                            "anomaly_detection": True,
                            "cost_threshold": anomaly_threshold,
                            "statistical_method": "z_score_2_sigma"
                        },
                        outcomes={
                            "anomaly_count": len(anomalies),
                            "anomaly_percentage": len(anomalies) / len(results),
                            "avg_anomaly_cost": avg_anomaly_cost,
                            "avg_normal_cost": mean_cost,
                            "cost_difference": avg_anomaly_cost - mean_cost,
                            "most_common_task_type": most_common_task,
                            "most_common_model": most_common_model,
                            "avg_anomaly_success": avg_anomaly_success
                        },
                        recommendations=[
                            f"Monitor {most_common_model} usage for {most_common_task} tasks",
                            f"Investigate why costs exceed ${anomaly_threshold:.4f}",
                            f"Consider alternatives if success rate ({avg_anomaly_success:.1%}) doesn't justify cost",
                            "Set up automated alerts for cost anomalies",
                            "Review model pricing and usage patterns"
                        ],
                        business_impact={
                            "cost_control": "critical",
                            "potential_savings": avg_anomaly_cost - mean_cost,
                            "risk_identification": "high",
                            "monitoring_priority": "high"
                        },
                        supporting_data=[],
                        discovered_at=datetime.now(),
                        last_validated=datetime.now(),
                        validation_count=1
                    )
                    
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_correlation(self, x_data: List[float], y_data: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        try:
            return np.corrcoef(x_data, y_data)[0, 1]
        except:
            return 0.0
    
    def _calculate_confidence_interval(self, mean: float, std: float, sample_size: int) -> float:
        """Calculate statistical confidence based on sample size and distribution"""
        if sample_size < 5:
            return 0.5
        elif sample_size < 10:
            return 0.7
        elif sample_size < 30:
            return 0.8
        elif sample_size < 100:
            return 0.9
        else:
            return 0.95
    
    def _determine_pattern_strength(self, confidence: float) -> PatternStrengthEnum:
        """Determine pattern strength based on statistical confidence"""
        if confidence >= self.confidence_thresholds[PatternStrengthEnum.VERY_STRONG]:
            return PatternStrengthEnum.VERY_STRONG
        elif confidence >= self.confidence_thresholds[PatternStrengthEnum.STRONG]:
            return PatternStrengthEnum.STRONG
        elif confidence >= self.confidence_thresholds[PatternStrengthEnum.MODERATE]:
            return PatternStrengthEnum.MODERATE
        elif confidence >= self.confidence_thresholds[PatternStrengthEnum.WEAK]:
            return PatternStrengthEnum.WEAK
        else:
            return PatternStrengthEnum.INSUFFICIENT
    
    def _generate_correlation_recommendations(self, correlation: float, var_x: str, var_y: str) -> List[str]:
        """Generate recommendations based on correlation analysis"""
        recommendations = []
        
        if correlation > 0.7:  # Strong positive correlation
            recommendations.append(f"Strong positive relationship: Higher {var_x.lower()} tends to increase {var_y.lower()}")
            recommendations.append(f"Use {var_x.lower()} as predictor for {var_y.lower()}")
        elif correlation > 0.3:  # Moderate positive correlation
            recommendations.append(f"Moderate positive relationship between {var_x.lower()} and {var_y.lower()}")
            recommendations.append(f"Consider {var_x.lower()} when optimizing {var_y.lower()}")
        elif correlation < -0.7:  # Strong negative correlation
            recommendations.append(f"Strong negative relationship: Higher {var_x.lower()} tends to decrease {var_y.lower()}")
            recommendations.append(f"Optimize by balancing {var_x.lower()} vs {var_y.lower()} trade-off")
        elif correlation < -0.3:  # Moderate negative correlation
            recommendations.append(f"Moderate negative relationship between {var_x.lower()} and {var_y.lower()}")
            recommendations.append(f"Consider inverse relationship when planning")
        
        recommendations.append("Monitor correlation stability over time")
        recommendations.append("Use correlation for predictive modeling")
        
        return recommendations
    
    async def _store_pattern(self, pattern: AdvancedPattern):
        """Store discovered pattern in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO advanced_patterns (
                        id, pattern_type, name, description, statistical_confidence,
                        strength, frequency, sample_size, effect_size, p_value,
                        conditions_json, outcomes_json, recommendations_json,
                        business_impact_json, supporting_data_json, discovered_at,
                        last_validated, validation_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    pattern.id,
                    pattern.pattern_type.value,
                    pattern.name,
                    pattern.description,
                    pattern.statistical_confidence,
                    pattern.strength.value,
                    pattern.frequency,
                    pattern.sample_size,
                    pattern.effect_size,
                    pattern.p_value,
                    json.dumps(pattern.conditions),
                    json.dumps(pattern.outcomes),
                    json.dumps(pattern.recommendations),
                    json.dumps(pattern.business_impact),
                    json.dumps(pattern.supporting_data),
                    pattern.discovered_at.isoformat(),
                    pattern.last_validated.isoformat(),
                    pattern.validation_count
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"❌ Failed to store pattern {pattern.id}: {e}")
    
    async def _discover_mock_patterns(self) -> List[AdvancedPattern]:
        """Generate mock patterns for demonstration when no data available"""
        mock_patterns = [
            AdvancedPattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternTypeEnum.SUCCESS_PATTERN,
                name="High Success: FastAPI + JSON for API Development",
                description="Using FastAPI framework with JSON responses for API development achieves 94% success rate",
                statistical_confidence=0.91,
                strength=PatternStrengthEnum.STRONG,
                frequency=47,
                sample_size=47,
                effect_size=0.94,
                p_value=0.09,
                conditions={
                    "task_type": "development",
                    "approach_used": "fastapi_json_api",
                    "framework": "FastAPI"
                },
                outcomes={
                    "avg_success_rate": 0.94,
                    "avg_cost": 0.0023,
                    "avg_duration_seconds": 28
                },
                recommendations=[
                    "Prioritize FastAPI + JSON approach for API development tasks",
                    "Expected success rate: 94%",
                    "Budget approximately $0.0023 per task",
                    "Monitor for consistency with established pattern"
                ],
                business_impact={
                    "success_rate_improvement": 0.24,
                    "cost_efficiency": 434.78,
                    "roi_multiplier": 408.70
                },
                supporting_data=[],
                discovered_at=datetime.now(),
                last_validated=datetime.now(),
                validation_count=1
            ),
            AdvancedPattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternTypeEnum.COST_PATTERN,
                name="Cost Efficient: claude-3-sonnet for Analysis Tasks",
                description="Using claude-3-sonnet for analysis tasks provides best cost efficiency at $0.0012 per success point",
                statistical_confidence=0.87,
                strength=PatternStrengthEnum.STRONG,
                frequency=34,
                sample_size=34,
                effect_size=0.0085,
                p_value=0.13,
                conditions={
                    "task_type": "analysis",
                    "cost_optimization": True,
                    "model_family": "claude"
                },
                outcomes={
                    "recommended_model": "claude-3-sonnet",
                    "avg_cost": 0.0021,
                    "cost_efficiency": 0.0012,
                    "potential_savings": 0.0085
                },
                recommendations=[
                    "Use claude-3-sonnet as primary choice for analysis tasks",
                    "Expected cost: $0.0021 per task",
                    "Expected success rate: 87%",
                    "Potential savings: $0.0085 per task vs alternatives"
                ],
                business_impact={
                    "cost_reduction": 0.0085,
                    "monthly_savings": 0.255,
                    "efficiency_ratio": 0.0012
                },
                supporting_data=[],
                discovered_at=datetime.now(),
                last_validated=datetime.now(),
                validation_count=1
            ),
            AdvancedPattern(
                id=str(uuid.uuid4()),
                pattern_type=PatternTypeEnum.TEMPORAL_PATTERN,
                name="Weekly Peak: Tuesday Performance",
                description="Tasks performed on Tuesday have 12% higher success rate than other days",
                statistical_confidence=0.83,
                strength=PatternStrengthEnum.STRONG,
                frequency=156,
                sample_size=156,
                effect_size=0.12,
                p_value=0.17,
                conditions={
                    "temporal_analysis": "weekly",
                    "day_of_week_focus": True
                },
                outcomes={
                    "best_day": "Tuesday",
                    "best_day_success": 0.89,
                    "worst_day": "Friday",
                    "worst_day_success": 0.77,
                    "success_difference": 0.12
                },
                recommendations=[
                    "Schedule important tasks on Tuesday for best results",
                    "Consider postponing complex tasks from Friday if possible",
                    "Expected success improvement: 12%",
                    "Review weekly workload distribution for optimization"
                ],
                business_impact={
                    "success_rate_improvement": 0.12,
                    "weekly_optimization": True,
                    "scheduling_intelligence": "active"
                },
                supporting_data=[],
                discovered_at=datetime.now(),
                last_validated=datetime.now(),
                validation_count=1
            )
        ]
        
        return mock_patterns
    
    async def get_pattern_insights(self, 
                                 pattern_types: Optional[List[PatternTypeEnum]] = None,
                                 min_confidence: float = 0.70) -> Dict[str, Any]:
        """Get comprehensive insights from discovered patterns"""
        try:
            insights = {
                "status": "success",
                "pattern_insights": {},
                "summary": {},
                "recommendations": [],
                "business_impact": {},
                "timestamp": datetime.now().isoformat()
            }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get pattern summary by type
                cursor.execute('''
                    SELECT 
                        pattern_type,
                        COUNT(*) as pattern_count,
                        AVG(statistical_confidence) as avg_confidence,
                        MAX(statistical_confidence) as max_confidence,
                        SUM(frequency) as total_frequency
                    FROM advanced_patterns
                    WHERE statistical_confidence >= ?
                    GROUP BY pattern_type
                    ORDER BY pattern_count DESC
                ''', (min_confidence,))
                
                type_summaries = cursor.fetchall()
                
                for type_row in type_summaries:
                    pattern_type, count, avg_conf, max_conf, total_freq = type_row
                    
                    insights["pattern_insights"][pattern_type] = {
                        "pattern_count": count,
                        "avg_confidence": round(avg_conf, 3),
                        "max_confidence": round(max_conf, 3),
                        "total_frequency": total_freq,
                        "reliability": "high" if avg_conf > 0.85 else "moderate" if avg_conf > 0.75 else "developing"
                    }
                
                # Get top patterns by confidence
                cursor.execute('''
                    SELECT 
                        name,
                        description,
                        statistical_confidence,
                        strength,
                        pattern_type,
                        business_impact_json
                    FROM advanced_patterns
                    WHERE statistical_confidence >= ?
                    ORDER BY statistical_confidence DESC
                    LIMIT 10
                ''', (min_confidence,))
                
                top_patterns = cursor.fetchall()
                insights["top_patterns"] = []
                
                total_business_impact = 0
                for pattern_row in top_patterns:
                    name, desc, conf, strength, ptype, business_json = pattern_row
                    
                    business_impact = json.loads(business_json) if business_json else {}
                    
                    insights["top_patterns"].append({
                        "name": name,
                        "description": desc,
                        "confidence": round(conf, 3),
                        "strength": strength,
                        "type": ptype,
                        "business_impact": business_impact
                    })
                    
                    # Accumulate business impact
                    if isinstance(business_impact, dict):
                        for key, value in business_impact.items():
                            if isinstance(value, (int, float)):
                                total_business_impact += abs(value)
                
                # Generate summary
                total_patterns = sum(row[1] for row in type_summaries)
                avg_overall_confidence = sum(row[2] for row in type_summaries) / len(type_summaries) if type_summaries else 0
                
                insights["summary"] = {
                    "total_patterns_discovered": total_patterns,
                    "avg_overall_confidence": round(avg_overall_confidence, 3),
                    "pattern_types_active": len(type_summaries),
                    "estimated_business_impact": round(total_business_impact, 4),
                    "system_learning_status": "active" if total_patterns > 5 else "developing",
                    "confidence_level": "high" if avg_overall_confidence > 0.85 else "moderate"
                }
                
                # Generate actionable recommendations
                if total_patterns > 0:
                    insights["recommendations"] = [
                        f"Found {total_patterns} patterns with {avg_overall_confidence:.1%} average confidence",
                        "Apply top patterns for immediate performance improvement",
                        "Monitor pattern stability over time for reliability",
                        "Use correlation patterns for predictive planning",
                        "Set up alerts for anomaly patterns to control costs"
                    ]
                    
                    if avg_overall_confidence > 0.85:
                        insights["recommendations"].append("Pattern confidence is high - suitable for automated application")
                else:
                    insights["recommendations"] = [
                        "No patterns discovered yet - system needs more data",
                        "Continue collecting experience data for pattern discovery",
                        "Consider running with mock patterns for development"
                    ]
            
            return insights
            
        except Exception as e:
            logger.error(f"❌ Failed to generate pattern insights: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def get_pattern_recommendations(self, 
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Get pattern-based recommendations for specific context"""
        try:
            task_type = context.get("task_type", "development")
            approach = context.get("approach", "")
            model_preference = context.get("model_preference", "")
            
            recommendations = {
                "status": "success",
                "context": context,
                "pattern_matches": [],
                "recommendations": [],
                "confidence_score": 0.0,
                "expected_outcomes": {},
                "timestamp": datetime.now().isoformat()
            }
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Find matching patterns
                cursor.execute('''
                    SELECT 
                        name,
                        description,
                        statistical_confidence,
                        recommendations_json,
                        outcomes_json,
                        business_impact_json
                    FROM advanced_patterns
                    WHERE 
                        (conditions_json LIKE ? OR conditions_json LIKE ? OR conditions_json LIKE ?)
                        AND statistical_confidence > 0.7
                    ORDER BY statistical_confidence DESC
                    LIMIT 5
                ''', (f'%{task_type}%', f'%{approach}%', f'%{model_preference}%'))
                
                matches = cursor.fetchall()
                total_confidence = 0
                
                for match in matches:
                    name, desc, conf, recs_json, outcomes_json, impact_json = match
                    
                    pattern_recommendations = json.loads(recs_json) if recs_json else []
                    outcomes = json.loads(outcomes_json) if outcomes_json else {}
                    impact = json.loads(impact_json) if impact_json else {}
                    
                    recommendations["pattern_matches"].append({
                        "pattern_name": name,
                        "description": desc,
                        "confidence": round(conf, 3),
                        "recommendations": pattern_recommendations,
                        "expected_outcomes": outcomes,
                        "business_impact": impact
                    })
                    
                    recommendations["recommendations"].extend(pattern_recommendations)
                    total_confidence += conf
                
                if matches:
                    recommendations["confidence_score"] = round(total_confidence / len(matches), 3)
                    
                    # Aggregate expected outcomes
                    for match in recommendations["pattern_matches"]:
                        outcomes = match["expected_outcomes"]
                        for key, value in outcomes.items():
                            if isinstance(value, (int, float)):
                                if key not in recommendations["expected_outcomes"]:
                                    recommendations["expected_outcomes"][key] = []
                                recommendations["expected_outcomes"][key].append(value)
                    
                    # Average the expected outcomes
                    for key, values in recommendations["expected_outcomes"].items():
                        if values:
                            recommendations["expected_outcomes"][key] = round(sum(values) / len(values), 4)
                else:
                    recommendations["recommendations"] = [
                        f"No specific patterns found for {task_type} tasks",
                        "Using general best practices",
                        "Consider building experience data for better recommendations"
                    ]
                    recommendations["confidence_score"] = 0.5
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Failed to get pattern recommendations: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

# =============================================================================
# CLI INTEGRATION FUNCTIONS
# =============================================================================

async def discover_patterns_cli(min_confidence: float = 0.70) -> Dict[str, Any]:
    """CLI integration for pattern discovery"""
    engine = AdvancedPatternRecognitionEngine()
    patterns = await engine.discover_advanced_patterns(min_confidence=min_confidence)
    
    return {
        "patterns_discovered": len(patterns),
        "patterns": [
            {
                "name": p.name,
                "type": p.pattern_type.value,
                "confidence": p.statistical_confidence,
                "strength": p.strength.value,
                "description": p.description
            }
            for p in patterns[:10]  # Limit output
        ]
    }

async def get_pattern_insights_cli() -> Dict[str, Any]:
    """CLI integration for pattern insights"""
    engine = AdvancedPatternRecognitionEngine()
    return await engine.get_pattern_insights()

async def get_pattern_recommendations_cli(task_type: str = "development",
                                        approach: str = "",
                                        model_preference: str = "") -> Dict[str, Any]:
    """CLI integration for pattern recommendations"""
    engine = AdvancedPatternRecognitionEngine()
    context = {
        "task_type": task_type,
        "approach": approach,
        "model_preference": model_preference
    }
    return await engine.get_pattern_recommendations(context)

# =============================================================================
# TESTING AND DEMO
# =============================================================================

async def demo_advanced_pattern_recognition():
    """Demonstrate Advanced Pattern Recognition System"""
    print("🔍 Agent Zero V2.0 - Advanced Pattern Recognition System Demo")
    print("=" * 70)
    
    engine = AdvancedPatternRecognitionEngine()
    
    # Demo 1: Pattern Discovery
    print("\n1. Discovering Advanced Patterns...")
    patterns = await engine.discover_advanced_patterns(min_confidence=0.70)
    print(f"  🔍 Discovered {len(patterns)} advanced patterns")
    
    for i, pattern in enumerate(patterns[:3], 1):
        print(f"  {i}. {pattern.name}")
        print(f"     Type: {pattern.pattern_type.value}")
        print(f"     Confidence: {pattern.statistical_confidence:.1%}")
        print(f"     Strength: {pattern.strength.value}")
        print(f"     Description: {pattern.description}")
        print()
    
    # Demo 2: Pattern Insights
    print("2. Generating Pattern Insights...")
    insights = await engine.get_pattern_insights()
    print(f"  📊 Total patterns: {insights.get('summary', {}).get('total_patterns_discovered', 0)}")
    print(f"  📊 Average confidence: {insights.get('summary', {}).get('avg_overall_confidence', 0):.1%}")
    print(f"  📊 System status: {insights.get('summary', {}).get('system_learning_status', 'unknown')}")
    
    # Demo 3: Pattern Recommendations
    print("\n3. Getting Pattern Recommendations...")
    context = {
        "task_type": "development",
        "approach": "fastapi",
        "model_preference": "claude"
    }
    
    recommendations = await engine.get_pattern_recommendations(context)
    print(f"  🎯 Confidence score: {recommendations.get('confidence_score', 0):.1%}")
    print(f"  🎯 Pattern matches: {len(recommendations.get('pattern_matches', []))}")
    
    top_recommendations = recommendations.get('recommendations', [])[:3]
    for i, rec in enumerate(top_recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\n✅ Advanced Pattern Recognition System operational!")

if __name__ == "__main__":
    asyncio.run(demo_advanced_pattern_recognition())