#!/usr/bin/env python3
"""
Agent Zero V1 - Success/Failure Evaluator
V2.0 Intelligence Layer - Multi-dimensional Success Classification

Author: Developer A (Backend Architect)
Date: 10 października 2025
Linear Issue: A0-28
"""

import ast
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuccessLevel(Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    FAILURE = "FAILURE"

class TaskOutputType(Enum):
    CODE = "code"
    TEXT = "text"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"

@dataclass
class SuccessEvaluation:
    level: SuccessLevel
    overall_score: float
    dimension_scores: Dict[str, float]
    recommendations: List[str]
    reasoning: str
    confidence: float
    improvement_suggestions: List[str]
    cost_effectiveness: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['level'] = self.level.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class TaskResult:
    task_id: str
    task_type: str
    output_type: TaskOutputType
    output_content: str
    expected_requirements: List[str]
    context: Dict[str, Any]
    execution_time_ms: int
    cost_usd: float
    model_used: str
    human_feedback: Optional[int] = None

class SuccessEvaluator:
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = Path(db_path)
        
        self.dimensions = {
            'correctness': 0.5,
            'efficiency': 0.2,
            'cost': 0.15,
            'latency': 0.15
        }
        
        self._init_database()
        
        self.thresholds = {
            'success_threshold': 0.85,
            'partial_threshold': 0.65,
            'improvement_threshold': 0.45,
            'max_acceptable_cost': 0.05,
            'max_acceptable_latency': 30000,
        }

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS v2_success_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    task_type TEXT NOT NULL,
                    success_level TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    correctness_score REAL NOT NULL,
                    efficiency_score REAL NOT NULL,
                    cost_score REAL NOT NULL,
                    latency_score REAL NOT NULL,
                    confidence REAL NOT NULL,
                    cost_effectiveness REAL NOT NULL,
                    model_used TEXT NOT NULL,
                    execution_time_ms INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()

    def evaluate_task(self, task_result: TaskResult) -> SuccessEvaluation:
        logger.info(f"🔍 Evaluating task {task_result.task_id}")
        
        dimension_scores = {}
        dimension_scores['correctness'] = self._evaluate_correctness(task_result)
        dimension_scores['efficiency'] = self._evaluate_efficiency(task_result)
        dimension_scores['cost'] = self._evaluate_cost(task_result)
        dimension_scores['latency'] = self._evaluate_latency(task_result)
        
        overall_score = sum(
            score * self.dimensions[dimension]
            for dimension, score in dimension_scores.items()
        )
        
        success_level = self._classify_success_level(overall_score)
        recommendations = self._generate_recommendations(dimension_scores, task_result)
        improvement_suggestions = self._generate_improvements(dimension_scores, task_result)
        reasoning = self._generate_reasoning(dimension_scores, overall_score, task_result)
        
        confidence = self._calculate_confidence(dimension_scores, task_result)
        cost_effectiveness = self._calculate_cost_effectiveness(overall_score, task_result.cost_usd)
        
        evaluation = SuccessEvaluation(
            level=success_level,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            recommendations=recommendations,
            reasoning=reasoning,
            confidence=confidence,
            improvement_suggestions=improvement_suggestions,
            cost_effectiveness=cost_effectiveness,
            timestamp=datetime.now()
        )
        
        self._store_evaluation(evaluation, task_result)
        
        logger.info(f"✅ Evaluation complete: {success_level.value} (score: {overall_score:.2f})")
        return evaluation

    def _evaluate_correctness(self, task_result: TaskResult) -> float:
        if task_result.output_type == TaskOutputType.CODE:
            try:
                ast.parse(task_result.output_content)
                syntax_score = 1.0
            except SyntaxError:
                syntax_score = 0.3
            
            requirement_score = 0.8 if task_result.expected_requirements else 0.7
            return (syntax_score * 0.6 + requirement_score * 0.4)
        
        return 0.7  # Default for non-code

    def _evaluate_efficiency(self, task_result: TaskResult) -> float:
        return 0.8  # Base efficiency score

    def _evaluate_cost(self, task_result: TaskResult) -> float:
        if task_result.cost_usd == 0:
            return 1.0
        elif task_result.cost_usd <= 0.01:
            return 0.9
        elif task_result.cost_usd <= 0.03:
            return 0.7
        else:
            return max(0.0, 0.5 - (task_result.cost_usd - 0.05) * 5)

    def _evaluate_latency(self, task_result: TaskResult) -> float:
        if task_result.execution_time_ms <= 2000:
            return 1.0
        elif task_result.execution_time_ms <= 5000:
            return 0.8
        else:
            return max(0.0, 0.8 - (task_result.execution_time_ms - 5000) / 10000)

    def _classify_success_level(self, overall_score: float) -> SuccessLevel:
        if overall_score >= self.thresholds['success_threshold']:
            return SuccessLevel.SUCCESS
        elif overall_score >= self.thresholds['partial_threshold']:
            return SuccessLevel.PARTIAL
        elif overall_score >= self.thresholds['improvement_threshold']:
            return SuccessLevel.NEEDS_IMPROVEMENT
        else:
            return SuccessLevel.FAILURE

    def _generate_recommendations(self, dimension_scores: Dict[str, float], task_result: TaskResult) -> List[str]:
        recommendations = []
        
        if dimension_scores['correctness'] < 0.7:
            recommendations.append("🎯 Focus on requirement fulfillment and technical accuracy")
        
        if dimension_scores['cost'] < 0.7:
            recommendations.append("💰 Explore cost-effective model alternatives")
        
        return recommendations

    def _generate_improvements(self, dimension_scores: Dict[str, float], task_result: TaskResult) -> List[str]:
        improvements = []
        
        weakest_dimension = min(dimension_scores, key=dimension_scores.get)
        if dimension_scores[weakest_dimension] < 0.5:
            improvements.append(f"📋 Improve {weakest_dimension} performance")
        
        return improvements

    def _generate_reasoning(self, dimension_scores: Dict[str, float], overall_score: float, task_result: TaskResult) -> str:
        reasoning_parts = []
        
        success_level = self._classify_success_level(overall_score)
        reasoning_parts.append(f"Overall: {success_level.value} (score: {overall_score:.2f})")
        
        for dimension, score in dimension_scores.items():
            weight = self.dimensions[dimension]
            reasoning_parts.append(f"{dimension.title()}: {score:.2f} (weight: {weight:.0%})")
        
        return " | ".join(reasoning_parts)

    def _calculate_confidence(self, dimension_scores: Dict[str, float], task_result: TaskResult) -> float:
        return 0.8  # Base confidence

    def _calculate_cost_effectiveness(self, overall_score: float, cost: float) -> float:
        if cost == 0:
            return overall_score
        return overall_score / max(0.001, cost)

    def _store_evaluation(self, evaluation: SuccessEvaluation, task_result: TaskResult):
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO v2_success_evaluations (
                        task_id, task_type, success_level, overall_score,
                        correctness_score, efficiency_score, cost_score, latency_score,
                        confidence, cost_effectiveness, model_used, execution_time_ms, cost_usd,
                        reasoning
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    task_result.task_id,
                    task_result.task_type,
                    evaluation.level.value,
                    evaluation.overall_score,
                    evaluation.dimension_scores['correctness'],
                    evaluation.dimension_scores['efficiency'],
                    evaluation.dimension_scores['cost'],
                    evaluation.dimension_scores['latency'],
                    evaluation.confidence,
                    evaluation.cost_effectiveness,
                    task_result.model_used,
                    task_result.execution_time_ms,
                    task_result.cost_usd,
                    evaluation.reasoning
                ))
                conn.commit()
        except sqlite3.Error as e:
            logger.error(f"Failed to store evaluation: {e}")

if __name__ == "__main__":
    evaluator = SuccessEvaluator()
    
    task_result = TaskResult(
        task_id="test_001",
        task_type="code_generation",
        output_type=TaskOutputType.CODE,
        output_content="def hello():\n    return 'Hello World'",
        expected_requirements=["Create a hello function"],
        context={"complexity": "simple"},
        execution_time_ms=1500,
        cost_usd=0.008,
        model_used="gpt-4o-mini"
    )
    
    evaluation = evaluator.evaluate_task(task_result)
    print(f"Success Level: {evaluation.level.value}")
    print(f"Overall Score: {evaluation.overall_score:.2f}")
    print(f"Reasoning: {evaluation.reasoning}")
