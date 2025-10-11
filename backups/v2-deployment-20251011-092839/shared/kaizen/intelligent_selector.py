#!/usr/bin/env python3
"""
Agent Zero V1 - Intelligent Model Selector
V2.0 Intelligence Layer - AI-First Decision System

Author: Developer A (Backend Architect)
Date: 10 października 2025
Linear Issue: A0-28
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    LOCAL_OLLAMA = "local_ollama"
    OPENAI_GPT = "openai_gpt"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    MISTRAL_AI = "mistral_ai"

class TaskType(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    BUSINESS_ANALYSIS = "business_analysis"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    ARCHITECTURE_DESIGN = "architecture_design"

@dataclass
class ModelRecommendation:
    model_name: str
    model_type: ModelType
    confidence_score: float
    reasoning: str
    alternatives: List[str]
    estimated_cost: float
    estimated_latency_ms: int
    success_probability: float
    context_factors: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['model_type'] = self.model_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class DecisionCriteria:
    success_rate: float = 0.4
    cost_efficiency: float = 0.3
    latency: float = 0.2
    human_acceptance: float = 0.1
    
    def __post_init__(self):
        total = self.success_rate + self.cost_efficiency + self.latency + self.human_acceptance
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Criteria weights must sum to 1.0, got {total}")

class IntelligentModelSelector:
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = Path(db_path)
        self.criteria = DecisionCriteria()
        self.models = self._initialize_models()
        self._init_database()

    def _initialize_models(self) -> Dict[str, Dict[str, Any]]:
        return {
            "llama3.2:3b": {
                "type": ModelType.LOCAL_OLLAMA,
                "base_cost": 0.0,
                "base_latency_ms": 2000,
                "strengths": ["code_generation", "debugging"],
                "max_context": 8192,
                "quality_tier": "fast"
            },
            "llama3.2:70b": {
                "type": ModelType.LOCAL_OLLAMA, 
                "base_cost": 0.0,
                "base_latency_ms": 8000,
                "strengths": ["architecture_design", "business_analysis"],
                "max_context": 32768,
                "quality_tier": "high"
            },
            "gpt-4o-mini": {
                "type": ModelType.OPENAI_GPT,
                "base_cost": 0.015,
                "base_latency_ms": 1500,
                "strengths": ["code_review", "documentation"],
                "max_context": 128000,
                "quality_tier": "balanced"
            },
            "gpt-4o": {
                "type": ModelType.OPENAI_GPT,
                "base_cost": 0.06,
                "base_latency_ms": 3000,
                "strengths": ["architecture_design", "business_analysis"],
                "max_context": 128000,
                "quality_tier": "premium"
            }
        }

    def _init_database(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS v2_model_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    task_type TEXT NOT NULL,
                    recommended_model TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    reasoning TEXT NOT NULL,
                    alternatives TEXT NOT NULL,
                    estimated_cost REAL NOT NULL,
                    estimated_latency_ms INTEGER NOT NULL,
                    context_factors TEXT NOT NULL
                )
            ''')
            conn.commit()

    def recommend_model(self, task_type: str, context: Dict[str, Any]) -> ModelRecommendation:
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                logger.warning(f"Unknown task type: {task_type}, using CODE_GENERATION")
                task_type = TaskType.CODE_GENERATION
        
        logger.info(f"🤖 Generating recommendation for {task_type.value}")
        
        # Score all models
        model_scores = {}
        for model_name, model_config in self.models.items():
            score = self._calculate_model_score(model_name, model_config, task_type, context)
            model_scores[model_name] = score
        
        # Select best model
        best_model = max(model_scores, key=lambda x: model_scores[x]['total_score'])
        best_score = model_scores[best_model]
        
        # Generate alternatives
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1]['total_score'], reverse=True)
        alternatives = [m[0] for m in sorted_models[1:4]]
        
        recommendation = ModelRecommendation(
            model_name=best_model,
            model_type=self.models[best_model]['type'],
            confidence_score=best_score['total_score'],
            reasoning=self._generate_reasoning(best_model, best_score, task_type, context),
            alternatives=alternatives,
            estimated_cost=best_score['cost'],
            estimated_latency_ms=best_score['latency_ms'],
            success_probability=best_score['success_rate'],
            context_factors=context,
            timestamp=datetime.now()
        )
        
        logger.info(f"✅ Recommended {best_model} (confidence: {best_score['total_score']:.2f})")
        return recommendation

    def _calculate_model_score(self, model_name: str, model_config: Dict[str, Any], 
                              task_type: TaskType, context: Dict[str, Any]) -> Dict[str, float]:
        
        success_rate = 0.7  # Base success rate
        
        # Cost calculation
        base_cost = model_config['base_cost']
        complexity_multiplier = context.get('complexity', 1.0)
        estimated_cost = base_cost * complexity_multiplier
        cost_score = 1.0 - min(estimated_cost / 0.1, 1.0)
        
        # Latency calculation  
        base_latency = model_config['base_latency_ms']
        urgency_factor = context.get('urgency', 1.0)
        estimated_latency = base_latency * urgency_factor
        latency_score = 1.0 - min(estimated_latency / 10000, 1.0)
        
        human_score = 0.8  # Base human acceptance
        
        # Task affinity bonus
        task_affinity = 1.0
        if task_type.value in model_config.get('strengths', []):
            task_affinity = 1.2
        
        # Calculate weighted total
        total_score = (
            self.criteria.success_rate * success_rate * task_affinity +
            self.criteria.cost_efficiency * cost_score +
            self.criteria.latency * latency_score +
            self.criteria.human_acceptance * human_score
        )
        
        return {
            'total_score': total_score,
            'success_rate': success_rate,
            'cost_score': cost_score,
            'latency_score': latency_score, 
            'human_score': human_score,
            'cost': estimated_cost,
            'latency_ms': int(estimated_latency),
            'task_affinity': task_affinity
        }

    def _generate_reasoning(self, model_name: str, score_breakdown: Dict[str, float],
                           task_type: TaskType, context: Dict[str, Any]) -> str:
        
        reasoning_parts = []
        
        reasoning_parts.append(
            f"Selected {model_name} for {task_type.value} with confidence {score_breakdown['total_score']:.2f}"
        )
        
        if score_breakdown['cost'] < 0.01:
            reasoning_parts.append("✅ Cost-effective: Free local execution")
        elif score_breakdown['cost'] < 0.05:
            reasoning_parts.append(f"💰 Moderate cost: ~${score_breakdown['cost']:.3f}")
        else:
            reasoning_parts.append(f"💸 Premium cost: ~${score_breakdown['cost']:.3f} (high quality)")
        
        if score_breakdown['success_rate'] > 0.8:
            reasoning_parts.append(f"🎯 High success rate: {score_breakdown['success_rate']:.1%}")
        else:
            reasoning_parts.append(f"📊 Good success rate: {score_breakdown['success_rate']:.1%}")
        
        if score_breakdown['latency_ms'] < 2000:
            reasoning_parts.append(f"⚡ Fast response: ~{score_breakdown['latency_ms']}ms")
        else:
            reasoning_parts.append(f"🚀 Moderate speed: ~{score_breakdown['latency_ms']}ms")
        
        if score_breakdown['task_affinity'] > 1.0:
            reasoning_parts.append(f"🎨 Specialized for {task_type.value}")
        
        return " | ".join(reasoning_parts)

if __name__ == "__main__":
    selector = IntelligentModelSelector()
    context = {'complexity': 1.2, 'urgency': 1.0, 'budget': 'medium'}
    recommendation = selector.recommend_model(TaskType.CODE_GENERATION, context)
    print(f"Recommendation: {recommendation.model_name}")
    print(f"Reasoning: {recommendation.reasoning}")
