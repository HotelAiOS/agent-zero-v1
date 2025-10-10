#!/usr/bin/env python3
"""
Agent Zero V1 - Intelligent Model Selector
V2.0 Intelligence Layer - AI-First Decision System

This module implements the core AI decision-making system for Agent Zero V1,
providing intelligent model selection based on multi-criteria scoring.

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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types in Agent Zero V1"""
    LOCAL_OLLAMA = "local_ollama"
    OPENAI_GPT = "openai_gpt"
    ANTHROPIC_CLAUDE = "anthropic_claude"
    MISTRAL_AI = "mistral_ai"

class TaskType(Enum):
    """Task categories for intelligent selection"""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    BUSINESS_ANALYSIS = "business_analysis"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    ARCHITECTURE_DESIGN = "architecture_design"

@dataclass
class ModelRecommendation:
    """Data structure for AI model recommendations"""
    model_name: str
    model_type: ModelType
    confidence_score: float  # 0.0 to 1.0
    reasoning: str
    alternatives: List[str]
    estimated_cost: float
    estimated_latency_ms: int
    success_probability: float
    context_factors: Dict[str, Any]
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['model_type'] = self.model_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class DecisionCriteria:
    """Multi-criteria scoring weights"""
    success_rate: float = 0.4      # Historical success rate weight
    cost_efficiency: float = 0.3   # Cost optimization weight
    latency: float = 0.2           # Response time weight  
    human_acceptance: float = 0.1  # Human feedback weight
    
    def __post_init__(self):
        """Validate weights sum to 1.0"""
        total = self.success_rate + self.cost_efficiency + self.latency + self.human_acceptance
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"Criteria weights must sum to 1.0, got {total}")

class IntelligentModelSelector:
    """
    Core AI decision system for Agent Zero V1
    
    Implements intelligent model selection based on:
    - Historical performance analysis
    - Multi-criteria decision scoring
    - Transparent reasoning generation
    - Human feedback integration
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = Path(db_path)
        self.criteria = DecisionCriteria()
        self.models = self._initialize_models()
        self._init_database()
        
        # Load existing SimpleTracker integration
        try:
            from shared.utils.simple_tracker import SimpleTracker
            self.tracker = SimpleTracker()
            logger.info("✅ SimpleTracker integration loaded")
        except ImportError:
            logger.warning("⚠️ SimpleTracker not available, using fallback")
            self.tracker = None

    def _initialize_models(self) -> Dict[str, Dict[str, Any]]:
        """Initialize available models with their characteristics"""
        return {
            "llama3.2:3b": {
                "type": ModelType.LOCAL_OLLAMA,
                "base_cost": 0.0,  # Free local execution
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
                "base_cost": 0.015,  # Per 1K tokens
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
            },
            "claude-3-haiku": {
                "type": ModelType.ANTHROPIC_CLAUDE,
                "base_cost": 0.025,
                "base_latency_ms": 2000,
                "strengths": ["documentation", "code_review"],
                "max_context": 200000,
                "quality_tier": "balanced"
            }
        }

    def _init_database(self):
        """Initialize SQLite database for V2.0 tracking"""
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
                    context_factors TEXT NOT NULL,
                    human_feedback INTEGER DEFAULT NULL,
                    actual_cost REAL DEFAULT NULL,
                    actual_latency_ms INTEGER DEFAULT NULL,
                    actual_success INTEGER DEFAULT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS v2_model_performance (
                    model_name TEXT,
                    task_type TEXT,
                    success_count INTEGER DEFAULT 0,
                    failure_count INTEGER DEFAULT 0,
                    total_cost REAL DEFAULT 0.0,
                    avg_latency_ms REAL DEFAULT 0.0,
                    human_approval_rate REAL DEFAULT 0.0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_name, task_type)
                )
            ''')
            
            conn.commit()

    def recommend_model(self, task_type: str, context: Dict[str, Any]) -> ModelRecommendation:
        """
        Generate intelligent model recommendation
        
        Args:
            task_type: Type of task (TaskType enum value or string)
            context: Additional context including complexity, budget, etc.
            
        Returns:
            ModelRecommendation with detailed reasoning
        """
        # Normalize task type
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                logger.warning(f"Unknown task type: {task_type}, using CODE_GENERATION")
                task_type = TaskType.CODE_GENERATION
        
        logger.info(f"🤖 Generating recommendation for {task_type.value}")
        
        # Analyze historical performance
        historical_data = self._get_historical_performance(task_type)
        
        # Score all models
        model_scores = {}
        for model_name, model_config in self.models.items():
            score = self._calculate_model_score(
                model_name, model_config, task_type, context, historical_data
            )
            model_scores[model_name] = score
        
        # Select best model
        best_model = max(model_scores, key=lambda x: model_scores[x]['total_score'])
        best_score = model_scores[best_model]
        
        # Generate alternatives
        sorted_models = sorted(
            model_scores.items(),
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        alternatives = [m[0] for m in sorted_models[1:4]]  # Top 3 alternatives
        
        # Create recommendation
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
        
        # Store decision for learning
        self._store_decision(recommendation, task_type)
        
        logger.info(f"✅ Recommended {best_model} (confidence: {best_score['total_score']:.2f})")
        return recommendation

    def _calculate_model_score(
        self, 
        model_name: str, 
        model_config: Dict[str, Any],
        task_type: TaskType,
        context: Dict[str, Any],
        historical_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate multi-criteria score for a model"""
        
        # Base scores
        success_rate = historical_data.get(model_name, {}).get('success_rate', 0.5)
        
        # Cost calculation
        base_cost = model_config['base_cost']
        complexity_multiplier = context.get('complexity', 1.0)
        estimated_cost = base_cost * complexity_multiplier
        cost_score = 1.0 - min(estimated_cost / 0.1, 1.0)  # Normalize to 0-1
        
        # Latency calculation  
        base_latency = model_config['base_latency_ms']
        urgency_factor = context.get('urgency', 1.0)
        estimated_latency = base_latency * urgency_factor
        latency_score = 1.0 - min(estimated_latency / 10000, 1.0)  # Normalize to 0-1
        
        # Human acceptance
        human_score = historical_data.get(model_name, {}).get('human_approval', 0.5)
        
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

    def _get_historical_performance(self, task_type: TaskType) -> Dict[str, Dict[str, float]]:
        """Retrieve historical performance data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('''
                    SELECT model_name, success_count, failure_count, human_approval_rate
                    FROM v2_model_performance 
                    WHERE task_type = ?
                ''', (task_type.value,))
                
                results = {}
                for row in cursor.fetchall():
                    model_name, success_count, failure_count, human_approval = row
                    total_tasks = success_count + failure_count
                    success_rate = success_count / total_tasks if total_tasks > 0 else 0.5
                    
                    results[model_name] = {
                        'success_rate': success_rate,
                        'human_approval': human_approval or 0.5,
                        'total_tasks': total_tasks
                    }
                
                return results
                
        except sqlite3.Error as e:
            logger.warning(f"Database error retrieving performance: {e}")
            return {}

    def _generate_reasoning(
        self, 
        model_name: str,
        score_breakdown: Dict[str, float],
        task_type: TaskType,
        context: Dict[str, Any]
    ) -> str:
        """Generate transparent reasoning for the recommendation"""
        
        model_config = self.models[model_name]
        reasoning_parts = []
        
        # Main selection reason
        reasoning_parts.append(
            f"Selected {model_name} for {task_type.value} with confidence {score_breakdown['total_score']:.2f}"
        )
        
        # Cost analysis
        if score_breakdown['cost'] < 0.01:
            reasoning_parts.append("✅ Cost-effective: Free local execution")
        elif score_breakdown['cost'] < 0.05:
            reasoning_parts.append(f"💰 Moderate cost: ~${score_breakdown['cost']:.3f}")
        else:
            reasoning_parts.append(f"💸 Premium cost: ~${score_breakdown['cost']:.3f} (high quality)")
        
        # Performance analysis
        if score_breakdown['success_rate'] > 0.8:
            reasoning_parts.append(f"🎯 High success rate: {score_breakdown['success_rate']:.1%}")
        elif score_breakdown['success_rate'] > 0.6:
            reasoning_parts.append(f"📊 Good success rate: {score_breakdown['success_rate']:.1%}")
        else:
            reasoning_parts.append(f"⚠️ Learning phase: {score_breakdown['success_rate']:.1%} success rate")
        
        # Speed analysis
        if score_breakdown['latency_ms'] < 2000:
            reasoning_parts.append(f"⚡ Fast response: ~{score_breakdown['latency_ms']}ms")
        elif score_breakdown['latency_ms'] < 5000:
            reasoning_parts.append(f"🚀 Moderate speed: ~{score_breakdown['latency_ms']}ms")
        else:
            reasoning_parts.append(f"🔄 Thoughtful processing: ~{score_breakdown['latency_ms']}ms")
        
        # Task affinity
        if score_breakdown['task_affinity'] > 1.0:
            reasoning_parts.append(f"🎨 Specialized for {task_type.value}")
        
        # Context factors
        if context.get('budget') == 'low':
            reasoning_parts.append("💡 Optimized for low budget")
        if context.get('urgency') == 'high':
            reasoning_parts.append("⏰ Prioritized for speed")
        
        return " | ".join(reasoning_parts)

    def _store_decision(self, recommendation: ModelRecommendation, task_type: TaskType):
        """Store decision for future learning"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO v2_model_decisions (
                        task_type, recommended_model, confidence_score, reasoning,
                        alternatives, estimated_cost, estimated_latency_ms, context_factors
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    task_type.value,
                    recommendation.model_name,
                    recommendation.confidence_score,
                    recommendation.reasoning,
                    json.dumps(recommendation.alternatives),
                    recommendation.estimated_cost,
                    recommendation.estimated_latency_ms,
                    json.dumps(recommendation.context_factors)
                ))
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to store decision: {e}")

    def record_feedback(
        self, 
        decision_id: int, 
        human_feedback: int,  # 1-5 scale
        actual_cost: Optional[float] = None,
        actual_latency_ms: Optional[int] = None,
        actual_success: bool = True
    ):
        """Record human feedback and actual performance for learning"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Update decision record
                conn.execute('''
                    UPDATE v2_model_decisions
                    SET human_feedback = ?, actual_cost = ?, 
                        actual_latency_ms = ?, actual_success = ?
                    WHERE id = ?
                ''', (human_feedback, actual_cost, actual_latency_ms, 
                      1 if actual_success else 0, decision_id))
                
                # Update model performance
                cursor = conn.execute('''
                    SELECT task_type, recommended_model FROM v2_model_decisions WHERE id = ?
                ''', (decision_id,))
                
                row = cursor.fetchone()
                if row:
                    task_type, model_name = row
                    
                    # Update or insert performance record
                    conn.execute('''
                        INSERT OR REPLACE INTO v2_model_performance 
                        (model_name, task_type, success_count, failure_count, human_approval_rate)
                        VALUES (?, ?, 
                            COALESCE((SELECT success_count FROM v2_model_performance WHERE model_name = ? AND task_type = ?), 0) + ?,
                            COALESCE((SELECT failure_count FROM v2_model_performance WHERE model_name = ? AND task_type = ?), 0) + ?,
                            COALESCE((SELECT human_approval_rate FROM v2_model_performance WHERE model_name = ? AND task_type = ?), 2.5) * 0.9 + ? * 0.1
                        )
                    ''', (
                        model_name, task_type,
                        model_name, task_type, 1 if actual_success else 0,
                        model_name, task_type, 0 if actual_success else 1,
                        model_name, task_type, human_feedback
                    ))
                
                conn.commit()
                logger.info(f"✅ Recorded feedback for decision {decision_id}")
                
        except sqlite3.Error as e:
            logger.error(f"Failed to record feedback: {e}")

    def get_performance_insights(self) -> Dict[str, Any]:
        """Generate performance insights for Kaizen reporting"""
        insights = {
            'top_performing_models': {},
            'cost_optimization_opportunities': [],
            'learning_recommendations': [],
            'success_trends': {}
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Top performing models by task type
                cursor = conn.execute('''
                    SELECT task_type, model_name, success_count, failure_count,
                           human_approval_rate
                    FROM v2_model_performance
                    WHERE success_count + failure_count >= 5
                    ORDER BY 
                        (success_count::REAL / (success_count + failure_count)) * human_approval_rate DESC
                ''')
                
                for row in cursor.fetchall():
                    task_type, model_name, success_count, failure_count, approval_rate = row
                    success_rate = success_count / (success_count + failure_count)
                    
                    if task_type not in insights['top_performing_models']:
                        insights['top_performing_models'][task_type] = []
                    
                    insights['top_performing_models'][task_type].append({
                        'model': model_name,
                        'success_rate': success_rate,
                        'approval_rate': approval_rate,
                        'total_tasks': success_count + failure_count
                    })
                
                # Cost optimization opportunities
                cursor = conn.execute('''
                    SELECT recommended_model, AVG(estimated_cost), COUNT(*), task_type
                    FROM v2_model_decisions
                    WHERE estimated_cost > 0.02
                    GROUP BY recommended_model, task_type
                    HAVING COUNT(*) >= 3
                    ORDER BY AVG(estimated_cost) DESC
                ''')
                
                for row in cursor.fetchall():
                    model, avg_cost, count, task_type = row
                    insights['cost_optimization_opportunities'].append({
                        'model': model,
                        'task_type': task_type,
                        'avg_cost': avg_cost,
                        'frequency': count,
                        'potential_savings': avg_cost * count * 0.3  # Estimated 30% savings
                    })
                
        except sqlite3.Error as e:
            logger.error(f"Failed to generate insights: {e}")
        
        return insights

# Example usage and testing
def main():
    """Test the IntelligentModelSelector"""
    print("🚀 Testing Agent Zero V1 - IntelligentModelSelector")
    
    # Initialize selector
    selector = IntelligentModelSelector()
    
    # Test recommendation
    context = {
        'complexity': 1.2,
        'urgency': 1.0,
        'budget': 'medium'
    }
    
    recommendation = selector.recommend_model(TaskType.CODE_GENERATION, context)
    
    print(f"\n📋 Recommendation:")
    print(f"Model: {recommendation.model_name}")
    print(f"Confidence: {recommendation.confidence_score:.2f}")
    print(f"Reasoning: {recommendation.reasoning}")
    print(f"Alternatives: {', '.join(recommendation.alternatives)}")
    print(f"Est. Cost: ${recommendation.estimated_cost:.4f}")
    print(f"Est. Latency: {recommendation.estimated_latency_ms}ms")
    
    # Test performance insights
    insights = selector.get_performance_insights()
    print(f"\n📊 Performance Insights:")
    print(f"Top performers: {len(insights['top_performing_models'])} categories")
    print(f"Cost opportunities: {len(insights['cost_optimization_opportunities'])} found")

if __name__ == "__main__":
    main()