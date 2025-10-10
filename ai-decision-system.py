# AI-First Decision System - Strategic Module for Agent Zero V1
# Task: A0-22 AI-First Decision System (Week 43)
# Focus: Zastąpienie statycznego mapowania modeli inteligentnym selektorem
# Kluczowe: System proponuje → człowiek decyduje → system się uczy

"""
AI-First Decision System for Agent Zero V1
Intelligent model selector with continuous learning

This system provides:
- Dynamic model selection based on task characteristics
- Human feedback integration for model choice validation
- Continuous learning from decision outcomes
- Cost-quality optimization across model choices
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from pathlib import Path
import sqlite3
from contextlib import contextmanager
import statistics
import numpy as np
from collections import defaultdict, deque

# Import existing components
try:
    import sys
    sys.path.insert(0, '.')
    from simple_tracker import SimpleTracker
    from business_requirements_parser import BusinessRequirementsParser, IntentType, ComplexityLevel
    from feedback_loop_engine import FeedbackLoopEngine
except ImportError as e:
    logging.warning(f"Could not import existing components: {e}")
    # Fallback classes for testing
    class SimpleTracker:
        def track_event(self, event): pass
    class BusinessRequirementsParser:
        def parse_intent(self, text): return None

class ModelType(Enum):
    """Available AI model types"""
    LOCAL_SMALL = "local_small"      # Fast, cheap, basic capability
    LOCAL_MEDIUM = "local_medium"    # Balanced performance
    LOCAL_LARGE = "local_large"      # High capability, slower
    CLOUD_GPT35 = "cloud_gpt35"      # OpenAI GPT-3.5
    CLOUD_GPT4 = "cloud_gpt4"        # OpenAI GPT-4
    CLOUD_CLAUDE = "cloud_claude"    # Anthropic Claude
    CLOUD_GEMINI = "cloud_gemini"    # Google Gemini
    SPECIALIZED_CODE = "specialized_code"    # Code-focused models
    SPECIALIZED_DATA = "specialized_data"    # Data analysis models

class DecisionContext(Enum):
    """Context for decision making"""
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    TESTING = "testing"
    RESEARCH = "research"
    DEMO = "demo"

class QualityMetric(Enum):
    """Quality metrics for model evaluation"""
    ACCURACY = "accuracy"
    SPEED = "speed"
    COST = "cost"
    CREATIVITY = "creativity"
    RELIABILITY = "reliability"
    CONTEXT_UNDERSTANDING = "context_understanding"

@dataclass
class ModelCapabilities:
    """Model capability profile"""
    model_type: ModelType
    name: str
    description: str
    
    # Performance characteristics
    max_tokens: int = 4096
    avg_response_time: float = 2.0  # seconds
    cost_per_1k_tokens: float = 0.001
    availability_sla: float = 0.99
    
    # Capability scores (0-10)
    text_generation: int = 5
    code_generation: int = 5
    data_analysis: int = 5
    reasoning: int = 5
    creativity: int = 5
    factual_accuracy: int = 5
    instruction_following: int = 5
    
    # Technical constraints
    supports_functions: bool = False
    supports_streaming: bool = False
    requires_internet: bool = False
    local_deployment: bool = True
    
    # Usage patterns
    best_for_tasks: List[str] = field(default_factory=list)
    avoid_for_tasks: List[str] = field(default_factory=list)

@dataclass
class TaskCharacteristics:
    """Characteristics of a task that influence model selection"""
    task_id: str
    intent_type: Optional[IntentType] = None
    complexity_level: Optional[ComplexityLevel] = None
    estimated_tokens: int = 1000
    response_time_requirement: float = 30.0  # seconds max
    budget_constraint: float = 0.10  # max cost
    quality_requirements: List[QualityMetric] = field(default_factory=list)
    context: DecisionContext = DecisionContext.DEVELOPMENT
    
    # Task content analysis
    has_code: bool = False
    has_data: bool = False
    requires_creativity: bool = False
    requires_accuracy: bool = False
    language_complexity: int = 5  # 1-10
    domain_specific: bool = False
    
    # Environmental factors
    user_preference: Optional[ModelType] = None
    previous_model_performance: Dict[ModelType, float] = field(default_factory=dict)
    available_budget: float = 1.0
    deadline_pressure: bool = False

@dataclass
class ModelRecommendation:
    """Model recommendation with reasoning"""
    recommended_model: ModelType
    confidence_score: float  # 0-1
    expected_cost: float
    expected_duration: float
    expected_quality: float
    reasoning: str
    alternatives: List[Tuple[ModelType, float]] = field(default_factory=list)  # (model, score)

@dataclass
class DecisionRecord:
    """Record of a decision made by the system"""
    decision_id: str
    timestamp: datetime
    task_characteristics: TaskCharacteristics
    system_recommendation: ModelRecommendation
    human_decision: Optional[ModelType] = None
    human_reasoning: Optional[str] = None
    actual_outcome: Optional[Dict[str, float]] = None  # actual cost, duration, quality
    feedback_score: Optional[int] = None  # 1-5 human satisfaction
    learned_from: bool = False

class ModelRegistry:
    """Registry of available models with their capabilities"""
    
    def __init__(self):
        self.models: Dict[ModelType, ModelCapabilities] = {}
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize with default model configurations"""
        
        # Local models
        self.models[ModelType.LOCAL_SMALL] = ModelCapabilities(
            model_type=ModelType.LOCAL_SMALL,
            name="Llama 3.2 3B",
            description="Fast local model for simple tasks",
            max_tokens=2048,
            avg_response_time=1.0,
            cost_per_1k_tokens=0.0,
            text_generation=6,
            code_generation=4,
            reasoning=5,
            local_deployment=True,
            best_for_tasks=["simple_text", "basic_qa", "formatting"]
        )
        
        self.models[ModelType.LOCAL_MEDIUM] = ModelCapabilities(
            model_type=ModelType.LOCAL_MEDIUM,
            name="Llama 3.1 8B",
            description="Balanced local model",
            max_tokens=4096,
            avg_response_time=3.0,
            cost_per_1k_tokens=0.0,
            text_generation=7,
            code_generation=6,
            reasoning=7,
            local_deployment=True,
            best_for_tasks=["text_generation", "basic_coding", "analysis"]
        )
        
        self.models[ModelType.LOCAL_LARGE] = ModelCapabilities(
            model_type=ModelType.LOCAL_LARGE,
            name="Llama 3.1 70B",
            description="High-capability local model",
            max_tokens=8192,
            avg_response_time=8.0,
            cost_per_1k_tokens=0.0,
            text_generation=8,
            code_generation=8,
            reasoning=8,
            creativity=7,
            local_deployment=True,
            best_for_tasks=["complex_reasoning", "advanced_coding", "creative_writing"]
        )
        
        # Cloud models
        self.models[ModelType.CLOUD_GPT35] = ModelCapabilities(
            model_type=ModelType.CLOUD_GPT35,
            name="GPT-3.5 Turbo",
            description="Fast and cost-effective cloud model",
            max_tokens=4096,
            avg_response_time=2.0,
            cost_per_1k_tokens=0.0015,
            text_generation=7,
            code_generation=7,
            reasoning=7,
            instruction_following=8,
            requires_internet=True,
            local_deployment=False,
            supports_functions=True,
            best_for_tasks=["general_purpose", "api_calls", "structured_output"]
        )
        
        self.models[ModelType.CLOUD_GPT4] = ModelCapabilities(
            model_type=ModelType.CLOUD_GPT4,
            name="GPT-4 Turbo",
            description="Premium cloud model for complex tasks",
            max_tokens=128000,
            avg_response_time=5.0,
            cost_per_1k_tokens=0.01,
            text_generation=9,
            code_generation=9,
            reasoning=9,
            creativity=8,
            factual_accuracy=9,
            requires_internet=True,
            local_deployment=False,
            supports_functions=True,
            best_for_tasks=["complex_reasoning", "research", "high_quality_content"]
        )
        
        self.models[ModelType.CLOUD_CLAUDE] = ModelCapabilities(
            model_type=ModelType.CLOUD_CLAUDE,
            name="Claude 3.5 Sonnet",
            description="Anthropic's advanced reasoning model",
            max_tokens=200000,
            avg_response_time=4.0,
            cost_per_1k_tokens=0.003,
            text_generation=9,
            code_generation=8,
            reasoning=9,
            factual_accuracy=9,
            instruction_following=9,
            requires_internet=True,
            local_deployment=False,
            best_for_tasks=["analysis", "reasoning", "safety_critical"]
        )
        
        # Specialized models
        self.models[ModelType.SPECIALIZED_CODE] = ModelCapabilities(
            model_type=ModelType.SPECIALIZED_CODE,
            name="CodeLlama 34B",
            description="Specialized for code generation",
            max_tokens=16384,
            avg_response_time=5.0,
            cost_per_1k_tokens=0.0,
            text_generation=6,
            code_generation=9,
            reasoning=7,
            local_deployment=True,
            best_for_tasks=["code_generation", "debugging", "refactoring"],
            avoid_for_tasks=["creative_writing", "general_qa"]
        )
    
    def get_model(self, model_type: ModelType) -> Optional[ModelCapabilities]:
        """Get model capabilities by type"""
        return self.models.get(model_type)
    
    def list_models(self, context: Optional[DecisionContext] = None) -> List[ModelCapabilities]:
        """List available models, optionally filtered by context"""
        models = list(self.models.values())
        
        if context == DecisionContext.PRODUCTION:
            # Filter for production-ready models
            models = [m for m in models if m.availability_sla >= 0.99]
        elif context == DecisionContext.DEVELOPMENT:
            # Prefer local models for development
            models = sorted(models, key=lambda m: (not m.local_deployment, m.cost_per_1k_tokens))
        
        return models
    
    def add_model(self, capabilities: ModelCapabilities):
        """Add or update model in registry"""
        self.models[capabilities.model_type] = capabilities
    
    def update_model_performance(self, model_type: ModelType, 
                               metric: str, value: float):
        """Update model performance metrics based on real usage"""
        model = self.models.get(model_type)
        if model and hasattr(model, metric):
            # Exponential moving average for updates
            current_value = getattr(model, metric)
            updated_value = 0.8 * current_value + 0.2 * value
            setattr(model, metric, updated_value)

class DecisionEngine:
    """Core decision engine for model selection"""
    
    def __init__(self, model_registry: ModelRegistry):
        self.model_registry = model_registry
        self.logger = logging.getLogger(__name__)
        
        # Scoring weights for different factors
        self.scoring_weights = {
            'capability_match': 0.25,
            'cost_efficiency': 0.20,
            'performance_speed': 0.15,
            'historical_success': 0.20,
            'availability': 0.10,
            'user_preference': 0.10
        }
    
    def recommend_model(self, task_characteristics: TaskCharacteristics) -> ModelRecommendation:
        """Recommend best model for given task characteristics"""
        
        available_models = self.model_registry.list_models(task_characteristics.context)
        
        if not available_models:
            raise ValueError("No models available for the given context")
        
        # Score each model
        model_scores = []
        
        for model in available_models:
            score = self._calculate_model_score(model, task_characteristics)
            model_scores.append((model.model_type, score))
        
        # Sort by score descending
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not model_scores:
            raise ValueError("No suitable models found")
        
        # Best recommendation
        best_model_type, best_score = model_scores[0]
        best_model = self.model_registry.get_model(best_model_type)
        
        # Calculate expected metrics
        expected_cost = self._calculate_expected_cost(best_model, task_characteristics)
        expected_duration = self._calculate_expected_duration(best_model, task_characteristics)
        expected_quality = self._calculate_expected_quality(best_model, task_characteristics)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(best_model, task_characteristics, best_score)
        
        # Alternative suggestions
        alternatives = model_scores[1:4]  # Top 3 alternatives
        
        return ModelRecommendation(
            recommended_model=best_model_type,
            confidence_score=min(best_score / 10.0, 1.0),
            expected_cost=expected_cost,
            expected_duration=expected_duration,
            expected_quality=expected_quality,
            reasoning=reasoning,
            alternatives=alternatives
        )
    
    def _calculate_model_score(self, model: ModelCapabilities, 
                             task: TaskCharacteristics) -> float:
        """Calculate composite score for model-task fit"""
        
        scores = {}
        
        # 1. Capability match score
        capability_score = self._score_capability_match(model, task)
        scores['capability_match'] = capability_score
        
        # 2. Cost efficiency score
        cost_score = self._score_cost_efficiency(model, task)
        scores['cost_efficiency'] = cost_score
        
        # 3. Performance speed score
        speed_score = self._score_performance_speed(model, task)
        scores['performance_speed'] = speed_score
        
        # 4. Historical success score
        history_score = self._score_historical_performance(model, task)
        scores['historical_success'] = history_score
        
        # 5. Availability score
        availability_score = self._score_availability(model, task)
        scores['availability'] = availability_score
        
        # 6. User preference score
        preference_score = self._score_user_preference(model, task)
        scores['user_preference'] = preference_score
        
        # Calculate weighted total
        total_score = sum(
            scores[factor] * weight 
            for factor, weight in self.scoring_weights.items()
        )
        
        return total_score
    
    def _score_capability_match(self, model: ModelCapabilities, 
                              task: TaskCharacteristics) -> float:
        """Score how well model capabilities match task requirements"""
        
        score = 5.0  # Base score
        
        # Intent type matching
        if task.intent_type:
            if task.intent_type == IntentType.CREATE:
                if task.has_code:
                    score = model.code_generation
                else:
                    score = model.text_generation
            elif task.intent_type == IntentType.ANALYZE:
                score = model.data_analysis
            elif task.intent_type == IntentType.GENERATE:
                score = (model.text_generation + model.creativity) / 2
            else:
                score = model.reasoning
        
        # Complexity adjustments
        if task.complexity_level:
            if task.complexity_level == ComplexityLevel.ENTERPRISE:
                score = (score + model.reasoning + model.factual_accuracy) / 3
            elif task.complexity_level == ComplexityLevel.SIMPLE:
                # Simple tasks don't need high capability
                score = min(score, 7)  # Cap at 7 for simple tasks
        
        # Quality requirements
        if QualityMetric.ACCURACY in task.quality_requirements:
            score = (score + model.factual_accuracy) / 2
        if QualityMetric.CREATIVITY in task.quality_requirements:
            score = (score + model.creativity) / 2
        
        # Task-specific adjustments
        if task.has_code:
            score = (score + model.code_generation) / 2
        if task.requires_creativity:
            score = (score + model.creativity) / 2
        
        return min(score, 10.0)
    
    def _score_cost_efficiency(self, model: ModelCapabilities, 
                             task: TaskCharacteristics) -> float:
        """Score cost efficiency for the task"""
        
        estimated_cost = model.cost_per_1k_tokens * (task.estimated_tokens / 1000)
        
        if estimated_cost <= 0:
            return 10.0  # Free local models get max score
        
        if estimated_cost > task.budget_constraint:
            return 0.0  # Over budget gets zero
        
        # Score based on budget utilization (lower is better)
        budget_utilization = estimated_cost / task.budget_constraint
        cost_score = 10 * (1 - budget_utilization)
        
        return max(0, min(10, cost_score))
    
    def _score_performance_speed(self, model: ModelCapabilities, 
                               task: TaskCharacteristics) -> float:
        """Score performance speed against requirements"""
        
        if model.avg_response_time <= task.response_time_requirement / 2:
            return 10.0  # Significantly faster than required
        elif model.avg_response_time <= task.response_time_requirement:
            return 8.0   # Meets requirement
        elif model.avg_response_time <= task.response_time_requirement * 1.5:
            return 5.0   # Slightly over requirement
        else:
            return 2.0   # Too slow
    
    def _score_historical_performance(self, model: ModelCapabilities, 
                                    task: TaskCharacteristics) -> float:
        """Score based on historical performance for similar tasks"""
        
        # Use previous performance data if available
        if model.model_type in task.previous_model_performance:
            return task.previous_model_performance[model.model_type] * 10
        
        # Default score based on model reliability
        return 5.0 + (model.availability_sla - 0.9) * 50  # Scale 0.9-0.99 to 5-9.5
    
    def _score_availability(self, model: ModelCapabilities, 
                          task: TaskCharacteristics) -> float:
        """Score model availability"""
        
        if task.context == DecisionContext.PRODUCTION:
            # High availability requirement for production
            return model.availability_sla * 10
        else:
            # Less critical for development/testing
            return min(10, model.availability_sla * 10 + 2)
    
    def _score_user_preference(self, model: ModelCapabilities, 
                             task: TaskCharacteristics) -> float:
        """Score user preference if specified"""
        
        if task.user_preference == model.model_type:
            return 10.0
        elif task.user_preference is None:
            return 5.0  # Neutral
        else:
            return 2.0  # Not preferred
    
    def _calculate_expected_cost(self, model: ModelCapabilities, 
                               task: TaskCharacteristics) -> float:
        """Calculate expected cost for the task"""
        return model.cost_per_1k_tokens * (task.estimated_tokens / 1000)
    
    def _calculate_expected_duration(self, model: ModelCapabilities, 
                                   task: TaskCharacteristics) -> float:
        """Calculate expected duration for the task"""
        
        base_duration = model.avg_response_time
        
        # Adjust for task complexity
        if task.complexity_level:
            complexity_multipliers = {
                ComplexityLevel.SIMPLE: 0.8,
                ComplexityLevel.MODERATE: 1.0,
                ComplexityLevel.COMPLEX: 1.5,
                ComplexityLevel.ENTERPRISE: 2.0
            }
            base_duration *= complexity_multipliers.get(task.complexity_level, 1.0)
        
        # Adjust for token count
        token_multiplier = max(1.0, task.estimated_tokens / 2000)
        
        return base_duration * token_multiplier
    
    def _calculate_expected_quality(self, model: ModelCapabilities, 
                                  task: TaskCharacteristics) -> float:
        """Calculate expected quality score"""
        
        # Average relevant capability scores
        if task.has_code:
            return (model.code_generation + model.reasoning) / 2
        elif task.intent_type == IntentType.ANALYZE:
            return (model.data_analysis + model.reasoning) / 2
        elif task.requires_creativity:
            return (model.creativity + model.text_generation) / 2
        else:
            return (model.text_generation + model.reasoning + model.instruction_following) / 3
    
    def _generate_reasoning(self, model: ModelCapabilities, 
                          task: TaskCharacteristics, score: float) -> str:
        """Generate human-readable reasoning for the recommendation"""
        
        reasons = []
        
        # Primary capability match
        if task.has_code and model.code_generation >= 7:
            reasons.append("Strong code generation capabilities")
        elif task.intent_type == IntentType.ANALYZE and model.data_analysis >= 7:
            reasons.append("Excellent data analysis skills")
        elif task.requires_creativity and model.creativity >= 7:
            reasons.append("High creativity for content generation")
        
        # Cost considerations
        if model.cost_per_1k_tokens == 0:
            reasons.append("Zero cost (local model)")
        elif self._calculate_expected_cost(model, task) <= task.budget_constraint * 0.5:
            reasons.append("Cost-effective within budget")
        
        # Performance considerations
        if model.avg_response_time <= task.response_time_requirement * 0.7:
            reasons.append("Fast response time")
        
        # Context appropriateness
        if task.context == DecisionContext.PRODUCTION and model.availability_sla >= 0.99:
            reasons.append("Production-ready reliability")
        elif task.context == DecisionContext.DEVELOPMENT and model.local_deployment:
            reasons.append("Local deployment suitable for development")
        
        # Fallback reasoning
        if not reasons:
            if score >= 7:
                reasons.append("Well-balanced capabilities for this task type")
            else:
                reasons.append("Best available option given constraints")
        
        return ". ".join(reasons) + f" (Score: {score:.1f}/10)"

class LearningEngine:
    """Continuous learning engine for improving decision quality"""
    
    def __init__(self, db_path: str = "ai_decision_learning.db"):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._initialize_database()
        
        # Learning parameters
        self.min_decisions_for_learning = 5
        self.learning_rate = 0.1
    
    def _initialize_database(self):
        """Initialize SQLite database for learning data"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS decision_records (
                    decision_id TEXT PRIMARY KEY,
                    timestamp TIMESTAMP,
                    task_json TEXT,
                    system_recommendation_json TEXT,
                    human_decision TEXT,
                    human_reasoning TEXT,
                    actual_outcome_json TEXT,
                    feedback_score INTEGER,
                    learned_from BOOLEAN DEFAULT FALSE
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_performance_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    model_type TEXT,
                    task_type TEXT,
                    complexity_level TEXT,
                    actual_cost REAL,
                    actual_duration REAL,
                    quality_score REAL,
                    success_rate REAL,
                    timestamp TIMESTAMP
                )
            """)
            
            conn.commit()
    
    def record_decision(self, record: DecisionRecord):
        """Record a decision for learning"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO decision_records
                (decision_id, timestamp, task_json, system_recommendation_json,
                 human_decision, human_reasoning, actual_outcome_json, feedback_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.decision_id,
                record.timestamp,
                json.dumps(asdict(record.task_characteristics)),
                json.dumps(asdict(record.system_recommendation)),
                record.human_decision.value if record.human_decision else None,
                record.human_reasoning,
                json.dumps(record.actual_outcome) if record.actual_outcome else None,
                record.feedback_score
            ))
            
            conn.commit()
    
    def learn_from_decisions(self, decision_engine: DecisionEngine) -> Dict[str, Any]:
        """Learn from recorded decisions and update models"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get unprocessed decisions with outcomes
            cursor.execute("""
                SELECT * FROM decision_records 
                WHERE learned_from = FALSE 
                AND actual_outcome_json IS NOT NULL
                AND feedback_score IS NOT NULL
            """)
            
            unprocessed = cursor.fetchall()
            
            if len(unprocessed) < self.min_decisions_for_learning:
                return {
                    'learned_decisions': 0,
                    'message': f'Need at least {self.min_decisions_for_learning} decisions to learn'
                }
        
        learning_stats = {
            'learned_decisions': 0,
            'weight_updates': 0,
            'model_updates': 0,
            'insights': []
        }
        
        # Analyze decisions by pattern
        decision_patterns = self._analyze_decision_patterns(unprocessed)
        
        # Update decision engine weights
        weight_updates = self._update_decision_weights(decision_engine, decision_patterns)
        learning_stats['weight_updates'] = len(weight_updates)
        learning_stats['insights'].extend(weight_updates)
        
        # Update model performance estimates
        model_updates = self._update_model_performance(decision_patterns)
        learning_stats['model_updates'] = len(model_updates)
        learning_stats['insights'].extend(model_updates)
        
        # Mark decisions as learned from
        decision_ids = [row[0] for row in unprocessed]
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' * len(decision_ids))
            cursor.execute(
                f"UPDATE decision_records SET learned_from = TRUE WHERE decision_id IN ({placeholders})",
                decision_ids
            )
            conn.commit()
        
        learning_stats['learned_decisions'] = len(decision_ids)
        
        self.logger.info(f"Learned from {learning_stats['learned_decisions']} decisions")
        return learning_stats
    
    def _analyze_decision_patterns(self, decisions: List) -> Dict[str, Any]:
        """Analyze patterns in decision data"""
        
        patterns = {
            'human_vs_system_agreement': 0,
            'feedback_by_model': defaultdict(list),
            'cost_accuracy': defaultdict(list),
            'duration_accuracy': defaultdict(list),
            'quality_accuracy': defaultdict(list)
        }
        
        for row in decisions:
            try:
                # Parse JSON data
                task = json.loads(row[2])
                recommendation = json.loads(row[3])
                outcome = json.loads(row[6]) if row[6] else {}
                
                system_model = recommendation.get('recommended_model')
                human_model = row[4]
                feedback_score = row[7]
                
                # Agreement tracking
                if system_model == human_model:
                    patterns['human_vs_system_agreement'] += 1
                
                # Feedback by model
                actual_model = human_model or system_model
                if actual_model and feedback_score:
                    patterns['feedback_by_model'][actual_model].append(feedback_score)
                
                # Prediction accuracy
                if outcome:
                    expected_cost = recommendation.get('expected_cost', 0)
                    actual_cost = outcome.get('cost', 0)
                    if expected_cost > 0 and actual_cost > 0:
                        accuracy = 1.0 - abs(expected_cost - actual_cost) / expected_cost
                        patterns['cost_accuracy'][actual_model].append(accuracy)
                    
                    expected_duration = recommendation.get('expected_duration', 0)
                    actual_duration = outcome.get('duration', 0)
                    if expected_duration > 0 and actual_duration > 0:
                        accuracy = 1.0 - abs(expected_duration - actual_duration) / expected_duration
                        patterns['duration_accuracy'][actual_model].append(accuracy)
            
            except (json.JSONDecodeError, KeyError) as e:
                self.logger.warning(f"Error parsing decision record: {e}")
                continue
        
        return patterns
    
    def _update_decision_weights(self, decision_engine: DecisionEngine, 
                               patterns: Dict[str, Any]) -> List[str]:
        """Update decision engine weights based on learning"""
        
        updates = []
        
        # Calculate agreement rate
        total_decisions = len(patterns['feedback_by_model'])
        if total_decisions > 0:
            agreement_rate = patterns['human_vs_system_agreement'] / total_decisions
            
            if agreement_rate < 0.6:  # Low agreement
                # Increase user preference weight
                old_weight = decision_engine.scoring_weights['user_preference']
                decision_engine.scoring_weights['user_preference'] = min(0.3, old_weight * 1.2)
                updates.append(f"Increased user preference weight due to low agreement ({agreement_rate:.1%})")
            
            elif agreement_rate > 0.8:  # High agreement  
                # Increase confidence in system decisions
                old_weight = decision_engine.scoring_weights['capability_match']
                decision_engine.scoring_weights['capability_match'] = min(0.4, old_weight * 1.1)
                updates.append(f"Increased capability matching weight due to high agreement ({agreement_rate:.1%})")
        
        # Analyze feedback scores
        for model_type, scores in patterns['feedback_by_model'].items():
            if len(scores) >= 3:  # Minimum for statistical significance
                avg_feedback = statistics.mean(scores)
                
                if avg_feedback < 3.0:  # Poor feedback
                    # This model is underperforming - increase cost sensitivity
                    old_weight = decision_engine.scoring_weights['cost_efficiency']
                    decision_engine.scoring_weights['cost_efficiency'] = min(0.3, old_weight * 1.1)
                    updates.append(f"Model {model_type} showing poor feedback ({avg_feedback:.1f}/5), increased cost sensitivity")
        
        return updates
    
    def _update_model_performance(self, patterns: Dict[str, Any]) -> List[str]:
        """Update model performance estimates"""
        
        updates = []
        
        # Update cost prediction accuracy
        for model_type, accuracies in patterns['cost_accuracy'].items():
            if len(accuracies) >= 3:
                avg_accuracy = statistics.mean(accuracies)
                
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO model_performance_history
                        (model_type, task_type, actual_cost, quality_score, timestamp)
                        VALUES (?, 'cost_prediction', ?, ?, ?)
                    """, (model_type, avg_accuracy, avg_accuracy, datetime.now()))
                    conn.commit()
                
                updates.append(f"Updated cost prediction accuracy for {model_type}: {avg_accuracy:.1%}")
        
        # Update duration prediction accuracy  
        for model_type, accuracies in patterns['duration_accuracy'].items():
            if len(accuracies) >= 3:
                avg_accuracy = statistics.mean(accuracies)
                updates.append(f"Updated duration prediction accuracy for {model_type}: {avg_accuracy:.1%}")
        
        return updates
    
    def get_model_insights(self, model_type: ModelType, days: int = 30) -> Dict[str, Any]:
        """Get insights about model performance over time"""
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Get recent performance data
            cursor.execute("""
                SELECT task_type, actual_cost, actual_duration, quality_score, timestamp
                FROM model_performance_history 
                WHERE model_type = ? AND timestamp > datetime('now', '-{} days')
            """.format(days), (model_type.value,))
            
            performance_data = cursor.fetchall()
            
            # Get recent feedback scores
            cursor.execute("""
                SELECT feedback_score, timestamp
                FROM decision_records
                WHERE (human_decision = ? OR (human_decision IS NULL AND system_recommendation_json LIKE ?))
                AND feedback_score IS NOT NULL
                AND timestamp > datetime('now', '-{} days')
            """.format(days), (model_type.value, f'%{model_type.value}%'))
            
            feedback_data = cursor.fetchall()
        
        insights = {
            'model_type': model_type.value,
            'period_days': days,
            'usage_count': len(performance_data) + len(feedback_data),
            'avg_feedback_score': 0,
            'cost_trend': 'stable',
            'duration_trend': 'stable',
            'quality_trend': 'stable',
            'recommendations': []
        }
        
        if feedback_data:
            scores = [row[0] for row in feedback_data]
            insights['avg_feedback_score'] = statistics.mean(scores)
            
            if insights['avg_feedback_score'] < 3.0:
                insights['recommendations'].append("Consider using alternative models due to poor feedback")
            elif insights['avg_feedback_score'] > 4.0:
                insights['recommendations'].append("High satisfaction model - good for similar tasks")
        
        if not performance_data and not feedback_data:
            insights['recommendations'].append("Insufficient data for insights - need more usage")
        
        return insights

class AIFirstDecisionSystem:
    """Main AI-First Decision System class"""
    
    def __init__(self, db_path: str = "ai_decision_system.db"):
        self.model_registry = ModelRegistry()
        self.decision_engine = DecisionEngine(self.model_registry)
        self.learning_engine = LearningEngine(db_path)
        self.logger = logging.getLogger(__name__)
        
        # Integration components
        try:
            self.tracker = SimpleTracker()
            self.business_parser = BusinessRequirementsParser()
        except:
            self.tracker = None
            self.business_parser = None
            self.logger.warning("Could not initialize integration components")
    
    async def recommend_model_for_task(self, business_request: str, 
                                     context: DecisionContext = DecisionContext.DEVELOPMENT,
                                     user_preferences: Optional[Dict] = None) -> ModelRecommendation:
        """Get model recommendation for a business task"""
        
        # Parse business request to extract characteristics
        task_characteristics = await self._extract_task_characteristics(
            business_request, context, user_preferences
        )
        
        # Get recommendation from decision engine
        recommendation = self.decision_engine.recommend_model(task_characteristics)
        
        # Track decision
        if self.tracker:
            self.tracker.track_event({
                'type': 'model_recommendation_generated',
                'business_request': business_request[:100],
                'recommended_model': recommendation.recommended_model.value,
                'confidence': recommendation.confidence_score,
                'context': context.value
            })
        
        return recommendation
    
    async def _extract_task_characteristics(self, business_request: str,
                                          context: DecisionContext,
                                          user_preferences: Optional[Dict]) -> TaskCharacteristics:
        """Extract task characteristics from business request"""
        
        # Default characteristics
        characteristics = TaskCharacteristics(
            task_id=f"task_{int(time.time())}",
            context=context,
            estimated_tokens=len(business_request.split()) * 2,  # Rough estimate
        )
        
        # Parse with business parser if available
        if self.business_parser:
            try:
                intent = self.business_parser.parse_intent(business_request)
                if intent:
                    characteristics.intent_type = intent.primary_action
                    characteristics.complexity_level = intent.complexity
                    
                    # Adjust token estimate based on complexity
                    complexity_multipliers = {
                        ComplexityLevel.SIMPLE: 1.0,
                        ComplexityLevel.MODERATE: 2.0,
                        ComplexityLevel.COMPLEX: 4.0,
                        ComplexityLevel.ENTERPRISE: 8.0
                    }
                    multiplier = complexity_multipliers.get(intent.complexity, 1.0)
                    characteristics.estimated_tokens = int(characteristics.estimated_tokens * multiplier)
            
            except Exception as e:
                self.logger.warning(f"Could not parse business request: {e}")
        
        # Analyze content for code/data patterns
        request_lower = business_request.lower()
        
        code_indicators = ['code', 'function', 'api', 'algorithm', 'programming', 'script']
        characteristics.has_code = any(indicator in request_lower for indicator in code_indicators)
        
        data_indicators = ['data', 'analysis', 'chart', 'report', 'statistics', 'metrics']
        characteristics.has_data = any(indicator in request_lower for indicator in data_indicators)
        
        creative_indicators = ['creative', 'story', 'poem', 'artistic', 'innovative', 'original']
        characteristics.requires_creativity = any(indicator in request_lower for indicator in creative_indicators)
        
        accuracy_indicators = ['accurate', 'precise', 'fact', 'research', 'analysis', 'critical']
        characteristics.requires_accuracy = any(indicator in request_lower for indicator in accuracy_indicators)
        
        # Apply user preferences
        if user_preferences:
            if 'preferred_model' in user_preferences:
                try:
                    characteristics.user_preference = ModelType(user_preferences['preferred_model'])
                except ValueError:
                    pass
            
            if 'budget_constraint' in user_preferences:
                characteristics.budget_constraint = user_preferences['budget_constraint']
            
            if 'max_response_time' in user_preferences:
                characteristics.response_time_requirement = user_preferences['max_response_time']
            
            if 'quality_requirements' in user_preferences:
                characteristics.quality_requirements = [
                    QualityMetric(req) for req in user_preferences['quality_requirements']
                    if req in [m.value for m in QualityMetric]
                ]
        
        return characteristics
    
    def record_human_decision(self, recommendation: ModelRecommendation, 
                            task_characteristics: TaskCharacteristics,
                            human_choice: ModelType, 
                            reasoning: Optional[str] = None) -> str:
        """Record human decision that overrides system recommendation"""
        
        decision_id = f"decision_{int(time.time())}"
        
        record = DecisionRecord(
            decision_id=decision_id,
            timestamp=datetime.now(),
            task_characteristics=task_characteristics,
            system_recommendation=recommendation,
            human_decision=human_choice,
            human_reasoning=reasoning
        )
        
        self.learning_engine.record_decision(record)
        
        # Track in SimpleTracker
        if self.tracker:
            self.tracker.track_event({
                'type': 'human_decision_recorded',
                'decision_id': decision_id,
                'system_recommendation': recommendation.recommended_model.value,
                'human_choice': human_choice.value,
                'agreement': recommendation.recommended_model == human_choice
            })
        
        return decision_id
    
    def record_task_outcome(self, decision_id: str, outcome: Dict[str, float],
                          feedback_score: int):
        """Record actual outcome and feedback for a task"""
        
        # Update decision record
        with sqlite3.connect(self.learning_engine.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE decision_records 
                SET actual_outcome_json = ?, feedback_score = ?
                WHERE decision_id = ?
            """, (json.dumps(outcome), feedback_score, decision_id))
            conn.commit()
        
        # Track in SimpleTracker
        if self.tracker:
            self.tracker.track_event({
                'type': 'task_outcome_recorded',
                'decision_id': decision_id,
                'feedback_score': feedback_score,
                'actual_cost': outcome.get('cost', 0),
                'actual_duration': outcome.get('duration', 0)
            })
    
    async def learn_and_improve(self) -> Dict[str, Any]:
        """Trigger learning process to improve future decisions"""
        
        learning_results = self.learning_engine.learn_from_decisions(self.decision_engine)
        
        # Track learning
        if self.tracker:
            self.tracker.track_event({
                'type': 'system_learning_completed',
                'learned_decisions': learning_results['learned_decisions'],
                'weight_updates': learning_results['weight_updates'],
                'model_updates': learning_results['model_updates']
            })
        
        return learning_results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        # Count available models
        models = self.model_registry.list_models()
        model_count = len(models)
        
        # Get decision statistics
        with sqlite3.connect(self.learning_engine.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM decision_records")
            total_decisions = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(*) FROM decision_records 
                WHERE human_decision = (
                    SELECT json_extract(system_recommendation_json, '$.recommended_model')
                    FROM decision_records d2 
                    WHERE d2.decision_id = decision_records.decision_id
                )
            """)
            agreed_decisions = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(feedback_score) FROM decision_records WHERE feedback_score IS NOT NULL")
            avg_feedback = cursor.fetchone()[0] or 0
        
        agreement_rate = agreed_decisions / total_decisions if total_decisions > 0 else 0
        
        return {
            'available_models': model_count,
            'total_decisions': total_decisions,
            'human_ai_agreement_rate': agreement_rate,
            'average_feedback_score': round(avg_feedback, 2),
            'learning_engine_ready': total_decisions >= self.learning_engine.min_decisions_for_learning,
            'decision_weights': self.decision_engine.scoring_weights
        }

# CLI interface for testing
async def main():
    """CLI interface for testing AI-First Decision System"""
    
    system = AIFirstDecisionSystem()
    
    print("🤖 Agent Zero V1 - AI-First Decision System")
    print("=" * 60)
    
    # Test scenarios
    test_requests = [
        {
            'request': "Create a REST API for user authentication with JWT tokens and password hashing",
            'context': DecisionContext.DEVELOPMENT,
            'preferences': {'budget_constraint': 0.05}
        },
        {
            'request': "Analyze quarterly sales data and generate executive dashboard with KPI insights",
            'context': DecisionContext.PRODUCTION,
            'preferences': {'quality_requirements': ['accuracy', 'reliability']}
        },
        {
            'request': "Write creative marketing copy for our new AI product launch campaign",
            'context': DecisionContext.DEVELOPMENT,
            'preferences': {'preferred_model': 'cloud_gpt4'}
        },
        {
            'request': "Debug complex memory leak in C++ application with performance profiling",
            'context': DecisionContext.PRODUCTION,
            'preferences': {'max_response_time': 10.0}
        }
    ]
    
    print(f"\n📋 Testing {len(test_requests)} scenarios...")
    
    decisions = []
    
    for i, scenario in enumerate(test_requests):
        print(f"\n🔍 Scenario {i+1}: {scenario['request'][:60]}...")
        
        # Get recommendation
        recommendation = await system.recommend_model_for_task(
            scenario['request'],
            scenario['context'],
            scenario['preferences']
        )
        
        print(f"   🎯 Recommended: {recommendation.recommended_model.value}")
        print(f"   📊 Confidence: {recommendation.confidence_score:.1%}")
        print(f"   💰 Expected Cost: ${recommendation.expected_cost:.4f}")
        print(f"   ⏱️  Expected Duration: {recommendation.expected_duration:.1f}s")
        print(f"   🏆 Expected Quality: {recommendation.expected_quality:.1f}/10")
        print(f"   💡 Reasoning: {recommendation.reasoning}")
        
        # Simulate human decision (sometimes agree, sometimes override)
        if i % 3 == 0:  # Override every 3rd recommendation
            human_choice = recommendation.alternatives[0][0] if recommendation.alternatives else recommendation.recommended_model
            decision_id = system.record_human_decision(
                recommendation, 
                await system._extract_task_characteristics(scenario['request'], scenario['context'], scenario['preferences']),
                human_choice,
                "Testing alternative model for comparison"
            )
            print(f"   👤 Human Override: {human_choice.value}")
        else:
            decision_id = system.record_human_decision(
                recommendation,
                await system._extract_task_characteristics(scenario['request'], scenario['context'], scenario['preferences']),
                recommendation.recommended_model,
                "Agreed with system recommendation"
            )
            print(f"   ✅ Human Agreed")
        
        # Simulate outcome
        simulated_outcome = {
            'cost': recommendation.expected_cost * (0.8 + 0.4 * (i / len(test_requests))),
            'duration': recommendation.expected_duration * (0.9 + 0.2 * (i / len(test_requests))),
            'quality': recommendation.expected_quality * (0.9 + 0.1 * (i / len(test_requests)))
        }
        feedback_score = 4 + (i % 2)  # Alternate between 4 and 5
        
        system.record_task_outcome(decision_id, simulated_outcome, feedback_score)
        decisions.append(decision_id)
        
        print(f"   📝 Outcome recorded (Feedback: {feedback_score}/5)")
    
    # Trigger learning
    print(f"\n🧠 Learning from {len(decisions)} decisions...")
    learning_results = await system.learn_and_improve()
    
    print(f"   ✅ Learned from: {learning_results['learned_decisions']} decisions")
    print(f"   🎛️  Weight updates: {learning_results['weight_updates']}")
    print(f"   📈 Model updates: {learning_results['model_updates']}")
    
    # Show insights
    if learning_results['insights']:
        print(f"\n💡 Key Insights:")
        for insight in learning_results['insights'][:3]:
            print(f"   • {insight}")
    
    # System status
    status = system.get_system_status()
    print(f"\n📊 System Status:")
    print(f"   Models Available: {status['available_models']}")
    print(f"   Total Decisions: {status['total_decisions']}")
    print(f"   Agreement Rate: {status['human_ai_agreement_rate']:.1%}")
    print(f"   Avg Feedback: {status['average_feedback_score']:.1f}/5")
    print(f"   Learning Ready: {status['learning_engine_ready']}")
    
    print(f"\n✅ AI-First Decision System test completed!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(main())