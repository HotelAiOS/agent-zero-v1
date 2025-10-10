#!/usr/bin/env python3
"""
Agent Zero V1 - Intelligent Model Selector
V2.0 Intelligence Layer Component - Week 43 Implementation

Dynamiczny wybór modeli AI z uczeniem maszynowym i multi-criteria decision algorithm.
Integruje z istniejącym SimpleTracker dla continuous learning.
"""

import json
import sqlite3
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from enum import Enum
import math

# Import existing components
import sys
sys.path.append('.')
from simple_tracker import SimpleTracker

class TaskType(Enum):
    """Typy zadań wspierane przez system"""
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    PIPELINE = "pipeline"
    BUSINESS_PARSING = "business_parsing"
    ORCHESTRATION = "orchestration"

class ModelProvider(Enum):
    """Dostawcy modeli AI"""
    LOCAL_OLLAMA = "local_ollama"
    CLOUD_OPENAI = "cloud_openai"
    CLOUD_ANTHROPIC = "cloud_anthropic"

@dataclass
class ModelCapabilities:
    """Definicja możliwości modelu AI"""
    name: str
    provider: ModelProvider
    cost_per_token: float
    avg_latency_ms: int
    quality_score: float  # 0.0-1.0
    specializations: List[TaskType]
    max_context: int
    supports_function_calling: bool
    reliability_score: float  # 0.0-1.0 based on uptime

@dataclass
class SelectionCriteria:
    """Kryteria wyboru modelu"""
    task_type: TaskType
    priority: str  # "cost", "quality", "speed", "balanced"
    max_cost_per_task: Optional[float] = None
    max_latency_ms: Optional[int] = None
    min_quality_threshold: float = 0.6
    context_requirements: int = 1000

@dataclass
class ModelRecommendation:
    """Rekomendacja modelu z uzasadnieniem"""
    recommended_model: str
    confidence_score: float
    reasoning: str
    alternatives: List[Tuple[str, float]]  # (model, score)
    cost_estimate: float
    latency_estimate: int
    quality_estimate: float
    decision_factors: Dict[str, float]

class DecisionEngine:
    """Silnik podejmowania decyzji z algorytmami ML"""
    
    def __init__(self):
        self.weights = {
            'cost': 0.15,
            'quality': 0.5,
            'latency': 0.15,
            'human_acceptance': 0.2
        }
        self.learning_rate = 0.1
    
    def calculate_score(self, model_stats: Dict, criteria: SelectionCriteria) -> float:
        """Oblicza wielokryterialny score dla modelu"""
        
        # Normalizacja metryk (0.0-1.0, gdzie wyższe = lepsze)
        cost_score = self._normalize_cost(model_stats.get('avg_cost', 0.01))
        quality_score = model_stats.get('avg_rating', 2.5) / 5.0
        latency_score = self._normalize_latency(model_stats.get('avg_latency', 1000))
        acceptance_score = model_stats.get('human_acceptance_rate', 0.5)
        
        # Aplikuj priority weighting
        adjusted_weights = self._adjust_weights_for_priority(criteria.priority)
        
        total_score = (
            cost_score * adjusted_weights['cost'] +
            quality_score * adjusted_weights['quality'] +
            latency_score * adjusted_weights['latency'] +
            acceptance_score * adjusted_weights['human_acceptance']
        )
        
        return min(max(total_score, 0.0), 1.0)
    
    def _normalize_cost(self, cost: float) -> float:
        """Normalizuje koszt (niższy koszt = wyższy score)"""
        if cost <= 0:
            return 1.0
        # Logarytmiczna normalizacja - koszt $0.001 = 1.0, $0.1 = 0.0
        return max(0.0, 1.0 - math.log10(cost * 1000 + 1) / 3.0)
    
    def _normalize_latency(self, latency_ms: int) -> float:
        """Normalizuje latencję (niższa latencja = wyższy score)"""
        # 0ms = 1.0, 5000ms = 0.0
        return max(0.0, 1.0 - (latency_ms / 5000.0))
    
    def _adjust_weights_for_priority(self, priority: str) -> Dict[str, float]:
        """Dostosowuje wagi na podstawie priorytetu"""
        if priority == "cost":
            return {'cost': 0.6, 'quality': 0.2, 'latency': 0.1, 'human_acceptance': 0.1}
        elif priority == "quality":
            return {'cost': 0.1, 'quality': 0.7, 'latency': 0.1, 'human_acceptance': 0.1}
        elif priority == "speed":
            return {'cost': 0.1, 'quality': 0.2, 'latency': 0.6, 'human_acceptance': 0.1}
        else:  # balanced
            return self.weights.copy()
    
    def update_weights_from_feedback(self, feedback_data: List[Dict]):
        """Aktualizuje wagi na podstawie human feedback"""
        if len(feedback_data) < 5:
            return  # Potrzebujemy minimum danych
        
        # Analiza wzorców w overrides i feedback
        for feedback in feedback_data[-20:]:  # Ostatnie 20 decyzji
            if feedback.get('was_overridden') and feedback.get('user_choice'):
                # Jeśli user wybrał inny model, dowiedz się dlaczego
                self._adjust_weights_based_on_override(feedback)
    
    def _adjust_weights_based_on_override(self, feedback: Dict):
        """Dostosowuje wagi na podstawie user override"""
        # Prosta heurystyka - można rozszerzyć o ML
        recommended_model = feedback.get('recommended_model')
        chosen_model = feedback.get('user_choice')
        
        # Analiza charakterystyk - jeśli user wybiera tańsze modele
        # zwiększ wagę cost, jeśli wybiera szybsze - zwiększ wagę speed
        pass  # TODO: Implementacja zaawansowanego uczenia

class IntelligentModelSelector:
    """
    Główna klasa intelligent model selection z continuous learning
    """
    
    def __init__(self, tracker: Optional[SimpleTracker] = None):
        self.tracker = tracker or SimpleTracker()
        self.decision_engine = DecisionEngine()
        self.logger = self._setup_logging()
        
        # Katalog dostępnych modeli
        self.available_models = self._initialize_model_catalog()
        
        # Cache dla performance
        self._performance_cache = {}
        self._cache_timestamp = datetime.now()
    
    def _setup_logging(self) -> logging.Logger:
        """Konfiguracja logowania"""
        logger = logging.getLogger('intelligent_selector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_model_catalog(self) -> Dict[str, ModelCapabilities]:
        """Inicjalizuje katalog dostępnych modeli"""
        return {
            # Local Ollama models
            "llama3.2-3b": ModelCapabilities(
                name="llama3.2-3b",
                provider=ModelProvider.LOCAL_OLLAMA,
                cost_per_token=0.0,
                avg_latency_ms=800,
                quality_score=0.75,
                specializations=[TaskType.CHAT, TaskType.ANALYSIS],
                max_context=4096,
                supports_function_calling=False,
                reliability_score=0.95
            ),
            "qwen2.5-coder:7b": ModelCapabilities(
                name="qwen2.5-coder:7b",
                provider=ModelProvider.LOCAL_OLLAMA,
                cost_per_token=0.0,
                avg_latency_ms=1200,
                quality_score=0.85,
                specializations=[TaskType.CODE_GENERATION],
                max_context=8192,
                supports_function_calling=True,
                reliability_score=0.90
            ),
            "mistral:7b": ModelCapabilities(
                name="mistral:7b",
                provider=ModelProvider.LOCAL_OLLAMA,
                cost_per_token=0.0,
                avg_latency_ms=900,
                quality_score=0.80,
                specializations=[TaskType.CHAT, TaskType.BUSINESS_PARSING],
                max_context=4096,
                supports_function_calling=False,
                reliability_score=0.92
            ),
            # Cloud models (dla porównania)
            "gpt-4": ModelCapabilities(
                name="gpt-4",
                provider=ModelProvider.CLOUD_OPENAI,
                cost_per_token=0.00003,
                avg_latency_ms=2000,
                quality_score=0.95,
                specializations=list(TaskType),
                max_context=8192,
                supports_function_calling=True,
                reliability_score=0.99
            ),
            "claude-3": ModelCapabilities(
                name="claude-3",
                provider=ModelProvider.CLOUD_ANTHROPIC,
                cost_per_token=0.000015,
                avg_latency_ms=1800,
                quality_score=0.92,
                specializations=[TaskType.ANALYSIS, TaskType.CODE_GENERATION],
                max_context=100000,
                supports_function_calling=True,
                reliability_score=0.98
            )
        }
    
    def select_optimal_model(
        self, 
        criteria: SelectionCriteria,
        context: Optional[Dict] = None
    ) -> ModelRecommendation:
        """
        Główna metoda wyboru optymalnego modelu
        
        Args:
            criteria: Kryteria wyboru modelu
            context: Dodatkowy kontekst (user preferences, project settings)
        
        Returns:
            ModelRecommendation z uzasadnieniem i alternatywami
        """
        
        self.logger.info(f"Selecting model for task: {criteria.task_type}")
        
        # 1. Filtruj modele spełniające podstawowe wymagania
        candidate_models = self._filter_candidate_models(criteria)
        
        if not candidate_models:
            # Fallback do podstawowego modelu
            return self._create_fallback_recommendation(criteria)
        
        # 2. Pobierz historical performance data
        performance_data = self._get_performance_data(list(candidate_models.keys()))
        
        # 3. Oblicz scores dla każdego modelu
        model_scores = {}
        detailed_factors = {}
        
        for model_name, capabilities in candidate_models.items():
            model_stats = performance_data.get(model_name, {})
            score = self.decision_engine.calculate_score(model_stats, criteria)
            
            model_scores[model_name] = score
            detailed_factors[model_name] = {
                'base_quality': capabilities.quality_score,
                'historical_rating': model_stats.get('avg_rating', 2.5) / 5.0,
                'cost_efficiency': self.decision_engine._normalize_cost(
                    capabilities.cost_per_token * 1000  # Estimate per task
                ),
                'speed_score': self.decision_engine._normalize_latency(
                    capabilities.avg_latency_ms
                ),
                'specialization_bonus': self._calculate_specialization_bonus(
                    capabilities, criteria.task_type
                )
            }
        
        # 4. Wybierz najlepszy model
        best_model = max(model_scores.items(), key=lambda x: x[1])
        recommended_model_name = best_model[0]
        confidence = best_model[1]
        
        # 5. Przygotuj alternatywy
        alternatives = sorted(
            [(name, score) for name, score in model_scores.items() 
             if name != recommended_model_name],
            key=lambda x: x[1], 
            reverse=True
        )[:3]  # Top 3 alternatywy
        
        # 6. Generate reasoning
        reasoning = self._generate_reasoning(
            recommended_model_name,
            candidate_models[recommended_model_name],
            detailed_factors[recommended_model_name],
            criteria
        )
        
        # 7. Estimate cost and performance
        recommended_model = candidate_models[recommended_model_name]
        cost_estimate = self._estimate_cost(recommended_model, criteria)
        
        return ModelRecommendation(
            recommended_model=recommended_model_name,
            confidence_score=confidence,
            reasoning=reasoning,
            alternatives=alternatives,
            cost_estimate=cost_estimate,
            latency_estimate=recommended_model.avg_latency_ms,
            quality_estimate=recommended_model.quality_score,
            decision_factors=detailed_factors[recommended_model_name]
        )
    
    def _filter_candidate_models(self, criteria: SelectionCriteria) -> Dict[str, ModelCapabilities]:
        """Filtruje modele spełniające podstawowe wymagania"""
        candidates = {}
        
        for name, model in self.available_models.items():
            # Sprawdź hard constraints
            if criteria.max_cost_per_task and model.cost_per_token * 1000 > criteria.max_cost_per_task:
                continue
            
            if criteria.max_latency_ms and model.avg_latency_ms > criteria.max_latency_ms:
                continue
            
            if model.quality_score < criteria.min_quality_threshold:
                continue
            
            if criteria.context_requirements > model.max_context:
                continue
            
            candidates[name] = model
        
        return candidates
    
    def _get_performance_data(self, model_names: List[str]) -> Dict[str, Dict]:
        """Pobiera dane o wydajności modeli z SimpleTracker"""
        
        # Sprawdź cache
        if (datetime.now() - self._cache_timestamp).seconds < 300:  # 5 min cache
            cached_data = {name: self._performance_cache.get(name, {}) 
                          for name in model_names}
            if all(cached_data.values()):
                return cached_data
        
        # Pobierz fresh data z trackera
        performance_data = {}
        
        if hasattr(self.tracker, 'get_model_comparison'):
            model_stats = self.tracker.get_model_comparison(days=7)
            
            for model_name in model_names:
                stats = model_stats.get(model_name, {})
                
                # Oblicz human acceptance rate
                human_acceptance = self._calculate_human_acceptance_rate(model_name)
                stats['human_acceptance_rate'] = human_acceptance
                
                performance_data[model_name] = stats
        
        # Update cache
        self._performance_cache.update(performance_data)
        self._cache_timestamp = datetime.now()
        
        return performance_data
    
    def _calculate_human_acceptance_rate(self, model_name: str) -> float:
        """Oblicza rate akceptacji przez użytkowników"""
        try:
            cursor = self.tracker.conn.execute('''
                SELECT 
                    COUNT(*) as total_recommendations,
                    SUM(CASE WHEN model_used = model_recommended THEN 1 ELSE 0 END) as accepted
                FROM tasks 
                WHERE model_recommended = ? AND timestamp >= datetime('now', '-30 days')
            ''', (model_name,))
            
            result = cursor.fetchone()
            if result and result[0] > 0:
                return result[1] / result[0]
        except Exception as e:
            self.logger.warning(f"Error calculating acceptance rate for {model_name}: {e}")
        
        return 0.5  # Default neutral value
    
    def _calculate_specialization_bonus(self, model: ModelCapabilities, task_type: TaskType) -> float:
        """Oblicza bonus za specjalizację modelu"""
        if task_type in model.specializations:
            return 0.15  # 15% bonus za specjalizację
        return 0.0
    
    def _estimate_cost(self, model: ModelCapabilities, criteria: SelectionCriteria) -> float:
        """Szacuje koszt wykonania zadania"""
        # Proste oszacowanie na podstawie typu zadania i modelu
        base_tokens = {
            TaskType.CHAT: 500,
            TaskType.CODE_GENERATION: 1500,
            TaskType.ANALYSIS: 2000,
            TaskType.PIPELINE: 800,
            TaskType.BUSINESS_PARSING: 1200,
            TaskType.ORCHESTRATION: 1000
        }
        
        estimated_tokens = base_tokens.get(criteria.task_type, 1000)
        return model.cost_per_token * estimated_tokens
    
    def _generate_reasoning(
        self, 
        model_name: str, 
        model: ModelCapabilities, 
        factors: Dict[str, float],
        criteria: SelectionCriteria
    ) -> str:
        """Generuje human-readable uzasadnienie wyboru"""
        
        reasons = []
        
        # Główne mocne strony
        if factors['base_quality'] > 0.8:
            reasons.append(f"wysoką jakość bazową ({factors['base_quality']:.1%})")
        
        if factors['cost_efficiency'] > 0.8:
            reasons.append("doskonałą efektywność kosztową")
        
        if factors['speed_score'] > 0.8:
            reasons.append(f"szybkość odpowiedzi ({model.avg_latency_ms}ms)")
        
        if factors['specialization_bonus'] > 0:
            reasons.append(f"specjalizację w {criteria.task_type.value}")
        
        # Historical performance
        if factors['historical_rating'] > 0.8:
            reasons.append("doskonałe oceny użytkowników")
        
        main_reason = f"Model {model_name} został wybrany ze względu na " + ", ".join(reasons)
        
        # Dodatkowe informacje
        additional_info = []
        
        if model.provider == ModelProvider.LOCAL_OLLAMA:
            additional_info.append("Działa lokalnie (zero kosztów API)")
        
        if model.supports_function_calling:
            additional_info.append("Wspiera function calling")
        
        if model.reliability_score > 0.95:
            additional_info.append(f"Bardzo wysoka niezawodność ({model.reliability_score:.1%})")
        
        if additional_info:
            main_reason += ". " + ". ".join(additional_info) + "."
        else:
            main_reason += "."
        
        return main_reason
    
    def _create_fallback_recommendation(self, criteria: SelectionCriteria) -> ModelRecommendation:
        """Tworzy fallback recommendation gdy żaden model nie spełnia kryteriów"""
        fallback_model = "llama3.2-3b"  # Safe default
        
        return ModelRecommendation(
            recommended_model=fallback_model,
            confidence_score=0.3,
            reasoning=f"Fallback do {fallback_model} - żaden model nie spełnił wszystkich kryteriów",
            alternatives=[],
            cost_estimate=0.0,
            latency_estimate=800,
            quality_estimate=0.75,
            decision_factors={}
        )
    
    def learn_from_feedback(self, task_id: str, actual_model_used: str, user_rating: int):
        """
        Uczy się z feedback użytkownika
        
        Args:
            task_id: ID zadania
            actual_model_used: Model faktycznie użyty
            user_rating: Ocena użytkownika (1-5)
        """
        
        try:
            # Pobierz oryginalne recommendation dla tego task
            cursor = self.tracker.conn.execute(
                "SELECT model_recommended FROM tasks WHERE id = ?",
                (task_id,)
            )
            result = cursor.fetchone()
            
            if result:
                recommended_model = result[0]
                was_overridden = recommended_model != actual_model_used
                
                feedback_data = {
                    'task_id': task_id,
                    'recommended_model': recommended_model,
                    'actual_model': actual_model_used,
                    'was_overridden': was_overridden,
                    'user_rating': user_rating,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Update decision engine weights
                self.decision_engine.update_weights_from_feedback([feedback_data])
                
                self.logger.info(f"Learning from feedback: task {task_id}, "
                               f"rating {user_rating}, override: {was_overridden}")
        
        except Exception as e:
            self.logger.error(f"Error learning from feedback: {e}")
    
    def get_model_recommendations_summary(self, days: int = 7) -> Dict:
        """Zwraca podsumowanie rekomendacji dla ostatnich dni"""
        
        try:
            cursor = self.tracker.conn.execute('''
                SELECT 
                    model_recommended,
                    COUNT(*) as recommendation_count,
                    SUM(CASE WHEN model_used = model_recommended THEN 1 ELSE 0 END) as acceptance_count,
                    AVG(CASE WHEN f.rating IS NOT NULL THEN f.rating ELSE NULL END) as avg_rating
                FROM tasks t
                LEFT JOIN feedback f ON t.id = f.task_id
                WHERE t.timestamp >= datetime('now', '-{} days')
                GROUP BY model_recommended
                ORDER BY recommendation_count DESC
            '''.format(days))
            
            results = {}
            for row in cursor.fetchall():
                model, rec_count, acc_count, avg_rating = row
                results[model] = {
                    'recommendations': rec_count,
                    'acceptance_rate': (acc_count / rec_count) if rec_count > 0 else 0.0,
                    'avg_rating': avg_rating or 0.0
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting recommendations summary: {e}")
            return {}


# === CLI Integration Functions ===

def enhance_cli_with_intelligent_selection():
    """
    Funkcje do integracji z CLI - gotowe do użycia w cli/__main__.py
    """
    
    def get_intelligent_model_recommendation(task_type: str, priority: str = "balanced") -> str:
        """
        Wrapper function for CLI integration
        
        Usage in CLI:
            model = get_intelligent_model_recommendation("code_generation", "quality")
        """
        
        selector = IntelligentModelSelector()
        
        # Convert string to enum
        try:
            task_enum = TaskType(task_type)
        except ValueError:
            task_enum = TaskType.CHAT  # Default fallback
        
        criteria = SelectionCriteria(
            task_type=task_enum,
            priority=priority
        )
        
        recommendation = selector.select_optimal_model(criteria)
        
        return recommendation.recommended_model
    
    def explain_model_choice(task_type: str, priority: str = "balanced") -> Dict:
        """
        Returns detailed explanation for CLI display
        
        Usage in CLI:
            explanation = explain_model_choice("code_generation", "quality")
            console.print(explanation['reasoning'])
        """
        
        selector = IntelligentModelSelector()
        
        try:
            task_enum = TaskType(task_type)
        except ValueError:
            task_enum = TaskType.CHAT
        
        criteria = SelectionCriteria(
            task_type=task_enum,
            priority=priority
        )
        
        recommendation = selector.select_optimal_model(criteria)
        
        return {
            'recommended_model': recommendation.recommended_model,
            'confidence': recommendation.confidence_score,
            'reasoning': recommendation.reasoning,
            'alternatives': recommendation.alternatives,
            'cost_estimate': recommendation.cost_estimate
        }

# === Testing Functions ===

def test_intelligent_selector():
    """Testy funkcjonalne dla intelligent selector"""
    
    print("=== TESTING INTELLIGENT MODEL SELECTOR ===\n")
    
    # Initialize
    selector = IntelligentModelSelector()
    
    # Test 1: Code generation task
    print("Test 1: Code generation (quality priority)")
    criteria = SelectionCriteria(
        task_type=TaskType.CODE_GENERATION,
        priority="quality"
    )
    
    recommendation = selector.select_optimal_model(criteria)
    print(f"Recommended: {recommendation.recommended_model}")
    print(f"Confidence: {recommendation.confidence_score:.2f}")
    print(f"Reasoning: {recommendation.reasoning}")
    print(f"Alternatives: {recommendation.alternatives}")
    print()
    
    # Test 2: Chat task with cost priority
    print("Test 2: Chat (cost priority)")
    criteria = SelectionCriteria(
        task_type=TaskType.CHAT,
        priority="cost"
    )
    
    recommendation = selector.select_optimal_model(criteria)
    print(f"Recommended: {recommendation.recommended_model}")
    print(f"Confidence: {recommendation.confidence_score:.2f}")
    print(f"Reasoning: {recommendation.reasoning}")
    print()
    
    # Test 3: Speed priority
    print("Test 3: Analysis (speed priority)")
    criteria = SelectionCriteria(
        task_type=TaskType.ANALYSIS,
        priority="speed",
        max_latency_ms=1000
    )
    
    recommendation = selector.select_optimal_model(criteria)
    print(f"Recommended: {recommendation.recommended_model}")
    print(f"Confidence: {recommendation.confidence_score:.2f}")
    print(f"Reasoning: {recommendation.reasoning}")
    print()
    
    # Test 4: CLI integration
    print("Test 4: CLI Integration Functions")
    model = enhance_cli_with_intelligent_selection()
    
    rec_model = get_intelligent_model_recommendation("code_generation", "quality")
    print(f"CLI Recommendation: {rec_model}")
    
    explanation = explain_model_choice("chat", "balanced")
    print(f"CLI Explanation: {explanation['reasoning']}")
    
    print("\n=== ALL TESTS COMPLETED ===")

if __name__ == "__main__":
    test_intelligent_selector()