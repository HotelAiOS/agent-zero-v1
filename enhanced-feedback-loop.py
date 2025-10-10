#!/usr/bin/env python3
"""
Agent Zero V1 - Enhanced Feedback Loop Engine  
V2.0 Intelligence Layer Component - Week 43 Implementation

Advanced Learning Capabilities:
- Pattern-based weight adjustment - adaptacja kryteriów decyzyjnych
- Human preference learning - uczenie się z wyborów użytkownika  
- Confidence score evolution - poprawa pewności rekomendacji
- Cross-context knowledge transfer - przenoszenie wiedzy między typami zadań

Rozszerza istniejący feedback-loop-engine.py o zaawansowane uczenie maszynowe.
"""

import json
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from enum import Enum
from collections import defaultdict
import logging
import math

# Import existing components
import sys
sys.path.append('.')
from simple_tracker import SimpleTracker

try:
    # Try to import existing feedback loop engine
    exec(open('feedback-loop-engine.py').read(), globals())
    has_existing_engine = True
except FileNotFoundError:
    has_existing_engine = False
    # Create minimal base class if needed
    class FeedbackLoopEngine:
        def __init__(self):
            pass

class LearningPattern(Enum):
    """Typy wzorców uczenia"""
    USER_PREFERENCE = "USER_PREFERENCE"         # Preferencje użytkownika
    TASK_CONTEXT = "TASK_CONTEXT"              # Kontekst zadania
    TEMPORAL = "TEMPORAL"                      # Wzorce czasowe
    MODEL_PERFORMANCE = "MODEL_PERFORMANCE"     # Wydajność modeli
    COST_QUALITY_TRADEOFF = "COST_QUALITY_TRADEOFF"  # Trade-off koszt vs jakość

@dataclass
class UserPreference:
    """Preferencje użytkownika lub projektu"""
    user_id: str
    preference_type: str  # "model_choice", "quality_threshold", "cost_sensitivity"
    preference_value: Any
    confidence: float     # 0.0-1.0 jak pewna jest ta preferencja
    learned_from_samples: int
    last_updated: datetime
    context: Optional[Dict] = None

@dataclass 
class ContextPattern:
    """Wzorzec kontekstowy w systemie"""
    pattern_id: str
    pattern_type: LearningPattern
    conditions: Dict[str, Any]  # Warunki gdy wzorzec się aplikuje
    outcomes: Dict[str, Any]    # Obserwowane rezultaty
    confidence: float
    sample_count: int
    success_rate: float
    last_seen: datetime

@dataclass
class KnowledgeTransfer:
    """Transfer wiedzy między kontekstami"""
    source_context: str
    target_context: str
    transferred_knowledge: Dict[str, Any]
    applicability_score: float  # 0.0-1.0 jak dobrze się aplikuje
    validation_samples: int

class PatternRecognition:
    """Rozpoznawanie wzorców w danych Kaizen"""
    
    def __init__(self, tracker: SimpleTracker):
        self.tracker = tracker
        self.logger = logging.getLogger('pattern_recognition')
    
    def discover_user_preferences(self, user_id: str = "default", days: int = 30) -> List[UserPreference]:
        """Odkrywa preferencje użytkownika z historii decyzji"""
        
        preferences = []
        
        try:
            # Analiza overrides - gdy user wybiera inny model niż AI
            cursor = self.tracker.conn.execute('''
                SELECT 
                    model_recommended,
                    model_used,
                    task_type,
                    COUNT(*) as override_count
                FROM tasks 
                WHERE model_used != model_recommended 
                AND timestamp >= datetime('now', '-{} days')
                GROUP BY model_recommended, model_used, task_type
                HAVING override_count >= 3
            '''.format(days))
            
            for row in cursor.fetchall():
                recommended, used, task_type, count = row
                
                # User prefers `used` over `recommended` for `task_type`
                preference = UserPreference(
                    user_id=user_id,
                    preference_type="model_choice", 
                    preference_value={
                        'preferred_model': used,
                        'over_model': recommended,
                        'for_task_type': task_type
                    },
                    confidence=min(count / 10.0, 1.0),  # 10 samples = full confidence
                    learned_from_samples=count,
                    last_updated=datetime.now(),
                    context={'discovery_method': 'override_analysis'}
                )
                preferences.append(preference)
            
            # Analiza quality thresholds - jaką jakość user akceptuje
            cursor = self.tracker.conn.execute('''
                SELECT 
                    task_type,
                    AVG(rating) as avg_accepted_quality,
                    MIN(rating) as min_accepted,
                    COUNT(*) as sample_count
                FROM tasks t
                JOIN feedback f ON t.id = f.task_id
                WHERE f.rating >= 3  -- Ratings >= 3 = accepted
                AND t.timestamp >= datetime('now', '-{} days')
                GROUP BY task_type
                HAVING sample_count >= 5
            '''.format(days))
            
            for row in cursor.fetchall():
                task_type, avg_quality, min_accepted, samples = row
                
                preference = UserPreference(
                    user_id=user_id,
                    preference_type="quality_threshold",
                    preference_value={
                        'task_type': task_type,
                        'min_acceptable_quality': min_accepted,
                        'preferred_quality': avg_quality
                    },
                    confidence=min(samples / 20.0, 1.0),  # 20 samples = full confidence
                    learned_from_samples=samples,
                    last_updated=datetime.now(),
                    context={'discovery_method': 'quality_analysis'}
                )
                preferences.append(preference)
        
        except Exception as e:
            self.logger.error(f"Error discovering user preferences: {e}")
        
        return preferences
    
    def identify_context_patterns(self, days: int = 30) -> List[ContextPattern]:
        """Identyfikuje wzorce kontekstowe w zadaniach"""
        
        patterns = []
        
        try:
            # Wzorzec: Typ zadania + model → sukces
            cursor = self.tracker.conn.execute('''
                SELECT 
                    t.task_type,
                    t.model_used,
                    COUNT(*) as total_count,
                    AVG(CASE WHEN f.rating >= 4 THEN 1.0 ELSE 0.0 END) as success_rate,
                    AVG(f.rating) as avg_rating,
                    AVG(t.cost_usd) as avg_cost,
                    AVG(t.latency_ms) as avg_latency
                FROM tasks t
                LEFT JOIN feedback f ON t.id = f.task_id
                WHERE t.timestamp >= datetime('now', '-{} days')
                GROUP BY t.task_type, t.model_used
                HAVING total_count >= 5
            '''.format(days))
            
            for row in cursor.fetchall():
                task_type, model, count, success_rate, avg_rating, avg_cost, avg_latency = row
                
                # Utwórz wzorzec tylko jeśli ma znaczące dane
                if success_rate is not None and count >= 5:
                    pattern = ContextPattern(
                        pattern_id=f"{task_type}_{model}_{datetime.now().strftime('%Y%m')}",
                        pattern_type=LearningPattern.MODEL_PERFORMANCE,
                        conditions={
                            'task_type': task_type,
                            'model_used': model
                        },
                        outcomes={
                            'success_rate': success_rate,
                            'avg_rating': avg_rating or 2.5,
                            'avg_cost': avg_cost or 0.0,
                            'avg_latency': avg_latency or 1000
                        },
                        confidence=min(count / 20.0, 1.0),
                        sample_count=count,
                        success_rate=success_rate,
                        last_seen=datetime.now()
                    )
                    patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error identifying context patterns: {e}")
        
        return patterns
    
    def detect_temporal_patterns(self, days: int = 30) -> List[ContextPattern]:
        """Wykrywa wzorce czasowe (np. wydajność w różnych porach)"""
        
        patterns = []
        
        try:
            # Analiza wydajności według godzin
            cursor = self.tracker.conn.execute('''
                SELECT 
                    strftime('%H', timestamp) as hour,
                    AVG(latency_ms) as avg_latency,
                    AVG(CASE WHEN f.rating IS NOT NULL THEN f.rating ELSE 3 END) as avg_rating,
                    COUNT(*) as count
                FROM tasks t
                LEFT JOIN feedback f ON t.id = f.task_id
                WHERE t.timestamp >= datetime('now', '-{} days')
                GROUP BY strftime('%H', timestamp)
                HAVING count >= 5
                ORDER BY hour
            '''.format(days))
            
            hourly_data = {}
            for row in cursor.fetchall():
                hour, avg_latency, avg_rating, count = row
                hourly_data[int(hour)] = {
                    'avg_latency': avg_latency,
                    'avg_rating': avg_rating,
                    'count': count
                }
            
            # Znajdź godziny z najlepszą/najgorszą wydajnością
            if len(hourly_data) >= 3:
                best_hours = sorted(hourly_data.items(), key=lambda x: x[1]['avg_rating'], reverse=True)[:3]
                worst_hours = sorted(hourly_data.items(), key=lambda x: x[1]['avg_rating'])[:3]
                
                # Wzorzec dla najlepszych godzin
                best_hours_list = [h[0] for h in best_hours]
                best_avg_rating = sum(h[1]['avg_rating'] for h in best_hours) / len(best_hours)
                
                pattern = ContextPattern(
                    pattern_id=f"peak_hours_{datetime.now().strftime('%Y%m')}",
                    pattern_type=LearningPattern.TEMPORAL,
                    conditions={
                        'hours': best_hours_list,
                        'condition': 'peak_performance'
                    },
                    outcomes={
                        'avg_rating': best_avg_rating,
                        'sample_hours': best_hours_list
                    },
                    confidence=0.7,
                    sample_count=sum(h[1]['count'] for h in best_hours),
                    success_rate=best_avg_rating / 5.0,
                    last_seen=datetime.now()
                )
                patterns.append(pattern)
        
        except Exception as e:
            self.logger.error(f"Error detecting temporal patterns: {e}")
        
        return patterns

class AdvancedLearningEngine:
    """Zaawansowany silnik uczenia z ML algorithms"""
    
    def __init__(self, tracker: SimpleTracker):
        self.tracker = tracker
        self.pattern_recognition = PatternRecognition(tracker)
        self.logger = logging.getLogger('learning_engine')
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_decay = 0.95  # Per week
        self.min_samples_for_learning = 5
    
    def update_decision_weights(
        self, 
        feedback_data: List[Dict],
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Aktualizuje wagi decyzyjne na podstawie pattern-based learning
        
        Args:
            feedback_data: Lista feedback data
            current_weights: Obecne wagi decyzyjne
        
        Returns:
            Zaktualizowane wagi
        """
        
        if len(feedback_data) < self.min_samples_for_learning:
            return current_weights
        
        new_weights = current_weights.copy()
        
        # Analizuj wzorce w feedback
        patterns = self._analyze_feedback_patterns(feedback_data)
        
        for pattern_type, adjustment in patterns.items():
            if pattern_type in new_weights:
                # Gradual adjustment z learning rate
                new_weights[pattern_type] += adjustment * self.learning_rate
        
        # Normalizuj wagi żeby suma = 1.0
        total_weight = sum(new_weights.values())
        if total_weight > 0:
            new_weights = {k: v / total_weight for k, v in new_weights.items()}
        
        return new_weights
    
    def _analyze_feedback_patterns(self, feedback_data: List[Dict]) -> Dict[str, float]:
        """Analizuje wzorce w feedback i zwraca adjustments dla wag"""
        
        adjustments = {}
        
        # Pattern 1: Jeśli users często override expensive models → increase cost weight
        cost_overrides = len([f for f in feedback_data if 
                             f.get('was_overridden') and 
                             f.get('recommended_model', '').startswith('gpt')])
        
        if cost_overrides > len(feedback_data) * 0.3:  # >30% cost-related overrides
            adjustments['cost'] = 0.05  # Increase cost sensitivity
        
        # Pattern 2: Jeśli low ratings correlate z high latency → increase speed weight  
        slow_low_ratings = len([f for f in feedback_data if
                               f.get('user_rating', 3) < 3 and
                               f.get('latency_ms', 1000) > 3000])
        
        if slow_low_ratings > len(feedback_data) * 0.2:  # >20% slow+bad
            adjustments['latency'] = 0.03  # Increase speed importance
        
        # Pattern 3: Consistent high ratings for specific model → increase quality weight
        high_quality_feedback = len([f for f in feedback_data if f.get('user_rating', 3) >= 4])
        if high_quality_feedback > len(feedback_data) * 0.7:  # >70% high ratings
            adjustments['quality'] = 0.02  # Slightly increase quality focus
        
        return adjustments
    
    def learn_cross_context_knowledge(self, days: int = 60) -> List[KnowledgeTransfer]:
        """Uczy się wiedzy aplikowalnej między różnymi kontekstami"""
        
        transfers = []
        
        try:
            # Znajdź successful patterns w jednym kontekście
            cursor = self.tracker.conn.execute('''
                SELECT 
                    t.task_type as source_context,
                    t.model_used,
                    AVG(f.rating) as avg_rating,
                    COUNT(*) as sample_count,
                    AVG(t.cost_usd) as avg_cost
                FROM tasks t
                JOIN feedback f ON t.id = f.task_id
                WHERE t.timestamp >= datetime('now', '-{} days')
                AND f.rating >= 4  -- High quality outcomes
                GROUP BY t.task_type, t.model_used
                HAVING sample_count >= 5 AND avg_rating >= 4.2
            '''.format(days))
            
            successful_patterns = cursor.fetchall()
            
            # Sprawdź czy te successful patterns mogą być użyte w innych kontekstach
            for source_pattern in successful_patterns:
                source_context, model, rating, samples, cost = source_pattern
                
                # Znajdź podobne task types gdzie ten model nie był jeszcze używany intensywnie
                cursor2 = self.tracker.conn.execute('''
                    SELECT 
                        task_type,
                        COUNT(*) as usage_count,
                        AVG(COALESCE(f.rating, 3)) as avg_rating
                    FROM tasks t
                    LEFT JOIN feedback f ON t.id = f.task_id
                    WHERE model_used = ?
                    AND task_type != ?
                    AND timestamp >= datetime('now', '-{} days')
                    GROUP BY task_type
                '''.format(days), (model, source_context))
                
                for target_row in cursor2.fetchall():
                    target_context, target_usage, target_rating = target_row
                    
                    # Jeśli model ma niski usage w target context ale wysoką jakość w source
                    if target_usage < samples // 2 and rating > target_rating + 0.5:
                        
                        # Oblicz applicability score
                        context_similarity = self._calculate_context_similarity(source_context, target_context)
                        
                        transfer = KnowledgeTransfer(
                            source_context=source_context,
                            target_context=target_context,
                            transferred_knowledge={
                                'model': model,
                                'expected_quality': rating,
                                'expected_cost': cost,
                                'success_indicators': f"High rating ({rating:.1f}) in {source_context}"
                            },
                            applicability_score=context_similarity,
                            validation_samples=samples
                        )
                        transfers.append(transfer)
        
        except Exception as e:
            self.logger.error(f"Error learning cross-context knowledge: {e}")
        
        return transfers
    
    def _calculate_context_similarity(self, context1: str, context2: str) -> float:
        """Oblicza podobieństwo między kontekstami zadań"""
        
        # Prosta heurystyka - można rozszerzyć o ML
        similarity_matrix = {
            ('chat', 'analysis'): 0.6,
            ('code_generation', 'pipeline'): 0.7,
            ('business_parsing', 'analysis'): 0.8,
            ('orchestration', 'pipeline'): 0.6
        }
        
        # Sprawdź obie kombinacje
        similarity = similarity_matrix.get((context1, context2)) or \
                    similarity_matrix.get((context2, context1)) or \
                    0.3  # Default low similarity
        
        return similarity
    
    def evolve_confidence_scores(self, historical_predictions: List[Dict]) -> Dict[str, float]:
        """Ewoluuje confidence scores na podstawie accuracy predictions"""
        
        confidence_adjustments = {}
        
        try:
            # Grupuj predictions by model
            model_predictions = defaultdict(list)
            
            for pred in historical_predictions:
                model = pred.get('model')
                predicted_success = pred.get('predicted_success_prob', 0.5)
                actual_rating = pred.get('actual_rating', 3)
                actual_success = 1.0 if actual_rating >= 4 else 0.0
                
                if model:
                    model_predictions[model].append((predicted_success, actual_success))
            
            # Oblicz accuracy dla każdego modelu
            for model, predictions in model_predictions.items():
                if len(predictions) >= 5:
                    
                    # Calculate prediction accuracy
                    accuracies = []
                    for pred_prob, actual_success in predictions:
                        # Threshold predictions at 0.5
                        pred_success = 1.0 if pred_prob > 0.5 else 0.0
                        accuracy = 1.0 if pred_success == actual_success else 0.0
                        accuracies.append(accuracy)
                    
                    avg_accuracy = sum(accuracies) / len(accuracies)
                    
                    # Adjust confidence based on accuracy
                    if avg_accuracy > 0.8:
                        confidence_adjustments[model] = 0.1  # Increase confidence
                    elif avg_accuracy < 0.6:
                        confidence_adjustments[model] = -0.1  # Decrease confidence
                    else:
                        confidence_adjustments[model] = 0.0  # Keep stable
        
        except Exception as e:
            self.logger.error(f"Error evolving confidence scores: {e}")
        
        return confidence_adjustments

class EnhancedFeedbackLoopEngine:
    """
    Rozszerzony Feedback Loop Engine z zaawansowanymi capabilities
    
    Integruje z istniejącym feedback-loop-engine.py jeśli dostępny
    """
    
    def __init__(self, tracker: Optional[SimpleTracker] = None):
        self.tracker = tracker or SimpleTracker()
        self.pattern_recognition = PatternRecognition(self.tracker)
        self.learning_engine = AdvancedLearningEngine(self.tracker)
        self.logger = self._setup_logging()
        
        # Initialize base engine if available
        if has_existing_engine:
            self.base_engine = FeedbackLoopEngine()
        
        # Learning state
        self.user_preferences = {}
        self.context_patterns = []
        self.knowledge_transfers = []
        self.confidence_scores = defaultdict(float)
        
        # Decision weights (will be learned and adjusted)
        self.decision_weights = {
            'cost': 0.15,
            'quality': 0.5, 
            'latency': 0.15,
            'human_acceptance': 0.2
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger('enhanced_feedback')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def process_feedback_with_learning(
        self,
        task_id: str,
        user_rating: int,
        model_used: str,
        model_recommended: str,
        task_type: str,
        cost: float,
        latency: int,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Proces feedback z zaawansowanym uczeniem
        
        Args:
            task_id: ID zadania
            user_rating: Rating użytkownika (1-5)
            model_used: Faktycznie użyty model
            model_recommended: Rekomendowany model
            task_type: Typ zadania
            cost: Koszt zadania
            latency: Latencja
            context: Dodatkowy kontekst
        
        Returns:
            Dict z wynikami uczenia i insights
        """
        
        self.logger.info(f"Processing enhanced feedback for task {task_id}")
        
        # 1. Zapisz standardowy feedback
        if hasattr(self.tracker, 'record_feedback'):
            self.tracker.record_feedback(task_id, user_rating, context.get('comment') if context else None)
        
        # 2. Analiza czy był override
        was_overridden = model_used != model_recommended
        
        # 3. Update user preferences
        self._update_user_preferences(
            user_rating, model_used, model_recommended, task_type, was_overridden
        )
        
        # 4. Update decision weights based on patterns
        recent_feedback = self._get_recent_feedback(days=7)
        self.decision_weights = self.learning_engine.update_decision_weights(
            recent_feedback, self.decision_weights
        )
        
        # 5. Update confidence scores
        prediction_history = self._get_prediction_history(model_used)
        confidence_updates = self.learning_engine.evolve_confidence_scores(prediction_history)
        self._apply_confidence_updates(confidence_updates)
        
        # 6. Cross-context learning
        if user_rating >= 4:  # High quality outcome
            self._record_successful_pattern(model_used, task_type, user_rating, cost, latency)
        
        # 7. Generate learning insights
        insights = self._generate_learning_insights(was_overridden, user_rating, model_used, task_type)
        
        return {
            'feedback_processed': True,
            'was_overridden': was_overridden,
            'learning_insights': insights,
            'updated_weights': self.decision_weights,
            'confidence_adjustment': confidence_updates.get(model_used, 0.0),
            'patterns_updated': len(self.context_patterns)
        }
    
    def _update_user_preferences(
        self,
        rating: int, 
        model_used: str, 
        model_recommended: str,
        task_type: str,
        was_overridden: bool
    ):
        """Updates user preferences based on feedback"""
        
        user_id = "default"  # Can be extended for multi-user
        
        # If user overrode recommendation and gave high rating
        if was_overridden and rating >= 4:
            pref_key = f"{task_type}_model_preference"
            
            if pref_key not in self.user_preferences:
                self.user_preferences[pref_key] = UserPreference(
                    user_id=user_id,
                    preference_type="model_choice",
                    preference_value={'preferred_models': defaultdict(int)},
                    confidence=0.0,
                    learned_from_samples=0,
                    last_updated=datetime.now()
                )
            
            # Increment preference for chosen model
            self.user_preferences[pref_key].preference_value['preferred_models'][model_used] += 1
            self.user_preferences[pref_key].learned_from_samples += 1
            self.user_preferences[pref_key].confidence = min(
                self.user_preferences[pref_key].learned_from_samples / 10.0, 1.0
            )
            self.user_preferences[pref_key].last_updated = datetime.now()
    
    def _get_recent_feedback(self, days: int = 7) -> List[Dict]:
        """Pobiera recent feedback dla learning"""
        
        try:
            cursor = self.tracker.conn.execute('''
                SELECT 
                    t.id, t.model_used, t.model_recommended, t.task_type,
                    t.cost_usd, t.latency_ms, f.rating, f.comment
                FROM tasks t
                LEFT JOIN feedback f ON t.id = f.task_id
                WHERE t.timestamp >= datetime('now', '-{} days')
                ORDER BY t.timestamp DESC
            '''.format(days))
            
            feedback_data = []
            for row in cursor.fetchall():
                task_id, model_used, model_rec, task_type, cost, latency, rating, comment = row
                
                feedback_data.append({
                    'task_id': task_id,
                    'model_used': model_used,
                    'model_recommended': model_rec,
                    'task_type': task_type,
                    'cost_usd': cost,
                    'latency_ms': latency,
                    'user_rating': rating or 3,
                    'comment': comment,
                    'was_overridden': model_used != model_rec
                })
            
            return feedback_data
        
        except Exception as e:
            self.logger.error(f"Error getting recent feedback: {e}")
            return []
    
    def _get_prediction_history(self, model_name: str, days: int = 30) -> List[Dict]:
        """Pobiera historię predictions dla modelu"""
        
        # Mock implementation - w przyszłości może być rozszerzone o tracking predictions
        return [
            {
                'model': model_name,
                'predicted_success_prob': 0.8,
                'actual_rating': 4,
                'timestamp': datetime.now() - timedelta(days=1)
            }
        ]
    
    def _apply_confidence_updates(self, updates: Dict[str, float]):
        """Aplikuje aktualizacje confidence scores"""
        
        for model, adjustment in updates.items():
            current_confidence = self.confidence_scores[model]
            new_confidence = max(0.1, min(1.0, current_confidence + adjustment))
            self.confidence_scores[model] = new_confidence
            
            self.logger.info(f"Updated confidence for {model}: {current_confidence:.2f} → {new_confidence:.2f}")
    
    def _record_successful_pattern(
        self,
        model: str,
        task_type: str, 
        rating: int,
        cost: float,
        latency: int
    ):
        """Zapisuje successful pattern dla future knowledge transfer"""
        
        pattern = ContextPattern(
            pattern_id=f"success_{model}_{task_type}_{datetime.now().strftime('%Y%m%d')}",
            pattern_type=LearningPattern.MODEL_PERFORMANCE,
            conditions={
                'model': model,
                'task_type': task_type
            },
            outcomes={
                'rating': rating,
                'cost': cost,
                'latency': latency,
                'success': True
            },
            confidence=0.8,
            sample_count=1,
            success_rate=1.0,
            last_seen=datetime.now()
        )
        
        self.context_patterns.append(pattern)
    
    def _generate_learning_insights(
        self,
        was_overridden: bool,
        rating: int, 
        model_used: str,
        task_type: str
    ) -> List[str]:
        """Generuje insights z procesu uczenia"""
        
        insights = []
        
        if was_overridden:
            if rating >= 4:
                insights.append(f"✅ Dobra decyzja user: {model_used} dla {task_type} dał rating {rating}/5")
                insights.append("🧠 System uczy się tej preferencji")
            else:
                insights.append(f"❌ Override nie pomógł: {model_used} nadal dał niski rating {rating}/5")
        
        # Insights about learning progress
        if len(self.user_preferences) > 0:
            insights.append(f"📚 Nauczone preferencje: {len(self.user_preferences)}")
        
        if len(self.context_patterns) > 0:
            insights.append(f"🔍 Wykryte wzorce: {len(self.context_patterns)}")
        
        return insights
    
    def get_learning_summary(self) -> Dict[str, Any]:
        """Zwraca podsumowanie stanu uczenia systemu"""
        
        return {
            'user_preferences_count': len(self.user_preferences),
            'context_patterns_count': len(self.context_patterns),
            'knowledge_transfers_count': len(self.knowledge_transfers),
            'current_decision_weights': self.decision_weights,
            'average_confidence': sum(self.confidence_scores.values()) / len(self.confidence_scores) if self.confidence_scores else 0.0,
            'models_tracked': list(self.confidence_scores.keys()),
            'last_learning_update': datetime.now().isoformat()
        }

# === CLI Integration Functions ===

def process_enhanced_feedback_cli(
    task_id: str,
    rating: int, 
    model_used: str,
    model_recommended: str,
    task_type: str,
    cost: float,
    latency: int,
    comment: str = None
) -> Dict:
    """
    CLI wrapper for enhanced feedback processing
    
    Usage:
        result = process_enhanced_feedback_cli(task_id, 4, "llama3.2-3b", "gpt-4", "chat", 0.0, 800)
        console.print(f"Learning insights: {result['learning_insights']}")
    """
    
    engine = EnhancedFeedbackLoopEngine()
    
    context = {'comment': comment} if comment else None
    
    result = engine.process_feedback_with_learning(
        task_id, rating, model_used, model_recommended, 
        task_type, cost, latency, context
    )
    
    return result

def get_learning_status_cli() -> Dict:
    """
    Returns learning system status for CLI
    
    Usage:
        status = get_learning_status_cli()
        console.print(f"Learned patterns: {status['context_patterns_count']}")
    """
    
    engine = EnhancedFeedbackLoopEngine()
    return engine.get_learning_summary()

def discover_user_patterns_cli(days: int = 30) -> Dict:
    """
    CLI wrapper for pattern discovery
    
    Usage:
        patterns = discover_user_patterns_cli(30)
        console.print(f"Discovered {patterns['preferences_count']} preferences")
    """
    
    engine = EnhancedFeedbackLoopEngine()
    
    # Discover patterns
    preferences = engine.pattern_recognition.discover_user_preferences(days=days)
    context_patterns = engine.pattern_recognition.identify_context_patterns(days=days)
    temporal_patterns = engine.pattern_recognition.detect_temporal_patterns(days=days)
    
    return {
        'preferences_count': len(preferences),
        'context_patterns_count': len(context_patterns),
        'temporal_patterns_count': len(temporal_patterns),
        'preferences': [
            {
                'type': p.preference_type,
                'value': p.preference_value,
                'confidence': p.confidence
            } for p in preferences[:5]  # Top 5
        ],
        'top_patterns': [
            {
                'conditions': p.conditions,
                'success_rate': p.success_rate,
                'confidence': p.confidence
            } for p in context_patterns[:5]  # Top 5
        ]
    }

# === Testing Functions ===

def test_enhanced_feedback_loop():
    """Testy funkcjonalne dla enhanced feedback loop"""
    
    print("=== TESTING ENHANCED FEEDBACK LOOP ENGINE ===\n")
    
    engine = EnhancedFeedbackLoopEngine()
    
    # Test 1: Process feedback with learning
    print("Test 1: Enhanced feedback processing")
    
    result = engine.process_feedback_with_learning(
        task_id="test-1",
        user_rating=4,
        model_used="llama3.2-3b",
        model_recommended="gpt-4", 
        task_type="chat",
        cost=0.0,
        latency=800,
        context={'comment': 'Good local model performance'}
    )
    
    print(f"Learning result: {result}")
    print(f"Was overridden: {result['was_overridden']}")
    print(f"Insights: {result['learning_insights']}")
    print()
    
    # Test 2: User preference discovery
    print("Test 2: User preference discovery")
    
    preferences = engine.pattern_recognition.discover_user_preferences(days=30)
    print(f"Discovered {len(preferences)} user preferences")
    for pref in preferences[:3]:
        print(f"  - {pref.preference_type}: confidence {pref.confidence:.2f}")
    print()
    
    # Test 3: Context pattern identification
    print("Test 3: Context pattern identification")
    
    patterns = engine.pattern_recognition.identify_context_patterns(days=30)
    print(f"Identified {len(patterns)} context patterns")
    for pattern in patterns[:3]:
        print(f"  - {pattern.conditions}: success rate {pattern.success_rate:.2f}")
    print()
    
    # Test 4: Learning summary
    print("Test 4: Learning system summary")
    
    summary = engine.get_learning_summary()
    print(f"Learning Summary:")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    print()
    
    # Test 5: CLI Integration
    print("Test 5: CLI Integration")
    
    cli_result = process_enhanced_feedback_cli(
        "cli-test", 5, "qwen2.5-coder:7b", "qwen2.5-coder:7b", "code_generation", 0.0, 1200
    )
    print(f"CLI Feedback Result: {cli_result['learning_insights']}")
    
    cli_status = get_learning_status_cli()
    print(f"CLI Learning Status: {cli_status['models_tracked']} models tracked")
    
    cli_patterns = discover_user_patterns_cli(30)
    print(f"CLI Patterns: {cli_patterns['preferences_count']} preferences discovered")
    
    print("\n=== ALL TESTS COMPLETED ===")

if __name__ == "__main__":
    test_enhanced_feedback_loop()