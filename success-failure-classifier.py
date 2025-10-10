#!/usr/bin/env python3
"""
Agent Zero V1 - Success/Failure Classification System
V2.0 Intelligence Layer Component - Week 43 Implementation

Multi-dimensional success criteria system dla Kaizen learning:
- Correctness (0.5 weight) - czy output jest poprawny?
- Efficiency (0.2 weight) - czy zrobiono optymalnie? 
- Cost (0.15 weight) - czy koszt akceptowalny?
- Latency (0.15 weight) - czy wykonano na czas?

Integruje z SimpleTracker i generuje predictive success probability.
"""

import json
import sqlite3
import ast
import re
import math
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
from enum import Enum
import logging

# Import existing components
import sys
sys.path.append('.')
from simple_tracker import SimpleTracker

class SuccessLevel(Enum):
    """Poziomy sukcesu zadania"""
    EXCELLENT = "EXCELLENT"      # 4.5-5.0
    GOOD = "GOOD"               # 3.5-4.5  
    ACCEPTABLE = "ACCEPTABLE"    # 2.5-3.5
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"  # 1.5-2.5
    FAILURE = "FAILURE"         # 0-1.5

class TaskType(Enum):
    """Typy zadań - zgodne z IntelligentModelSelector"""
    CHAT = "chat"
    CODE_GENERATION = "code_generation"
    ANALYSIS = "analysis"
    PIPELINE = "pipeline"
    BUSINESS_PARSING = "business_parsing"
    ORCHESTRATION = "orchestration"

@dataclass
class SuccessMetrics:
    """Metryki sukcesu dla zadania"""
    correctness_score: float      # 0.0-1.0
    efficiency_score: float       # 0.0-1.0  
    cost_score: float            # 0.0-1.0
    latency_score: float         # 0.0-1.0
    
    # Wagi wymiarów (suma = 1.0)
    correctness_weight: float = 0.5
    efficiency_weight: float = 0.2
    cost_weight: float = 0.15
    latency_weight: float = 0.15

@dataclass
class SuccessEvaluation:
    """Wynik oceny sukcesu zadania"""
    task_id: str
    overall_score: float         # 0.0-1.0
    success_level: SuccessLevel
    dimension_scores: SuccessMetrics
    recommendations: List[str]
    predicted_success_probability: Optional[float] = None
    confidence: float = 0.0
    timestamp: datetime = None

class CodeValidator:
    """Walidator kodu - sprawdza poprawność syntaktyczną"""
    
    @staticmethod
    def validate_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """
        Sprawdza poprawność składni Python
        
        Returns:
            (is_valid, error_message)
        """
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, f"Syntax Error: {str(e)}"
        except Exception as e:
            return False, f"Parse Error: {str(e)}"
    
    @staticmethod
    def validate_javascript_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Podstawowa walidacja JavaScript (można rozszerzyć)"""
        # Prosta heurystyka - sprawdź podstawowe błędy składni
        lines = code.split('\n')
        
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Sprawdź niezamknięte nawiasy/klamry
            open_parens = line.count('(') - line.count(')')
            open_braces = line.count('{') - line.count('}')
            
            if abs(open_parens) > 2 or abs(open_braces) > 2:
                return False, f"Unbalanced brackets on line {i}"
        
        return True, None
    
    @staticmethod 
    def validate_sql_syntax(code: str) -> Tuple[bool, Optional[str]]:
        """Podstawowa walidacja SQL"""
        # Sprawdź podstawowe słowa kluczowe SQL
        sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'CREATE', 'DROP', 'ALTER']
        
        code_upper = code.upper()
        has_sql_keyword = any(keyword in code_upper for keyword in sql_keywords)
        
        if not has_sql_keyword:
            return False, "No SQL keywords found"
        
        # Sprawdź podstawowe problemy
        if code_upper.count('SELECT') > 0 and 'FROM' not in code_upper:
            return False, "SELECT without FROM clause"
        
        return True, None
    
    @classmethod
    def validate_code_by_language(cls, code: str, language: str) -> Tuple[bool, Optional[str]]:
        """Uniwersalny walidator kodu"""
        language = language.lower()
        
        if language in ['python', 'py']:
            return cls.validate_python_syntax(code)
        elif language in ['javascript', 'js']:
            return cls.validate_javascript_syntax(code)
        elif language in ['sql']:
            return cls.validate_sql_syntax(code)
        else:
            # Dla nieznanych języków - podstawowe sprawdzenia
            if len(code.strip()) == 0:
                return False, "Empty code"
            return True, None

class LLMAsJudge:
    """
    LLM-as-a-Judge dla oceny jakości tekstowych odpowiedzi
    Lekka implementacja z lokalnym modelem
    """
    
    def __init__(self):
        self.evaluation_prompts = {
            'chat': """
Oceń jakość tej odpowiedzi na czacie (1-10):
- Czy odpowiada na pytanie?
- Czy jest pomocna i dokładna?
- Czy ma odpowiednią długość?

Odpowiedź: {response}
Pytanie: {question}

Ocena (tylko liczba 1-10):""",
            
            'analysis': """
Oceń jakość tej analizy (1-10):
- Czy jest dokładna i szczegółowa?
- Czy zawiera istotne insights?
- Czy wnioski są uzasadnione?

Analiza: {response}

Ocena (tylko liczba 1-10):""",
            
            'business_parsing': """
Oceń jakość parsowania wymagań biznesowych (1-10):
- Czy wszystkie wymagania zostały zidentyfikowane?
- Czy struktura jest logiczna?
- Czy nic nie zostało pominięte?

Wymagania: {response}

Ocena (tylko liczba 1-10):"""
        }
    
    def evaluate_response_quality(
        self, 
        response: str, 
        task_type: TaskType, 
        context: Dict = None
    ) -> float:
        """
        Ocenia jakość odpowiedzi using simple heuristics
        
        W produkcji można zastąpić prawdziwym LLM call
        
        Returns:
            Quality score 0.0-1.0
        """
        
        # Podstawowe heurystyki (można zastąpić LLM call)
        if not response or len(response.strip()) < 10:
            return 0.1
        
        score = 0.5  # Base score
        
        # Długość odpowiedzi
        length_score = min(len(response) / 500.0, 1.0) * 0.2
        score += length_score
        
        # Strukturalność (nowe linie, punkty)
        structure_score = 0.0
        if '\n' in response:
            structure_score += 0.1
        if any(marker in response for marker in ['1.', '2.', '-', '*']):
            structure_score += 0.1
        
        score += structure_score
        
        # Specjalne sprawdzenia dla różnych typów zadań
        if task_type == TaskType.CODE_GENERATION:
            # Sprawdź czy zawiera kod
            if any(keyword in response.lower() for keyword in ['def ', 'function', 'class ', 'import']):
                score += 0.2
        
        elif task_type == TaskType.ANALYSIS:
            # Sprawdź słowa analityczne
            analysis_keywords = ['analyze', 'conclusion', 'result', 'finding', 'insight']
            if any(keyword in response.lower() for keyword in analysis_keywords):
                score += 0.2
        
        return min(score, 1.0)

class SuccessEvaluator:
    """
    Główna klasa do oceny sukcesu zadań w systemie Agent Zero V1
    """
    
    def __init__(self, tracker: Optional[SimpleTracker] = None):
        self.tracker = tracker or SimpleTracker()
        self.code_validator = CodeValidator()
        self.llm_judge = LLMAsJudge()
        self.logger = self._setup_logging()
        
        # Thresholds dla różnych metryk
        self.cost_thresholds = {
            TaskType.CHAT: 0.01,
            TaskType.CODE_GENERATION: 0.05,
            TaskType.ANALYSIS: 0.03,
            TaskType.PIPELINE: 0.02,
            TaskType.BUSINESS_PARSING: 0.02,
            TaskType.ORCHESTRATION: 0.04
        }
        
        self.latency_thresholds = {
            TaskType.CHAT: 2000,        # 2s
            TaskType.CODE_GENERATION: 5000,  # 5s  
            TaskType.ANALYSIS: 3000,    # 3s
            TaskType.PIPELINE: 10000,   # 10s
            TaskType.BUSINESS_PARSING: 3000,  # 3s
            TaskType.ORCHESTRATION: 8000     # 8s
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Konfiguracja logowania"""
        logger = logging.getLogger('success_evaluator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def evaluate_task_success(
        self,
        task_id: str,
        task_type: Union[TaskType, str],
        output: str,
        cost_usd: float,
        latency_ms: int,
        context: Optional[Dict] = None
    ) -> SuccessEvaluation:
        """
        Główna metoda oceny sukcesu zadania
        
        Args:
            task_id: ID zadania
            task_type: Typ zadania
            output: Wyjście/rezultat zadania
            cost_usd: Koszt w USD
            latency_ms: Latencja w ms
            context: Dodatkowy kontekst (input, user_preferences, etc.)
        
        Returns:
            SuccessEvaluation z pełną analizą
        """
        
        # Convert string to enum if needed
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                task_type = TaskType.CHAT  # fallback
        
        self.logger.info(f"Evaluating success for task {task_id} ({task_type.value})")
        
        # Oblicz każdy wymiar sukcesu
        correctness_score = self._evaluate_correctness(output, task_type, context)
        efficiency_score = self._evaluate_efficiency(output, task_type, context)
        cost_score = self._evaluate_cost(cost_usd, task_type)
        latency_score = self._evaluate_latency(latency_ms, task_type)
        
        # Utwórz metryki
        metrics = SuccessMetrics(
            correctness_score=correctness_score,
            efficiency_score=efficiency_score,
            cost_score=cost_score,
            latency_score=latency_score
        )
        
        # Oblicz ogólny score
        overall_score = self._calculate_weighted_score(metrics)
        
        # Określ poziom sukcesu
        success_level = self._determine_success_level(overall_score)
        
        # Generuj rekomendacje
        recommendations = self._generate_recommendations(metrics, task_type)
        
        # Przewidywanie prawdopodobieństwa sukcesu (na podstawie historii)
        predicted_prob = self._predict_success_probability(task_type, cost_usd, latency_ms)
        
        evaluation = SuccessEvaluation(
            task_id=task_id,
            overall_score=overall_score,
            success_level=success_level,
            dimension_scores=metrics,
            recommendations=recommendations,
            predicted_success_probability=predicted_prob,
            confidence=self._calculate_confidence(metrics),
            timestamp=datetime.now()
        )
        
        # Zapisz do trackera
        self._save_evaluation(evaluation)
        
        return evaluation
    
    def _evaluate_correctness(self, output: str, task_type: TaskType, context: Dict = None) -> float:
        """Ocenia poprawność output"""
        
        if not output or len(output.strip()) < 5:
            return 0.0
        
        score = 0.5  # Base score
        
        if task_type == TaskType.CODE_GENERATION:
            # Sprawdź składnię kodu
            language = context.get('language', 'python') if context else 'python'
            is_valid, error = self.code_validator.validate_code_by_language(output, language)
            
            if is_valid:
                score = 0.9  # Poprawna składnia
            else:
                score = 0.2  # Błąd składni
                self.logger.warning(f"Code validation failed: {error}")
        
        else:
            # Dla zadań tekstowych - użyj LLM-as-judge
            llm_score = self.llm_judge.evaluate_response_quality(output, task_type, context)
            score = llm_score
        
        return min(max(score, 0.0), 1.0)
    
    def _evaluate_efficiency(self, output: str, task_type: TaskType, context: Dict = None) -> float:
        """Ocenia efektywność rozwiązania"""
        
        # Podstawowe metryki efektywności
        if not output:
            return 0.0
        
        score = 0.7  # Base efficiency score
        
        # Sprawdź długość odpowiedzi vs oczekiwania
        expected_lengths = {
            TaskType.CHAT: (50, 500),
            TaskType.CODE_GENERATION: (100, 2000),
            TaskType.ANALYSIS: (200, 1500),
            TaskType.PIPELINE: (100, 800),
            TaskType.BUSINESS_PARSING: (150, 1000),
            TaskType.ORCHESTRATION: (100, 600)
        }
        
        min_len, max_len = expected_lengths.get(task_type, (50, 500))
        output_len = len(output)
        
        if min_len <= output_len <= max_len:
            score += 0.2  # Optimal length
        elif output_len < min_len:
            score -= 0.3  # Too short
        elif output_len > max_len * 2:
            score -= 0.2  # Too verbose
        
        # Dodatkowe sprawdzenia dla kodu
        if task_type == TaskType.CODE_GENERATION:
            # Sprawdź czy kod ma komentarze
            if '#' in output or '//' in output or '"""' in output:
                score += 0.1
            
            # Sprawdź czy ma error handling
            if any(keyword in output.lower() for keyword in ['try:', 'except:', 'catch', 'error']):
                score += 0.1
        
        return min(max(score, 0.0), 1.0)
    
    def _evaluate_cost(self, cost_usd: float, task_type: TaskType) -> float:
        """Ocenia efektywność kosztową"""
        
        threshold = self.cost_thresholds.get(task_type, 0.02)
        
        if cost_usd <= 0:
            return 1.0  # Free = perfect score
        
        # Logarytmiczna skala - koszt poniżej threshold = score > 0.8
        if cost_usd <= threshold:
            return max(0.8, 1.0 - (cost_usd / threshold) * 0.2)
        else:
            # Koszt powyżej threshold - spadek score
            excess = cost_usd / threshold
            return max(0.1, 0.8 - math.log(excess) * 0.3)
    
    def _evaluate_latency(self, latency_ms: int, task_type: TaskType) -> float:
        """Ocenia wydajność czasową"""
        
        threshold = self.latency_thresholds.get(task_type, 3000)
        
        if latency_ms <= threshold:
            # Linear scale do threshold
            return max(0.8, 1.0 - (latency_ms / threshold) * 0.2)
        else:
            # Powyżej threshold - exponential decay
            excess_ratio = latency_ms / threshold
            return max(0.1, 0.8 / excess_ratio)
    
    def _calculate_weighted_score(self, metrics: SuccessMetrics) -> float:
        """Oblicza ważony score ogólny"""
        
        weighted_score = (
            metrics.correctness_score * metrics.correctness_weight +
            metrics.efficiency_score * metrics.efficiency_weight +
            metrics.cost_score * metrics.cost_weight +
            metrics.latency_score * metrics.latency_weight
        )
        
        return min(max(weighted_score, 0.0), 1.0)
    
    def _determine_success_level(self, score: float) -> SuccessLevel:
        """Określa poziom sukcesu na podstawie score"""
        
        if score >= 0.9:
            return SuccessLevel.EXCELLENT
        elif score >= 0.7:
            return SuccessLevel.GOOD
        elif score >= 0.5:
            return SuccessLevel.ACCEPTABLE
        elif score >= 0.3:
            return SuccessLevel.NEEDS_IMPROVEMENT
        else:
            return SuccessLevel.FAILURE
    
    def _generate_recommendations(self, metrics: SuccessMetrics, task_type: TaskType) -> List[str]:
        """Generuje rekomendacje na podstawie słabych punktów"""
        
        recommendations = []
        
        # Sprawdź każdy wymiar
        if metrics.correctness_score < 0.6:
            if task_type == TaskType.CODE_GENERATION:
                recommendations.append("Sprawdź składnię kodu i przetestuj podstawowe funkcje")
            else:
                recommendations.append("Popraw jakość i dokładność odpowiedzi")
        
        if metrics.efficiency_score < 0.6:
            recommendations.append("Zoptymalizuj długość i strukturę odpowiedzi")
        
        if metrics.cost_score < 0.6:
            recommendations.append("Rozważ użycie tańszego modelu AI lub skróć zadanie")
        
        if metrics.latency_score < 0.6:
            recommendations.append("Użyj szybszego modelu lub uprość zadanie")
        
        # Ogólne rekomendacje
        if not recommendations:
            recommendations.append("Doskonała jakość - kontynuuj obecne podejście")
        
        return recommendations
    
    def _predict_success_probability(self, task_type: TaskType, cost_usd: float, latency_ms: int) -> Optional[float]:
        """Przewiduje prawdopodobieństwo sukcesu na podstawie historii"""
        
        try:
            # Pobierz podobne zadania z historii
            cursor = self.tracker.conn.execute('''
                SELECT AVG(f.rating) as avg_rating, COUNT(*) as count
                FROM tasks t
                LEFT JOIN feedback f ON t.id = f.task_id
                WHERE t.task_type = ? 
                AND t.cost_usd BETWEEN ? AND ?
                AND t.latency_ms BETWEEN ? AND ?
                AND t.timestamp >= datetime('now', '-30 days')
            ''', (
                task_type.value,
                cost_usd * 0.5, cost_usd * 1.5,  # ±50% cost range
                latency_ms * 0.7, latency_ms * 1.3  # ±30% latency range
            ))
            
            result = cursor.fetchone()
            
            if result and result[1] >= 3:  # Min 3 samples
                avg_rating = result[0] or 2.5
                # Convert rating (1-5) to probability (0-1)
                probability = (avg_rating - 1) / 4.0
                return min(max(probability, 0.0), 1.0)
        
        except Exception as e:
            self.logger.warning(f"Error predicting success probability: {e}")
        
        return None
    
    def _calculate_confidence(self, metrics: SuccessMetrics) -> float:
        """Oblicza confidence level oceny"""
        
        # Confidence based on consistency of scores
        scores = [
            metrics.correctness_score,
            metrics.efficiency_score, 
            metrics.cost_score,
            metrics.latency_score
        ]
        
        avg_score = sum(scores) / len(scores)
        variance = sum((score - avg_score) ** 2 for score in scores) / len(scores)
        
        # High variance = low confidence
        confidence = max(0.3, 1.0 - variance * 2)
        
        return confidence
    
    def _save_evaluation(self, evaluation: SuccessEvaluation):
        """Zapisuje ocenę do bazy dla przyszłych analiz"""
        
        try:
            # Rozszerz schemat trackera o tabelę evaluations
            self.tracker.conn.execute('''
                CREATE TABLE IF NOT EXISTS evaluations (
                    task_id TEXT PRIMARY KEY,
                    overall_score REAL,
                    success_level TEXT,
                    correctness_score REAL,
                    efficiency_score REAL,
                    cost_score REAL,
                    latency_score REAL,
                    predicted_probability REAL,
                    confidence REAL,
                    recommendations TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (task_id) REFERENCES tasks(id)
                )
            ''')
            
            recommendations_json = json.dumps(evaluation.recommendations)
            
            self.tracker.conn.execute('''
                INSERT OR REPLACE INTO evaluations 
                (task_id, overall_score, success_level, correctness_score, 
                 efficiency_score, cost_score, latency_score, predicted_probability,
                 confidence, recommendations) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                evaluation.task_id,
                evaluation.overall_score,
                evaluation.success_level.value,
                evaluation.dimension_scores.correctness_score,
                evaluation.dimension_scores.efficiency_score,
                evaluation.dimension_scores.cost_score,
                evaluation.dimension_scores.latency_score,
                evaluation.predicted_success_probability,
                evaluation.confidence,
                recommendations_json
            ))
            
            self.tracker.conn.commit()
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation: {e}")
    
    def get_success_analytics(self, days: int = 7) -> Dict:
        """Zwraca analityki sukcesu za ostatnie dni"""
        
        try:
            cursor = self.tracker.conn.execute('''
                SELECT 
                    AVG(overall_score) as avg_score,
                    success_level,
                    COUNT(*) as count,
                    AVG(correctness_score) as avg_correctness,
                    AVG(efficiency_score) as avg_efficiency,
                    AVG(cost_score) as avg_cost,
                    AVG(latency_score) as avg_latency
                FROM evaluations 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY success_level
                ORDER BY avg_score DESC
            '''.format(days))
            
            results = {}
            for row in cursor.fetchall():
                level = row[1]
                results[level] = {
                    'avg_score': row[0],
                    'count': row[2],
                    'avg_correctness': row[3],
                    'avg_efficiency': row[4],
                    'avg_cost': row[5],
                    'avg_latency': row[6]
                }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting success analytics: {e}")
            return {}
    
    def get_improvement_recommendations(self, task_type: Optional[TaskType] = None) -> List[Dict]:
        """Zwraca rekomendacje poprawy na podstawie analizy"""
        
        recommendations = []
        
        try:
            # Znajdź najsłabsze wymiary
            where_clause = ""
            params = []
            
            if task_type:
                where_clause = "WHERE task_type = ?"
                params.append(task_type.value)
            
            # Pobierz tasks z evaluations
            cursor = self.tracker.conn.execute(f'''
                SELECT 
                    t.task_type,
                    AVG(e.correctness_score) as avg_correctness,
                    AVG(e.efficiency_score) as avg_efficiency,
                    AVG(e.cost_score) as avg_cost,
                    AVG(e.latency_score) as avg_latency,
                    COUNT(*) as task_count
                FROM evaluations e
                JOIN tasks t ON e.task_id = t.id
                {where_clause}
                GROUP BY t.task_type
                HAVING task_count >= 3
            ''', params)
            
            for row in cursor.fetchall():
                task_type_str, correctness, efficiency, cost, latency, count = row
                
                # Znajdź najsłabsze wymiary
                scores = {
                    'correctness': correctness,
                    'efficiency': efficiency,
                    'cost': cost,
                    'latency': latency
                }
                
                weak_dimensions = [(dim, score) for dim, score in scores.items() if score < 0.6]
                
                if weak_dimensions:
                    recommendations.append({
                        'task_type': task_type_str,
                        'weak_dimensions': weak_dimensions,
                        'sample_size': count,
                        'recommendation': f"Focus on improving {', '.join([d[0] for d in weak_dimensions])} for {task_type_str} tasks"
                    })
            
        except Exception as e:
            self.logger.error(f"Error getting improvement recommendations: {e}")
        
        return recommendations

# === CLI Integration Functions ===

def evaluate_task_from_cli(
    task_id: str,
    task_type: str,
    output: str,
    cost_usd: float,
    latency_ms: int
) -> Dict:
    """
    CLI wrapper for success evaluation
    
    Usage in CLI:
        result = evaluate_task_from_cli(task_id, "code_generation", code_output, 0.005, 1200)
        console.print(f"Success Level: {result['success_level']}")
    """
    
    evaluator = SuccessEvaluator()
    evaluation = evaluator.evaluate_task_success(
        task_id, task_type, output, cost_usd, latency_ms
    )
    
    return {
        'task_id': evaluation.task_id,
        'overall_score': evaluation.overall_score,
        'success_level': evaluation.success_level.value,
        'recommendations': evaluation.recommendations,
        'confidence': evaluation.confidence,
        'dimension_breakdown': {
            'correctness': evaluation.dimension_scores.correctness_score,
            'efficiency': evaluation.dimension_scores.efficiency_score,
            'cost': evaluation.dimension_scores.cost_score,
            'latency': evaluation.dimension_scores.latency_score
        }
    }

def get_success_summary() -> Dict:
    """
    Returns success summary for CLI display
    
    Usage:
        summary = get_success_summary()
        console.print(f"Overall success rate: {summary['overall_success_rate']:.1%}")
    """
    
    evaluator = SuccessEvaluator()
    analytics = evaluator.get_success_analytics(days=7)
    
    total_tasks = sum(data['count'] for data in analytics.values())
    successful_tasks = sum(
        data['count'] for level, data in analytics.items()
        if level in ['EXCELLENT', 'GOOD']
    )
    
    overall_success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0
    
    return {
        'total_tasks': total_tasks,
        'successful_tasks': successful_tasks,
        'overall_success_rate': overall_success_rate,
        'level_breakdown': analytics
    }

# === Testing Functions ===

def test_success_evaluator():
    """Testy funkcjonalne dla success evaluator"""
    
    print("=== TESTING SUCCESS/FAILURE CLASSIFIER ===\n")
    
    evaluator = SuccessEvaluator()
    
    # Test 1: Excellent code
    print("Test 1: Excellent code generation")
    excellent_code = '''
def fibonacci(n):
    """Calculate fibonacci number efficiently"""
    if n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b

# Test the function
print(fibonacci(10))
'''
    
    eval1 = evaluator.evaluate_task_success(
        task_id="test-1",
        task_type=TaskType.CODE_GENERATION,
        output=excellent_code,
        cost_usd=0.003,
        latency_ms=1200
    )
    
    print(f"Overall Score: {eval1.overall_score:.2f}")
    print(f"Success Level: {eval1.success_level.value}")
    print(f"Recommendations: {eval1.recommendations}")
    print()
    
    # Test 2: Poor code with syntax errors
    print("Test 2: Poor code with syntax errors")
    poor_code = '''
def broken_function(
    print "Hello World"
    return x + y  # undefined variables
'''
    
    eval2 = evaluator.evaluate_task_success(
        task_id="test-2",
        task_type=TaskType.CODE_GENERATION,
        output=poor_code,
        cost_usd=0.008,
        latency_ms=3000
    )
    
    print(f"Overall Score: {eval2.overall_score:.2f}")
    print(f"Success Level: {eval2.success_level.value}")
    print(f"Recommendations: {eval2.recommendations}")
    print()
    
    # Test 3: Chat task
    print("Test 3: Good chat response")
    chat_response = '''
    Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every scenario.

    Key components include:
    1. Data collection and preprocessing
    2. Algorithm selection (supervised, unsupervised, reinforcement learning)  
    3. Model training and validation
    4. Performance evaluation and optimization

    Popular applications include recommendation systems, image recognition, natural language processing, and predictive analytics.
    '''
    
    eval3 = evaluator.evaluate_task_success(
        task_id="test-3", 
        task_type=TaskType.CHAT,
        output=chat_response,
        cost_usd=0.001,
        latency_ms=800
    )
    
    print(f"Overall Score: {eval3.overall_score:.2f}")
    print(f"Success Level: {eval3.success_level.value}")
    print(f"Recommendations: {eval3.recommendations}")
    print()
    
    # Test 4: CLI Integration
    print("Test 4: CLI Integration")
    cli_result = evaluate_task_from_cli(
        "test-cli", "analysis", "This is a comprehensive analysis...", 0.002, 1500
    )
    print(f"CLI Result: {cli_result}")
    
    summary = get_success_summary()
    print(f"Success Summary: {summary}")
    
    print("\n=== ALL TESTS COMPLETED ===")

if __name__ == "__main__":
    test_success_evaluator()