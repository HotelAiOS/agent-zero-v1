#!/usr/bin/env python3
"""
Agent Zero V1 - Active Metrics Analyzer
V2.0 Intelligence Layer Component - Week 43 Implementation

Real-time Kaizen z alertami i optimization suggestions:
- Threshold monitoring (cost > $0.02, latency > 5s alerts)
- Daily Kaizen reports - automatyczne podsumowania ulepszeń
- Cost optimization engine - identyfikacja savings opportunities  
- Performance trend analysis - wykrywanie degradacji
"""

import json
import sqlite3
import statistics
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any, Set
from pathlib import Path
from enum import Enum
import logging
import math

# Import existing components
import sys
sys.path.append('.')
from simple_tracker import SimpleTracker

class AlertType(Enum):
    """Typy alertów systemu"""
    HIGH_COST = "HIGH_COST"
    HIGH_LATENCY = "HIGH_LATENCY"
    LOW_QUALITY = "LOW_QUALITY"
    FREQUENT_FAILURES = "FREQUENT_FAILURES"
    MODEL_DEGRADATION = "MODEL_DEGRADATION"
    COST_SPIKE = "COST_SPIKE"

class AlertSeverity(Enum):
    """Poziomy ważności alertów"""
    CRITICAL = "CRITICAL"    # Wymaga natychmiastowej akcji
    WARNING = "WARNING"      # Wymaga uwagi
    INFO = "INFO"           # Informacyjny

class TrendDirection(Enum):
    """Kierunki trendów"""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DEGRADING = "DEGRADING"

@dataclass
class Alert:
    """Alert systemowy"""
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    affected_model: Optional[str] = None
    affected_task_type: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[float] = None
    suggestion: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MetricTrend:
    """Trend metryki w czasie"""
    metric_name: str
    current_value: float
    previous_value: float
    change_percent: float
    direction: TrendDirection
    significance: float  # 0.0-1.0 - jak znacząca jest zmiana

@dataclass
class CostOptimization:
    """Sugestia optymalizacji kosztów"""
    description: str
    current_cost: float
    projected_savings: float
    confidence: float  # 0.0-1.0
    implementation_effort: str  # "LOW", "MEDIUM", "HIGH"
    affected_models: List[str]
    rationale: str

@dataclass
class KaizenReport:
    """Dzienny raport Kaizen"""
    report_date: datetime
    total_tasks: int
    total_cost: float
    avg_quality: float
    alerts: List[Alert]
    trends: List[MetricTrend]
    optimizations: List[CostOptimization]
    key_insights: List[str]
    action_items: List[str]

class ThresholdMonitor:
    """Monitor progów metryk z alertami"""
    
    def __init__(self):
        # Domyślne progi - można konfigurować
        self.thresholds = {
            'cost_per_task': {
                'warning': 0.02,
                'critical': 0.05
            },
            'latency_ms': {
                'warning': 3000,
                'critical': 8000
            },
            'quality_rating': {
                'warning': 2.5,  # Poniżej tego = warning
                'critical': 2.0  # Poniżej tego = critical
            },
            'success_rate': {
                'warning': 0.7,  # Poniżej 70%
                'critical': 0.5  # Poniżej 50%
            }
        }
    
    def check_thresholds(self, metrics: Dict[str, float]) -> List[Alert]:
        """Sprawdza progi i generuje alerty"""
        alerts = []
        
        for metric_name, value in metrics.items():
            if metric_name not in self.thresholds:
                continue
            
            thresholds = self.thresholds[metric_name]
            
            # Sprawdź critical threshold
            if self._exceeds_threshold(metric_name, value, thresholds['critical']):
                alerts.append(Alert(
                    alert_type=self._get_alert_type(metric_name),
                    severity=AlertSeverity.CRITICAL,
                    message=f"{metric_name} przekroczył krytyczny próg: {value} > {thresholds['critical']}",
                    metric_value=value,
                    threshold_value=thresholds['critical'],
                    suggestion=self._get_threshold_suggestion(metric_name, 'critical')
                ))
            
            # Sprawdź warning threshold
            elif self._exceeds_threshold(metric_name, value, thresholds['warning']):
                alerts.append(Alert(
                    alert_type=self._get_alert_type(metric_name),
                    severity=AlertSeverity.WARNING,
                    message=f"{metric_name} przekroczył próg ostrzeżenia: {value} > {thresholds['warning']}",
                    metric_value=value,
                    threshold_value=thresholds['warning'],
                    suggestion=self._get_threshold_suggestion(metric_name, 'warning')
                ))
        
        return alerts
    
    def _exceeds_threshold(self, metric_name: str, value: float, threshold: float) -> bool:
        """Sprawdza czy wartość przekracza próg (uwzględnia kierunek)"""
        if metric_name in ['quality_rating', 'success_rate']:
            # Dla tych metryk - poniżej progu = problem
            return value < threshold
        else:
            # Dla cost, latency - powyżej progu = problem
            return value > threshold
    
    def _get_alert_type(self, metric_name: str) -> AlertType:
        """Mapuje metrykę na typ alertu"""
        mapping = {
            'cost_per_task': AlertType.HIGH_COST,
            'latency_ms': AlertType.HIGH_LATENCY,
            'quality_rating': AlertType.LOW_QUALITY,
            'success_rate': AlertType.FREQUENT_FAILURES
        }
        return mapping.get(metric_name, AlertType.HIGH_COST)
    
    def _get_threshold_suggestion(self, metric_name: str, level: str) -> str:
        """Generuje sugestię dla przekroczonego progu"""
        suggestions = {
            'cost_per_task': {
                'warning': "Rozważ użycie tańszego modelu dla prostych zadań",
                'critical': "PILNE: Sprawdź konfigurację modeli - koszt jest zbyt wysoki"
            },
            'latency_ms': {
                'warning': "Sprawdź wydajność modeli i rozważ optymalizację",
                'critical': "PILNE: System działa bardzo wolno - sprawdź infrastrukturę"
            },
            'quality_rating': {
                'warning': "Jakość odpowiedzi spada - sprawdź feedback użytkowników",
                'critical': "PILNE: Bardzo niska jakość - wymagana natychmiastowa akcja"
            },
            'success_rate': {
                'warning': "Zwiększona liczba niepowodzeń - sprawdź modele",
                'critical': "PILNE: Wysoki wskaźnik niepowodzeń - system wymaga naprawy"
            }
        }
        
        return suggestions.get(metric_name, {}).get(level, "Sprawdź tę metrykę")

class TrendAnalyzer:
    """Analizator trendów metryk w czasie"""
    
    def analyze_trends(self, historical_data: Dict[str, List[Tuple[datetime, float]]]) -> List[MetricTrend]:
        """
        Analizuje trendy dla każdej metryki
        
        Args:
            historical_data: Dict[metric_name -> List[(timestamp, value)]]
        
        Returns:
            Lista MetricTrend objects
        """
        trends = []
        
        for metric_name, data_points in historical_data.items():
            if len(data_points) < 2:
                continue
            
            # Sortuj po czasie
            data_points.sort(key=lambda x: x[0])
            
            # Podziel na dwie części - poprzedni okres vs obecny
            split_point = len(data_points) // 2
            if split_point == 0:
                continue
            
            previous_values = [point[1] for point in data_points[:split_point]]
            current_values = [point[1] for point in data_points[split_point:]]
            
            if not previous_values or not current_values:
                continue
            
            prev_avg = statistics.mean(previous_values)
            curr_avg = statistics.mean(current_values)
            
            if prev_avg == 0:
                change_percent = 0
            else:
                change_percent = ((curr_avg - prev_avg) / prev_avg) * 100
            
            # Określ kierunek trendu
            direction = self._determine_trend_direction(metric_name, change_percent)
            
            # Oblicz znaczenie zmiany
            significance = self._calculate_significance(change_percent, previous_values, current_values)
            
            trend = MetricTrend(
                metric_name=metric_name,
                current_value=curr_avg,
                previous_value=prev_avg,
                change_percent=change_percent,
                direction=direction,
                significance=significance
            )
            
            trends.append(trend)
        
        return trends
    
    def _determine_trend_direction(self, metric_name: str, change_percent: float) -> TrendDirection:
        """Określa kierunek trendu uwzględniając semantykę metryki"""
        
        # Progi znaczących zmian
        threshold = 10.0  # 10%
        
        if abs(change_percent) < threshold:
            return TrendDirection.STABLE
        
        # Dla metryk gdzie wyższe = lepsze (quality, success_rate)
        positive_metrics = {'quality_rating', 'success_rate', 'user_satisfaction'}
        
        if metric_name in positive_metrics:
            return TrendDirection.IMPROVING if change_percent > 0 else TrendDirection.DEGRADING
        else:
            # Dla metryk gdzie niższe = lepsze (cost, latency)
            return TrendDirection.IMPROVING if change_percent < 0 else TrendDirection.DEGRADING
    
    def _calculate_significance(self, change_percent: float, prev_values: List[float], curr_values: List[float]) -> float:
        """Oblicza znaczenie zmiany (0.0-1.0)"""
        
        # Im większa zmiana procentowa, tym większe znaczenie
        percent_significance = min(abs(change_percent) / 50.0, 1.0)  # 50% = max significance
        
        # Im mniejsza wariancja, tym większa pewność
        try:
            prev_var = statistics.variance(prev_values) if len(prev_values) > 1 else 1.0
            curr_var = statistics.variance(curr_values) if len(curr_values) > 1 else 1.0
            variance_factor = 1.0 / (1.0 + (prev_var + curr_var) / 2.0)
        except:
            variance_factor = 0.5
        
        significance = (percent_significance + variance_factor) / 2.0
        return min(significance, 1.0)

class CostOptimizer:
    """Engine optymalizacji kosztów"""
    
    def __init__(self, tracker: SimpleTracker):
        self.tracker = tracker
        
        # Model cost mapping (cost per 1000 tokens)
        self.model_costs = {
            'llama3.2-3b': 0.0,
            'qwen2.5-coder:7b': 0.0,
            'mistral:7b': 0.0,
            'gpt-4': 0.03,
            'claude-3': 0.015,
            'gpt-3.5-turbo': 0.002
        }
    
    def find_cost_optimizations(self, days: int = 7) -> List[CostOptimization]:
        """Znajduje możliwości optymalizacji kosztów"""
        
        optimizations = []
        
        # 1. Identify expensive models with low quality
        expensive_low_quality = self._find_expensive_low_quality_models(days)
        optimizations.extend(expensive_low_quality)
        
        # 2. Find overused expensive models for simple tasks
        overused_expensive = self._find_overused_expensive_models(days)
        optimizations.extend(overused_expensive)
        
        # 3. Identify models with high cost variance
        cost_variance_issues = self._find_cost_variance_issues(days)
        optimizations.extend(cost_variance_issues)
        
        return optimizations
    
    def _find_expensive_low_quality_models(self, days: int) -> List[CostOptimization]:
        """Znajduje drogie modele z niską jakością"""
        
        optimizations = []
        
        try:
            cursor = self.tracker.conn.execute('''
                SELECT 
                    t.model_used,
                    AVG(t.cost_usd) as avg_cost,
                    AVG(f.rating) as avg_rating,
                    COUNT(*) as usage_count,
                    SUM(t.cost_usd) as total_cost
                FROM tasks t
                LEFT JOIN feedback f ON t.id = f.task_id
                WHERE t.timestamp >= datetime('now', '-{} days')
                AND t.cost_usd > 0
                GROUP BY t.model_used
                HAVING usage_count >= 5 AND avg_cost > 0.01 AND avg_rating < 3.5
            '''.format(days))
            
            for row in cursor.fetchall():
                model, avg_cost, avg_rating, usage_count, total_cost = row
                
                # Znajdź tańszą alternatywę
                alternative_models = self._find_alternative_models(model, avg_rating or 2.5)
                
                if alternative_models:
                    best_alternative = alternative_models[0]
                    projected_savings = total_cost * 0.7  # Assume 70% savings
                    
                    optimization = CostOptimization(
                        description=f"Zastąp {model} tańszą alternatywą",
                        current_cost=total_cost,
                        projected_savings=projected_savings,
                        confidence=0.8,
                        implementation_effort="MEDIUM",
                        affected_models=[model],
                        rationale=f"{model} ma wysokie koszty (${avg_cost:.4f}/task) przy niskiej jakości ({avg_rating:.1f}/5). Alternatywa: {best_alternative}"
                    )
                    
                    optimizations.append(optimization)
        
        except Exception as e:
            logging.error(f"Error finding expensive low quality models: {e}")
        
        return optimizations
    
    def _find_overused_expensive_models(self, days: int) -> List[CostOptimization]:
        """Znajduje nadmiernie używane drogie modele"""
        
        optimizations = []
        
        try:
            # Znajdź drogie modele używane do prostych zadań
            cursor = self.tracker.conn.execute('''
                SELECT 
                    t.model_used,
                    t.task_type,
                    COUNT(*) as usage_count,
                    AVG(t.cost_usd) as avg_cost,
                    SUM(t.cost_usd) as total_cost
                FROM tasks t
                WHERE t.timestamp >= datetime('now', '-{} days')
                AND t.cost_usd > 0.005
                AND t.task_type IN ('chat', 'analysis')
                GROUP BY t.model_used, t.task_type
                HAVING usage_count >= 10
            '''.format(days))
            
            for row in cursor.fetchall():
                model, task_type, usage_count, avg_cost, total_cost = row
                
                # Dla prostych zadań sugeruj tańsze modele
                if task_type in ['chat'] and avg_cost > 0.002:
                    projected_savings = total_cost * 0.8  # 80% savings with local models
                    
                    optimization = CostOptimization(
                        description=f"Użyj lokalnych modeli dla zadań {task_type}",
                        current_cost=total_cost,
                        projected_savings=projected_savings,
                        confidence=0.9,
                        implementation_effort="LOW",
                        affected_models=[model],
                        rationale=f"Zadania {task_type} nie wymagają drogich modeli cloud. Lokalne modele dają podobną jakość za darmo."
                    )
                    
                    optimizations.append(optimization)
        
        except Exception as e:
            logging.error(f"Error finding overused expensive models: {e}")
        
        return optimizations
    
    def _find_cost_variance_issues(self, days: int) -> List[CostOptimization]:
        """Znajduje problemy z wysoką wariancją kosztów"""
        
        optimizations = []
        
        try:
            # Modele z bardzo różnymi kosztami za podobne zadania
            cursor = self.tracker.conn.execute('''
                SELECT 
                    model_used,
                    task_type,
                    AVG(cost_usd) as avg_cost,
                    MIN(cost_usd) as min_cost,
                    MAX(cost_usd) as max_cost,
                    COUNT(*) as count
                FROM tasks
                WHERE timestamp >= datetime('now', '-{} days')
                AND cost_usd > 0
                GROUP BY model_used, task_type
                HAVING count >= 5 AND (max_cost - min_cost) > avg_cost
            '''.format(days))
            
            for row in cursor.fetchall():
                model, task_type, avg_cost, min_cost, max_cost, count = row
                
                variance_ratio = (max_cost - min_cost) / avg_cost
                
                if variance_ratio > 2.0:  # Bardzo wysoka wariancja
                    optimization = CostOptimization(
                        description=f"Standaryzuj użycie {model} dla {task_type}",
                        current_cost=avg_cost * count,
                        projected_savings=avg_cost * count * 0.3,  # 30% savings
                        confidence=0.6,
                        implementation_effort="MEDIUM",
                        affected_models=[model],
                        rationale=f"Koszt {model} dla {task_type} waha się od ${min_cost:.4f} do ${max_cost:.4f}. Standaryzacja może obniżyć koszty."
                    )
                    
                    optimizations.append(optimization)
        
        except Exception as e:
            logging.error(f"Error finding cost variance issues: {e}")
        
        return optimizations
    
    def _find_alternative_models(self, expensive_model: str, min_quality: float) -> List[str]:
        """Znajduje tańsze alternatywy dla drogiego modelu"""
        
        # Proste mapowanie - można rozszerzyć o inteligentne wyszukiwanie
        alternatives = {
            'gpt-4': ['claude-3', 'llama3.2-3b'],
            'claude-3': ['llama3.2-3b', 'mistral:7b'],
            'gpt-3.5-turbo': ['llama3.2-3b']
        }
        
        return alternatives.get(expensive_model, ['llama3.2-3b'])

class ActiveMetricsAnalyzer:
    """
    Główna klasa Active Metrics Analyzer - real-time Kaizen analytics
    """
    
    def __init__(self, tracker: Optional[SimpleTracker] = None):
        self.tracker = tracker or SimpleTracker()
        self.threshold_monitor = ThresholdMonitor()
        self.trend_analyzer = TrendAnalyzer()
        self.cost_optimizer = CostOptimizer(self.tracker)
        self.logger = self._setup_logging()
        
        # Cache dla wydajności
        self._metrics_cache = {}
        self._cache_timestamp = datetime.now()
        self._cache_duration = timedelta(minutes=5)
    
    def _setup_logging(self) -> logging.Logger:
        """Konfiguracja logowania"""
        logger = logging.getLogger('active_metrics')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_task_completion(self, task_id: str, model_used: str, cost_usd: float, latency_ms: int) -> List[Alert]:
        """
        Analizuje zakończone zadanie i generuje alerty w czasie rzeczywistym
        
        Args:
            task_id: ID zadania
            model_used: Użyty model
            cost_usd: Koszt zadania
            latency_ms: Latencja zadania
        
        Returns:
            Lista alertów do natychmiastowego działania
        """
        
        alerts = []
        
        # Sprawdź progi dla tego konkretnego zadania
        task_metrics = {
            'cost_per_task': cost_usd,
            'latency_ms': latency_ms
        }
        
        threshold_alerts = self.threshold_monitor.check_thresholds(task_metrics)
        alerts.extend(threshold_alerts)
        
        # Dodaj informacje o modelu i zadaniu
        for alert in threshold_alerts:
            alert.affected_model = model_used
        
        # Check for cost spikes
        if self._is_cost_spike(model_used, cost_usd):
            alerts.append(Alert(
                alert_type=AlertType.COST_SPIKE,
                severity=AlertSeverity.WARNING,
                message=f"Wykryto skok kosztów dla {model_used}: ${cost_usd:.4f}",
                affected_model=model_used,
                metric_value=cost_usd,
                suggestion=f"Sprawdź dlaczego {model_used} generuje wyższe koszty niż zwykle"
            ))
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"Alert: {alert.message}")
        
        return alerts
    
    def _is_cost_spike(self, model_name: str, current_cost: float) -> bool:
        """Sprawdza czy koszt jest nietypowo wysoki dla tego modelu"""
        
        try:
            cursor = self.tracker.conn.execute('''
                SELECT AVG(cost_usd) as avg_cost, COUNT(*) as count
                FROM tasks 
                WHERE model_used = ? 
                AND timestamp >= datetime('now', '-7 days')
                AND cost_usd > 0
            ''', (model_name,))
            
            result = cursor.fetchone()
            
            if result and result[1] >= 5:  # Min 5 samples
                avg_cost = result[0]
                # Spike if current cost is 3x average
                return current_cost > avg_cost * 3
        
        except Exception as e:
            self.logger.warning(f"Error checking cost spike: {e}")
        
        return False
    
    def generate_daily_kaizen_report(self, date: Optional[datetime] = None) -> KaizenReport:
        """
        Generuje dzienny raport Kaizen z insights i action items
        
        Args:
            date: Data raportu (domyślnie dzisiaj)
        
        Returns:
            KaizenReport z kompletną analizą
        """
        
        if date is None:
            date = datetime.now()
        
        self.logger.info(f"Generating Kaizen report for {date.date()}")
        
        # Pobierz metryki dnia
        daily_metrics = self._get_daily_metrics(date)
        
        # Sprawdź alerty
        alerts = self._get_daily_alerts(date)
        
        # Analiza trendów
        trends = self._analyze_daily_trends(date)
        
        # Optymalizacje kosztów
        optimizations = self.cost_optimizer.find_cost_optimizations(days=1)
        
        # Generuj insights
        key_insights = self._generate_key_insights(daily_metrics, trends, alerts)
        
        # Generuj action items
        action_items = self._generate_action_items(alerts, optimizations, trends)
        
        report = KaizenReport(
            report_date=date,
            total_tasks=daily_metrics.get('total_tasks', 0),
            total_cost=daily_metrics.get('total_cost', 0.0),
            avg_quality=daily_metrics.get('avg_quality', 0.0),
            alerts=alerts,
            trends=trends,
            optimizations=optimizations,
            key_insights=key_insights,
            action_items=action_items
        )
        
        return report
    
    def _get_daily_metrics(self, date: datetime) -> Dict[str, Any]:
        """Pobiera metryki dla danego dnia"""
        
        try:
            date_str = date.strftime('%Y-%m-%d')
            
            cursor = self.tracker.conn.execute('''
                SELECT 
                    COUNT(*) as total_tasks,
                    SUM(cost_usd) as total_cost,
                    AVG(cost_usd) as avg_cost,
                    AVG(latency_ms) as avg_latency,
                    AVG(f.rating) as avg_rating,
                    COUNT(f.rating) as feedback_count
                FROM tasks t
                LEFT JOIN feedback f ON t.id = f.task_id
                WHERE DATE(t.timestamp) = ?
            ''', (date_str,))
            
            result = cursor.fetchone()
            
            if result:
                return {
                    'total_tasks': result[0] or 0,
                    'total_cost': result[1] or 0.0,
                    'avg_cost': result[2] or 0.0,
                    'avg_latency': result[3] or 0,
                    'avg_quality': result[4] or 0.0,
                    'feedback_count': result[5] or 0
                }
        
        except Exception as e:
            self.logger.error(f"Error getting daily metrics: {e}")
        
        return {}
    
    def _get_daily_alerts(self, date: datetime) -> List[Alert]:
        """Generuje alerty na podstawie danych dnia"""
        
        daily_metrics = self._get_daily_metrics(date)
        
        # Convert to format expected by threshold monitor
        threshold_metrics = {
            'cost_per_task': daily_metrics.get('avg_cost', 0.0),
            'latency_ms': daily_metrics.get('avg_latency', 0),
            'quality_rating': daily_metrics.get('avg_quality', 5.0)
        }
        
        alerts = self.threshold_monitor.check_thresholds(threshold_metrics)
        
        return alerts
    
    def _analyze_daily_trends(self, date: datetime) -> List[MetricTrend]:
        """Analizuje trendy dla danego dnia vs poprzednie dni"""
        
        try:
            # Pobierz dane z ostatnich 14 dni
            cursor = self.tracker.conn.execute('''
                SELECT 
                    DATE(timestamp) as date,
                    AVG(cost_usd) as avg_cost,
                    AVG(latency_ms) as avg_latency,
                    AVG(COALESCE(f.rating, 3)) as avg_rating
                FROM tasks t
                LEFT JOIN feedback f ON t.id = f.task_id
                WHERE timestamp >= datetime('now', '-14 days')
                GROUP BY DATE(timestamp)
                ORDER BY date
            ''')
            
            # Organizuj dane według metryk
            historical_data = {
                'avg_cost': [],
                'avg_latency': [], 
                'avg_rating': []
            }
            
            for row in cursor.fetchall():
                date_str, cost, latency, rating = row
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                
                historical_data['avg_cost'].append((date_obj, cost or 0))
                historical_data['avg_latency'].append((date_obj, latency or 0))
                historical_data['avg_rating'].append((date_obj, rating or 3))
            
            trends = self.trend_analyzer.analyze_trends(historical_data)
            return trends
        
        except Exception as e:
            self.logger.error(f"Error analyzing daily trends: {e}")
            return []
    
    def _generate_key_insights(self, daily_metrics: Dict, trends: List[MetricTrend], alerts: List[Alert]) -> List[str]:
        """Generuje kluczowe insights z danych"""
        
        insights = []
        
        # Insights z metryk
        total_tasks = daily_metrics.get('total_tasks', 0)
        if total_tasks > 0:
            avg_cost = daily_metrics.get('avg_cost', 0)
            insights.append(f"Wykonano {total_tasks} zadań ze średnim kosztem ${avg_cost:.4f}")
        
        # Insights z trendów
        for trend in trends:
            if trend.significance > 0.7:  # Significant trends only
                direction_text = {
                    TrendDirection.IMPROVING: "poprawił się",
                    TrendDirection.DEGRADING: "pogorszył się", 
                    TrendDirection.STABLE: "pozostał stabilny"
                }
                
                insights.append(
                    f"{trend.metric_name} {direction_text[trend.direction]} o {abs(trend.change_percent):.1f}%"
                )
        
        # Insights z alertów
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        if critical_alerts:
            insights.append(f"🚨 {len(critical_alerts)} krytycznych alertów wymaga natychmiastowej uwagi")
        
        if not insights:
            insights.append("System działa stabilnie bez znaczących problemów")
        
        return insights
    
    def _generate_action_items(self, alerts: List[Alert], optimizations: List[CostOptimization], trends: List[MetricTrend]) -> List[str]:
        """Generuje konkretne action items"""
        
        action_items = []
        
        # Action items z alertów
        critical_alerts = [a for a in alerts if a.severity == AlertSeverity.CRITICAL]
        for alert in critical_alerts:
            if alert.suggestion:
                action_items.append(f"🚨 PILNE: {alert.suggestion}")
        
        # Action items z optymalizacji
        high_impact_optimizations = [o for o in optimizations if o.projected_savings > 0.01]
        for opt in high_impact_optimizations[:3]:  # Top 3
            action_items.append(f"💰 {opt.description} (oszczędności: ${opt.projected_savings:.3f})")
        
        # Action items z trendów
        degrading_trends = [t for t in trends if t.direction == TrendDirection.DEGRADING and t.significance > 0.6]
        for trend in degrading_trends:
            action_items.append(f"📉 Zbadaj pogorszenie w {trend.metric_name} ({trend.change_percent:.1f}%)")
        
        if not action_items:
            action_items.append("✅ Brak krytycznych akcji - system działa optymalnie")
        
        return action_items
    
    def get_cost_analysis(self, days: int = 7) -> Dict:
        """Zwraca analizę kosztów z sugestiami optymalizacji"""
        
        try:
            # Całkowite koszty
            cursor = self.tracker.conn.execute('''
                SELECT 
                    SUM(cost_usd) as total_cost,
                    AVG(cost_usd) as avg_cost,
                    COUNT(*) as total_tasks
                FROM tasks 
                WHERE timestamp >= datetime('now', '-{} days')
                AND cost_usd > 0
            '''.format(days))
            
            result = cursor.fetchone()
            total_cost, avg_cost, total_tasks = result or (0, 0, 0)
            
            # Koszty per model
            cursor = self.tracker.conn.execute('''
                SELECT 
                    model_used,
                    SUM(cost_usd) as model_cost,
                    COUNT(*) as model_tasks,
                    AVG(cost_usd) as avg_model_cost
                FROM tasks 
                WHERE timestamp >= datetime('now', '-{} days')
                AND cost_usd > 0
                GROUP BY model_used
                ORDER BY model_cost DESC
            '''.format(days))
            
            model_costs = {}
            for row in cursor.fetchall():
                model, cost, tasks, avg_cost = row
                model_costs[model] = {
                    'total_cost': cost,
                    'tasks': tasks,
                    'avg_cost': avg_cost,
                    'percentage': (cost / total_cost * 100) if total_cost > 0 else 0
                }
            
            # Znajdź optymalizacje
            optimizations = self.cost_optimizer.find_cost_optimizations(days)
            total_savings = sum(opt.projected_savings for opt in optimizations)
            
            return {
                'period_days': days,
                'total_cost': total_cost or 0,
                'avg_cost_per_task': avg_cost or 0,
                'total_tasks': total_tasks or 0,
                'model_breakdown': model_costs,
                'optimization_opportunities': len(optimizations),
                'projected_savings': total_savings,
                'savings_percentage': (total_savings / total_cost * 100) if total_cost > 0 else 0,
                'top_optimizations': optimizations[:5]  # Top 5
            }
        
        except Exception as e:
            self.logger.error(f"Error getting cost analysis: {e}")
            return {'error': str(e)}

# === CLI Integration Functions ===

def generate_kaizen_report_cli(format: str = "summary") -> Dict:
    """
    CLI wrapper for Kaizen report generation
    
    Usage:
        report = generate_kaizen_report_cli("detailed")
        console.print(report['summary'])
    
    Args:
        format: "summary" lub "detailed"
    """
    
    analyzer = ActiveMetricsAnalyzer()
    report = analyzer.generate_daily_kaizen_report()
    
    if format == "summary":
        return {
            'date': report.report_date.strftime('%Y-%m-%d'),
            'summary': f"📊 {report.total_tasks} zadań, ${report.total_cost:.4f} koszt, {len(report.alerts)} alertów",
            'key_insights': report.key_insights[:3],
            'top_actions': report.action_items[:3],
            'alerts_count': len(report.alerts),
            'critical_alerts': len([a for a in report.alerts if a.severity == AlertSeverity.CRITICAL])
        }
    else:
        return {
            'date': report.report_date.strftime('%Y-%m-%d'),
            'total_tasks': report.total_tasks,
            'total_cost': report.total_cost,
            'avg_quality': report.avg_quality,
            'alerts': [{'type': a.alert_type.value, 'severity': a.severity.value, 'message': a.message} for a in report.alerts],
            'insights': report.key_insights,
            'action_items': report.action_items,
            'optimizations': [{'description': o.description, 'savings': o.projected_savings} for o in report.optimizations]
        }

def get_cost_analysis_cli(days: int = 7) -> Dict:
    """
    CLI wrapper for cost analysis
    
    Usage:
        analysis = get_cost_analysis_cli(7)
        console.print(f"Total cost: ${analysis['total_cost']:.4f}")
    """
    
    analyzer = ActiveMetricsAnalyzer()
    return analyzer.get_cost_analysis(days)

def check_real_time_alerts_cli(task_id: str, model: str, cost: float, latency: int) -> List[Dict]:
    """
    CLI wrapper for real-time alerts
    
    Usage:
        alerts = check_real_time_alerts_cli(task_id, "gpt-4", 0.05, 8000)
        for alert in alerts:
            console.print(f"⚠️ {alert['message']}")  
    """
    
    analyzer = ActiveMetricsAnalyzer()
    alerts = analyzer.analyze_task_completion(task_id, model, cost, latency)
    
    return [
        {
            'type': alert.alert_type.value,
            'severity': alert.severity.value,
            'message': alert.message,
            'suggestion': alert.suggestion
        } for alert in alerts
    ]

# === Testing Functions ===

def test_active_metrics_analyzer():
    """Testy funkcjonalne dla active metrics analyzer"""
    
    print("=== TESTING ACTIVE METRICS ANALYZER ===\n")
    
    analyzer = ActiveMetricsAnalyzer()
    
    # Test 1: Real-time alert checking
    print("Test 1: Real-time alert checking")
    alerts = analyzer.analyze_task_completion(
        task_id="test-1",
        model_used="gpt-4", 
        cost_usd=0.08,  # High cost
        latency_ms=9000  # High latency
    )
    
    print(f"Generated {len(alerts)} alerts:")
    for alert in alerts:
        print(f"  - {alert.severity.value}: {alert.message}")
    print()
    
    # Test 2: Daily Kaizen report
    print("Test 2: Daily Kaizen report generation")
    report = analyzer.generate_daily_kaizen_report()
    
    print(f"Report for {report.report_date.date()}:")
    print(f"  - Total tasks: {report.total_tasks}")
    print(f"  - Total cost: ${report.total_cost:.4f}")
    print(f"  - Alerts: {len(report.alerts)}")
    print(f"  - Key insights: {report.key_insights}")
    print(f"  - Action items: {report.action_items}")
    print()
    
    # Test 3: Cost analysis
    print("Test 3: Cost analysis")
    cost_analysis = analyzer.get_cost_analysis(days=7)
    
    print(f"Cost Analysis (7 days):")
    print(f"  - Total cost: ${cost_analysis.get('total_cost', 0):.4f}")
    print(f"  - Avg per task: ${cost_analysis.get('avg_cost_per_task', 0):.4f}")
    print(f"  - Optimization opportunities: {cost_analysis.get('optimization_opportunities', 0)}")
    print(f"  - Projected savings: ${cost_analysis.get('projected_savings', 0):.4f}")
    print()
    
    # Test 4: CLI Integration
    print("Test 4: CLI Integration Functions")
    
    cli_report = generate_kaizen_report_cli("summary")
    print(f"CLI Report Summary: {cli_report}")
    
    cli_cost = get_cost_analysis_cli(7)
    print(f"CLI Cost Analysis: Total ${cli_cost.get('total_cost', 0):.4f}")
    
    cli_alerts = check_real_time_alerts_cli("test-cli", "expensive-model", 0.1, 10000)
    print(f"CLI Alerts: {len(cli_alerts)} alerts generated")
    
    print("\n=== ALL TESTS COMPLETED ===")

if __name__ == "__main__":
    test_active_metrics_analyzer()