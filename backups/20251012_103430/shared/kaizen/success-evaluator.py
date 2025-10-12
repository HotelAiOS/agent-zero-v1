#!/usr/bin/env python3
"""
Agent Zero V1 - Success/Failure Evaluator
V2.0 Intelligence Layer - Multi-dimensional Success Classification

This module implements comprehensive success evaluation for Agent Zero V1,
providing multi-dimensional scoring and actionable recommendations.

Author: Developer A (Backend Architect)
Date: 10 października 2025
Linear Issue: A0-28
"""

import ast
import json
import sqlite3
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuccessLevel(Enum):
    """Success classification levels"""
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    NEEDS_IMPROVEMENT = "NEEDS_IMPROVEMENT"
    FAILURE = "FAILURE"

class TaskOutputType(Enum):
    """Types of task outputs to evaluate"""
    CODE = "code"
    TEXT = "text"
    ANALYSIS = "analysis"
    DOCUMENTATION = "documentation"
    ARCHITECTURE = "architecture"

@dataclass
class SuccessEvaluation:
    """Comprehensive success evaluation result"""
    level: SuccessLevel
    overall_score: float  # 0.0 to 1.0
    dimension_scores: Dict[str, float]
    recommendations: List[str]
    reasoning: str
    confidence: float
    improvement_suggestions: List[str]
    cost_effectiveness: float
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = asdict(self)
        result['level'] = self.level.value
        result['timestamp'] = self.timestamp.isoformat()
        return result

@dataclass
class TaskResult:
    """Input data for success evaluation"""
    task_id: str
    task_type: str
    output_type: TaskOutputType
    output_content: str
    expected_requirements: List[str]
    context: Dict[str, Any]
    execution_time_ms: int
    cost_usd: float
    model_used: str
    human_feedback: Optional[int] = None  # 1-5 scale

class SuccessEvaluator:
    """
    Multi-dimensional success evaluator for Agent Zero V1
    
    Evaluates task results across four key dimensions:
    - Correctness (50%): Technical accuracy and requirement fulfillment
    - Efficiency (20%): Optimal implementation and resource usage
    - Cost (15%): Budget effectiveness and value delivery
    - Latency (15%): Time performance and responsiveness
    """
    
    def __init__(self, db_path: str = "agent_zero.db"):
        self.db_path = Path(db_path)
        
        # Success dimension weights
        self.dimensions = {
            'correctness': 0.5,    # Technical accuracy
            'efficiency': 0.2,     # Resource optimization
            'cost': 0.15,         # Budget effectiveness
            'latency': 0.15       # Time performance
        }
        
        # Initialize database
        self._init_database()
        
        # Load thresholds from config
        self.thresholds = {
            'success_threshold': 0.85,
            'partial_threshold': 0.65,
            'improvement_threshold': 0.45,
            'max_acceptable_cost': 0.05,  # $0.05 per task
            'max_acceptable_latency': 30000,  # 30 seconds
        }

    def _init_database(self):
        """Initialize database tables for success tracking"""
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
                    recommendations TEXT NOT NULL,
                    improvement_suggestions TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS v2_quality_metrics (
                    task_type TEXT,
                    model_name TEXT,
                    avg_correctness REAL DEFAULT 0.0,
                    avg_efficiency REAL DEFAULT 0.0,
                    avg_cost_score REAL DEFAULT 0.0,
                    avg_latency_score REAL DEFAULT 0.0,
                    success_rate REAL DEFAULT 0.0,
                    evaluation_count INTEGER DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (task_type, model_name)
                )
            ''')
            
            conn.commit()

    def evaluate_task(self, task_result: TaskResult) -> SuccessEvaluation:
        """
        Perform comprehensive multi-dimensional evaluation
        
        Args:
            task_result: Task output and metadata to evaluate
            
        Returns:
            SuccessEvaluation with detailed scoring and recommendations
        """
        logger.info(f"🔍 Evaluating task {task_result.task_id} ({task_result.output_type.value})")
        
        # Calculate dimension scores
        dimension_scores = {}
        
        # 1. Correctness evaluation (50%)
        dimension_scores['correctness'] = self._evaluate_correctness(task_result)
        
        # 2. Efficiency evaluation (20%)
        dimension_scores['efficiency'] = self._evaluate_efficiency(task_result)
        
        # 3. Cost evaluation (15%)
        dimension_scores['cost'] = self._evaluate_cost(task_result)
        
        # 4. Latency evaluation (15%)
        dimension_scores['latency'] = self._evaluate_latency(task_result)
        
        # Calculate weighted overall score
        overall_score = sum(
            score * self.dimensions[dimension]
            for dimension, score in dimension_scores.items()
        )
        
        # Determine success level
        success_level = self._classify_success_level(overall_score)
        
        # Generate recommendations and improvements
        recommendations = self._generate_recommendations(dimension_scores, task_result)
        improvement_suggestions = self._generate_improvements(dimension_scores, task_result)
        reasoning = self._generate_reasoning(dimension_scores, overall_score, task_result)
        
        # Calculate confidence and cost effectiveness
        confidence = self._calculate_confidence(dimension_scores, task_result)
        cost_effectiveness = self._calculate_cost_effectiveness(overall_score, task_result.cost_usd)
        
        # Create evaluation result
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
        
        # Store evaluation for learning
        self._store_evaluation(evaluation, task_result)
        
        logger.info(f"✅ Evaluation complete: {success_level.value} (score: {overall_score:.2f})")
        return evaluation

    def _evaluate_correctness(self, task_result: TaskResult) -> float:
        """Evaluate technical correctness and requirement fulfillment"""
        score = 0.0
        
        if task_result.output_type == TaskOutputType.CODE:
            # Code-specific validation
            syntax_score = self._validate_code_syntax(task_result.output_content)
            requirement_score = self._check_requirements_fulfillment(
                task_result.output_content, 
                task_result.expected_requirements
            )
            quality_score = self._assess_code_quality(task_result.output_content)
            
            score = (syntax_score * 0.4 + requirement_score * 0.4 + quality_score * 0.2)
            
        elif task_result.output_type == TaskOutputType.TEXT:
            # Text quality assessment using LLM-as-judge
            score = self._llm_judge_text_quality(task_result.output_content, task_result.expected_requirements)
            
        elif task_result.output_type == TaskOutputType.ANALYSIS:
            # Analysis completeness and accuracy
            completeness_score = self._assess_analysis_completeness(
                task_result.output_content,
                task_result.expected_requirements
            )
            accuracy_score = self._assess_analysis_accuracy(task_result.output_content)
            
            score = (completeness_score * 0.6 + accuracy_score * 0.4)
            
        else:
            # Generic content evaluation
            score = self._generic_content_evaluation(
                task_result.output_content,
                task_result.expected_requirements
            )
        
        # Factor in human feedback if available
        if task_result.human_feedback is not None:
            human_score = task_result.human_feedback / 5.0  # Convert 1-5 to 0-1
            score = score * 0.7 + human_score * 0.3  # Blend with human judgment
        
        return max(0.0, min(1.0, score))

    def _validate_code_syntax(self, code: str) -> float:
        """Validate code syntax using AST parsing"""
        try:
            # Try to parse as Python first
            ast.parse(code)
            return 1.0
        except SyntaxError:
            # Try basic syntax checks for other languages
            if self._basic_syntax_check(code):
                return 0.8
            else:
                return 0.3  # Partial credit for attempt
        except Exception:
            return 0.5  # Unknown parsing error

    def _basic_syntax_check(self, code: str) -> bool:
        """Basic syntax validation for non-Python code"""
        # Check for balanced brackets
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets:
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets.get(stack.pop()) != char:
                    return False
        
        return len(stack) == 0

    def _check_requirements_fulfillment(self, content: str, requirements: List[str]) -> float:
        """Check how well the output fulfills stated requirements"""
        if not requirements:
            return 0.8  # Default score when no specific requirements
        
        fulfilled_count = 0
        for requirement in requirements:
            # Simple keyword matching - could be enhanced with NLP
            keywords = re.findall(r'\b\w+\b', requirement.lower())
            content_lower = content.lower()
            
            if any(keyword in content_lower for keyword in keywords if len(keyword) > 3):
                fulfilled_count += 1
        
        return fulfilled_count / len(requirements)

    def _assess_code_quality(self, code: str) -> float:
        """Assess code quality metrics"""
        score = 0.0
        
        # Length appropriateness (not too short, not too verbose)
        line_count = len(code.split('\n'))
        if 10 <= line_count <= 100:
            score += 0.3
        elif 5 <= line_count <= 200:
            score += 0.2
        else:
            score += 0.1
        
        # Comments presence
        comment_lines = len([line for line in code.split('\n') if line.strip().startswith('#')])
        if comment_lines > 0:
            score += 0.2
        
        # Function/class structure
        if 'def ' in code or 'class ' in code:
            score += 0.3
        
        # Variable naming (basic check)
        if re.search(r'\b[a-z_][a-z0-9_]*\b', code):
            score += 0.2
        
        return min(1.0, score)

    def _llm_judge_text_quality(self, content: str, requirements: List[str]) -> float:
        """Use LLM-as-judge for text quality evaluation"""
        # Simplified evaluation - in production, would use actual LLM call
        score = 0.5  # Base score
        
        # Length appropriateness
        word_count = len(content.split())
        if 50 <= word_count <= 500:
            score += 0.2
        elif 20 <= word_count <= 1000:
            score += 0.1
        
        # Structure indicators
        if any(marker in content for marker in ['\n\n', '1.', '2.', '-', '*']):
            score += 0.2
        
        # Requirement coverage
        if requirements:
            requirement_score = self._check_requirements_fulfillment(content, requirements)
            score += requirement_score * 0.3
        
        return min(1.0, score)

    def _evaluate_efficiency(self, task_result: TaskResult) -> float:
        """Evaluate resource efficiency and optimization"""
        score = 0.8  # Base efficiency score
        
        # Execution time efficiency
        expected_time = self._get_expected_execution_time(task_result.task_type)
        if expected_time > 0:
            time_efficiency = min(1.0, expected_time / task_result.execution_time_ms)
            score = score * 0.6 + time_efficiency * 0.4
        
        # Code efficiency (if applicable)
        if task_result.output_type == TaskOutputType.CODE:
            code_efficiency = self._assess_code_efficiency(task_result.output_content)
            score = score * 0.7 + code_efficiency * 0.3
        
        return max(0.0, min(1.0, score))

    def _assess_code_efficiency(self, code: str) -> float:
        """Assess code efficiency indicators"""
        score = 0.5
        
        # Avoid obvious inefficiencies
        inefficiency_patterns = [
            r'for.*for.*for',  # Triple nested loops
            r'while.*while.*while',  # Triple nested while
            r'time\.sleep\(\d+\)',  # Long sleeps
        ]
        
        for pattern in inefficiency_patterns:
            if re.search(pattern, code):
                score -= 0.2
        
        # Good efficiency patterns
        efficiency_patterns = [
            r'list comprehension',
            r'enumerate\(',
            r'zip\(',
            r'\.join\(',
        ]
        
        for pattern in efficiency_patterns:
            if re.search(pattern, code):
                score += 0.1
        
        return max(0.0, min(1.0, score))

    def _evaluate_cost(self, task_result: TaskResult) -> float:
        """Evaluate cost effectiveness"""
        if task_result.cost_usd == 0:
            return 1.0  # Free execution gets perfect score
        
        # Cost per quality unit
        if task_result.cost_usd <= 0.01:
            return 0.9
        elif task_result.cost_usd <= 0.03:
            return 0.7
        elif task_result.cost_usd <= 0.05:
            return 0.5
        else:
            return max(0.0, 0.5 - (task_result.cost_usd - 0.05) * 5)

    def _evaluate_latency(self, task_result: TaskResult) -> float:
        """Evaluate response time performance"""
        if task_result.execution_time_ms <= 2000:  # 2 seconds
            return 1.0
        elif task_result.execution_time_ms <= 5000:  # 5 seconds
            return 0.8
        elif task_result.execution_time_ms <= 10000:  # 10 seconds
            return 0.6
        elif task_result.execution_time_ms <= 30000:  # 30 seconds
            return 0.4
        else:
            return max(0.0, 0.4 - (task_result.execution_time_ms - 30000) / 100000)

    def _classify_success_level(self, overall_score: float) -> SuccessLevel:
        """Classify overall success level based on score"""
        if overall_score >= self.thresholds['success_threshold']:
            return SuccessLevel.SUCCESS
        elif overall_score >= self.thresholds['partial_threshold']:
            return SuccessLevel.PARTIAL
        elif overall_score >= self.thresholds['improvement_threshold']:
            return SuccessLevel.NEEDS_IMPROVEMENT
        else:
            return SuccessLevel.FAILURE

    def _generate_recommendations(self, dimension_scores: Dict[str, float], task_result: TaskResult) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Correctness recommendations
        if dimension_scores['correctness'] < 0.7:
            recommendations.append("🎯 Focus on requirement fulfillment and technical accuracy")
            if task_result.output_type == TaskOutputType.CODE:
                recommendations.append("🔧 Add syntax validation and testing")
        
        # Efficiency recommendations
        if dimension_scores['efficiency'] < 0.7:
            recommendations.append("⚡ Optimize resource usage and execution efficiency")
            recommendations.append("🚀 Consider faster algorithms or approaches")
        
        # Cost recommendations
        if dimension_scores['cost'] < 0.7:
            recommendations.append("💰 Explore cost-effective model alternatives")
            recommendations.append("💡 Consider local model execution for routine tasks")
        
        # Latency recommendations
        if dimension_scores['latency'] < 0.7:
            recommendations.append("⏰ Optimize for faster response times")
            recommendations.append("🔄 Consider breaking down complex tasks")
        
        return recommendations

    def _generate_improvements(self, dimension_scores: Dict[str, float], task_result: TaskResult) -> List[str]:
        """Generate specific improvement suggestions"""
        improvements = []
        
        # Find weakest dimension
        weakest_dimension = min(dimension_scores, key=dimension_scores.get)
        weakest_score = dimension_scores[weakest_dimension]
        
        if weakest_score < 0.5:
            if weakest_dimension == 'correctness':
                improvements.append(f"📋 Add validation steps for {task_result.output_type.value} outputs")
                improvements.append("🧪 Implement automated testing")
            elif weakest_dimension == 'efficiency':
                improvements.append("📊 Profile performance bottlenecks")
                improvements.append("🔧 Refactor for optimization")
            elif weakest_dimension == 'cost':
                improvements.append("💵 Evaluate cost-performance tradeoffs")
                improvements.append("🏠 Consider local processing options")
            elif weakest_dimension == 'latency':
                improvements.append("⚡ Implement caching mechanisms")
                improvements.append("🔀 Use asynchronous processing")
        
        return improvements

    def _generate_reasoning(self, dimension_scores: Dict[str, float], overall_score: float, task_result: TaskResult) -> str:
        """Generate transparent reasoning for the evaluation"""
        reasoning_parts = []
        
        # Overall assessment
        success_level = self._classify_success_level(overall_score)
        reasoning_parts.append(f"Overall: {success_level.value} (score: {overall_score:.2f})")
        
        # Dimension breakdown
        for dimension, score in dimension_scores.items():
            weight = self.dimensions[dimension]
            reasoning_parts.append(f"{dimension.title()}: {score:.2f} (weight: {weight:.0%})")
        
        # Key factors
        if task_result.cost_usd == 0:
            reasoning_parts.append("💚 Cost-effective: Free execution")
        elif task_result.cost_usd > 0.03:
            reasoning_parts.append(f"💰 High cost: ${task_result.cost_usd:.3f}")
        
        if task_result.execution_time_ms < 3000:
            reasoning_parts.append("⚡ Fast execution")
        elif task_result.execution_time_ms > 10000:
            reasoning_parts.append("⏳ Slow execution")
        
        return " | ".join(reasoning_parts)

    def _calculate_confidence(self, dimension_scores: Dict[str, float], task_result: TaskResult) -> float:
        """Calculate confidence in the evaluation"""
        # Base confidence from score consistency
        scores = list(dimension_scores.values())
        score_variance = sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)
        consistency_confidence = max(0.0, 1.0 - score_variance * 2)
        
        # Adjust based on available data
        data_confidence = 0.7  # Base confidence
        if task_result.human_feedback is not None:
            data_confidence += 0.2
        if task_result.expected_requirements:
            data_confidence += 0.1
        
        return min(1.0, (consistency_confidence + data_confidence) / 2)

    def _calculate_cost_effectiveness(self, overall_score: float, cost: float) -> float:
        """Calculate cost effectiveness ratio"""
        if cost == 0:
            return overall_score  # Perfect cost effectiveness for free
        
        return overall_score / max(0.001, cost)  # Score per dollar

    def _get_expected_execution_time(self, task_type: str) -> int:
        """Get expected execution time for task type (in ms)"""
        expected_times = {
            'code_generation': 5000,
            'code_review': 3000,
            'documentation': 4000,
            'analysis': 6000,
            'debugging': 8000,
            'architecture_design': 10000
        }
        return expected_times.get(task_type, 5000)

    def _generic_content_evaluation(self, content: str, requirements: List[str]) -> float:
        """Generic content evaluation for non-specific types"""
        score = 0.5
        
        # Basic completeness
        if len(content.strip()) > 50:
            score += 0.3
        
        # Structure indicators
        if any(indicator in content for indicator in ['\n', '.', ':', '-']):
            score += 0.2
        
        return min(1.0, score)

    def _assess_analysis_completeness(self, content: str, requirements: List[str]) -> float:
        """Assess completeness of analysis output"""
        score = 0.5
        
        # Look for analysis keywords
        analysis_keywords = ['analysis', 'conclusion', 'recommendation', 'finding', 'result']
        keyword_count = sum(1 for keyword in analysis_keywords if keyword in content.lower())
        score += min(0.4, keyword_count * 0.1)
        
        # Requirement fulfillment
        if requirements:
            req_score = self._check_requirements_fulfillment(content, requirements)
            score += req_score * 0.3
        
        return min(1.0, score)

    def _assess_analysis_accuracy(self, content: str) -> float:
        """Assess accuracy of analysis (simplified)"""
        # Basic accuracy indicators
        score = 0.6
        
        # Look for supporting evidence
        evidence_indicators = ['because', 'due to', 'based on', 'evidence', 'data shows']
        if any(indicator in content.lower() for indicator in evidence_indicators):
            score += 0.2
        
        # Balanced perspective
        balance_indicators = ['however', 'although', 'on the other hand', 'alternatively']
        if any(indicator in content.lower() for indicator in balance_indicators):
            score += 0.2
        
        return min(1.0, score)

    def _store_evaluation(self, evaluation: SuccessEvaluation, task_result: TaskResult):
        """Store evaluation results for learning and reporting"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store detailed evaluation
                conn.execute('''
                    INSERT INTO v2_success_evaluations (
                        task_id, task_type, success_level, overall_score,
                        correctness_score, efficiency_score, cost_score, latency_score,
                        confidence, cost_effectiveness, model_used, execution_time_ms, cost_usd,
                        reasoning, recommendations, improvement_suggestions
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    evaluation.reasoning,
                    json.dumps(evaluation.recommendations),
                    json.dumps(evaluation.improvement_suggestions)
                ))
                
                # Update quality metrics
                conn.execute('''
                    INSERT OR REPLACE INTO v2_quality_metrics (
                        task_type, model_name, avg_correctness, avg_efficiency,
                        avg_cost_score, avg_latency_score, success_rate, evaluation_count
                    ) VALUES (?, ?, 
                        COALESCE((SELECT avg_correctness * evaluation_count FROM v2_quality_metrics WHERE task_type = ? AND model_name = ?), 0) + ?,
                        COALESCE((SELECT avg_efficiency * evaluation_count FROM v2_quality_metrics WHERE task_type = ? AND model_name = ?), 0) + ?,
                        COALESCE((SELECT avg_cost_score * evaluation_count FROM v2_quality_metrics WHERE task_type = ? AND model_name = ?), 0) + ?,
                        COALESCE((SELECT avg_latency_score * evaluation_count FROM v2_quality_metrics WHERE task_type = ? AND model_name = ?), 0) + ?,
                        COALESCE((SELECT success_rate * evaluation_count FROM v2_quality_metrics WHERE task_type = ? AND model_name = ?), 0) + ?,
                        COALESCE((SELECT evaluation_count FROM v2_quality_metrics WHERE task_type = ? AND model_name = ?), 0) + 1
                    ) / (COALESCE((SELECT evaluation_count FROM v2_quality_metrics WHERE task_type = ? AND model_name = ?), 0) + 1)
                ''', (
                    task_result.task_type, task_result.model_used,
                    task_result.task_type, task_result.model_used, evaluation.dimension_scores['correctness'],
                    task_result.task_type, task_result.model_used, evaluation.dimension_scores['efficiency'],
                    task_result.task_type, task_result.model_used, evaluation.dimension_scores['cost'],
                    task_result.task_type, task_result.model_used, evaluation.dimension_scores['latency'],
                    task_result.task_type, task_result.model_used, 1.0 if evaluation.level in [SuccessLevel.SUCCESS, SuccessLevel.PARTIAL] else 0.0,
                    task_result.task_type, task_result.model_used,
                    task_result.task_type, task_result.model_used
                ))
                
                conn.commit()
                
        except sqlite3.Error as e:
            logger.error(f"Failed to store evaluation: {e}")

    def get_quality_insights(self) -> Dict[str, Any]:
        """Generate quality insights for reporting"""
        insights = {
            'success_rates': {},
            'quality_trends': {},
            'improvement_opportunities': [],
            'top_performers': {}
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Success rates by model and task type
                cursor = conn.execute('''
                    SELECT task_type, model_name, success_rate, evaluation_count
                    FROM v2_quality_metrics
                    WHERE evaluation_count >= 3
                    ORDER BY success_rate DESC
                ''')
                
                for row in cursor.fetchall():
                    task_type, model_name, success_rate, count = row
                    if task_type not in insights['success_rates']:
                        insights['success_rates'][task_type] = []
                    
                    insights['success_rates'][task_type].append({
                        'model': model_name,
                        'success_rate': success_rate,
                        'sample_size': count
                    })
                
                # Improvement opportunities
                cursor = conn.execute('''
                    SELECT task_type, model_name, avg_correctness, avg_efficiency,
                           avg_cost_score, avg_latency_score, evaluation_count
                    FROM v2_quality_metrics
                    WHERE evaluation_count >= 5
                    AND (avg_correctness < 0.7 OR avg_efficiency < 0.7 OR avg_cost_score < 0.7 OR avg_latency_score < 0.7)
                    ORDER BY (avg_correctness + avg_efficiency + avg_cost_score + avg_latency_score) ASC
                ''')
                
                for row in cursor.fetchall():
                    task_type, model_name, correctness, efficiency, cost_score, latency_score, count = row
                    
                    # Identify primary weakness
                    scores = {
                        'correctness': correctness,
                        'efficiency': efficiency,
                        'cost': cost_score,
                        'latency': latency_score
                    }
                    weakest = min(scores, key=scores.get)
                    
                    insights['improvement_opportunities'].append({
                        'task_type': task_type,
                        'model': model_name,
                        'primary_weakness': weakest,
                        'weakness_score': scores[weakest],
                        'sample_size': count
                    })
                
        except sqlite3.Error as e:
            logger.error(f"Failed to generate quality insights: {e}")
        
        return insights

# Example usage and testing
def main():
    """Test the SuccessEvaluator"""
    print("🚀 Testing Agent Zero V1 - SuccessEvaluator")
    
    evaluator = SuccessEvaluator()
    
    # Test code evaluation
    task_result = TaskResult(
        task_id="test_001",
        task_type="code_generation",
        output_type=TaskOutputType.CODE,
        output_content='''
def fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
print(fibonacci(10))
        ''',
        expected_requirements=["Create a fibonacci function", "Include documentation", "Add test"],
        context={"complexity": "medium"},
        execution_time_ms=3500,
        cost_usd=0.015,
        model_used="gpt-4o-mini",
        human_feedback=4  # 4/5 stars
    )
    
    evaluation = evaluator.evaluate_task(task_result)
    
    print(f"\n📊 Evaluation Results:")
    print(f"Success Level: {evaluation.level.value}")
    print(f"Overall Score: {evaluation.overall_score:.2f}")
    print(f"Confidence: {evaluation.confidence:.2f}")
    print(f"Cost Effectiveness: {evaluation.cost_effectiveness:.2f}")
    
    print(f"\n📈 Dimension Scores:")
    for dimension, score in evaluation.dimension_scores.items():
        print(f"  {dimension.title()}: {score:.2f}")
    
    print(f"\n💡 Recommendations:")
    for rec in evaluation.recommendations:
        print(f"  {rec}")
    
    print(f"\n🔧 Improvements:")
    for imp in evaluation.improvement_suggestions:
        print(f"  {imp}")
        
    print(f"\n🎯 Reasoning: {evaluation.reasoning}")

if __name__ == "__main__":
    main()