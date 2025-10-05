"""
Learning Module
System uczenia się organizacji - analiza projektów, pattern detection, evolution
"""

from .post_mortem import (
    PostMortemAnalyzer, 
    ProjectAnalysis, 
    AnalysisInsight,
    ProjectStatus,
    AnalysisCategory
)
from .pattern_detector import PatternDetector, Pattern, PatternType
from .anti_patterns import AntiPatternDetector, AntiPattern, AntiPatternSeverity
from .recommendations import RecommendationEngine, Recommendation, RecommendationCategory
from .evolution import AgentEvolutionEngine, EvolutionResult

__all__ = [
    'PostMortemAnalyzer',
    'ProjectAnalysis',
    'AnalysisInsight',
    'ProjectStatus',
    'AnalysisCategory',
    'PatternDetector',
    'Pattern',
    'PatternType',
    'AntiPatternDetector',
    'AntiPattern',
    'AntiPatternSeverity',
    'RecommendationEngine',
    'Recommendation',
    'RecommendationCategory',
    'AgentEvolutionEngine',
    'EvolutionResult'
]

__version__ = '1.0.0'
