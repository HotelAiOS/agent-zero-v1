"""
Agent Zero V2.0 - Production Kaizen Intelligence Layer
Replaces all mock implementations with real ML-powered components
"""

# Import production components
try:
    from .intelligence_layer import (
        IntelligentModelSelector,
        SuccessEvaluator,
        ActiveMetricsAnalyzer,
        EnhancedFeedbackLoopEngine,
        TaskContext,
        TaskResult,
        ModelCandidate,
        ModelType,
        TaskComplexity,
        SuccessLevel,
        get_intelligent_model_recommendation,
        evaluate_task_from_cli,
        generate_kaizen_report_cli,
        get_cost_analysis_cli,
        discover_user_patterns_cli,
        get_success_summary
    )
    
    PRODUCTION_MODE = True
    print("✅ Agent Zero V2.0 - Production Kaizen Intelligence Layer loaded")
    
except ImportError as e:
    print(f"❌ Failed to load production layer: {e}")
    # Fallback to mock implementations
    from .mock_fallback import *
    PRODUCTION_MODE = False

# Export all components
__all__ = [
    'IntelligentModelSelector',
    'SuccessEvaluator', 
    'ActiveMetricsAnalyzer',
    'EnhancedFeedbackLoopEngine',
    'TaskContext',
    'TaskResult',
    'ModelCandidate',
    'get_intelligent_model_recommendation',
    'evaluate_task_from_cli',
    'generate_kaizen_report_cli',
    'get_cost_analysis_cli',
    'discover_user_patterns_cli',
    'get_success_summary',
    'PRODUCTION_MODE'
]
