#!/usr/bin/env python3
"""
Agent Zero V1 - V2.0 Integration Test
Tests all V2.0 components working together

Author: Developer A (Backend Architect)
Date: 10 października 2025
Linear Issue: A0-28
"""

import sys
import os
import sqlite3
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_database_setup():
    """Test that V2.0 database tables are created correctly"""
    print("🔍 Testing database setup...")
    
    # Import components
    from shared.kaizen.intelligent_selector import IntelligentModelSelector
    from shared.kaizen.success_evaluator import SuccessEvaluator
    from shared.kaizen.metrics_analyzer import ActiveMetricsAnalyzer
    
    # Initialize components (this should create tables)
    db_path = "test_agent_zero.db"
    
    try:
        selector = IntelligentModelSelector(db_path)
        evaluator = SuccessEvaluator(db_path)
        analyzer = ActiveMetricsAnalyzer(db_path)
        
        # Check tables exist
        with sqlite3.connect(db_path) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            expected_tables = [
                'v2_model_decisions',
                'v2_success_evaluations', 
                'v2_active_metrics',
                'v2_alerts'
            ]
            
            for table in expected_tables:
                if table in tables:
                    print(f"  ✅ Table {table} exists")
                else:
                    print(f"  ❌ Table {table} missing")
                    return False
        
        print("  ✅ Database setup test passed")
        return True
        
    except Exception as e:
        print(f"  ❌ Database setup test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(db_path):
            os.remove(db_path)

def test_intelligent_selector():
    """Test IntelligentModelSelector functionality"""
    print("🤖 Testing IntelligentModelSelector...")
    
    try:
        from shared.kaizen.intelligent_selector import IntelligentModelSelector, TaskType
        
        selector = IntelligentModelSelector("test_selector.db")
        
        # Test recommendation
        context = {'complexity': 1.2, 'urgency': 1.0, 'budget': 'medium'}
        recommendation = selector.recommend_model(TaskType.CODE_GENERATION, context)
        
        if recommendation.model_name:
            print(f"  ✅ Recommendation generated: {recommendation.model_name}")
            print(f"  ✅ Confidence: {recommendation.confidence_score:.2f}")
            print(f"  ✅ Reasoning: {recommendation.reasoning[:60]}...")
            return True
        else:
            print("  ❌ No recommendation generated")
            return False
            
    except Exception as e:
        print(f"  ❌ IntelligentSelector test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists("test_selector.db"):
            os.remove("test_selector.db")

def test_success_evaluator():
    """Test SuccessEvaluator functionality"""
    print("📊 Testing SuccessEvaluator...")
    
    try:
        from shared.kaizen.success_evaluator import SuccessEvaluator, TaskResult, TaskOutputType
        
        evaluator = SuccessEvaluator("test_evaluator.db")
        
        # Create test task result
        task_result = TaskResult(
            task_id="test_001",
            task_type="code_generation", 
            output_type=TaskOutputType.CODE,
            output_content="def hello():\n    return 'Hello World'",
            expected_requirements=["Create a hello function"],
            context={"complexity": "simple"},
            execution_time_ms=1500,
            cost_usd=0.008,
            model_used="gpt-4o-mini"
        )
        
        evaluation = evaluator.evaluate_task(task_result)
        
        if evaluation.overall_score > 0:
            print(f"  ✅ Evaluation completed: {evaluation.level.value}")
            print(f"  ✅ Overall score: {evaluation.overall_score:.2f}")
            print(f"  ✅ Confidence: {evaluation.confidence:.2f}")
            return True
        else:
            print("  ❌ Invalid evaluation score")
            return False
            
    except Exception as e:
        print(f"  ❌ SuccessEvaluator test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists("test_evaluator.db"):
            os.remove("test_evaluator.db")

def test_metrics_analyzer():
    """Test ActiveMetricsAnalyzer functionality"""
    print("📈 Testing ActiveMetricsAnalyzer...")
    
    try:
        from shared.kaizen.metrics_analyzer import ActiveMetricsAnalyzer
        
        analyzer = ActiveMetricsAnalyzer("test_analyzer.db")
        
        # Test task completion analysis
        task_result = {
            'cost_usd': 0.015,
            'execution_time_ms': 3500, 
            'success': True,
            'human_override': False
        }
        
        analyzer.analyze_task_completion(task_result)
        
        # Get current metrics
        metrics = analyzer.get_current_metrics()
        
        print(f"  ✅ Metrics analysis completed")
        print(f"  ✅ Current metrics: {len(metrics)} tracked")
        return True
        
    except Exception as e:
        print(f"  ❌ MetricsAnalyzer test failed: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists("test_analyzer.db"):
            os.remove("test_analyzer.db")

def test_cli_import():
    """Test CLI module import"""
    print("💻 Testing Enhanced CLI import...")
    
    try:
        from cli import AgentZeroCLI
        
        cli = AgentZeroCLI("test_cli.db")
        print("  ✅ CLI import successful")
        return True
        
    except Exception as e:
        print(f"  ❌ CLI import test failed: {e}")
        return False

def run_full_integration_test():
    """Run complete V2.0 integration test"""
    print("🚀 Agent Zero V1 - V2.0 Integration Test")
    print("=" * 50)
    
    tests = [
        test_database_setup,
        test_intelligent_selector, 
        test_success_evaluator,
        test_metrics_analyzer,
        test_cli_import
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("📋 Test Results:")
    print(f"  ✅ Passed: {passed}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📊 Success Rate: {passed/(passed+failed):.1%}")
    
    if failed == 0:
        print("\n🎉 All V2.0 components working correctly!")
        print("🚀 Ready for deployment to Agent Zero V1")
        return True
    else:
        print(f"\n⚠️ {failed} tests failed - check components before deployment")
        return False

if __name__ == "__main__":
    success = run_full_integration_test()
    sys.exit(0 if success else 1)
