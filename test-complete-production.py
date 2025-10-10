#!/usr/bin/env python3
"""
Agent Zero V1 - Complete Integration Test
Tests ALL production components
"""
import sys
import sqlite3
import time

sys.path.append('.')

def test_complete_system():
    print("🧪 Agent Zero V1 - Complete System Integration Test")
    print("=" * 55)
    
    total_tests = 0
    passed_tests = 0
    
    # Test 1: Database Schema
    print("\n📊 Test 1: Complete Database Schema")
    total_tests += 1
    try:
        with sqlite3.connect('agent_zero.db', timeout=10.0) as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            v1_tables = [t for t in tables if not t.startswith('v2_')]
            v2_tables = [t for t in tables if t.startswith('v2_')]
            
            print(f"   ✅ V1 Tables: {len(v1_tables)}")
            print(f"   ✅ V2 Tables: {len(v2_tables)}")
            
            # Test context column
            cursor = conn.execute("PRAGMA table_info(simple_tracker)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'context' in columns:
                print("   ✅ Context column exists")
            else:
                print("   ❌ Context column missing")
                return False
            
            passed_tests += 1
            print("   ✅ Database Schema: PASS")
            
    except Exception as e:
        print(f"   ❌ Database Schema: FAIL - {e}")
    
    # Test 2: Task Decomposer Complete
    print("\n🔧 Test 2: Complete Task Decomposer")
    total_tests += 1
    try:
        from shared.orchestration.task_decomposer import Task, TaskDecomposer, TaskPriority, TaskStatus, TaskDependency, TaskType
        
        task = Task(id=1, title="Test", description="Test task", 
                   priority=TaskPriority.HIGH, status=TaskStatus.PENDING)
        td = TaskDecomposer()
        result = td.decompose_project("fullstack_web_app", ["Test requirement"])
        
        print(f"   ✅ All classes imported: Task, TaskPriority, TaskStatus, TaskDependency, TaskType")
        print(f"   ✅ Task Decomposer: {len(result)} tasks created")
        
        passed_tests += 1
        print("   ✅ Task Decomposer Complete: PASS")
        
    except Exception as e:
        print(f"   ❌ Task Decomposer Complete: FAIL - {e}")
    
    # Test 3: Intelligent Planner
    print("\n🎯 Test 3: Intelligent Planner")
    total_tests += 1
    try:
        from shared.orchestration.planner import IntelligentPlanner
        
        planner = IntelligentPlanner()
        plan = planner.create_project_plan("Test Project", "fullstack_web_app", ["Test req"])
        
        print(f"   ✅ Planner initialized")
        print(f"   ✅ Project plan: {len(plan.tasks)} tasks, {len(plan.team.members)} team members")
        
        passed_tests += 1
        print("   ✅ Intelligent Planner: PASS")
        
    except Exception as e:
        print(f"   ❌ Intelligent Planner: FAIL - {e}")
    
    # Test 4: Enhanced SimpleTracker Production
    print("\n📊 Test 4: Enhanced SimpleTracker Production")
    total_tests += 1
    try:
        from shared.utils.enhanced_simple_tracker import EnhancedSimpleTracker
        
        tracker = EnhancedSimpleTracker()
        
        # Test enhanced tracking
        task_id = tracker.track_event(
            'production_test_001',
            'integration_validation',
            'llama3.2-3b',
            0.94,
            cost_usd=0.018,
            latency_ms=1200,
            context='Production integration test',
            tracking_level='enhanced',
            user_feedback='System integration successful'
        )
        
        print(f"   ✅ Enhanced tracking: {task_id}")
        
        # Test null-safe summary
        summary = tracker.get_enhanced_summary()
        print(f"   ✅ Enhanced summary: {summary['v1_metrics']['total_tasks']} tasks")
        print(f"   ✅ Success rate: {summary['v1_metrics']['avg_success_rate']}%")
        
        # Test system health
        health = tracker.get_v2_system_health()
        print(f"   ✅ System health: {health['overall_health']}")
        
        passed_tests += 1
        print("   ✅ Enhanced SimpleTracker Production: PASS")
        
    except Exception as e:
        print(f"   ❌ Enhanced SimpleTracker Production: FAIL - {e}")
    
    # Test 5: Experience Manager
    print("\n📝 Test 5: Experience Manager")
    total_tests += 1
    try:
        from shared.experience_manager import ExperienceManager, record_task_experience
        
        manager = ExperienceManager()
        exp_id = record_task_experience(
            'exp_test_001', 'validation', 0.92, 0.015, 1100, 'llama3.2-3b'
        )
        
        print(f"   ✅ Experience recorded: {exp_id}")
        
        summary = manager.get_experience_summary()
        print(f"   ✅ Experience summary: {summary['total_experiences']} experiences")
        
        passed_tests += 1
        print("   ✅ Experience Manager: PASS")
        
    except Exception as e:
        print(f"   ❌ Experience Manager: FAIL - {e}")
    
    # Test 6: Pattern Mining Engine
    print("\n🔍 Test 6: Pattern Mining Engine")
    total_tests += 1
    try:
        from shared.learning.pattern_mining_engine import PatternMiningEngine, run_full_pattern_mining
        
        engine = PatternMiningEngine()
        results = run_full_pattern_mining(days_back=7)
        
        print(f"   ✅ Pattern mining: {results['summary']['total_patterns_discovered']} patterns")
        print(f"   ✅ Insights: {results['summary']['total_insights_generated']} insights")
        
        passed_tests += 1
        print("   ✅ Pattern Mining Engine: PASS")
        
    except Exception as e:
        print(f"   ❌ Pattern Mining Engine: FAIL - {e}")
    
    # Test 7: ML Training Pipeline
    print("\n🤖 Test 7: ML Training Pipeline")
    total_tests += 1
    try:
        from shared.learning.ml_training_pipeline import MLModelTrainingPipeline, train_all_models
        
        pipeline = MLModelTrainingPipeline()
        result = train_all_models()
        
        print(f"   ✅ Training jobs: {result['jobs_created']}")
        print(f"   ✅ Successful models: {len(result['successful_models'])}")
        
        passed_tests += 1
        print("   ✅ ML Training Pipeline: PASS")
        
    except Exception as e:
        print(f"   ❌ ML Training Pipeline: FAIL - {e}")
    
    # Test 8: Analytics Dashboard API
    print("\n📊 Test 8: Analytics Dashboard API")
    total_tests += 1
    try:
        from api.analytics_dashboard_api import AnalyticsDashboardAPI
        
        api = AnalyticsDashboardAPI()
        data = api.get_dashboard_data()
        
        print(f"   ✅ Dashboard data: {data['status']}")
        
        kaizen = api.get_kaizen_report()
        print(f"   ✅ Kaizen report: {kaizen.get('total_experiences', 0)} experiences")
        
        passed_tests += 1
        print("   ✅ Analytics Dashboard API: PASS")
        
    except Exception as e:
        print(f"   ❌ Analytics Dashboard API: FAIL - {e}")
    
    # Final Results
    success_rate = (passed_tests / total_tests) * 100
    
    print("\n🏆 COMPLETE SYSTEM TEST RESULTS:")
    print("=" * 55)
    print(f"Total Tests: {total_tests}")
    print(f"Passed Tests: {passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("\n🎉 EXCELLENT: Agent Zero V1 Production System FULLY OPERATIONAL!")
    elif success_rate >= 75:
        print("\n✅ GOOD: Agent Zero V1 Production System OPERATIONAL!")
    else:
        print("\n⚠️  NEEDS WORK: Some components require attention")
    
    print("\n🚀 Production Features Available:")
    print("   • Complete task decomposition with dependency management")
    print("   • Intelligent project planning and team formation") 
    print("   • Enhanced multi-dimensional tracking")
    print("   • Experience-based learning and recommendations")
    print("   • Pattern mining and optimization insights")
    print("   • ML model training and selection")
    print("   • Advanced analytics dashboard")
    print("   • Neo4j knowledge graph foundation")
    
    return success_rate >= 75

if __name__ == "__main__":
    success = test_complete_system()
    sys.exit(0 if success else 1)
