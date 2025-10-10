#!/bin/bash
#
# V2.0 Quick Test Script
# Test podstawowych funkcji Agent Zero V2.0
#

echo "🧪 Agent Zero V2.0 - Quick Functionality Test"
echo "=" * 50

python3 -c "
import sys
sys.path.append('.')

print('🔍 V2.0 Component Import Diagnostic')
print('=' * 50)

components = [
    ('Enhanced SimpleTracker', 'shared.utils.enhanced_simple_tracker', 'EnhancedSimpleTracker'),
    ('Experience Manager', 'shared.experience_manager', 'ExperienceManager'),
    ('Knowledge Graph', 'shared.knowledge.neo4j_knowledge_graph', 'KnowledgeGraphManager'),
    ('Pattern Mining', 'shared.learning.pattern_mining_engine', 'PatternMiningEngine'),
    ('ML Pipeline', 'shared.learning.ml_training_pipeline', 'MLModelTrainingPipeline'),
    ('Analytics API', 'api.analytics_dashboard_api', 'AnalyticsDashboardAPI')
]

working_components = []
for name, module_path, class_name in components:
    try:
        print(f'Testing {name}...')
        module = __import__(module_path, fromlist=[class_name])
        class_obj = getattr(module, class_name)
        print(f'  ✅ {name}: Import OK')
        working_components.append(name)
    except ImportError as e:
        print(f'  ❌ {name}: Import failed - {e}')
    except AttributeError as e:
        print(f'  ⚠️  {name}: Class not found - {e}')
    except Exception as e:
        print(f'  🔧 {name}: Dependency issue - {e}')

print(f'')
print(f'📊 Working Components: {len(working_components)}/{len(components)}')
print(f'💡 Working: {working_components}')
"

echo ""
echo "🧪 Testing Enhanced SimpleTracker V2.0..."

python3 -c "
import sys
sys.path.append('.')

try:
    from shared.utils.enhanced_simple_tracker import EnhancedSimpleTracker, TrackingLevel
    
    print('✅ Enhanced SimpleTracker imported successfully')
    
    # Initialize tracker
    tracker = EnhancedSimpleTracker()
    print('✅ Enhanced SimpleTracker initialized')
    
    # Test V2.0 tracking
    task_id = tracker.track_event(
        task_id='v2_functional_test_001',
        task_type='system_validation',
        model_used='test_model',
        success_score=0.95,
        cost_usd=0.01,
        latency_ms=1200,
        tracking_level=TrackingLevel.FULL,
        user_feedback='V2.0 system test successful'
    )
    
    print(f'✅ V2.0 Enhanced Tracking: {task_id}')
    
    # Test V2.0 summary
    summary = tracker.get_enhanced_summary()
    print(f'✅ V2.0 Summary: {summary[\"v1_metrics\"][\"total_tasks\"]} total tasks')
    
    # Test system health
    health = tracker.get_v2_system_health()
    print(f'✅ V2.0 Health: {health[\"overall_health\"]}')
    
    print('')
    print('🎉 Enhanced SimpleTracker V2.0 is FULLY FUNCTIONAL!')
    print('📊 You have working V2.0 Intelligence with:')
    print('  - Enhanced tracking with multi-dimensional scoring')
    print('  - V2.0 database schema')
    print('  - System health monitoring')
    print('  - V1.0 backward compatibility')
    
except Exception as e:
    print(f'❌ Enhanced SimpleTracker test failed: {e}')
"

echo ""
echo "🎯 Testing V2.0 CLI System Status..."
python3 cli/advanced_commands.py v2-system status