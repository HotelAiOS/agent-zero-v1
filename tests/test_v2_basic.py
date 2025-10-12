#!/usr/bin/env python3
"""Basic V2.0 integration test"""

import asyncio
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

async def test_basic_imports():
    """Test that V2.0 components can be imported"""
    try:
        from shared.experience.enhanced_tracker import V2ExperienceTracker
        print("✅ Enhanced Experience Tracker import - OK")
    except ImportError as e:
        print(f"❌ Enhanced Experience Tracker import failed: {e}")
        return False
    
    try:
        from shared.learning.pattern_mining_engine import PatternMiningEngine
        print("✅ Pattern Mining Engine import - OK")
    except ImportError as e:
        print(f"❌ Pattern Mining Engine import failed: {e}")
        return False
    
    try:
        from shared.learning.ml_training_pipeline import MLTrainingPipeline
        print("✅ ML Training Pipeline import - OK")
    except ImportError as e:
        print(f"❌ ML Training Pipeline import failed: {e}")
        return False
    
    return True

async def test_basic_functionality():
    """Test basic V2.0 functionality"""
    try:
        # Test Experience Tracker
        from shared.experience.enhanced_tracker import V2ExperienceTracker
        tracker = V2ExperienceTracker()
        
        test_exp = {
            'task_id': 'test_001',
            'task_type': 'integration_test',
            'model_used': 'test_model',
            'success_score': 0.9,
            'cost_usd': 0.001,
            'latency_ms': 500
        }
        
        result = await tracker.track_experience(test_exp)
        if result.get('status') == 'success':
            print("✅ Experience tracking - OK")
        else:
            print(f"❌ Experience tracking failed: {result}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

async def main():
    """Run all basic tests"""
    print("🧪 Agent Zero V2.0 - Basic Integration Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = await test_basic_imports()
    if not imports_ok:
        print("❌ Import tests failed")
        return 1
    
    # Test basic functionality
    functionality_ok = await test_basic_functionality()
    if not functionality_ok:
        print("❌ Functionality tests failed")
        return 1
    
    print("\n✅ All basic tests passed!")
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
