#!/usr/bin/env python3
"""
Test fixes for Agent Zero V2.0
Addresses ML training pipeline test issue
"""

import asyncio
import sys
import os

# Add project to path
sys.path.insert(0, os.getcwd())

async def fix_ml_training_test():
    """Fix ML training pipeline test expectation"""
    try:
        from shared.learning.ml_training_pipeline import MLTrainingPipeline
        
        # Mock Neo4j client with proper data
        class FixedMockClient:
            async def execute_query(self, query, params=None):
                # Return more realistic training data
                return [
                    {
                        'task_type': f'test_task_{i}',
                        'model': f'test_model_{i % 3}',
                        'success_score': 0.8 + (i * 0.01),
                        'cost_usd': 0.001 + (i * 0.0001),
                        'latency_ms': 1000 + (i * 10),
                        'feedback_length': i % 10
                    }
                    for i in range(60)  # Enough samples for training
                ]
        
        pipeline = MLTrainingPipeline(FixedMockClient())
        result = await pipeline.train_models()
        
        print("🔧 ML Training Test Fix:")
        print(f"   Status: {result.get('status', 'unknown')}")
        print(f"   Training samples: {result.get('training_samples', 0)}")
        
        if result.get('status') == 'success':
            print("✅ ML Training Pipeline test should now pass")
            return True
        else:
            print(f"⚠️  Result: {result}")
            return False
            
    except Exception as e:
        print(f"❌ ML Training test fix failed: {e}")
        return False

async def test_components():
    """Test all components"""
    print("🧪 Testing Agent Zero V2.0 Components...")
    
    # Test imports
    try:
        from shared.experience.enhanced_tracker import V2ExperienceTracker
        print("✅ Experience Tracker import - OK")
    except Exception as e:
        print(f"❌ Experience Tracker: {e}")
    
    try:
        from shared.learning.pattern_mining_engine import PatternMiningEngine
        print("✅ Pattern Mining Engine import - OK")
    except Exception as e:
        print(f"❌ Pattern Mining Engine: {e}")
    
    try:
        from shared.learning.ml_training_pipeline import MLTrainingPipeline
        print("✅ ML Training Pipeline import - OK")
    except Exception as e:
        print(f"❌ ML Training Pipeline: {e}")
    
    # Test ML fix
    await fix_ml_training_test()

if __name__ == "__main__":
    asyncio.run(test_components())
