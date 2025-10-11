import asyncio
import logging
from ai_brain import ai_brain

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

async def test_ai_brain():
    """Test advanced AI Brain system"""
    print("🧠 Testing Advanced AI Brain System...")
    print("=" * 60)
    
    test_tasks = [
        {
            "task": "Write a Python function to calculate fibonacci numbers",
            "expected_model": "deepseek-coder:33b"
        },
        {
            "task": "Explain the trade-offs between microservices and monolithic architecture",
            "expected_model": "deepseek-r1:32b"
        },
        {
            "task": "Write engaging documentation for a REST API",
            "expected_model": "mixtral:8x7b"
        },
        {
            "task": "Hello world in Python",
            "expected_model": "phi3:mini"
        }
    ]
    
    for i, test_case in enumerate(test_tasks, 1):
        task = test_case["task"]
        expected = test_case["expected_model"]
        
        print(f"\n🧪 TEST {i}: {task}")
        print("-" * 60)
        
        # Think with AI Brain
        result = await ai_brain.think(task)
        
        # Show results
        print(f"📊 Classification: {result.classification['complexity']}")
        print(f"🤖 Model Used: {result.model_used}")
        print(f"⏱️  Processing Time: {result.processing_time:.2f} seconds")
        print(f"💫 Confidence: {result.confidence:.2f}")
        print(f"🎯 Expected Model: {expected}")
        print(f"✅ Model Match: {'✅' if expected in result.model_used else '❌'}")
        
        if result.reasoning_steps:
            print(f"🧠 Reasoning Steps Found: {len(result.reasoning_steps)}")
            for step in result.reasoning_steps[:2]:  # Show first 2 steps
                print(f"  -  {step[:80]}...")
        
        print(f"📝 Response Preview: {result.response[:150]}...")
        
    print("\n" + "=" * 60)
    print("📊 PERFORMANCE STATISTICS:")
    stats = ai_brain.get_performance_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Advanced AI Brain System Test Completed!")

if __name__ == "__main__":
    asyncio.run(test_ai_brain())
