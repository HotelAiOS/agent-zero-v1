import asyncio
from task_classifier import task_classifier

async def test_intelligent_classification():
    """Test sophisticated AI classification"""
    print("🧠 Testing Intelligent Task Classifier...")
    
    test_tasks = [
        "Write a simple hello world function in Python",
        "Design a microservices architecture for e-commerce platform with event sourcing",
        "Debug performance issues in React application with 10,000+ components",
        "Create a machine learning model for fraud detection using scikit-learn"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n🧪 TEST {i}: {task[:60]}...")
        
        profile = await task_classifier.classify_task(task)
        
        print(f"  📊 Complexity: {profile.complexity}")
        print(f"  🏷️  Domain: {profile.domain}")
        print(f"  🔢 Tokens: {profile.token_count}")
        print(f"  🤖 Model: {profile.recommended_model}")
        print(f"  💫 Confidence: {profile.confidence:.2f}")
        print(f"  💭 Reasoning: {profile.reasoning[:80]}...")
        
        print("  " + "="*60)

if __name__ == "__main__":
    asyncio.run(test_intelligent_classification())
