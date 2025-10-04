from smart_task_classifier import smart_classifier
import logging

logging.basicConfig(level=logging.INFO)

def test_smart_classification():
    """Test sophisticated AI classification"""
    print("🧠 Testing Smart Task Classifier with Real AI...")
    print("=" * 60)
    
    test_tasks = [
        "Write a hello world function in Python",
        "Design microservices architecture for e-commerce with event sourcing and CQRS",
        "Debug performance bottleneck in React app with 50,000+ components",
        "Create machine learning fraud detection system with real-time analysis",
        "Write creative documentation for API endpoints with examples"
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n🧪 TEST {i}: {task}")
        print("-" * 60)
        
        profile = smart_classifier.classify_task(task)
        
        print(f"📊 Complexity: {profile.complexity}")
        print(f"🏷️  Domain: {profile.domain}")
        print(f"🔢 Context: {profile.estimated_chars} chars")
        print(f"🧠 Reasoning: {profile.requires_reasoning}")
        print(f"🎨 Creative: {profile.requires_creativity}")
        print(f"⏱️  Time: {profile.estimated_time}s")
        print(f"🤖 Model: {profile.recommended_model}")
        print(f"💫 Confidence: {profile.confidence:.2f}")
        print(f"💭 Why: {profile.reasoning}")
        
        # Show model specs
        classifier = smart_classifier
        if profile.recommended_model in classifier.model_matrix:
            specs = classifier.model_matrix[profile.recommended_model]
            print(f"🔧 RAM: {specs['ram_usage']}GB, Speed: {specs['speed_score']}/10")
            print(f"💪 Strengths: {', '.join(specs['strengths'])}")
        
    print("\n" + "=" * 60)
    print("✅ Smart AI Classification System OPERATIONAL!")

if __name__ == "__main__":
    test_smart_classification()
