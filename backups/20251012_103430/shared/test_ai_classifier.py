from simple_classifier import classifier

def test_ai():
    print("🧠 Testing AI Classification...")
    
    tasks = [
        "Write hello world in Python",
        "Create REST API with authentication", 
        "Design microservices architecture",
        "Write creative API documentation"
    ]
    
    for task in tasks:
        print(f"\nTask: {task}")
        
        # Test AI classification
        ai_result = classifier.classify_with_ai(task)
        print(f"  🤖 AI Result: {ai_result['complexity']} → {ai_result['model']}")
        print(f"  💭 Reasoning: {ai_result['reasoning']}")

if __name__ == "__main__":
    test_ai()
