from simple_classifier import classifier

def test():
    print("🧪 Testing Simple Classifier...")
    
    tasks = [
        "Write a simple hello world function",
        "Debug complex React performance issue"
    ]
    
    for task in tasks:
        result = classifier.classify_task(task)
        print(f"Task: {task[:40]}...")
        print(f"  Result: {result['complexity']}")
        print(f"  Model: {result['model']}")
        print(f"  Why: {result['reasoning']}")
        print()

if __name__ == "__main__":
    test()
