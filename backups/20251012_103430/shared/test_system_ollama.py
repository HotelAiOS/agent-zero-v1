import logging
from ollama_client import ollama_client, ollama, chat

# Setup logging
logging.basicConfig(level=logging.INFO)

def test_system_ollama():
    """Test system ollama integration"""
    print("🧪 Testing System Ollama Integration...")
    print("=" * 50)
    
    # 1. Show available models
    models = ollama_client.available_models
    print(f"📋 Available Models ({len(models)}):")
    for i, model in enumerate(models[:10], 1):  # Show first 10
        print(f"  {i:2d}. {model}")
    
    if not models:
        print("❌ No ollama models found! Check: ollama list")
        return False
    
    # 2. Test fastest model (phi3:mini if available)
    test_model = None
    for preferred in ["phi3:mini", "llama3.2:latest", models[0]]:
        if ollama_client.is_model_available(preferred):
            test_model = preferred
            break
    
    print(f"\n🔬 Testing with model: {test_model}")
    print("-" * 50)
    
    # 3. Simple test
    test_messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Respond briefly."},
        {"role": "user", "content": "Say 'AI Brain operational' in exactly 3 words"}
    ]
    
    print("📤 Sending test message...")
    response = ollama.chat(test_model, test_messages)
    
    # 4. Show results
    if 'error' not in response:
        content = response['message']['content']
        print(f"✅ SUCCESS! Response: '{content}'")
        print(f"📊 Model: {response.get('model', 'unknown')}")
        print(f"✅ System ollama integration WORKING!")
        return True
    else:
        print(f"❌ ERROR: {response['error']}")
        return False

if __name__ == "__main__":
    success = test_system_ollama()
    print("\n" + "=" * 50)
    print("🎯 RESULT:", "✅ SUCCESS" if success else "❌ FAILED")
