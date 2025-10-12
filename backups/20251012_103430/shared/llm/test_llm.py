"""
Test LLM Integration
Test Ollama multi-model client
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ollama_client import OllamaClient, ModelConfig
from prompt_builder import PromptBuilder, PromptContext
from response_parser import ResponseParser


def test_client_initialization():
    """Test 1: Client initialization and config loading"""
    print("="*70)
    print("🧪 TEST 1: Client Initialization")
    print("="*70)
    
    try:
        # Initialize client
        client = OllamaClient(config_path="../../config.yaml")
        
        print(f"\n✅ Client initialized")
        print(f"   Base URL: {client.base_url}")
        print(f"   Agent models: {len(client.agent_models)}")
        print(f"   Protocol models: {len(client.protocol_models)}")
        
        # List agent models
        print(f"\n📋 Agent Model Assignment:")
        for agent_type, config in client.agent_models.items():
            print(f"   {agent_type:12} → {config.model:30} (temp={config.temperature}, ctx={config.num_ctx})")
        
        # List protocol models
        print(f"\n📋 Protocol Model Assignment:")
        for protocol_type, config in client.protocol_models.items():
            print(f"   {protocol_type:15} → {config.model}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_model_verification():
    """Test 2: Verify models are available"""
    print("\n" + "="*70)
    print("🧪 TEST 2: Model Verification")
    print("="*70)
    
    try:
        client = OllamaClient(config_path="../../config.yaml")
        
        # List available models
        print(f"\n📦 Available Ollama models:")
        available = client.list_available_models()
        for i, model in enumerate(available[:10], 1):
            print(f"   {i}. {model}")
        if len(available) > 10:
            print(f"   ... and {len(available) - 10} more")
        
        # Verify configured models
        print(f"\n✅ Verifying configured models:")
        results = client.verify_models()
        
        available_count = sum(1 for v in results.values() if v)
        missing_count = sum(1 for v in results.values() if not v)
        
        print(f"   Available: {available_count}/{len(results)}")
        print(f"   Missing: {missing_count}/{len(results)}")
        
        if missing_count > 0:
            print(f"\n⚠️  Missing models:")
            for model, available in results.items():
                if not available:
                    print(f"   - {model}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_prompt_building():
    """Test 3: Prompt building"""
    print("\n" + "="*70)
    print("🧪 TEST 3: Prompt Building")
    print("="*70)
    
    try:
        # Build task prompt
        context = PromptContext(
            agent_type='backend',
            task_name='Create REST API endpoint',
            task_description='Create a GET /users endpoint that returns user list',
            tech_stack=['Python', 'FastAPI', 'SQLAlchemy'],
            requirements=[
                'Return JSON response',
                'Include pagination',
                'Add error handling'
            ]
        )
        
        messages = PromptBuilder.build_task_prompt(context)
        
        print(f"\n📋 Generated prompt:")
        print(f"   Messages: {len(messages)}")
        print(f"   System prompt length: {len(messages[0]['content'])} chars")
        print(f"   User prompt length: {len(messages[1]['content'])} chars")
        
        print(f"\n📝 System prompt preview:")
        print(messages[0]['content'][:150] + "...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_ollama_chat():
    """Test 4: Real Ollama chat (requires Ollama running)"""
    print("\n" + "="*70)
    print("🧪 TEST 4: Ollama Chat (Real LLM Call)")
    print("="*70)
    
    try:
        client = OllamaClient(config_path="../../config.yaml")
        
        # Simple test message
        messages = [
            {
                'role': 'system',
                'content': 'Jesteś pomocnym asystentem. Odpowiadaj krótko i zwięźle.'
            },
            {
                'role': 'user',
                'content': 'Napisz prostą funkcję Python która dodaje dwie liczby. Zwróć tylko kod.'
            }
        ]
        
        print(f"\n⚡ Calling Ollama (backend agent: deepseek-coder:33b)...")
        print(f"   This may take 10-30 seconds...\n")
        
        # Call with backend agent model
        response = client.chat(messages, agent_type='backend')
        
        # Extract response
        content = response['message']['content']
        
        print(f"\n✅ Response received:")
        print(f"   Length: {len(content)} chars")
        print(f"   Tokens: {response.get('eval_count', 0)} output, {response.get('prompt_eval_count', 0)} input")
        
        print(f"\n📄 Response preview:")
        print(content[:300] + "..." if len(content) > 300 else content)
        
        # Try to extract code
        code_blocks = ResponseParser.extract_code_blocks(content)
        if code_blocks:
            print(f"\n💻 Extracted {len(code_blocks)} code block(s)")
            print(code_blocks[0]['code'])
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print(f"   Make sure Ollama is running: ollama serve")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🧪 TEST LLM INTEGRATION - AGENT ZERO V1".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    results = []
    
    # Test 1: Initialization
    results.append(("Initialization", test_client_initialization()))
    
    # Test 2: Model verification
    results.append(("Model Verification", test_model_verification()))
    
    # Test 3: Prompt building
    results.append(("Prompt Building", test_prompt_building()))
    
    # Test 4: Real Ollama call (optional)
    print("\n" + "="*70)
    print("⚠️  Test 4 requires Ollama running on localhost:11434")
    response = input("Run Test 4 (Ollama Chat)? [y/N]: ")
    if response.lower() == 'y':
        results.append(("Ollama Chat", test_ollama_chat()))
    else:
        print("Skipped Test 4")
    
    # Summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name:30} {status}")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    print(f"\n   Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED!")
        print("="*70)
        print("\n🚀 Ollama Integration Ready!")
        print("\nNext steps:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Integrate with Agent Factory")
        print("3. Test real task execution")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
