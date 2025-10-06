"""
Test Multi-LLM Factory
"""
import sys
from pathlib import Path
import asyncio

sys.path.insert(0, str(Path(__file__).parent / "shared"))

from llm import LLMFactory

async def main():
    print("🧪 Testing Multi-LLM Factory\n")
    
    # Load config
    print("1️⃣ Loading config...")
    LLMFactory.load_config("shared/llm/config.yaml")
    print(f"   ✅ Default provider: {LLMFactory.get_default_provider()}")
    print(f"   ✅ Available providers: {LLMFactory.list_providers()}\n")
    
    # Test Ollama (should work)
    print("2️⃣ Testing Ollama client...")
    try:
        ollama = LLMFactory.create("ollama")
        print(f"   ✅ Ollama client created: {ollama}")
        
        models = ollama.list_available_models()
        print(f"   ✅ Available models: {len(models)}")
        
        # Test simple chat
        messages = [
            {"role": "user", "content": "Say 'Hello from Ollama' in one line"}
        ]
        response = ollama.chat(messages, agent_type="backend")
        print(f"   ✅ Chat response: {response.content[:100]}...")
        print(f"   ✅ Tokens: {response.tokens_used}\n")
    except Exception as e:
        print(f"   ❌ Ollama error: {e}\n")
    
    # Test OpenAI (will fail without API key - expected)
    print("3️⃣ Testing OpenAI client...")
    try:
        openai = LLMFactory.create("openai")
        print(f"   ⚠️ OpenAI client created (but no API key)\n")
    except Exception as e:
        print(f"   ℹ️ OpenAI not available: {e}\n")
    
    # Test Anthropic (will fail without API key - expected)
    print("4️⃣ Testing Anthropic client...")
    try:
        anthropic = LLMFactory.create("anthropic")
        print(f"   ⚠️ Anthropic client created (but no API key)\n")
    except Exception as e:
        print(f"   ℹ️ Anthropic not available: {e}\n")
    
    # Test automatic fallback
    print("5️⃣ Testing automatic fallback...")
    try:
        client = await LLMFactory.create_with_fallback("openai")
        print(f"   ✅ Fallback client: {client}")
        print(f"   ✅ Provider: {client.provider_name}\n")
    except Exception as e:
        print(f"   ❌ Fallback failed: {e}\n")
    
    print("✅ Multi-LLM Factory test complete!")

if __name__ == "__main__":
    asyncio.run(main())
