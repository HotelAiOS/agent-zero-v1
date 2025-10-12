import asyncio
from ai_brain import ai_brain

async def quick_success_test():
    print("🧠 Testing with Fast Model...")
    
    # Force phi3:mini (fastest, always works)
    result = await ai_brain.think("Hello world Python function")
    
    print(f"Model: {result.model_used}")
    print(f"Time: {result.processing_time:.2f}s")
    print(f"Response length: {len(result.response)} chars")
    print(f"First 100 chars: {result.response[:100]}...")
    
    return result

if __name__ == "__main__":
    asyncio.run(quick_success_test())
