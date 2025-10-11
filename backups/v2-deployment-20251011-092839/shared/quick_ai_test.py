import asyncio
from ai_brain import ai_brain

async def quick_test():
    print("🧠 Quick AI Brain Test...")
    
    result = await ai_brain.think("Write hello world in Python")
    
    print(f"Model used: {result.model_used}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Response: {result.response[:100]}...")
    
    stats = ai_brain.get_stats()
    print(f"Stats: {stats}")

if __name__ == "__main__":
    asyncio.run(quick_test())
