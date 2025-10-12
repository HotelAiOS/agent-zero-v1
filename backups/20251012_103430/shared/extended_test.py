import asyncio
from ai_brain import ai_brain

async def extended_quality_test():
    print("🎯 EXTENDED QUALITY TEST - 5 Minute Timeout")
    print("=" * 50)
    
    # Complex task for quality demonstration
    task = "Create a complete Python FastAPI user registration endpoint with bcrypt password hashing and SQLAlchemy database integration"
    
    print(f"📋 TASK: {task}")
    print("🧠 AI Brain thinking... (extended timeout: 5 minutes)")
    print("⏳ Please wait for quality results...")
    
    result = await ai_brain.think(task)
    
    print(f"\n✅ RESULT:")
    print(f"🤖 Model: {result.model_used}")  
    print(f"⏱️ Time: {result.processing_time:.1f}s")
    print(f"📊 Confidence: {result.confidence:.2f}")
    print(f"📝 Length: {len(result.response)} chars")
    
    print(f"\n📄 RESPONSE PREVIEW:")
    print("-" * 50)
    preview = result.response[:300] if len(result.response) > 300 else result.response
    print(preview)
    if len(result.response) > 300:
        print("... (truncated)")
    print("-" * 50)
    
    # Quality indicators
    indicators = []
    if len(result.response) > 200:
        indicators.append("✅ Comprehensive")
    if "def " in result.response:
        indicators.append("✅ Functions")  
    if "import " in result.response:
        indicators.append("✅ Imports")
    if result.processing_time > 10:
        indicators.append("✅ Thoughtful")
        
    print(f"\n🏆 QUALITY: {len(indicators)}/4 indicators")
    for indicator in indicators:
        print(f"  {indicator}")

if __name__ == "__main__":
    asyncio.run(extended_quality_test())
