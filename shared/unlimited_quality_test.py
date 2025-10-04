import asyncio
from ai_brain import ai_brain

async def unlimited_test():
    print("🎯 UNLIMITED TIMEOUT QUALITY TEST")
    print("⏳ PATIENCE: Large models need unlimited time!")
    print("=" * 60)
    
    task = "Create a complete FastAPI user registration endpoint with JWT authentication, SQLAlchemy models, bcrypt password hashing, input validation, and comprehensive error handling"
    
    print(f"📋 ENTERPRISE TASK: {task[:70]}...")
    print("🧠 AI Brain thinking with UNLIMITED timeout...")
    print("⏳ This may take 10-20 minutes for quality results!")
    
    result = await ai_brain.think(task)
    
    print(f"\n🎉 UNLIMITED QUALITY RESULT!")
    print(f"🤖 Model: {result.model_used}")
    print(f"⏱️ Time: {result.processing_time:.1f}s ({result.processing_time/60:.1f} min)")
    print(f"📊 Confidence: {result.confidence:.2f}")
    print(f"📝 Length: {len(result.response)} chars")
    
    print(f"\n📄 QUALITY PREVIEW:")
    print("-" * 50)
    print(result.response[:400])
    if len(result.response) > 400:
        print("... (comprehensive solution continues)")
    print("-" * 50)
    
    if result.model_used != "fallback":
        print(f"🎉 SUCCESS: {result.model_used} delivered quality solution!")
    else:
        print("⚠️ Fallback used - model may need more time")

if __name__ == "__main__":
    asyncio.run(unlimited_test())
