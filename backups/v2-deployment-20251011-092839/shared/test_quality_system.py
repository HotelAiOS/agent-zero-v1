import asyncio
from ai_brain_quality import quality_ai_brain

async def test_quality_ai():
    print("🎯 TESTING QUALITY-FIRST AI SYSTEM")
    print("⏳ Patience required - Quality takes time!")
    print("=" * 60)
    
    # Test realistic development task
    task = "Create a Python FastAPI endpoint for user registration with email validation, password hashing, and database storage using SQLAlchemy"
    
    print(f"📋 TASK: {task}")
    print("🧠 AI Brain analyzing and generating quality solution...")
    print("⏳ This may take 3-5 minutes for comprehensive result...")
    
    result = await quality_ai_brain.think(task)
    
    print(f"\n✅ QUALITY RESULT:")
    print(f"🤖 Model: {result.model_used}")
    print(f"⏱️ Processing time: {result.processing_time:.1f} seconds")
    print(f"📊 Confidence: {result.confidence:.2f}")
    print(f"📝 Response length: {len(result.response)} characters")
    print(f"🧠 Reasoning steps: {len(result.reasoning_steps)}")
    
    if result.reasoning_steps:
        print(f"\n💭 REASONING PROCESS:")
        for step in result.reasoning_steps[:3]:
            print(f"  -  {step}")
    
    print(f"\n📄 RESPONSE PREVIEW:")
    print("─" * 60)
    print(result.response[:500] + "...")
    print("─" * 60)
    
    print(f"\n📊 SYSTEM STATS:")
    stats = quality_ai_brain.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    asyncio.run(test_quality_ai())
