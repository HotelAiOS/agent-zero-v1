import asyncio
from ai_brain import ai_brain
from agent_factory import agent_factory

async def quick_practical_demo():
    print("🚀 PRACTICAL DEMO: Multi-Agent Development (Optimized)")
    print("=" * 60)
    
    # Tworzymy team
    backend = agent_factory.create_agent("backend_developer", "Alice_Backend")
    frontend = agent_factory.create_agent("frontend_developer", "Bob_Frontend")
    
    print(f"👥 Team: {backend['name']} + {frontend['name']}")
    print()
    
    # REAL DEVELOPMENT TASKS (simplified for speed)
    tasks = [
        ("Backend", "Create a simple Python REST API endpoint for getting products"),
        ("Frontend", "Create a React component for displaying a product list"),
        ("Backend", "Write Python function to validate user login credentials"),
        ("Frontend", "Design a simple shopping cart component in React")
    ]
    
    results = []
    total_time = 0
    
    for i, (role, task) in enumerate(tasks, 1):
        print(f"🔧 Task {i} ({role}): {task[:50]}...")
        
        # AI Brain thinks (will use fastest available model)
        result = await ai_brain.think(task)
        
        print(f"  🤖 Model: {result.model_used}")
        print(f"  ⏱️ Time: {result.processing_time:.1f}s")
        print(f"  📝 Response: {len(result.response)} chars")
        
        # Show actual generated code preview
        if result.response != "Error":
            lines = result.response.split('\n')
            code_lines = [line for line in lines if line.strip() and not line.startswith('#')]
            if code_lines:
                print(f"  💻 Preview: {code_lines[0][:50]}...")
        
        results.append(result)
        total_time += result.processing_time
        print()
    
    # SUMMARY
    print("=" * 60)
    print("📊 DEVELOPMENT SESSION SUMMARY")
    print("=" * 60)
    
    working_results = [r for r in results if r.model_used != "fallback"]
    fallback_results = [r for r in results if r.model_used == "fallback"]
    
    print(f"✅ Successful AI generations: {len(working_results)}")
    print(f"⚠️ Fallback responses: {len(fallback_results)}")
    print(f"⏱️ Total development time: {total_time:.1f}s")
    
    if working_results:
        avg_confidence = sum(r.confidence for r in working_results) / len(working_results)
        models_used = set(r.model_used for r in working_results)
        total_code = sum(len(r.response) for r in working_results)
        
        print(f"📊 Average confidence: {avg_confidence:.2f}")
        print(f"🤖 AI models used: {', '.join(models_used)}")
        print(f"📝 Code generated: ~{total_code // 50} lines")
    
    # Performance stats from AI brain
    try:
        stats = ai_brain.get_stats()
        print(f"🧠 AI Brain stats: {stats}")
    except:
        print("🧠 AI Brain: Active and processing")
    
    print("\n🎉 PRACTICAL DEMO COMPLETED!")
    print("💡 System shows intelligent task routing even with model timeouts")

if __name__ == "__main__":
    asyncio.run(quick_practical_demo())
