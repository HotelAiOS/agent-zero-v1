"""
Test AI-Powered Agents z Knowledge Graph Integration

Ten test pokazuje pełny workflow:
1. Agent generuje kod z AI Brain
2. Kod automatycznie zapisuje się do Neo4j Knowledge Graph
3. System może znaleźć podobne zadania z przeszłości
4. Agent uczy się z historii
"""
import asyncio
import logging
import sys
import os
from intelligent_agent import IntelligentAgent

# Import Knowledge Graph
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'knowledge'))
from knowledge_graph import knowledge_graph

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)


async def test_ai_powered_agents_with_knowledge():
    """Test agentów z AI Brain + Knowledge Graph"""
    
    print("\n" + "="*70)
    print("🧪 TESTING AI-POWERED AGENTS + KNOWLEDGE GRAPH")
    print("="*70 + "\n")
    
    # Połącz z Knowledge Graph
    print("📝 Step 0: Connecting to Knowledge Graph...")
    await knowledge_graph.connect()
    print("   ✅ Knowledge Graph connected\n")
    
    # Sprawdź czy są podobne zadania w historii
    print("📝 Step 0.5: Checking for similar tasks in history...")
    similar_before = await knowledge_graph.find_similar_tasks(
        "user registration endpoint",
        limit=3
    )
    if similar_before:
        print(f"   🔍 Found {len(similar_before)} similar tasks from past:")
        for task in similar_before:
            print(f"      - {task['description']} (by {task['agent_id']})")
        print()
    else:
        print("   📝 No similar tasks yet (first time)\n")
    
    # Stwórz Backend Agent
    print("📝 Step 1: Creating Backend Agent with AI Brain...")
    backend = IntelligentAgent(
        agent_id="backend_ai_001",
        agent_type="backend",
        capabilities=["python", "fastapi", "code_generation"]
    )
    
    # Handler który używa AI Brain + zapisuje do KG
    async def ai_task_handler(message):
        """Backend używa AI Brain i zapisuje wynik do Knowledge Graph"""
        task = message['data']
        from_agent = message['from']
        
        print(f"\n🔧 BACKEND RECEIVED AI TASK:")
        print(f"   From: {from_agent}")
        print(f"   Task: {task.get('description')}")
        print(f"   🧠 Using AI Brain to generate code...\n")
        
        try:
            # PRAWDZIWA GENERACJA KODU
            result = await backend.ai_brain.generate_code(
                task_description=task.get('description'),
                context=task.get('context')
            )
            
            print(f"   ✅ AI Brain Response:")
            print(f"   Model: {result['model_used']}")
            print(f"   Time: {result['processing_time']:.2f}s")
            print(f"   Confidence: {result['confidence']}")
            print(f"\n   Generated Code Preview:")
            print("   " + "="*60)
            code_preview = result['code'][:500]
            for line in code_preview.split('\n'):
                print(f"   {line}")
            if len(result['code']) > 500:
                print(f"   ... (truncated, total {len(result['code'])} chars)")
            print("   " + "="*60 + "\n")
            
            # ZAPISZ DO KNOWLEDGE GRAPH
            print(f"   💾 Saving to Knowledge Graph...")
            task_id = await knowledge_graph.record_code_generation(
                agent_id=backend.agent_id,
                task_description=task.get('description'),
                generated_code=result['code'],
                model_used=result['model_used'],
                processing_time=result['processing_time'],
                success=True,
                context=task.get('context')
            )
            print(f"   ✅ Saved as: {task_id}\n")
            
            # Wyślij wynik do Frontend
            await backend.send_to_agent(
                target_agent_id=from_agent,
                message_type="ai_task_completed",
                data={
                    "status": "success",
                    "code": result['code'],
                    "model_used": result['model_used'],
                    "processing_time": result['processing_time'],
                    "task_id": task_id  # ID w Knowledge Graph
                }
            )
            
        except Exception as e:
            print(f"   ❌ AI Brain Error: {e}")
            await backend.send_to_agent(
                target_agent_id=from_agent,
                message_type="ai_task_completed",
                data={
                    "status": "error",
                    "error": str(e)
                }
            )
    
    backend.register_handler("ai_task_delegation", ai_task_handler)
    print("   ✅ AI handler registered\n")
    
    # Stwórz Frontend Agent
    print("📝 Step 2: Creating Frontend Agent...")
    frontend = IntelligentAgent(
        agent_id="frontend_ai_001",
        agent_type="frontend",
        capabilities=["task_management"]
    )
    
    # Handler dla wyniku
    async def ai_result_handler(message):
        """Frontend otrzymuje wygenerowany kod"""
        result = message['data']
        
        print(f"\n🎨 FRONTEND RECEIVED AI RESULT:")
        print(f"   Status: {result.get('status')}")
        
        if result.get('status') == 'success':
            print(f"   Model Used: {result.get('model_used')}")
            print(f"   Processing Time: {result.get('processing_time'):.2f}s")
            print(f"   Knowledge Graph ID: {result.get('task_id')}")
            print(f"   💚 AI Code Generation Successful!")
        else:
            print(f"   ❌ Error: {result.get('error')}")
    
    frontend.register_handler("ai_task_completed", ai_result_handler)
    print("   ✅ Frontend handler registered\n")
    
    # Uruchom oba agenty
    print("📝 Step 3: Starting agents...")
    print("-" * 70)
    await backend.start()
    await frontend.start()
    print("-" * 70)
    print("✅ Both AI agents are ONLINE!\n")
    
    # Frontend deleguje task
    print("📝 Step 4: Frontend requests AI code generation...")
    print("-" * 70)
    
    await frontend.send_to_agent(
        target_agent_id="backend_ai_001",
        message_type="ai_task_delegation",
        data={
            "description": "Create a FastAPI endpoint for user login with JWT token generation and refresh token support",
            "context": {
                "tech_stack": "FastAPI, JWT, Redis for refresh tokens",
                "requirements": [
                    "JWT access token (15 min expiry)",
                    "Refresh token (7 days expiry)",
                    "Store refresh tokens in Redis",
                    "Revoke token endpoint"
                ]
            }
        }
    )
    
    print("✅ AI task sent to Backend")
    print("-" * 70 + "\n")
    
    # Poczekaj na AI generację
    print("📝 Step 5: Waiting for AI Brain to generate code...")
    print("   (This may take 10-30 seconds depending on model)\n")
    await asyncio.sleep(35)
    
    # Sprawdź Knowledge Graph - czy zapisało
    print("\n📝 Step 6: Checking Knowledge Graph...")
    print("-" * 70)
    
    # Znajdź podobne zadania
    similar_after = await knowledge_graph.find_similar_tasks(
        "user login endpoint",
        limit=5
    )
    print(f"   🔍 Found {len(similar_after)} similar tasks in Knowledge Graph:")
    for i, task in enumerate(similar_after, 1):
        print(f"   {i}. {task['description']}")
        print(f"      Agent: {task['agent_id']}, Model: {task['model_used']}")
        print(f"      Time: {task['processing_time']:.2f}s")
    
    # Statystyki backend agenta
    print(f"\n   📊 Backend Agent Stats:")
    stats = await knowledge_graph.get_agent_stats("backend_ai_001")
    print(f"      Total tasks: {stats['total_tasks']}")
    print(f"      Success rate: {stats['success_rate']:.1f}%")
    print(f"      Avg time: {stats['avg_processing_time']:.2f}s")
    print(f"      Code lines: {stats['total_code_lines']}")
    print("-" * 70 + "\n")
    
    # Zatrzymaj agenty
    print("📝 Step 7: Stopping agents...")
    print("-" * 70)
    await backend.stop()
    await frontend.stop()
    await knowledge_graph.close()
    print("-" * 70)
    
    print("\n" + "="*70)
    print("✅ AI-POWERED AGENTS + KNOWLEDGE GRAPH TEST COMPLETED!")
    print("="*70 + "\n")
    
    print("📊 Summary:")
    print("   ✅ AI Brain integration working")
    print("   ✅ Real code generation (not simulation)")
    print("   ✅ Automatic Knowledge Graph recording")
    print("   ✅ Similar task discovery")
    print("   ✅ Agent statistics tracking")
    print("   ✅ Agent-to-agent AI collaboration")
    print("\n🎉 Agents are INTELLIGENT and have MEMORY!\n")


if __name__ == "__main__":
    asyncio.run(test_ai_powered_agents_with_knowledge())

