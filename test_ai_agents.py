"""
Test stabilności i odporności platformy Agent Zero (szybki test)
Generuje prosty endpoint FastAPI /hello z autoryzacją JWT (mock)
oraz test jednostkowy pytest.
"""
import asyncio
import logging
from intelligent_agent import IntelligentAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

async def test_stabilnosc_systemu():
    print("\n" + "="*70)
    print("🧪 SZYBKI TEST STABILNOŚCI PLATFORMY AGENT ZERO")
    print("="*70 + "\n")

    # Backend Agent
    backend = IntelligentAgent(
        agent_id="backend_ai_001",
        agent_type="backend",
        capabilities=["python", "fastapi", "code_generation"]
    )

    # Handler AI (pełny wynik na ekranie)
    async def ai_task_handler(message):
        task = message['data']
        from_agent = message['from']
        print(f"\n🔧 BACKEND OTRZYMAŁ ZADANIE OD {from_agent}")
        print(f"   Opis: {task.get('description')}")
        print(f"   🧠 Rozpoczynam generację kodu...\n")
        try:
            result = await backend.ai_brain.generate_code(
                task_description=task.get('description'),
                context=task.get('context')
            )
            print(f"   ✅ Wynik AI:")
            print(f"   Model: {result['model_used']}")
            print(f"   Time: {result['processing_time']:.2f}s\n")
            print("   == Kod (fragment) ==")
            preview = result['code'][:350]
            print(preview + ('...' if len(result['code']) > 350 else ''))
            print("   ===================")
            await backend.send_to_agent(
                target_agent_id=from_agent,
                message_type="ai_task_completed",
                data={
                    "status": "success",
                    "code": result['code'],
                    "model_used": result['model_used'],
                    "processing_time": result['processing_time']
                }
            )
        except Exception as e:
            print(f"   ❌ Błąd AI: {e}")
            await backend.send_to_agent(
                target_agent_id=from_agent,
                message_type="ai_task_completed",
                data={
                    "status": "error",
                    "error": str(e)
                }
            )

    backend.register_handler("ai_task_delegation", ai_task_handler)
    print("   ✅ Handler AI gotowy\n")

    # Frontend Agent
    frontend = IntelligentAgent(
        agent_id="frontend_ai_001",
        agent_type="frontend",
        capabilities=["task_management"]
    )

    async def ai_result_handler(message):
        result = message['data']
        print(f"\n🎨 FRONTEND OTRZYMAŁ WYNIK:")
        print(f"   Status: {result.get('status')}")
        if result.get('status') == 'success':
            print(f"   Model: {result.get('model_used')}")
            print(f"   Time: {result.get('processing_time'):.2f}s")
            print(f"   ✅ Kod wygenerowany poprawnie.")
        else:
            print(f"   ❌ Błąd: {result.get('error')}")

    frontend.register_handler("ai_task_completed", ai_result_handler)
    print("   ✅ Handler Frontendu gotowy\n")

    print("📝 Startujemy agentów...")
    await backend.start()
    await frontend.start()
    print("   ✅ Agenci ONLINE.")

    print("📝 Sending quick test task to backend...\n")
    await frontend.send_to_agent(
        target_agent_id="backend_ai_001",
        message_type="ai_task_delegation",
        data={
            "description": "Stwórz endpoint FastAPI /hello, autoryzacja JWT (z mockiem, niepełna walidacja), zwróć {'message': 'Hello user!'} oraz jeden test jednostkowy pytest.",
            "context": {}
        }
    )

    print("   ✅ Zadanie wysłane. Oczekuję na wygenerowanie kodu...\n")
    # Czekaj, ale znacznie krócej niż przy dużych zadaniach
    await asyncio.sleep(65)

    # Stop
    print("\n📝 Wyłączanie agentów...")
    await backend.stop()
    await frontend.stop()
    print("   ✅ KONIEC TESTU\n")

if __name__ == "__main__":
    asyncio.run(test_stabilnosc_systemu())
