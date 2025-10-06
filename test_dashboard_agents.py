"""
Test Dashboard - tworzy agentów i zadania
"""
import sys
from pathlib import Path
import asyncio
import time

sys.path.insert(0, str(Path(__file__).parent / "shared"))

from agent_factory.factory import AgentFactory
from agent_factory.lifecycle import AgentState

async def main():
    print("🚀 Tworzenie agentów testowych dla dashboard...")
    
    # Inicjalizuj factory
    factory = AgentFactory()
    
    # Utwórz 3 agentów różnych typów
    print("\n1️⃣ Tworzenie Backend Agent...")
    backend = factory.create_agent("backend")
    print(f"   ✅ Utworzono: {backend.agent_id}")
    
    print("\n2️⃣ Tworzenie Frontend Agent...")
    frontend = factory.create_agent("frontend")
    print(f"   ✅ Utworzono: {frontend.agent_id}")
    
    print("\n3️⃣ Tworzenie Database Agent...")
    database = factory.create_agent("database")
    print(f"   ✅ Utworzono: {database.agent_id}")
    
    # Symuluj aktywność
    print("\n📊 Symulacja aktywności agentów...")
    
    # Backend wykonuje zadanie
    backend.metrics.tasks_completed = 5
    backend.metrics.messages_sent = 12
    backend.metrics.messages_received = 8
    backend.metrics.uptime_seconds = 120.5
    factory.lifecycle_manager.transition_state(backend.agent_id, AgentState.BUSY)
    
    # Frontend w stanie READY
    frontend.metrics.tasks_completed = 3
    frontend.metrics.messages_sent = 7
    frontend.metrics.messages_received = 5
    factory.lifecycle_manager.transition_state(frontend.agent_id, AgentState.READY)
    
    # Database IDLE
    database.metrics.tasks_completed = 8
    database.metrics.messages_sent = 15
    database.metrics.messages_received = 12
    database.metrics.uptime_seconds = 200.0
    factory.lifecycle_manager.transition_state(database.agent_id, AgentState.IDLE)
    
    print("\n✅ Agenci aktywni! Sprawdź dashboard: http://localhost:5000")
    print("\n💡 Dashboard powinien teraz pokazywać:")
    print("   - 3 agentów")
    print("   - 16 zadań ukończonych")
    print("   - 59 wiadomości")
    print("   - Stany: 1x BUSY, 1x READY, 1x IDLE")
    
    print("\n⏳ Trzymam agentów aktywnych przez 60 sekund...")
    print("   (Odśwież dashboard żeby zobaczyć zmiany)")
    
    await asyncio.sleep(60)
    
    print("\n🛑 Zamykanie agentów...")
    factory.lifecycle_manager.terminate_agent(backend.agent_id)
    factory.lifecycle_manager.terminate_agent(frontend.agent_id)
    factory.lifecycle_manager.terminate_agent(database.agent_id)
    
    print("✅ Test zakończony!")

if __name__ == "__main__":
    asyncio.run(main())
