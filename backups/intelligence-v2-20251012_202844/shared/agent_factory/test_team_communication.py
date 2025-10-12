"""
Test: Zespół agentów komunikuje się przez Factory
"""

import sys
import time
from pathlib import Path

sys.path.append(str(Path.cwd().parent))

print("\n" + "="*70)
print("🧪 TEST: Zespół Agentów z Factory + Messaging")
print("="*70 + "\n")

# Import Factory
from agent_factory.factory import AgentFactory

print("1️⃣  Tworzenie Agent Factory z messaging...")
factory = AgentFactory()
print(f"   ✅ Factory gotowa ({len(factory.templates)} typów)\n")

print("2️⃣  Tworzenie zespołu (Backend, Database, Frontend)...")

# Stwórz Backend Agent
backend = factory.create_agent("backend", agent_id="backend_team")
print(f"   ✅ {backend.agent_id} utworzony\n")

# Stwórz Database Agent
database = factory.create_agent("database", agent_id="database_team")
print(f"   ✅ {database.agent_id} utworzony\n")

# Stwórz Frontend Agent
frontend = factory.create_agent("frontend", agent_id="frontend_team")
print(f"   ✅ {frontend.agent_id} utworzony\n")

print("3️⃣  Uruchamianie nasłuchiwania...")

# Wszyscy zaczynają nasłuchiwać
backend.start_listening()
database.start_listening()
frontend.start_listening()

print("   ✅ Wszyscy nasłuchują\n")

# Poczekaj żeby consumers byli gotowi
time.sleep(1)

print("4️⃣  Backend → Database: Poproś o schema...")
backend.send_message(
    recipient_id="database_team",
    subject="Schema Request",
    content="Need users table schema with auth fields"
)
print("   ✅ Wysłano\n")

time.sleep(2)

print("5️⃣  Frontend → Backend: Poproś o API endpoints...")
frontend.send_message(
    recipient_id="backend_team",
    subject="API Request",
    content="Need REST endpoints for user management"
)
print("   ✅ Wysłano\n")

time.sleep(2)

print("6️⃣  Backend → ALL: Broadcast status...")
backend.broadcast(
    subject="Status Update",
    content="User management module in progress"
)
print("   ✅ Broadcast wysłany\n")

time.sleep(2)

print("7️⃣  Statystyki zespołu:")
for agent_id in ["backend_team", "database_team", "frontend_team"]:
    metrics = factory.lifecycle_manager.get_agent_metrics(agent_id)
    if metrics:
        print(f"   {agent_id}:")
        print(f"      Wysłanych: {metrics.messages_sent}")
        print(f"      Odebranych: {metrics.messages_received}")

print("\n8️⃣  System Health:")
health = factory.lifecycle_manager.get_system_health()
print(f"   Status: {health['status']}")
print(f"   Agentów: {health['total_agents']}")
print(f"   Wiadomości: {health['total_messages']}")

print("\n9️⃣  Cleanup...")
for agent_id in ["backend_team", "database_team", "frontend_team"]:
    factory.lifecycle_manager.terminate_agent(agent_id)
print("   ✅ Zespół zakończony")

print("\n" + "="*70)
print("✅ TEST ZAKOŃCZONY!")
print("="*70 + "\n")
