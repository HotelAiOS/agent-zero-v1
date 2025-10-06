"""
Test AgentLifecycleManager z Neo4j Knowledge Integration
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "shared"))

from agent_factory import AgentLifecycleManager
from knowledge import Neo4jClient

def main():
    print("🧪 Testing AgentLifecycleManager with Neo4j Integration\n")
    
    # Połącz z Neo4j
    print("1️⃣ Connecting to Neo4j...")
    neo4j = Neo4jClient()
    print("   ✅ Neo4j connected\n")
    
    # Utwórz lifecycle manager z Neo4j
    print("2️⃣ Creating lifecycle manager with Neo4j...")
    manager = AgentLifecycleManager(
        enable_messaging=False,  # Wyłącz RabbitMQ dla czystego testu
        neo4j_client=neo4j
    )
    print("   ✅ Manager created\n")
    
    # Utwórz agenta - powinien zapisać się do Neo4j
    print("3️⃣ Creating agent (will save to Neo4j)...")
    agent = manager.create_agent(
        agent_id="test_agent_1",
        agent_type="backend",
        capabilities=["python", "fastapi", "postgresql"]
    )
    print(f"   ✅ Agent created: {agent.agent_id}")
    print(f"      State: {agent.state}\n")
    
    # Sprawdź w Neo4j
    print("4️⃣ Verifying agent in Neo4j...")
    neo4j_agent = neo4j.get_agent("test_agent_1")
    if neo4j_agent:
        print(f"   ✅ Agent found in Neo4j!")
        print(f"      Type: {neo4j_agent['agent_type']}")
        print(f"      Capabilities: {neo4j_agent['capabilities']}\n")
    else:
        print("   ❌ Agent not found in Neo4j!\n")
    
    # Przypisz task
    print("5️⃣ Assigning task...")
    manager.transition_state("test_agent_1", manager.agents["test_agent_1"].state.__class__.READY)
    success = manager.assign_task(
        agent_id="test_agent_1",
        task_id="task_001",
        task_description="Build REST API with FastAPI"
    )
    print(f"   ✅ Task assigned: {success}\n")
    
    # Complete task z experience
    print("6️⃣ Completing task (will store experience)...")
    manager.complete_task(
        agent_id="test_agent_1",
        task_id="task_001",
        success=True,
        outcome="Successfully created FastAPI REST API with proper error handling",
        tokens_used=1500,
        response_time=3.5,
        store_experience=True
    )
    print("   ✅ Task completed and experience stored\n")
    
    # Pobierz experiences
    print("7️⃣ Retrieving similar experiences...")
    experiences = manager.get_agent_experiences(
        agent_id="test_agent_1",
        keywords=["API", "FastAPI"],
        limit=5
    )
    print(f"   ✅ Found {len(experiences)} experiences")
    if experiences:
        exp = experiences[0]
        print(f"      Context: {exp.get('context', 'N/A')}")
        print(f"      Success: {exp.get('success', 'N/A')}\n")
    
    # Sprawdź stats
    print("8️⃣ Getting agent stats from Neo4j...")
    stats = neo4j.get_agent_stats("test_agent_1")
    print(f"   Total tasks: {stats.get('total_tasks', 0)}")
    print(f"   Successful tasks: {stats.get('successful_tasks', 0)}")
    print(f"   Experiences: {stats.get('experiences', 0)}\n")
    
    # System health
    print("9️⃣ System health check...")
    health = manager.get_system_health()
    print(f"   Status: {health['status']}")
    print(f"   Total agents: {health['total_agents']}")
    print(f"   Neo4j enabled: {health['neo4j_enabled']}\n")
    
    print("✅ Neo4j Integration test complete!")
    
    # Cleanup
    neo4j.close()

if __name__ == "__main__":
    main()
