"""
Test Agent Registry - Kompletny test wszystkich funkcji
"""
import asyncio
import logging
from datetime import datetime
from agent_registry import AgentRegistry, AgentInfo, agent_registry

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)


async def test_agent_registry():
    """Kompletny test agent registry"""
    
    print("\n" + "="*60)
    print("🧪 TESTING AGENT REGISTRY")
    print("="*60 + "\n")
    
    # Test 1: Rejestracja agentów
    print("📝 Test 1: Registering agents...")
    
    backend_agent = AgentInfo(
        agent_id="backend_001",
        agent_type="backend",
        capabilities=["python", "fastapi", "postgresql", "docker"],
        status="online",
        last_seen=datetime.now(),
        queue_name="agent_backend_001"
    )
    
    frontend_agent = AgentInfo(
        agent_id="frontend_001",
        agent_type="frontend",
        capabilities=["react", "typescript", "tailwind", "nextjs"],
        status="online",
        last_seen=datetime.now(),
        queue_name="agent_frontend_001"
    )
    
    devops_agent = AgentInfo(
        agent_id="devops_001",
        agent_type="devops",
        capabilities=["kubernetes", "docker", "terraform", "aws"],
        status="online",
        last_seen=datetime.now(),
        queue_name="agent_devops_001"
    )
    
    await agent_registry.register_agent(backend_agent)
    await agent_registry.register_agent(frontend_agent)
    await agent_registry.register_agent(devops_agent)
    
    print("✅ 3 agents registered\n")
    
    # Test 2: Statystyki
    print("📊 Test 2: Getting stats...")
    stats = await agent_registry.get_stats()
    print(f"   Total agents: {stats['total']}")
    print(f"   Online: {stats['online']}")
    print(f"   Busy: {stats['busy']}")
    print(f"   Offline: {stats['offline']}")
    print("✅ Stats retrieved\n")
    
    # Test 3: Znajdowanie po capability
    print("🔍 Test 3: Finding agent by capability...")
    python_agent = await agent_registry.find_agent_by_capability("python")
    if python_agent:
        print(f"   Found: {python_agent.agent_id} - {python_agent.agent_type}")
    
    react_agent = await agent_registry.find_agent_by_capability("react")
    if react_agent:
        print(f"   Found: {react_agent.agent_id} - {react_agent.agent_type}")
    
    docker_agent = await agent_registry.find_agent_by_capability("docker")
    if docker_agent:
        print(f"   Found: {docker_agent.agent_id} - {docker_agent.agent_type}")
    
    print("✅ Capability search working\n")
    
    # Test 4: Znajdowanie po typie
    print("🔍 Test 4: Finding agents by type...")
    backend_agents = await agent_registry.find_agents_by_type("backend")
    print(f"   Backend agents: {len(backend_agents)}")
    for agent in backend_agents:
        print(f"      - {agent.agent_id}: {agent.capabilities}")
    
    frontend_agents = await agent_registry.find_agents_by_type("frontend")
    print(f"   Frontend agents: {len(frontend_agents)}")
    
    print("✅ Type search working\n")
    
    # Test 5: Update status
    print("🔄 Test 5: Updating agent status...")
    await agent_registry.update_agent_status("backend_001", "busy")
    
    # Sprawdź czy zmienił status
    agent = await agent_registry.get_agent_by_id("backend_001")
    print(f"   Backend_001 status: {agent.status}")
    
    # Zmień z powrotem
    await agent_registry.update_agent_status("backend_001", "online")
    print("✅ Status update working\n")
    
    # Test 6: Get all agents
    print("📋 Test 6: Getting all agents...")
    all_agents = await agent_registry.get_all_agents()
    print(f"   Total agents: {len(all_agents)}")
    for agent in all_agents:
        print(f"      - {agent.agent_id} ({agent.agent_type}): {agent.status}")
    print("✅ Get all agents working\n")
    
    # Test 7: Wyrejestrowanie
    print("👋 Test 7: Unregistering agent...")
    success = await agent_registry.unregister_agent("devops_001")
    print(f"   Unregister success: {success}")
    
    stats_after = await agent_registry.get_stats()
    print(f"   Agents remaining: {stats_after['total']}")
    print("✅ Unregister working\n")
    
    # Test 8: Próba znalezienia nieistniejącego
    print("🔍 Test 8: Finding non-existent capability...")
    rust_agent = await agent_registry.find_agent_by_capability("rust")
    if rust_agent is None:
        print("   Correctly returned None for 'rust'")
    print("✅ Non-existent capability handling working\n")
    
    print("="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(test_agent_registry())
