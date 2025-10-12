"""
Test Agent Factory
Prosty test pokazujący działanie fabryki agentów
"""

import sys
from pathlib import Path

# Dodaj parent directory do path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_factory import AgentFactory
from agent_factory.capabilities import TechStack, SkillLevel


def main():
    print("=" * 70)
    print("🏭 TEST AGENT FACTORY")
    print("=" * 70)
    
    # Utwórz fabrykę
    factory = AgentFactory()
    
    # Pokaż dostępne typy agentów
    print(f"\n📋 Dostępne typy agentów: {factory.list_available_types()}\n")
    
    # Utwórz pojedynczego agenta
    print("🤖 Tworzenie agenta Backend Developer...")
    backend_agent = factory.create_agent('backend', 'backend_1')
    
    if backend_agent:
        print(f"✅ Agent utworzony: {backend_agent.agent_id}")
        print(f"   Typ: {backend_agent.agent_type}")
        print(f"   Stan: {backend_agent.state.value}")
        
        # Pokaż capabilities
        tech_stack = factory.capability_matcher.get_agent_tech_stack('backend_1')
        print(f"   Tech stack: {[t.value for t in tech_stack]}")
    
    print("\n" + "-" * 70 + "\n")
    
    # Utwórz zespół
    print("👥 Tworzenie zespołu Full Stack...")
    team_types = ['architect', 'backend', 'frontend', 'database', 'tester', 'devops']
    team = factory.create_team(team_types)
    
    print(f"\n✅ Zespół utworzony z {len(team)} agentów:")
    for agent_id, agent in team.items():
        print(f"   - {agent_id}: {agent.agent_type} ({agent.state.value})")
    
    print("\n" + "-" * 70 + "\n")
    
    # Test znajdowania agentów z określoną technologią
    print("🔍 Wyszukiwanie agentów z FastAPI...")
    fastapi_agents = factory.capability_matcher.find_agents_with_capability(
        TechStack.FASTAPI,
        SkillLevel.ADVANCED
    )
    print(f"   Znaleziono: {fastapi_agents}")
    
    print("\n" + "-" * 70 + "\n")
    
    # Test rekomendacji zespołu
    print("💡 Rekomendacja zespołu dla projektu API...")
    recommended_team = factory.recommend_team_for_project(
        required_technologies=['FastAPI', 'PostgreSQL', 'JWT'],
        project_type='api'
    )
    print(f"   Rekomendowany zespół: {recommended_team}")
    
    print("\n" + "-" * 70 + "\n")
    
    # Pokaż informacje o konkretnym typie agenta
    print("ℹ️  Informacje o agencie Security Auditor:")
    security_info = factory.get_agent_info('security')
    if security_info:
        print(f"   Nazwa: {security_info['name']}")
        print(f"   Model AI: {security_info['ai_model']}")
        print(f"   Capabilities: {len(security_info['capabilities'])}")
        print(f"   Współpracuje z: {security_info['collaborates_with']}")
    
    print("\n" + "-" * 70 + "\n")
    
    # Health check systemu
    print("🏥 Health Check systemu agentów:")
    health = factory.lifecycle_manager.get_system_health()
    print(f"   Status: {health['status']}")
    print(f"   Całkowita liczba agentów: {health['total_agents']}")
    print(f"   Rozkład stanów: {health['state_distribution']}")
    
    print("\n" + "=" * 70)
    print("✅ Test zakończony pomyślnie!")
    print("=" * 70)


if __name__ == "__main__":
    main()
