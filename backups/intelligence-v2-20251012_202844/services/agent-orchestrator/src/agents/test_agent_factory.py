import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from services.agent_orchestrator.src.agents.agent_factory import agent_factory
import logging

logging.basicConfig(level=logging.INFO)

def test_agent_factory():
    """Test Agent Factory functionality"""
    print("🧪 Testing Agent Factory...")
    
    # 1. List available templates
    print("\n📋 Available Templates:")
    templates = agent_factory.list_available_templates()
    for template in templates:
        print(f"  - {template}")
    
    # 2. Create Backend Developer agent
    print("\n🔧 Creating Backend Developer agent...")
    backend_agent = agent_factory.create_agent(
        template_name="backend_developer",
        agent_name="Senior_Backend_Alice"
    )
    
    if backend_agent:
        print(f"  ✅ Created: {backend_agent['name']} ({backend_agent['id'][:8]})")
        print(f"  📋 Capabilities: {len(backend_agent['capabilities'])} skills")
        print(f"  🔗 Listens to: {backend_agent['message_patterns']['listens_to']}")
    
    # 3. Create Frontend Developer agent
    print("\n🎨 Creating Frontend Developer agent...")
    frontend_agent = agent_factory.create_agent(
        template_name="frontend_developer", 
        agent_name="Senior_Frontend_Bob"
    )
    
    if frontend_agent:
        print(f"  ✅ Created: {frontend_agent['name']} ({frontend_agent['id'][:8]})")
        print(f"  🛠️ Tech Stack: {list(frontend_agent['tech_stack']['frameworks'])}")
    
    # 4. Create a team
    print("\n👥 Creating Development Team...")
    team = agent_factory.create_team({
        "name": "Full-Stack Development Team",
        "description": "Complete web application development team",
        "agents": [
            {"template": "backend_developer", "name": "Backend_Lead"},
            {"template": "frontend_developer", "name": "Frontend_Lead"}
        ]
    })
    
    print(f"  ✅ Created team: {team['name']}")
    print(f"  👥 Team size: {len(team['agents'])} agents")
    
    # 5. List all created agents
    print("\n🤖 All Created Agents:")
    agents = agent_factory.list_created_agents()
    for agent in agents:
        print(f"  - {agent['name']} ({agent['agent_type']})")
    
    print("\n✅ Agent Factory test completed!")

if __name__ == "__main__":
    test_agent_factory()
