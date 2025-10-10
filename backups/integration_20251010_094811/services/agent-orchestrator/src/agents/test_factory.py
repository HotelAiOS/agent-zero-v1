from agent_factory import agent_factory
import logging

logging.basicConfig(level=logging.INFO)

def test_factory():
    print("🧪 Testing Agent Factory...")
    
    # List templates
    templates = agent_factory.list_available_templates()
    print(f"📋 Found templates: {templates}")
    
    if not templates:
        print("⚠️ No templates found!")
        return
    
    # Create agent from first template
    template_name = templates[0] 
    agent = agent_factory.create_agent(template_name, f"Test_{template_name}")
    
    if agent:
        print(f"✅ Created: {agent['name']}")
        print(f"🔧 Type: {agent['agent_type']}")
    else:
        print("❌ Failed to create agent")

if __name__ == "__main__":
    test_factory()
