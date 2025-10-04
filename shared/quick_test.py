from agent_factory import agent_factory
print("🧪 Quick test")
templates = agent_factory.list_available_templates() 
print(f"Templates: {templates}")
if templates:
    agent = agent_factory.create_agent(templates[0])
    print(f"Created: {agent['name']}")
    print(f"Type: {agent['agent_type']}")
