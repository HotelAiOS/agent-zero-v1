"""
Test AgentFactory z Multi-LLM Support
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "shared"))

from agent_factory import AgentFactory

def main():
    print("🧪 Testing AgentFactory with Multi-LLM Support\n")
    
    # Inicjalizuj factory - automatycznie ładuje LLMFactory
    factory = AgentFactory()
    
    print(f"✅ AgentFactory initialized")
    print(f"   LLM Client: {type(factory.llm_client).__name__}")
    print(f"   Templates loaded: {len(factory.templates)}\n")
    
    # Utwórz agenta backend
    print("Creating backend agent...")
    agent = factory.create_agent("backend")
    
    print(f"✅ Agent created: {agent.agent_id}")
    print(f"   Type: {agent.agent_type}")
    print(f"   State: {agent.state}")
    print(f"   LLM Client: {type(agent.llm_client).__name__}\n")
    
    # Test czy agent ma LLM client
    if hasattr(agent, 'llm_client') and agent.llm_client:
        print("✅ Agent has LLM client attached")
        print(f"   Provider: {agent.llm_client.provider if hasattr(agent.llm_client, 'provider') else 'ollama'}")
    else:
        print("❌ Agent missing LLM client!")
    
    print("\n✅ Multi-LLM Integration test complete!")

if __name__ == "__main__":
    main()
