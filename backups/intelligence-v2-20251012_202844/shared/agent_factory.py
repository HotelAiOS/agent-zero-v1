from pathlib import Path
import yaml
import uuid
import logging

class AgentFactory:
    def __init__(self):
        self.templates_dir = Path(__file__).parent.parent / "config" / "agent_templates"
    
    def list_available_templates(self):
        return [f.stem for f in self.templates_dir.glob("*.yaml")]
    
    def create_agent(self, template_name, agent_name=None):
        template_path = self.templates_dir / f"{template_name}.yaml"
        with open(template_path) as f:
            template = yaml.safe_load(f)
        
        agent_id = str(uuid.uuid4())
        return {
            "id": agent_id,
            "name": agent_name or f"{template_name}_{agent_id[:8]}",
            "template": template_name,
            "agent_type": template.get("agent_type"),
            "capabilities": template.get("capabilities", [])
        }

agent_factory = AgentFactory()
