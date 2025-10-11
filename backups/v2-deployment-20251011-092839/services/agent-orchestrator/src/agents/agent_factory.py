import yaml
import uuid
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory for creating agents from YAML templates"""
    
    def __init__(self, templates_dir: str = "../../../../config/agent_templates"):
        # Path relative to services/agent-orchestrator/src/agents/
        self.templates_dir = Path(__file__).parent / templates_dir
        self.created_agents: Dict[str, Dict[str, Any]] = {}
    
    def list_available_templates(self) -> List[str]:
        """List all available agent templates"""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory {self.templates_dir} does not exist")
            return []
        
        templates = []
        for yaml_file in self.templates_dir.glob("*.yaml"):
            templates.append(yaml_file.stem)
        return templates
    
    def load_template(self, template_name: str) -> Optional[Dict[str, Any]]:
        """Load agent template from YAML file"""
        template_path = self.templates_dir / f"{template_name}.yaml"
        
        if not template_path.exists():
            logger.error(f"Template {template_name} not found at {template_path}")
            return None
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                template = yaml.safe_load(f)
            logger.info(f"✅ Loaded template: {template_name}")
            return template
        except Exception as e:
            logger.error(f"❌ Error loading template {template_name}: {e}")
            return None
    
    def create_agent(
        self, 
        template_name: str, 
        agent_name: Optional[str] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Create new agent instance from template"""
        
        # Load template
        template = self.load_template(template_name)
        if not template:
            return None
        
        # Generate unique agent ID and name
        agent_id = str(uuid.uuid4())
        if not agent_name:
            agent_name = f"{template.get('name', template_name)}_{agent_id[:8]}"
        
        # Create agent instance
        agent_instance = {
            "id": agent_id,
            "name": agent_name,
            "template": template_name,
            "created_at": str(uuid.uuid1().time),
            "status": "created",
            
            # Core configuration from template
            "agent_type": template.get("agent_type"),
            "description": template.get("description"),
            "capabilities": template.get("capabilities", []),
            "tech_stack": template.get("tech_stack", {}),
            "system_prompt": template.get("system_prompt"),
            
            # Communication setup
            "message_patterns": template.get("message_patterns", {}),
            "collaboration": template.get("collaboration", {}),
            
            # Configuration
            "config": template.get("config", {}),
            
            # Runtime state
            "current_tasks": [],
            "performance_metrics": {
                "tasks_completed": 0,
                "success_rate": 0.0,
                "average_response_time": 0.0
            }
        }
        
        # Apply custom configuration overrides
        if custom_config:
            agent_instance.update(custom_config)
        
        # Store created agent
        self.created_agents[agent_id] = agent_instance
        
        logger.info(f"🤖 Created agent: {agent_name} ({agent_id[:8]}) from template {template_name}")
        return agent_instance
    
    def get_agent(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get agent instance by ID"""
        return self.created_agents.get(agent_id)
    
    def list_created_agents(self) -> List[Dict[str, Any]]:
        """List all created agent instances"""
        return list(self.created_agents.values())
    
    def create_team(self, team_config: Dict[str, Any]) -> Dict[str, Any]:
        """Create team of agents for specific task"""
        team_id = str(uuid.uuid4())
        team_name = team_config.get("name", f"Team_{team_id[:8]}")
        
        team_agents = []
        for agent_spec in team_config.get("agents", []):
            if isinstance(agent_spec, str):
                # Simple template name
                agent = self.create_agent(agent_spec)
            else:
                # Detailed agent specification
                agent = self.create_agent(
                    agent_spec.get("template"),
                    agent_spec.get("name"),
                    agent_spec.get("config")
                )
            
            if agent:
                team_agents.append(agent)
        
        team = {
            "id": team_id,
            "name": team_name,
            "description": team_config.get("description"),
            "agents": team_agents,
            "created_at": str(uuid.uuid1().time),
            "status": "active"
        }
        
        logger.info(f"👥 Created team: {team_name} with {len(team_agents)} agents")
        return team

# Global factory instance
agent_factory = AgentFactory()
