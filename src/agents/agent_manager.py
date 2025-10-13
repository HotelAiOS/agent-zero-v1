"""Agent Manager for Agent Zero V1"""
from .agent_base import BaseAgent

class AgentManager:
    def __init__(self):
        self.agents = {}
        
    def register_agent(self, agent):
        self.agents[agent.name] = agent
        
    def get_agent(self, name):
        return self.agents.get(name)
