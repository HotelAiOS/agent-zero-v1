"""
Team Builder - Production Version
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TeamMember:
    agent_id: str
    agent_type: str
    role: str
    capabilities: List[str] = field(default_factory=list)
    current_workload: float = 0.0
    max_workload: float = 40.0

@dataclass
class TeamComposition:
    team_id: str
    project_id: str
    members: List[TeamMember]
    created_at: datetime = field(default_factory=datetime.now)

class TeamBuilder:
    def __init__(self):
        self.teams: Dict[str, TeamComposition] = {}
        self.agent_pool = {
            'architect': ['arch001', 'arch002'],
            'backend': ['be001', 'be002', 'be003'],
            'frontend': ['fe001', 'fe002'],
            'database': ['db001', 'db002'],
            'tester': ['test001', 'test002'],
            'devops': ['ops001']
        }
        logger.info("TeamBuilder initialized")
    
    def build_team(self, project_id: str, required_roles: List[str]) -> TeamComposition:
        """Build team for project"""
        team_id = f"team_{project_id}"
        members = []
        
        for role in required_roles:
            if role in self.agent_pool and self.agent_pool[role]:
                agent_id = self.agent_pool[role][0]
                member = TeamMember(
                    agent_id=agent_id,
                    agent_type=role,
                    role=role.title(),
                    capabilities=[role]
                )
                members.append(member)
        
        team = TeamComposition(team_id=team_id, project_id=project_id, members=members)
        self.teams[team_id] = team
        
        logger.info(f"Built team {team_id} with {len(members)} members")
        return team
