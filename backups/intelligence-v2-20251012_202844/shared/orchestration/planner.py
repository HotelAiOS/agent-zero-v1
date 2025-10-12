"""
Intelligent Planner - Complete Production Version
"""
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import logging

from .task_decomposer import TaskDecomposer, Task, TaskType
from .team_builder import TeamBuilder, TeamComposition

logger = logging.getLogger(__name__)

@dataclass
class ProjectPlan:
    project_id: str
    project_name: str
    project_type: str
    tasks: List[Task]
    team: TeamComposition
    created_at: datetime = field(default_factory=datetime.now)
    estimated_duration_hours: float = 0.0
    estimated_duration_days: float = 0.0
    total_cost_estimate: float = 0.0
    status: str = "planned"
    progress: float = 0.0

class IntelligentPlanner:
    def __init__(self):
        self.task_decomposer = TaskDecomposer()
        self.team_builder = TeamBuilder()
        self.plans: Dict[str, ProjectPlan] = {}
        logger.info("✅ IntelligentPlanner initialized")
    
    def create_project_plan(self, project_name: str, project_type: str, 
                          business_requirements: List[str], **kwargs) -> ProjectPlan:
        """Create complete project plan"""
        project_id = f"proj_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🎯 Creating project plan: {project_name}")
        
        # 1. Decompose into tasks
        tasks = self.task_decomposer.decompose_project(project_type, business_requirements)
        
        # 2. Build team
        required_roles = list(set(task.required_agent_type for task in tasks))
        team = self.team_builder.build_team(project_id, required_roles)
        
        # 3. Estimates
        total_hours = sum(task.estimated_hours for task in tasks)
        total_days = total_hours / 8
        total_cost = total_hours * 100
        
        # 4. Create plan
        plan = ProjectPlan(
            project_id=project_id,
            project_name=project_name,
            project_type=project_type,
            tasks=tasks,
            team=team,
            estimated_duration_hours=total_hours,
            estimated_duration_days=total_days,
            total_cost_estimate=total_cost
        )
        
        self.plans[project_id] = plan
        
        logger.info(f"✅ Project plan created: {len(tasks)} tasks, {len(team.members)} team members")
        return plan
    
    def get_plan(self, project_id: str) -> Optional[ProjectPlan]:
        return self.plans.get(project_id)
    
    def list_plans(self) -> List[Dict]:
        return [
            {
                'project_id': p.project_id,
                'project_name': p.project_name,
                'status': p.status,
                'tasks': len(p.tasks),
                'team_size': len(p.team.members),
                'estimated_days': p.estimated_duration_days
            }
            for p in self.plans.values()
        ]
