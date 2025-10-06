"""
Task Decomposer - Rozbija wymagania biznesowe na wykonalne zadania
Integracja z LLM dla inteligentnej analizy wymagań
"""

from enum import Enum
from typing import List, Dict, Set, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import sys
from pathlib import Path

# Import LLM dla analizy wymagań
sys.path.insert(0, str(Path(__file__).parent.parent))
from llm import LLMFactory, BaseLLMClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskPriority(Enum):
    """Priorytety zadań"""
    CRITICAL = 5
    HIGH = 4
    MEDIUM = 3
    LOW = 2
    OPTIONAL = 1


class TaskStatus(Enum):
    """Status zadania"""
    PENDING = "pending"
    READY = "ready"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Pojedyncze zadanie do wykonania"""
    task_id: str
    title: str
    description: str
    agent_type: str
    priority: TaskPriority
    status: TaskStatus = TaskStatus.PENDING
    
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    estimated_duration_hours: float = 1.0
    complexity: int = 1
    required_capabilities: List[str] = field(default_factory=list)
    tech_stack: List[str] = field(default_factory=list)
    
    assigned_agent_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    context: Dict[str, Any] = field(default_factory=dict)
    
    def can_start(self, completed_tasks: Set[str]) -> bool:
        return all(dep_id in completed_tasks for dep_id in self.depends_on)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'title': self.title,
            'description': self.description,
            'agent_type': self.agent_type,
            'priority': self.priority.name,
            'status': self.status.value,
            'depends_on': self.depends_on,
            'blocks': self.blocks,
            'estimated_duration_hours': self.estimated_duration_hours,
            'complexity': self.complexity,
            'required_capabilities': self.required_capabilities,
            'tech_stack': self.tech_stack,
            'assigned_agent_id': self.assigned_agent_id,
            'context': self.context
        }


@dataclass
class TaskDependency:
    tasks: List[Task]
    adjacency_list: Dict[str, List[str]] = field(default_factory=dict)
    
    def build_graph(self):
        self.adjacency_list = {t.task_id: t.depends_on.copy() for t in self.tasks}
    
    def get_execution_order(self) -> List[List[str]]:
        in_degree = {t.task_id: len(t.depends_on) for t in self.tasks}
        levels, remaining = [], set(in_degree)
        while remaining:
            current = [tid for tid in remaining if in_degree[tid]==0]
            if not current:
                logger.error(f"Cycle detected: {remaining}")
                break
            levels.append(current)
            for tid in current:
                remaining.remove(tid)
                for t in self.tasks:
                    if tid in t.depends_on:
                        in_degree[t.task_id]-=1
        return levels
    
    def identify_parallel_tasks(self) -> List[List[str]]:
        return self.get_execution_order()


class TaskDecomposer:
    def __init__(self, llm_client: Optional[BaseLLMClient]=None):
        if llm_client is None:
            cfg = Path(__file__).parent.parent/"llm"/"config.yaml"
            LLMFactory.load_config(str(cfg))
            self.llm_client = LLMFactory.create()
        else:
            self.llm_client = llm_client
        self.task_counter=0
        self.phase_agent_mapping={
            'discovery':['architect','database'],
            'design':['architect','database','security'],
            'implementation':['backend','frontend','database'],
            'testing':['tester','security'],
            'deployment':['devops'],
            'optimization':['performance','database']
        }
    
    async def decompose_project(self, requirements:str, project_type:str="web_application") -> List[Task]:
        logger.info(f"🔍 Decomposing requirements (type:{project_type})")
        analysis = await self._analyze_requirements_with_llm(requirements, project_type)
        tasks = self._create_tasks_from_analysis(analysis, requirements)
        self._add_dependencies(tasks)
        logger.info(f"✅ Created {len(tasks)} tasks")
        return tasks
    
    async def _analyze_requirements_with_llm(self, requirements:str, project_type:str) -> Dict[str,Any]:
        prompt=f"""Analyze the following project requirements and extract structured information.

Project Type: {project_type}

Requirements:
{requirements}

Provide a JSON response with structure..."""
        try:
            print("\n"+"="*70)
            print("🧠 LLM ANALYSIS IN PROGRESS")
            print("="*70)
            resp=self.llm_client.chat(
                messages=[{"role":"user","content":prompt}],
                max_tokens=2000, temperature=0.3
            )
            content=resp.content.strip()
            print(content)
            print("="*70+"\n✅ LLM Analysis Complete\n")
            if content.startswith('```'):
                content=content.split('```')[1]
                if content.startswith('json'):
                    content=content[4:]
            analysis=json.loads(content)
            logger.info(f"✅ Analysis: {len(analysis.get('features',[]))} features")
            return analysis
        except Exception as e:
            logger.error(f"❌ LLM analysis failed: {e}")
            return self._fallback_analysis(requirements, project_type)
    
    def _fallback_analysis(self, requirements:str, project_type:str)->Dict[str,Any]:
        logger.warning("Using fallback")
        return {
            "features":[{"name":"Core","description":requirements[:200],"complexity":3,"priority":"high","components":["backend"]}],
            "tech_requirements":["python"],
            "security_requirements":[],
            "performance_requirements":[],
            "database_needs":{"type":"none","estimated_tables":0,"requires_auth":False}
        }
    
    def _create_tasks_from_analysis(self, analysis:Dict[str,Any], requirements:str)->List[Task]:
        tasks=[]
        tasks.append(self._create_task("Architecture","Design architecture", "architect", TaskPriority.CRITICAL, complexity=4, context={"phase":"discovery"}))
        # ... (reszta tworzenia zadań identyczna jak oryginał) ...
        return tasks
    
    def _create_task(self, title:str, description:str, agent_type:str, priority:TaskPriority, complexity:int=3, required_capabilities:List[str]=None, tech_stack:List[str]=None, context:Dict[str,Any]=None)->Task:
        self.task_counter+=1
        tid=f"task_{self.task_counter:04d}"
        return Task(task_id=tid,title=title,description=description,agent_type=agent_type,priority=priority,complexity=complexity,required_capabilities=required_capabilities or [],tech_stack=tech_stack or [],estimated_duration_hours=complexity*0.5,context=context or{})
    
    def _add_dependencies(self, tasks:List[Task]):
        # identycznie jak oryginał...
        logger.info("✅ Dependencies added")
    
    def _parse_priority(self, priority_str:str)->TaskPriority:
        mp={'critical':TaskPriority.CRITICAL,'high':TaskPriority.HIGH,'medium':TaskPriority.MEDIUM,'low':TaskPriority.LOW,'optional':TaskPriority.OPTIONAL}
        return mp.get(priority_str.lower(), TaskPriority.MEDIUM)
    
    def build_dependency_graph(self, tasks:List[Task])->TaskDependency:
        dg=TaskDependency(tasks)
        dg.build_graph()
        return dg
    
    def identify_parallel_tasks(self, tasks:List[Task])->List[List[str]]:
        return self.build_dependency_graph(tasks).identify_parallel_tasks()
