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
        """Sprawdź czy zadanie może być rozpoczęte"""
        return all(dep_id in completed_tasks for dep_id in self.depends_on)
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwersja do dict dla serializacji"""
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
    """Graf zależności między zadaniami"""
    tasks: List[Task]
    adjacency_list: Dict[str, List[str]] = field(default_factory=dict)
    
    def build_graph(self):
        """Zbuduj graf zależności"""
        self.adjacency_list = {}
        for task in self.tasks:
            self.adjacency_list[task.task_id] = task.depends_on.copy()
    
    def get_execution_order(self) -> List[List[str]]:
        """Zwróć zadania w kolejności wykonania (topological sort)"""
        in_degree = {}
        for task in self.tasks:
            in_degree[task.task_id] = len(task.depends_on)
        
        levels = []
        remaining = set(t.task_id for t in self.tasks)
        
        while remaining:
            current_level = [tid for tid in remaining if in_degree[tid] == 0]
            
            if not current_level:
                logger.error(f"Cycle detected in task dependencies! Remaining: {remaining}")
                break
            
            levels.append(current_level)
            
            for tid in current_level:
                remaining.remove(tid)
                for task in self.tasks:
                    if tid in task.depends_on:
                        in_degree[task.task_id] -= 1
        
        return levels
    
    def identify_parallel_tasks(self) -> List[List[str]]:
        """Zwróć grupy zadań które mogą być wykonane równolegle"""
        return self.get_execution_order()


class TaskDecomposer:
    """Dekompozycja wymagań biznesowych na wykonalne zadania"""
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None):
        if llm_client is None:
            config_path = Path(__file__).parent.parent / "llm" / "config.yaml"
            LLMFactory.load_config(str(config_path))
            self.llm_client = LLMFactory.create()
        else:
            self.llm_client = llm_client
        
        self.task_counter = 0
        
        self.phase_agent_mapping = {
            'discovery': ['architect', 'database'],
            'design': ['architect', 'database', 'security'],
            'implementation': ['backend', 'frontend', 'database'],
            'testing': ['tester', 'security'],
            'deployment': ['devops'],
            'optimization': ['performance', 'database']
        }
    
    async def decompose_project(self, requirements: str, project_type: str = "web_application") -> List[Task]:
        """Rozbij wymagania biznesowe na zadania"""
        logger.info(f"🔍 Decomposing project requirements (type: {project_type})...")
        
        analysis = await self._analyze_requirements_with_llm(requirements, project_type)
        tasks = self._create_tasks_from_analysis(analysis, requirements)
        self._add_dependencies(tasks)
        
        logger.info(f"✅ Created {len(tasks)} tasks with dependencies")
        return tasks
    
    async def _analyze_requirements_with_llm(self, requirements: str, project_type: str) -> Dict[str, Any]:
        """Użyj LLM do analizy wymagań"""
        
        prompt = f"""Analyze the following project requirements and extract structured information.

Project Type: {project_type}

Requirements:
{requirements}

Provide a JSON response with the following structure:
{{
    "features": [
        {{
            "name": "feature name",
            "description": "detailed description",
            "complexity": 1-5,
            "priority": "critical|high|medium|low",
            "components": ["backend", "frontend", "database"]
        }}
    ],
    "tech_requirements": ["list of required technologies"],
    "security_requirements": ["list of security features needed"],
    "performance_requirements": ["list of performance constraints"],
    "database_needs": {{
        "type": "relational|nosql|graph",
        "estimated_tables": 5,
        "requires_auth": true
    }}
}}

Focus on actionable, technical breakdown. Be specific."""

        try:
            response_obj = self.llm_client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            
            content = response_obj.content.strip()
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
            
            analysis = json.loads(content)
            logger.info(f"✅ LLM analysis complete: {len(analysis.get('features', []))} features identified")
            return analysis
            
        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(requirements, project_type)
    
    def _fallback_analysis(self, requirements: str, project_type: str) -> Dict[str, Any]:
        """Prosta analiza bez LLM (fallback)"""
        logger.warning("Using fallback analysis (no LLM)")
        
        return {
            "features": [
                {
                    "name": "Core Application",
                    "description": requirements[:200],
                    "complexity": 3,
                    "priority": "high",
                    "components": ["backend", "frontend", "database"]
                }
            ],
            "tech_requirements": ["python", "fastapi", "react", "postgresql"],
            "security_requirements": ["authentication", "authorization"],
            "performance_requirements": ["caching", "database_optimization"],
            "database_needs": {
                "type": "relational",
                "estimated_tables": 5,
                "requires_auth": True
            }
        }
    
    def _create_tasks_from_analysis(self, analysis: Dict[str, Any], requirements: str) -> List[Task]:
        """Utwórz zadania na podstawie analizy"""
        tasks = []
        
        tasks.append(self._create_task(
            title="System Architecture Design",
            description=f"Design overall system architecture based on requirements: {requirements[:200]}",
            agent_type="architect",
            priority=TaskPriority.CRITICAL,
            complexity=4,
            context={"requirements": requirements, "analysis": analysis, "phase": "discovery"}
        ))
        
        if analysis.get('database_needs'):
            tasks.append(self._create_task(
                title="Database Schema Design",
                description=f"Design database schema for {analysis['database_needs'].get('estimated_tables', 5)} tables",
                agent_type="database",
                priority=TaskPriority.CRITICAL,
                complexity=3,
                tech_stack=[analysis['database_needs'].get('type', 'relational')],
                context={"database_needs": analysis['database_needs'], "phase": "design"}
            ))
        
        if analysis.get('security_requirements'):
            tasks.append(self._create_task(
                title="Security Architecture",
                description=f"Design security: {', '.join(analysis['security_requirements'])}",
                agent_type="security",
                priority=TaskPriority.HIGH,
                complexity=3,
                required_capabilities=['authentication', 'authorization'],
                context={"security_requirements": analysis['security_requirements'], "phase": "design"}
            ))
        
        for feature in analysis.get('features', []):
            if 'backend' in feature.get('components', []):
                tasks.append(self._create_task(
                    title=f"Backend: {feature['name']}",
                    description=feature['description'],
                    agent_type="backend",
                    priority=self._parse_priority(feature.get('priority', 'medium')),
                    complexity=feature.get('complexity', 3),
                    tech_stack=analysis.get('tech_requirements', []),
                    context={"feature": feature, "phase": "implementation"}
                ))
            
            if 'frontend' in feature.get('components', []):
                tasks.append(self._create_task(
                    title=f"Frontend: {feature['name']}",
                    description=f"Implement UI for {feature['name']}",
                    agent_type="frontend",
                    priority=self._parse_priority(feature.get('priority', 'medium')),
                    complexity=feature.get('complexity', 3),
                    context={"feature": feature, "phase": "implementation"}
                ))
        
        tasks.append(self._create_task(
            title="Integration Testing",
            description="Test all integrated components",
            agent_type="tester",
            priority=TaskPriority.HIGH,
            complexity=3,
            context={"phase": "testing"}
        ))
        
        tasks.append(self._create_task(
            title="Security Testing",
            description="Security audit and penetration testing",
            agent_type="security",
            priority=TaskPriority.HIGH,
            complexity=4,
            context={"phase": "testing"}
        ))
        
        if analysis.get('performance_requirements'):
            tasks.append(self._create_task(
                title="Performance Optimization",
                description=f"Optimize: {', '.join(analysis['performance_requirements'])}",
                agent_type="performance",
                priority=TaskPriority.MEDIUM,
                complexity=3,
                context={"performance_requirements": analysis['performance_requirements'], "phase": "optimization"}
            ))
        
        tasks.append(self._create_task(
            title="Deployment Setup",
            description="Configure CI/CD and deployment infrastructure",
            agent_type="devops",
            priority=TaskPriority.HIGH,
            complexity=3,
            tech_stack=['docker', 'kubernetes'],
            context={"phase": "deployment"}
        ))
        
        return tasks
    
    def _create_task(self, title: str, description: str, agent_type: str, priority: TaskPriority,
                     complexity: int = 3, required_capabilities: List[str] = None,
                     tech_stack: List[str] = None, context: Dict[str, Any] = None) -> Task:
        """Helper do tworzenia zadania"""
        self.task_counter += 1
        task_id = f"task_{self.task_counter:04d}"
        
        return Task(
            task_id=task_id,
            title=title,
            description=description,
            agent_type=agent_type,
            priority=priority,
            complexity=complexity,
            required_capabilities=required_capabilities or [],
            tech_stack=tech_stack or [],
            estimated_duration_hours=complexity * 0.5,
            context=context or {}
        )
    
    def _add_dependencies(self, tasks: List[Task]):
        """Dodaj dependencies między zadaniami"""
        task_map = {t.task_id: t for t in tasks}
        
        architecture_tasks = [t for t in tasks if t.agent_type == 'architect']
        database_tasks = [t for t in tasks if t.agent_type == 'database']
        security_design_tasks = [t for t in tasks if t.agent_type == 'security' and 'design' in t.context.get('phase', '')]
        backend_tasks = [t for t in tasks if t.agent_type == 'backend']
        frontend_tasks = [t for t in tasks if t.agent_type == 'frontend']
        testing_tasks = [t for t in tasks if t.agent_type == 'tester']
        security_test_tasks = [t for t in tasks if t.agent_type == 'security' and 'testing' in t.context.get('phase', '')]
        performance_tasks = [t for t in tasks if t.agent_type == 'performance']
        devops_tasks = [t for t in tasks if t.agent_type == 'devops']
        
        for task in database_tasks + security_design_tasks:
            if architecture_tasks:
                task.depends_on.append(architecture_tasks[0].task_id)
        
        for task in backend_tasks:
            for dep_task in database_tasks + security_design_tasks:
                task.depends_on.append(dep_task.task_id)
        
        for task in frontend_tasks:
            if backend_tasks:
                matching_backend = None
                for bt in backend_tasks:
                    if task.context.get('feature', {}).get('name') == bt.context.get('feature', {}).get('name'):
                        matching_backend = bt
                        break
                
                if matching_backend:
                    task.depends_on.append(matching_backend.task_id)
                elif backend_tasks:
                    task.depends_on.append(backend_tasks[0].task_id)
        
        for task in testing_tasks:
            for impl_task in backend_tasks + frontend_tasks:
                task.depends_on.append(impl_task.task_id)
        
        for task in security_test_tasks:
            if testing_tasks:
                task.depends_on.append(testing_tasks[0].task_id)
        
        for task in performance_tasks:
            if testing_tasks:
                task.depends_on.append(testing_tasks[0].task_id)
        
        for task in devops_tasks:
            for dep_task in testing_tasks + security_test_tasks + performance_tasks:
                if dep_task.task_id not in task.depends_on:
                    task.depends_on.append(dep_task.task_id)
        
        for task in tasks:
            for dep_id in task.depends_on:
                if dep_id in task_map:
                    task_map[dep_id].blocks.append(task.task_id)
        
        logger.info("✅ Dependencies added to all tasks")
    
    def _parse_priority(self, priority_str: str) -> TaskPriority:
        """Parse priority string do TaskPriority enum"""
        mapping = {
            'critical': TaskPriority.CRITICAL,
            'high': TaskPriority.HIGH,
            'medium': TaskPriority.MEDIUM,
            'low': TaskPriority.LOW,
            'optional': TaskPriority.OPTIONAL
        }
        return mapping.get(priority_str.lower(), TaskPriority.MEDIUM)
    
    def build_dependency_graph(self, tasks: List[Task]) -> TaskDependency:
        """Zbuduj graf zależności"""
        dep_graph = TaskDependency(tasks=tasks)
        dep_graph.build_graph()
        return dep_graph
    
    def identify_parallel_tasks(self, tasks: List[Task]) -> List[List[str]]:
        """Zidentyfikuj zadania które mogą być wykonane równolegle"""
        dep_graph = self.build_dependency_graph(tasks)
        return dep_graph.identify_parallel_tasks()
