import logging
import networkx as nx
from typing import List, Dict, Any, Optional
from ..agents.base_agent import AgentTask, AgentCapability, TaskStatus

logger = logging.getLogger(__name__)

class TaskPlanner:
    """Planner zadań z rozwiązywaniem zależności"""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
    
    def add_task(self, task: AgentTask, dependencies: List[str] = None):
        """Dodaj zadanie do grafu z zależnościami"""
        self.dependency_graph.add_node(task.id, task=task)
        
        if dependencies:
            for dep_id in dependencies:
                if self.dependency_graph.has_node(dep_id):
                    self.dependency_graph.add_edge(dep_id, task.id)
                else:
                    logger.warning(f"Dependency {dep_id} not found for task {task.id}")
    
    def get_execution_order(self) -> List[AgentTask]:
        """Zwróć zadania w kolejności wykonania (topological sort)"""
        try:
            # Topological sort - kolejność wykonania
            sorted_ids = list(nx.topological_sort(self.dependency_graph))
            
            tasks = []
            for task_id in sorted_ids:
                task_data = self.dependency_graph.nodes[task_id]
                if 'task' in task_data:
                    tasks.append(task_data['task'])
            
            return tasks
            
        except nx.NetworkXError as e:
            logger.error(f"Circular dependency detected: {e}")
            return []
    
    def can_execute(self, task_id: str) -> bool:
        """Sprawdź czy zadanie może być wykonane (dependencies completed)"""
        if not self.dependency_graph.has_node(task_id):
            return False
        
        # Sprawdź wszystkie dependencies
        for dep_id in self.dependency_graph.predecessors(task_id):
            dep_task = self.dependency_graph.nodes[dep_id].get('task')
            if dep_task and dep_task.status != TaskStatus.COMPLETED:
                return False
        
        return True
    
    def get_ready_tasks(self) -> List[AgentTask]:
        """Zwróć zadania gotowe do wykonania"""
        ready = []
        
        for node_id in self.dependency_graph.nodes():
            task = self.dependency_graph.nodes[node_id].get('task')
            if task and task.status == TaskStatus.PENDING:
                if self.can_execute(node_id):
                    ready.append(task)
        
        return ready
    
    def clear(self):
        """Wyczyść planner"""
        self.dependency_graph.clear()
