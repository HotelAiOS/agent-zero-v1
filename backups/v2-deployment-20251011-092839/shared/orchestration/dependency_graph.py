"""
Dependency Graph
Zarządzanie zależnościami między zadaniami
"""

from enum import Enum
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
import logging

from .task_decomposer import Task, TaskStatus

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DependencyType(Enum):
    """Typy zależności między zadaniami"""
    FINISH_TO_START = "finish_to_start"
    START_TO_START = "start_to_start"
    FINISH_TO_FINISH = "finish_to_finish"
    BLOCKS = "blocks"


@dataclass
class Dependency:
    """Reprezentacja zależności"""
    from_task: str
    to_task: str
    dependency_type: DependencyType
    reason: Optional[str] = None


class DependencyGraph:
    """Graf zależności między zadaniami"""
    
    def __init__(self):
        self.dependencies: List[Dependency] = []
        self.adjacency: Dict[str, List[str]] = {}
        logger.info("DependencyGraph zainicjalizowany")
    
    def add_dependency(self, from_task: str, to_task: str, 
                      dep_type: DependencyType = DependencyType.FINISH_TO_START,
                      reason: Optional[str] = None):
        """Dodaj zależność"""
        dep = Dependency(from_task, to_task, dep_type, reason)
        self.dependencies.append(dep)
        
        if from_task not in self.adjacency:
            self.adjacency[from_task] = []
        self.adjacency[from_task].append(to_task)
    
    def build_from_tasks(self, tasks: List[Task]):
        """Zbuduj graf z listy zadań"""
        self.dependencies.clear()
        self.adjacency.clear()
        
        for task in tasks:
            for dep_task_id in task.depends_on:
                self.add_dependency(dep_task_id, task.task_id, 
                                  DependencyType.FINISH_TO_START, "Task dependency")
        
        logger.info(f"Zbudowano graf z {len(self.dependencies)} zależności")
    
    def has_cycle(self) -> Tuple[bool, Optional[List[str]]]:
        """Wykryj cykl w grafie"""
        visited = set()
        rec_stack = set()
        
        def dfs(node: str, path: List[str]) -> Optional[List[str]]:
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in self.adjacency.get(node, []):
                if neighbor not in visited:
                    result = dfs(neighbor, path.copy())
                    if result:
                        return result
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    return path[cycle_start:] + [neighbor]
            
            rec_stack.remove(node)
            return None
        
        for node in self.adjacency.keys():
            if node not in visited:
                cycle = dfs(node, [])
                if cycle:
                    return True, cycle
        
        return False, None
    
    def topological_sort(self) -> Optional[List[str]]:
        """Sortowanie topologiczne"""
        has_cycle, _ = self.has_cycle()
        if has_cycle:
            return None
        
        in_degree = {task: 0 for task in self.adjacency.keys()}
        
        for deps in self.adjacency.values():
            for task in deps:
                if task not in in_degree:
                    in_degree[task] = 0
        
        for task, neighbors in self.adjacency.items():
            for neighbor in neighbors:
                in_degree[neighbor] += 1
        
        queue = [task for task, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            task = queue.pop(0)
            result.append(task)
            
            for neighbor in self.adjacency.get(task, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == len(in_degree) else None
    
    def get_critical_path(self, tasks: List[Task]) -> Tuple[List[str], float]:
        """Oblicz critical path"""
        task_map = {t.task_id: t for t in tasks}
        est = {}
        order = self.topological_sort()
        
        if not order:
            return [], 0.0
        
        for task_id in order:
            if task_id not in task_map:
                continue
            
            max_pred = 0.0
            for dep in self.dependencies:
                if dep.to_task == task_id and dep.from_task in est:
                    pred_task = task_map[dep.from_task]
                    max_pred = max(max_pred, est[dep.from_task] + pred_task.estimated_hours)
            
            est[task_id] = max_pred
        
        critical_tasks = []
        total_duration = max(est.values()) if est else 0.0
        
        return critical_tasks, total_duration
    
    def get_independent_tasks(self, completed: List[str]) -> List[str]:
        """Znajdź zadania gotowe do wykonania"""
        ready = []
        all_tasks = set(self.adjacency.keys())
        for deps in self.adjacency.values():
            all_tasks.update(deps)
        
        for task_id in all_tasks:
            if task_id in completed:
                continue
            
            dependencies = [d.from_task for d in self.dependencies if d.to_task == task_id]
            if all(dep in completed for dep in dependencies):
                ready.append(task_id)
        
        return ready
    
    def get_blocked_tasks(self, in_progress: List[str]) -> List[str]:
        """Znajdź zadania zablokowane"""
        return []
    
    def visualize(self) -> str:
        """Reprezentacja tekstowa grafu"""
        return "Graph visualization"
    
    def get_statistics(self) -> Dict[str, any]:
        """Statystyki grafu"""
        has_cycle, cycle = self.has_cycle()
        
        all_tasks = set(self.adjacency.keys())
        for deps in self.adjacency.values():
            all_tasks.update(deps)
        
        return {
            'total_dependencies': len(self.dependencies),
            'total_tasks': len(all_tasks),
            'has_cycle': has_cycle,
            'cycle_path': cycle if has_cycle else None,
            'max_depth': 0
        }


def create_dependency_graph() -> DependencyGraph:
    """Utwórz DependencyGraph"""
    return DependencyGraph()
