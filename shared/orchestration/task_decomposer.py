"""
Task Decomposer
Dekompozycja wymagań biznesowych na zadania techniczne
"""

from enum import Enum
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Typy zadań technicznych"""
    ARCHITECTURE = "architecture"
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    TESTING = "testing"
    DEVOPS = "devops"
    SECURITY = "security"
    PERFORMANCE = "performance"
    DOCUMENTATION = "documentation"
    INTEGRATION = "integration"


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
    REVIEW = "review"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Reprezentacja zadania technicznego"""
    task_id: str
    title: str
    description: str
    task_type: TaskType
    priority: TaskPriority
    estimated_hours: float
    required_agent_type: str
    status: TaskStatus = TaskStatus.PENDING
    
    # Dependencies
    depends_on: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    
    # Assignment
    assigned_agent: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Metadata
    tags: List[str] = field(default_factory=list)
    acceptance_criteria: List[str] = field(default_factory=list)
    deliverables: List[str] = field(default_factory=list)
    
    # Results
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.task_id:
            self.task_id = f"task_{uuid.uuid4().hex[:8]}"
    
    def is_ready(self, completed_tasks: List[str]) -> bool:
        """Sprawdź czy zadanie jest gotowe do wykonania"""
        if self.status != TaskStatus.PENDING:
            return False
        
        # Sprawdź czy wszystkie zależności są ukończone
        for dep in self.depends_on:
            if dep not in completed_tasks:
                return False
        
        return True
    
    def can_run_parallel_with(self, other: 'Task') -> bool:
        """Sprawdź czy zadanie może być wykonane równolegle z innym"""
        # Nie może jeśli jedno zależy od drugiego
        if self.task_id in other.depends_on or other.task_id in self.depends_on:
            return False
        
        # Nie może jeśli blokują się nawzajem
        if self.task_id in other.blocks or other.task_id in self.blocks:
            return False
        
        return True


class TaskDecomposer:
    """
    Dekompozycja wymagań biznesowych na zadania techniczne
    """
    
    def __init__(self):
        self.task_templates: Dict[str, List[Dict]] = self._load_task_templates()
        logger.info("TaskDecomposer zainicjalizowany")
    
    def _load_task_templates(self) -> Dict[str, List[Dict]]:
        """Załaduj szablony zadań dla różnych typów projektów"""
        return {
            'fullstack_web_app': self._get_fullstack_template(),
            'api_backend': self._get_api_backend_template(),
            'microservices': self._get_microservices_template(),
            'mobile_backend': self._get_mobile_backend_template()
        }
    
    def _get_fullstack_template(self) -> List[Dict]:
        """Szablon Full Stack Web App"""
        return [
            {
                'title': 'Projekt architektury systemu',
                'type': TaskType.ARCHITECTURE,
                'agent': 'architect',
                'priority': TaskPriority.CRITICAL,
                'hours': 8,
                'description': 'Zaprojektuj architekturę systemu, diagramy C4, ADR',
                'deliverables': ['Diagram C4', 'ADR documents', 'Tech stack decision'],
                'depends_on': []
            },
            {
                'title': 'Projekt schematu bazy danych',
                'type': TaskType.DATABASE,
                'agent': 'database',
                'priority': TaskPriority.CRITICAL,
                'hours': 6,
                'description': 'Zaprojektuj schemat bazy danych, ERD, migracje',
                'deliverables': ['ERD diagram', 'Initial migrations'],
                'depends_on': ['Projekt architektury systemu']
            },
            {
                'title': 'Setup projektu backend',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.HIGH,
                'hours': 4,
                'description': 'Inicjalizacja projektu FastAPI, struktura folderów',
                'deliverables': ['Project structure', 'requirements.txt'],
                'depends_on': ['Projekt architektury systemu']
            },
            {
                'title': 'Implementacja modeli danych',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.HIGH,
                'hours': 6,
                'description': 'SQLAlchemy models, Pydantic schemas',
                'deliverables': ['Models code', 'Schemas code'],
                'depends_on': ['Projekt schematu bazy danych', 'Setup projektu backend']
            },
            {
                'title': 'Implementacja API endpoints',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.HIGH,
                'hours': 12,
                'description': 'REST API endpoints, routing, business logic',
                'deliverables': ['API routes', 'OpenAPI spec'],
                'depends_on': ['Implementacja modeli danych']
            },
            {
                'title': 'Implementacja autoryzacji',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.CRITICAL,
                'hours': 8,
                'description': 'JWT authentication, OAuth2, permissions',
                'deliverables': ['Auth middleware', 'JWT implementation'],
                'depends_on': ['Implementacja API endpoints']
            },
            {
                'title': 'Setup projektu frontend',
                'type': TaskType.FRONTEND,
                'agent': 'frontend',
                'priority': TaskPriority.HIGH,
                'hours': 4,
                'description': 'React + TypeScript setup, routing, struktura',
                'deliverables': ['React project', 'package.json', 'Routing setup'],
                'depends_on': ['Projekt architektury systemu']
            },
            {
                'title': 'Implementacja UI komponentów',
                'type': TaskType.FRONTEND,
                'agent': 'frontend',
                'priority': TaskPriority.MEDIUM,
                'hours': 16,
                'description': 'React components, forms, layouts',
                'deliverables': ['React components', 'Storybook stories'],
                'depends_on': ['Setup projektu frontend']
            },
            {
                'title': 'Integracja frontend-backend',
                'type': TaskType.INTEGRATION,
                'agent': 'frontend',
                'priority': TaskPriority.HIGH,
                'hours': 8,
                'description': 'API calls, state management, error handling',
                'deliverables': ['API integration', 'State management'],
                'depends_on': ['Implementacja UI komponentów', 'Implementacja API endpoints']
            },
            {
                'title': 'Testy jednostkowe backend',
                'type': TaskType.TESTING,
                'agent': 'tester',
                'priority': TaskPriority.HIGH,
                'hours': 8,
                'description': 'Pytest tests dla API, models, business logic',
                'deliverables': ['Pytest tests', 'Coverage report'],
                'depends_on': ['Implementacja API endpoints']
            },
            {
                'title': 'Testy jednostkowe frontend',
                'type': TaskType.TESTING,
                'agent': 'tester',
                'priority': TaskPriority.MEDIUM,
                'hours': 6,
                'description': 'Jest/React Testing Library tests',
                'deliverables': ['Jest tests', 'Coverage report'],
                'depends_on': ['Implementacja UI komponentów']
            },
            {
                'title': 'Testy E2E',
                'type': TaskType.TESTING,
                'agent': 'tester',
                'priority': TaskPriority.MEDIUM,
                'hours': 8,
                'description': 'Playwright E2E tests dla critical paths',
                'deliverables': ['E2E tests', 'Test report'],
                'depends_on': ['Integracja frontend-backend']
            },
            {
                'title': 'Audyt bezpieczeństwa',
                'type': TaskType.SECURITY,
                'agent': 'security',
                'priority': TaskPriority.CRITICAL,
                'hours': 6,
                'description': 'Security audit, OWASP Top 10, vulnerability scanning',
                'deliverables': ['Security report', 'Remediation plan'],
                'depends_on': ['Implementacja autoryzacji']
            },
            {
                'title': 'Setup CI/CD',
                'type': TaskType.DEVOPS,
                'agent': 'devops',
                'priority': TaskPriority.HIGH,
                'hours': 6,
                'description': 'GitHub Actions, automated testing, deployment',
                'deliverables': ['CI/CD pipeline', 'Deployment scripts'],
                'depends_on': ['Testy jednostkowe backend', 'Testy jednostkowe frontend']
            },
            {
                'title': 'Dockerization',
                'type': TaskType.DEVOPS,
                'agent': 'devops',
                'priority': TaskPriority.HIGH,
                'hours': 4,
                'description': 'Dockerfile, docker-compose, multi-stage builds',
                'deliverables': ['Dockerfile', 'docker-compose.yml'],
                'depends_on': ['Setup projektu backend', 'Setup projektu frontend']
            },
            {
                'title': 'Dokumentacja użytkownika',
                'type': TaskType.DOCUMENTATION,
                'agent': 'backend',
                'priority': TaskPriority.LOW,
                'hours': 4,
                'description': 'README, installation guide, API docs',
                'deliverables': ['README.md', 'API documentation'],
                'depends_on': ['Integracja frontend-backend']
            }
        ]
    
    def _get_api_backend_template(self) -> List[Dict]:
        """Szablon API Backend Only"""
        return [
            {
                'title': 'Projekt architektury API',
                'type': TaskType.ARCHITECTURE,
                'agent': 'architect',
                'priority': TaskPriority.CRITICAL,
                'hours': 6,
                'description': 'API architecture, OpenAPI design, data flow',
                'deliverables': ['Architecture diagram', 'OpenAPI spec'],
                'depends_on': []
            },
            {
                'title': 'Projekt bazy danych',
                'type': TaskType.DATABASE,
                'agent': 'database',
                'priority': TaskPriority.CRITICAL,
                'hours': 6,
                'description': 'Database schema, indexes, migrations',
                'deliverables': ['ERD', 'Migrations'],
                'depends_on': ['Projekt architektury API']
            },
            {
                'title': 'FastAPI setup',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.HIGH,
                'hours': 3,
                'description': 'FastAPI project initialization',
                'deliverables': ['Project structure', 'Dependencies'],
                'depends_on': ['Projekt architektury API']
            },
            {
                'title': 'Models i schemas',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.HIGH,
                'hours': 6,
                'description': 'SQLAlchemy models, Pydantic schemas',
                'deliverables': ['Models', 'Schemas'],
                'depends_on': ['Projekt bazy danych', 'FastAPI setup']
            },
            {
                'title': 'CRUD operations',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.HIGH,
                'hours': 8,
                'description': 'Create, Read, Update, Delete endpoints',
                'deliverables': ['CRUD API', 'Tests'],
                'depends_on': ['Models i schemas']
            },
            {
                'title': 'Authentication',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.CRITICAL,
                'hours': 6,
                'description': 'JWT, OAuth2, permissions',
                'deliverables': ['Auth system', 'Middleware'],
                'depends_on': ['CRUD operations']
            },
            {
                'title': 'Rate limiting i throttling',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.MEDIUM,
                'hours': 3,
                'description': 'API rate limiting, request throttling',
                'deliverables': ['Rate limiter', 'Config'],
                'depends_on': ['Authentication']
            },
            {
                'title': 'API testing',
                'type': TaskType.TESTING,
                'agent': 'tester',
                'priority': TaskPriority.HIGH,
                'hours': 8,
                'description': 'Integration tests, API tests',
                'deliverables': ['Test suite', 'Coverage report'],
                'depends_on': ['Authentication']
            },
            {
                'title': 'Security audit',
                'type': TaskType.SECURITY,
                'agent': 'security',
                'priority': TaskPriority.CRITICAL,
                'hours': 4,
                'description': 'OWASP checks, vulnerability scan',
                'deliverables': ['Security report'],
                'depends_on': ['Authentication']
            },
            {
                'title': 'API documentation',
                'type': TaskType.DOCUMENTATION,
                'agent': 'backend',
                'priority': TaskPriority.MEDIUM,
                'hours': 3,
                'description': 'OpenAPI docs, examples, guides',
                'deliverables': ['API docs', 'README'],
                'depends_on': ['CRUD operations']
            },
            {
                'title': 'Docker setup',
                'type': TaskType.DEVOPS,
                'agent': 'devops',
                'priority': TaskPriority.HIGH,
                'hours': 3,
                'description': 'Dockerfile, docker-compose',
                'deliverables': ['Docker files'],
                'depends_on': ['FastAPI setup']
            },
            {
                'title': 'CI/CD pipeline',
                'type': TaskType.DEVOPS,
                'agent': 'devops',
                'priority': TaskPriority.HIGH,
                'hours': 4,
                'description': 'GitHub Actions, auto-deploy',
                'deliverables': ['CI/CD config'],
                'depends_on': ['API testing', 'Docker setup']
            }
        ]
    
    def _get_microservices_template(self) -> List[Dict]:
        """Szablon Microservices Architecture"""
        return [
            {
                'title': 'Microservices architecture design',
                'type': TaskType.ARCHITECTURE,
                'agent': 'architect',
                'priority': TaskPriority.CRITICAL,
                'hours': 12,
                'description': 'Service boundaries, communication patterns, data consistency',
                'deliverables': ['Architecture diagram', 'Service contracts', 'ADRs'],
                'depends_on': []
            },
            {
                'title': 'API Gateway setup',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.CRITICAL,
                'hours': 8,
                'description': 'Kong/NGINX gateway, routing, auth',
                'deliverables': ['Gateway config', 'Routing rules'],
                'depends_on': ['Microservices architecture design']
            },
            {
                'title': 'Service mesh setup',
                'type': TaskType.DEVOPS,
                'agent': 'devops',
                'priority': TaskPriority.HIGH,
                'hours': 8,
                'description': 'Istio/Linkerd service mesh',
                'deliverables': ['Service mesh config'],
                'depends_on': ['Microservices architecture design']
            },
            {
                'title': 'Message broker setup',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.CRITICAL,
                'hours': 6,
                'description': 'RabbitMQ/Kafka for async communication',
                'deliverables': ['Message broker setup', 'Topics/queues'],
                'depends_on': ['Microservices architecture design']
            },
            {
                'title': 'Service discovery',
                'type': TaskType.DEVOPS,
                'agent': 'devops',
                'priority': TaskPriority.HIGH,
                'hours': 4,
                'description': 'Consul/Eureka service registry',
                'deliverables': ['Service registry'],
                'depends_on': ['Service mesh setup']
            },
            {
                'title': 'Distributed tracing',
                'type': TaskType.DEVOPS,
                'agent': 'devops',
                'priority': TaskPriority.MEDIUM,
                'hours': 6,
                'description': 'Jaeger/Zipkin tracing',
                'deliverables': ['Tracing setup', 'Dashboards'],
                'depends_on': ['Service mesh setup']
            },
            {
                'title': 'Kubernetes deployment',
                'type': TaskType.DEVOPS,
                'agent': 'devops',
                'priority': TaskPriority.CRITICAL,
                'hours': 12,
                'description': 'K8s manifests, Helm charts',
                'deliverables': ['K8s configs', 'Helm charts'],
                'depends_on': ['Service discovery']
            }
        ]
    
    def _get_mobile_backend_template(self) -> List[Dict]:
        """Szablon Mobile Backend"""
        return [
            {
                'title': 'Mobile backend architecture',
                'type': TaskType.ARCHITECTURE,
                'agent': 'architect',
                'priority': TaskPriority.CRITICAL,
                'hours': 8,
                'description': 'REST/GraphQL API, push notifications, file storage',
                'deliverables': ['Architecture', 'API design'],
                'depends_on': []
            },
            {
                'title': 'API endpoints',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.HIGH,
                'hours': 12,
                'description': 'REST API for mobile clients',
                'deliverables': ['API code', 'OpenAPI spec'],
                'depends_on': ['Mobile backend architecture']
            },
            {
                'title': 'Push notifications',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.HIGH,
                'hours': 6,
                'description': 'FCM/APNS integration',
                'deliverables': ['Push notification service'],
                'depends_on': ['API endpoints']
            },
            {
                'title': 'File upload service',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.MEDIUM,
                'hours': 6,
                'description': 'S3/Cloud Storage for images/files',
                'deliverables': ['Upload service', 'CDN config'],
                'depends_on': ['API endpoints']
            },
            {
                'title': 'Real-time features',
                'type': TaskType.BACKEND,
                'agent': 'backend',
                'priority': TaskPriority.MEDIUM,
                'hours': 8,
                'description': 'WebSocket for real-time updates',
                'deliverables': ['WebSocket server'],
                'depends_on': ['API endpoints']
            }
        ]
    
    def decompose_project(
        self,
        project_type: str,
        business_requirements: List[str],
        additional_features: Optional[List[str]] = None
    ) -> List[Task]:
        """
        Dekompozycja projektu na zadania techniczne
        
        Args:
            project_type: Typ projektu ('fullstack_web_app', 'api_backend', etc.)
            business_requirements: Lista wymagań biznesowych
            additional_features: Dodatkowe feature do implementacji
        
        Returns:
            Lista zadań technicznych (Task objects)
        """
        if project_type not in self.task_templates:
            logger.error(f"Nieznany typ projektu: {project_type}")
            logger.info(f"Dostępne typy: {list(self.task_templates.keys())}")
            return []
        
        template = self.task_templates[project_type]
        tasks = []
        task_title_to_id = {}
        
        # Utwórz zadania z template
        for task_data in template:
            task = Task(
                task_id=f"task_{uuid.uuid4().hex[:8]}",
                title=task_data['title'],
                description=task_data['description'],
                task_type=task_data['type'],
                priority=task_data['priority'],
                estimated_hours=task_data['hours'],
                required_agent_type=task_data['agent'],
                deliverables=task_data['deliverables'],
                depends_on=[]  # Wypełnimy później
            )
            tasks.append(task)
            task_title_to_id[task_data['title']] = task.task_id
        
        # Utwórz dependencies (title → task_id)
        for i, task_data in enumerate(template):
            for dep_title in task_data.get('depends_on', []):
                if dep_title in task_title_to_id:
                    tasks[i].depends_on.append(task_title_to_id[dep_title])
        
        logger.info(
            f"Dekompozycja projektu {project_type}: "
            f"{len(tasks)} zadań, "
            f"~{sum(t.estimated_hours for t in tasks):.1f}h całkowity czas"
        )
        
        return tasks
    
    def get_parallel_tasks(self, tasks: List[Task]) -> List[List[Task]]:
        """
        Pogrupuj zadania które mogą być wykonywane równolegle
        
        Returns:
            Lista grup zadań do równoległego wykonania
        """
        completed = []
        parallel_groups = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            # Znajdź zadania które są ready (dependencies completed)
            ready_tasks = [
                t for t in remaining_tasks 
                if t.is_ready([task.task_id for task in completed])
            ]
            
            if not ready_tasks:
                # Dead lock - zadania czekają na siebie nawzajem
                logger.warning("Wykryto cykl zależności w zadaniach!")
                break
            
            parallel_groups.append(ready_tasks)
            
            # Symuluj completion tych zadań
            for task in ready_tasks:
                completed.append(task)
                remaining_tasks.remove(task)
        
        logger.info(
            f"Podzielono na {len(parallel_groups)} grup równoległych zadań"
        )
        return parallel_groups
    
    def estimate_project_duration(
        self,
        tasks: List[Task],
        available_agents: Dict[str, int]
    ) -> float:
        """
        Oszacuj czas trwania projektu
        
        Args:
            tasks: Lista zadań
            available_agents: Dict {agent_type: count}
        
        Returns:
            Szacowany czas w godzinach
        """
        parallel_groups = self.get_parallel_tasks(tasks)
        total_duration = 0.0
        
        for group in parallel_groups:
            # Pogrupuj zadania z grupy po typie agenta
            agent_tasks: Dict[str, List[Task]] = {}
            for task in group:
                agent_type = task.required_agent_type
                if agent_type not in agent_tasks:
                    agent_tasks[agent_type] = []
                agent_tasks[agent_type].append(task)
            
            # Dla każdego typu agenta oblicz czas
            group_duration = 0.0
            for agent_type, agent_task_list in agent_tasks.items():
                agent_count = available_agents.get(agent_type, 1)
                total_hours = sum(t.estimated_hours for t in agent_task_list)
                duration = total_hours / agent_count
                group_duration = max(group_duration, duration)
            
            total_duration += group_duration
        
        logger.info(
            f"Szacowany czas projektu: {total_duration:.1f}h "
            f"({total_duration/8:.1f} dni roboczych)"
        )
        return total_duration


# Szybka funkcja do tworzenia decomposera
def create_decomposer() -> TaskDecomposer:
    """Utwórz TaskDecomposer"""
    return TaskDecomposer()
