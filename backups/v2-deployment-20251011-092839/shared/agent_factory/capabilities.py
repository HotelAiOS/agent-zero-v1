"""
Capabilities Management System
System zarządzania możliwościami agentów
"""

from enum import Enum
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SkillLevel(Enum):
    """Poziomy umiejętności agenta"""
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


class TechStack(Enum):
    """Dostępne technologie"""
    # Backend Frameworks
    PYTHON = "python"
    FASTAPI = "fastapi"
    DJANGO = "django"
    FLASK = "flask"
    NODEJS = "nodejs"
    EXPRESS = "express"
    
    # Frontend
    REACT = "react"
    VUE = "vue"
    ANGULAR = "angular"
    TYPESCRIPT = "typescript"
    NEXTJS = "nextjs"
    TAILWINDCSS = "tailwindcss"
    CSS = "css"
    
    # State Management
    REDUX = "redux"
    ZUSTAND = "zustand"
    REACT_QUERY = "react_query"
    
    # API & Communication
    AXIOS = "axios"
    FETCH = "fetch"
    WEBSOCKETS = "websockets"
    
    # Database
    POSTGRESQL = "postgresql"
    MONGODB = "mongodb"
    NEO4J = "neo4j"
    REDIS = "redis"
    SQL = "sql"
    
    # Database Tools
    ALEMBIC = "alembic"
    FLYWAY = "flyway"
    LIQUIBASE = "liquibase"
    INDEXING = "indexing"
    EXPLAIN_ANALYZE = "explain_analyze"
    POSTGRESQL_REPLICATION = "postgresql_replication"
    MONGODB_REPLICA_SET = "mongodb_replica_set"
    
    # Authentication & Security
    JWT = "jwt"
    OAUTH2 = "oauth2"
    PASSLIB = "passlib"
    MFA = "mfa"
    
    # Message Queues & Async
    RABBITMQ = "rabbitmq"
    CELERY = "celery"
    ASYNCIO = "asyncio"
    
    # DevOps & Infrastructure
    DOCKER = "docker"
    DOCKER_COMPOSE = "docker_compose"
    KUBERNETES = "kubernetes"
    HELM = "helm"
    TERRAFORM = "terraform"
    ANSIBLE = "ansible"
    
    # CI/CD
    GITHUB_ACTIONS = "github_actions"
    GITLAB_CI = "gitlab_ci"
    JENKINS = "jenkins"
    
    # Monitoring & Observability
    PROMETHEUS = "prometheus"
    GRAFANA = "grafana"
    DATADOG = "datadog"
    
    # Testing
    PYTEST = "pytest"
    JEST = "jest"
    UNITTEST = "unittest"
    TESTCONTAINERS = "testcontainers"
    PLAYWRIGHT = "playwright"
    CYPRESS = "cypress"
    SELENIUM = "selenium"
    LOCUST = "locust"
    K6 = "k6"
    JMETER = "jmeter"
    
    # Security Tools
    SNYK = "snyk"
    BANDIT = "bandit"
    SAFETY = "safety"
    OWASP_TOP_10 = "owasp_top_10"
    SECURE_CODING = "secure_coding"
    SQLMAP = "sqlmap"
    ZAP = "zap"
    BURP_SUITE = "burp_suite"
    
    # Architecture & Design
    MICROSERVICES = "microservices"
    C4_MODEL = "c4_model"
    UML = "uml"
    MERMAID = "mermaid"
    DESIGN_PATTERNS = "design_patterns"
    CLEAN_ARCHITECTURE = "clean_architecture"
    HEXAGONAL_ARCHITECTURE = "hexagonal_architecture"
    
    # Performance
    PY_SPY = "py_spy"
    CPROFILE = "cprofile"
    MEMORY_PROFILER = "memory_profiler"
    MEMCACHED = "memcached"
    CDN = "cdn"
    NGINX = "nginx"
    HAPROXY = "haproxy"
    CONNECTION_POOLING = "connection_pooling"
    
    # UI/UX
    MATERIAL_UI = "material_ui"
    
    # AI/ML
    OLLAMA = "ollama"
    LANGCHAIN = "langchain"
    TENSORFLOW = "tensorflow"


@dataclass
class AgentCapability:
    """Reprezentacja możliwości agenta"""
    name: str
    category: str
    technologies: List[TechStack]
    skill_level: SkillLevel
    description: str
    requires: List[str] = field(default_factory=list)  # Wymagane inne capabilities
    
    def matches_requirement(self, required_tech: TechStack, min_level: SkillLevel) -> bool:
        """Sprawdź czy capability spełnia wymaganie"""
        return (
            required_tech in self.technologies and 
            self.skill_level.value >= min_level.value
        )


class CapabilityMatcher:
    """Dopasowywanie capabilities do wymagań projektu"""
    
    def __init__(self):
        self.capability_db: Dict[str, List[AgentCapability]] = {}
        logger.info("CapabilityMatcher zainicjalizowany")
    
    def register_agent_capabilities(
        self, 
        agent_id: str, 
        capabilities: List[AgentCapability]
    ):
        """Zarejestruj możliwości agenta"""
        self.capability_db[agent_id] = capabilities
        logger.info(
            f"Zarejestrowano {len(capabilities)} capabilities dla agenta {agent_id}"
        )
    
    def find_agents_with_capability(
        self,
        tech: TechStack,
        min_level: SkillLevel = SkillLevel.INTERMEDIATE,
        category: Optional[str] = None
    ) -> List[str]:
        """Znajdź agentów z określoną możliwością"""
        matching_agents = []
        
        for agent_id, capabilities in self.capability_db.items():
            for cap in capabilities:
                if cap.matches_requirement(tech, min_level):
                    if category is None or cap.category == category:
                        matching_agents.append(agent_id)
                        break
        
        logger.info(
            f"Znaleziono {len(matching_agents)} agentów z {tech.value} "
            f"(poziom >= {min_level.name})"
        )
        return matching_agents
    
    def find_best_agent_for_task(
        self,
        required_technologies: List[TechStack],
        task_category: str
    ) -> Optional[str]:
        """Znajdź najlepszego agenta do zadania"""
        agent_scores: Dict[str, int] = {}
        
        for agent_id, capabilities in self.capability_db.items():
            score = 0
            for tech in required_technologies:
                for cap in capabilities:
                    if cap.category == task_category and tech in cap.technologies:
                        score += cap.skill_level.value
            
            if score > 0:
                agent_scores[agent_id] = score
        
        if not agent_scores:
            logger.warning(
                f"Nie znaleziono agenta dla kategorii {task_category} "
                f"z technologiami {[t.value for t in required_technologies]}"
            )
            return None
        
        best_agent = max(agent_scores, key=agent_scores.get)
        logger.info(
            f"Najlepszy agent: {best_agent} (score: {agent_scores[best_agent]})"
        )
        return best_agent
    
    def get_agent_tech_stack(self, agent_id: str) -> Set[TechStack]:
        """Pobierz pełny stack technologiczny agenta"""
        if agent_id not in self.capability_db:
            return set()
        
        tech_stack = set()
        for cap in self.capability_db[agent_id]:
            tech_stack.update(cap.technologies)
        
        return tech_stack
    
    def check_coverage(
        self, 
        required_technologies: List[TechStack]
    ) -> Dict[str, bool]:
        """Sprawdź czy są agenci pokrywający wymagane technologie"""
        coverage = {}
        
        for tech in required_technologies:
            agents = self.find_agents_with_capability(tech)
            coverage[tech.value] = len(agents) > 0
        
        missing = [tech for tech, covered in coverage.items() if not covered]
        if missing:
            logger.warning(f"Brak pokrycia dla technologii: {missing}")
        
        return coverage


# Predefiniowane capability sets dla różnych typów agentów
ARCHITECT_CAPABILITIES = [
    AgentCapability(
        name="system_design",
        category="architecture",
        technologies=[TechStack.PYTHON, TechStack.NODEJS],
        skill_level=SkillLevel.EXPERT,
        description="Projektowanie architektury systemów",
        requires=[]
    ),
    AgentCapability(
        name="microservices",
        category="architecture",
        technologies=[TechStack.DOCKER, TechStack.KUBERNETES],
        skill_level=SkillLevel.ADVANCED,
        description="Architektury mikroserwisowe",
        requires=["system_design"]
    )
]

BACKEND_CAPABILITIES = [
    AgentCapability(
        name="api_development",
        category="backend",
        technologies=[TechStack.PYTHON, TechStack.FASTAPI],
        skill_level=SkillLevel.EXPERT,
        description="Tworzenie API RESTful",
        requires=[]
    ),
    AgentCapability(
        name="database_integration",
        category="backend",
        technologies=[TechStack.POSTGRESQL, TechStack.NEO4J],
        skill_level=SkillLevel.ADVANCED,
        description="Integracja z bazami danych",
        requires=["api_development"]
    )
]

FRONTEND_CAPABILITIES = [
    AgentCapability(
        name="ui_development",
        category="frontend",
        technologies=[TechStack.REACT, TechStack.TYPESCRIPT],
        skill_level=SkillLevel.EXPERT,
        description="Tworzenie interfejsów użytkownika",
        requires=[]
    ),
    AgentCapability(
        name="responsive_design",
        category="frontend",
        technologies=[TechStack.REACT, TechStack.VUE],
        skill_level=SkillLevel.ADVANCED,
        description="Responsywny design",
        requires=["ui_development"]
    )
]

DEVOPS_CAPABILITIES = [
    AgentCapability(
        name="containerization",
        category="devops",
        technologies=[TechStack.DOCKER],
        skill_level=SkillLevel.EXPERT,
        description="Konteneryzacja aplikacji",
        requires=[]
    ),
    AgentCapability(
        name="orchestration",
        category="devops",
        technologies=[TechStack.KUBERNETES],
        skill_level=SkillLevel.ADVANCED,
        description="Orkiestracja kontenerów",
        requires=["containerization"]
    ),
    AgentCapability(
        name="infrastructure_as_code",
        category="devops",
        technologies=[TechStack.TERRAFORM],
        skill_level=SkillLevel.ADVANCED,
        description="Infrastructure as Code",
        requires=[]
    )
]
