"""
Production Agent Factory - Agent Zero v1.0
Tworzy autonomicznych agentów AI zgodnie z architekturą z PDF
"""
import asyncio
import json
import uuid
import yaml
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from ollama_client import ollama

class AgentSpecialization(Enum):
    ARCHITECT = "architect"
    BACKEND_DEVELOPER = "backend_developer"
    FRONTEND_DEVELOPER = "frontend_developer"
    DEVOPS = "devops"
    TESTER = "tester"
    SECURITY = "security"
    DATABASE = "database"
    PROJECT_MANAGER = "project_manager"

class AgentStatus(Enum):
    IDLE = "idle"
    WORKING = "working"
    COLLABORATING = "collaborating"
    REVIEWING = "reviewing"
    LEARNING = "learning"
    ERROR = "error"

@dataclass
class AgentCapability:
    name: str
    description: str
    proficiency_level: float  # 0.0 - 1.0
    technologies: List[str]

@dataclass
class AgentTemplate:
    specialization: AgentSpecialization
    name: str
    description: str
    system_prompt: str
    capabilities: List[AgentCapability]
    preferred_model: str
    collaboration_style: str
    autonomy_level: float  # 0.0 - 1.0
    required_approvals: List[str]

@dataclass
class Agent:
    agent_id: str
    template: AgentTemplate
    status: AgentStatus
    current_task: Optional[str] = None
    team_id: Optional[str] = None
    knowledge_context: Dict[str, Any] = None
    performance_metrics: Dict[str, float] = None
    created_at: datetime = None
    last_activity: datetime = None

class ProjectTemplate:
    def __init__(self, name: str, description: str, required_agents: List[AgentSpecialization]):
        self.name = name
        self.description = description
        self.required_agents = required_agents
        self.phases = []
        self.quality_gates = []

class AgentFactory:
    """Production Agent Factory - tworzy i zarządza agentami"""
    
    def __init__(self, templates_dir: str = "./config/agent_templates"):
        self.templates_dir = Path(templates_dir)
        self.agent_templates: Dict[AgentSpecialization, AgentTemplate] = {}
        self.active_agents: Dict[str, Agent] = {}
        self.teams: Dict[str, List[str]] = {}
        self.project_templates: Dict[str, ProjectTemplate] = {}
        
        # Load predefined templates
        self._load_agent_templates()
        self._load_project_templates()
        
    def _load_agent_templates(self):
        """Ładuj szablony agentów z plików YAML"""
        
        # ARCHITECT AGENT
        architect_template = AgentTemplate(
            specialization=AgentSpecialization.ARCHITECT,
            name="System Architect",
            description="Expert w projektowaniu architektury systemów, wzorcach projektowych i decyzjach technologicznych",
            system_prompt="""Jesteś ekspertem architektem systemów IT. Twoja rola to:

1. PROJEKTOWANIE ARCHITEKTURY:
   - Analizuj wymagania biznesowe i techniczne
   - Projektuj skalowalne, maintainable architektury
   - Wybieraj odpowiednie wzorce projektowe
   - Twórz diagramy C4 i dokumentację techniczną

2. DECYZJE TECHNOLOGICZNE:
   - Rekomenduj stack technologiczny
   - Oceniaj trade-offs różnych rozwiązań
   - Planuj migracje i refactoring
   - Definiuj standardy kodowania

3. WSPÓŁPRACA:
   - Konsultuj decyzje z innymi agentami
   - Review kodu pod kątem architektury
   - Mentoruj junior developerów
   - Resolwuj konflikty techniczne

ZAWSZE:
- Myśl długoterminowo
- Priorytetyzuj maintainability
- Dokumentuj decyzje
- Uwzględniaj constraints biznesowe""",
            capabilities=[
                AgentCapability("Architecture Design", "Projektowanie architektury systemów", 0.95, 
                               ["Microservices", "Event-Driven", "CQRS", "DDD", "Clean Architecture"]),
                AgentCapability("Technology Selection", "Wybór technologii", 0.90,
                               ["Cloud Platforms", "Databases", "Frameworks", "Programming Languages"]),
                AgentCapability("Documentation", "Tworzenie dokumentacji", 0.85,
                               ["C4 Diagrams", "API Documentation", "Architecture Decision Records"])
            ],
            preferred_model="deepseek-coder:33b",
            collaboration_style="consultative",
            autonomy_level=0.8,
            required_approvals=["major_architecture_changes", "technology_stack_changes"]
        )
        
        # BACKEND DEVELOPER AGENT  
        backend_template = AgentTemplate(
            specialization=AgentSpecialization.BACKEND_DEVELOPER,
            name="Backend Developer",
            description="Expert w tworzeniu API, integracji z bazami danych i implementacji logiki biznesowej",
            system_prompt="""Jesteś ekspertem backend developerem. Twoja rola to:

1. ROZWÓJ API:
   - Implementuj RESTful i GraphQL API
   - Projektuj endpoints według best practices
   - Implementuj authentication i authorization
   - Twórz dokumentację API (OpenAPI/Swagger)

2. LOGIKA BIZNESOWA:
   - Implementuj business rules i workflows
   - Projektuj domain models
   - Zapewniaj data validation i error handling
   - Optymalizuj performance i scalability

3. INTEGRACJA:
   - Łącz z bazami danych (SQL/NoSQL)
   - Implementuj messaging (RabbitMQ, Kafka)
   - Integruj z zewnętrznymi API
   - Zarządzaj transakcjami

4. JAKOŚĆ:
   - Piszę czysty, testowalny kod
   - Implementuj unit i integration tests
   - Code review i refactoring
   - Monitoring i logging

ZAWSZE:
- Następuj SOLID principles
- Implementuj proper error handling
- Dokumentuj API changes
- Myśl o security implications""",
            capabilities=[
                AgentCapability("API Development", "Tworzenie API", 0.95,
                               ["FastAPI", "Django", "Express.js", "Spring Boot", "ASP.NET Core"]),
                AgentCapability("Database Integration", "Integracja z bazami danych", 0.90,
                               ["PostgreSQL", "MongoDB", "Redis", "SQLAlchemy", "Prisma"]),
                AgentCapability("Business Logic", "Implementacja logiki biznesowej", 0.88,
                               ["Domain Modeling", "Validation", "Workflows", "Event Handling"])
            ],
            preferred_model="deepseek-coder:33b",
            collaboration_style="cooperative",
            autonomy_level=0.85,
            required_approvals=["database_schema_changes", "breaking_api_changes"]
        )
        
        # DEVOPS AGENT
        devops_template = AgentTemplate(
            specialization=AgentSpecialization.DEVOPS,
            name="DevOps Engineer", 
            description="Expert w automatyzacji deployment, CI/CD, infrastrukturze i monitoring",
            system_prompt="""Jesteś ekspertem DevOps engineerem. Twoja rola to:

1. CI/CD PIPELINES:
   - Projektuj i implementuj automated pipelines
   - Konfiguruj GitHub Actions, GitLab CI, Jenkins
   - Automated testing, building, deployment
   - Rollback strategies i blue-green deployments

2. INFRASTRUCTURE AS CODE:
   - Terraform, Pulumi, CloudFormation
   - Kubernetes manifests i Helm charts
   - Docker containerization
   - Infrastructure monitoring

3. CLOUD PLATFORMS:
   - AWS, Azure, GCP deployment
   - Managed services configuration
   - Cost optimization
   - Security best practices

4. MONITORING & OBSERVABILITY:
   - Prometheus, Grafana setup
   - Logging aggregation (ELK, Fluentd)
   - APM tools (Datadog, New Relic)
   - Alerting i incident response

ZAWSZE:
- Automatyzuj wszystko co możliwe
- Priorytetyzuj security
- Optymalizuj koszty
- Dokumentuj infrastructure changes""",
            capabilities=[
                AgentCapability("CI/CD", "Continuous Integration/Deployment", 0.92,
                               ["GitHub Actions", "GitLab CI", "Jenkins", "ArgoCD"]),
                AgentCapability("Infrastructure", "Zarządzanie infrastrukturą", 0.90,
                               ["Kubernetes", "Docker", "Terraform", "Helm"]),
                AgentCapability("Cloud Platforms", "Platformy chmurowe", 0.85,
                               ["AWS", "Azure", "GCP", "DigitalOcean"])
            ],
            preferred_model="qwen2.5:14b",
            collaboration_style="supportive",
            autonomy_level=0.75,
            required_approvals=["production_deployments", "infrastructure_changes", "security_configs"]
        )
        
        # Store templates
        self.agent_templates[AgentSpecialization.ARCHITECT] = architect_template
        self.agent_templates[AgentSpecialization.BACKEND_DEVELOPER] = backend_template
        self.agent_templates[AgentSpecialization.DEVOPS] = devops_template
        
        print(f"✅ Loaded {len(self.agent_templates)} agent templates")
    
    def _load_project_templates(self):
        """Ładuj szablony projektów"""
        
        # SaaS Platform Template
        saas_template = ProjectTemplate(
            name="SaaS Platform",
            description="Kompletna platforma SaaS z authentication, subscription, API",
            required_agents=[
                AgentSpecialization.ARCHITECT,
                AgentSpecialization.BACKEND_DEVELOPER,
                AgentSpecialization.FRONTEND_DEVELOPER,
                AgentSpecialization.DEVOPS,
                AgentSpecialization.SECURITY
            ]
        )
        
        # E-commerce Template
        ecommerce_template = ProjectTemplate(
            name="E-commerce Platform",
            description="Platforma e-commerce z cart, payments, inventory",
            required_agents=[
                AgentSpecialization.ARCHITECT,
                AgentSpecialization.BACKEND_DEVELOPER,
                AgentSpecialization.FRONTEND_DEVELOPER,
                AgentSpecialization.DATABASE,
                AgentSpecialization.DEVOPS,
                AgentSpecialization.SECURITY
            ]
        )
        
        self.project_templates["saas"] = saas_template
        self.project_templates["ecommerce"] = ecommerce_template
        
        print(f"✅ Loaded {len(self.project_templates)} project templates")
    
    async def create_agent(self, specialization: AgentSpecialization, 
                          custom_config: Optional[Dict] = None) -> Agent:
        """Utwórz nowego agenta na podstawie template"""
        
        if specialization not in self.agent_templates:
            raise ValueError(f"Brak template dla specjalizacji: {specialization}")
        
        template = self.agent_templates[specialization]
        agent_id = f"{specialization.value}-{uuid.uuid4().hex[:8]}"
        
        # Apply custom configuration if provided
        if custom_config:
            # Deep copy template and apply customizations
            # TODO: Implement template customization
            pass
        
        agent = Agent(
            agent_id=agent_id,
            template=template,
            status=AgentStatus.IDLE,
            knowledge_context={},
            performance_metrics={
                "tasks_completed": 0,
                "success_rate": 0.0,
                "avg_task_time": 0.0,
                "collaboration_score": 0.0
            },
            created_at=datetime.now(),
            last_activity=datetime.now()
        )
        
        self.active_agents[agent_id] = agent
        
        print(f"✅ Created agent: {agent_id} ({specialization.value})")
        return agent
    
    async def form_team(self, project_type: str, custom_agents: Optional[List[AgentSpecialization]] = None) -> str:
        """Utwórz zespół agentów dla projektu"""
        
        team_id = f"team-{uuid.uuid4().hex[:8]}"
        
        if project_type in self.project_templates:
            # Use predefined project template
            template = self.project_templates[project_type]
            required_agents = template.required_agents
        elif custom_agents:
            # Use custom agent list
            required_agents = custom_agents
        else:
            raise ValueError("Musisz podać project_type lub custom_agents")
        
        team_members = []
        
        # Create required agents
        for specialization in required_agents:
            agent = await self.create_agent(specialization)
            agent.team_id = team_id
            team_members.append(agent.agent_id)
        
        self.teams[team_id] = team_members
        
        print(f"✅ Created team: {team_id} with {len(team_members)} agents")
        print(f"   Members: {[a.value for a in required_agents]}")
        
        return team_id
    
    async def assign_task(self, agent_id: str, task_description: str, 
                         context: Optional[Dict] = None) -> bool:
        """Przypisz zadanie agentowi"""
        
        if agent_id not in self.active_agents:
            print(f"❌ Agent {agent_id} nie istnieje")
            return False
        
        agent = self.active_agents[agent_id]
        
        if agent.status != AgentStatus.IDLE:
            print(f"❌ Agent {agent_id} jest zajęty ({agent.status.value})")
            return False
        
        agent.current_task = task_description
        agent.status = AgentStatus.WORKING
        agent.last_activity = datetime.now()
        
        if context:
            agent.knowledge_context.update(context)
        
        print(f"✅ Assigned task to {agent_id}: {task_description[:50]}...")
        
        # Start task execution in background
        asyncio.create_task(self._execute_agent_task(agent_id, task_description, context))
        
        return True
    
    async def _execute_agent_task(self, agent_id: str, task: str, context: Optional[Dict] = None):
        """Wykonaj zadanie agenta (symulacja + prawdziwa AI)"""
        
        agent = self.active_agents[agent_id]
        template = agent.template
        
        try:
            print(f"🚀 Agent {agent_id} rozpoczyna zadanie")
            
            # Construct prompt
            full_prompt = f"""
{template.system_prompt}

ZADANIE: {task}

KONTEKST: {json.dumps(context, indent=2) if context else "Brak dodatkowego kontekstu"}

INSTRUKCJE:
- Wykonaj zadanie zgodnie z twoją specjalizacją
- Jeśli potrzebujesz współpracy, zaznacz to w odpowiedzi
- Podaj konkretne deliverables
- Zasugeruj następne kroki

ODPOWIEDŹ:
"""
            
            # Execute with Ollama
            print(f"🧠 Using model: {template.preferred_model}")
            
            response = ollama.generate(
                model=template.preferred_model,
                prompt=full_prompt,
                stream=False
            )
            
            result = response['response'] if 'response' in response else "Brak odpowiedzi"
            
            # Update agent state
            agent.status = AgentStatus.IDLE
            agent.current_task = None
            agent.performance_metrics["tasks_completed"] += 1
            agent.last_activity = datetime.now()
            
            print(f"✅ Agent {agent_id} ukończył zadanie")
            print(f"📄 Wynik: {result[:200]}...")
            
            return result
            
        except Exception as e:
            agent.status = AgentStatus.ERROR
            print(f"❌ Agent {agent_id} error: {e}")
            return None
    
    def get_team_status(self, team_id: str) -> Dict:
        """Pobierz status zespołu"""
        
        if team_id not in self.teams:
            return {"error": "Team not found"}
        
        team_members = self.teams[team_id]
        status = {
            "team_id": team_id,
            "member_count": len(team_members),
            "members": []
        }
        
        for agent_id in team_members:
            if agent_id in self.active_agents:
                agent = self.active_agents[agent_id]
                status["members"].append({
                    "agent_id": agent_id,
                    "specialization": agent.template.specialization.value,
                    "status": agent.status.value,
                    "current_task": agent.current_task,
                    "tasks_completed": agent.performance_metrics.get("tasks_completed", 0)
                })
        
        return status
    
    def list_available_specializations(self) -> List[str]:
        """Lista dostępnych specjalizacji"""
        return [spec.value for spec in self.agent_templates.keys()]
    
    def list_project_templates(self) -> List[str]:
        """Lista dostępnych szablonów projektów"""
        return list(self.project_templates.keys())

# Production CLI Interface
class AgentZeroCLI:
    """Production CLI dla Agent Zero"""
    
    def __init__(self):
        self.factory = AgentFactory()
        
    async def run_interactive(self):
        """Interactive CLI session"""
        
        print("🚀 AGENT ZERO - PRODUCTION SYSTEM")
        print("=" * 50)
        print("✅ Agent Factory initialized")
        print(f"🤖 Available specializations: {', '.join(self.factory.list_available_specializations())}")
        print(f"📋 Project templates: {', '.join(self.factory.list_project_templates())}")
        print("🛑 Type 'exit' to quit")
        print()
        
        while True:
            try:
                print("\n🎮 COMMANDS:")
                print("1️⃣  create-agent <specialization>")
                print("2️⃣  form-team <project_type>")
                print("3️⃣  assign-task <agent_id> <task>")
                print("4️⃣  team-status <team_id>")
                print("5️⃣  list-agents")
                print("6️⃣  list-teams")
                print("0️⃣  exit")
                
                command = input("\n👉 Command: ").strip().lower()
                
                if command == "exit" or command == "0":
                    break
                elif command == "1" or command.startswith("create-agent"):
                    await self._create_agent_interactive()
                elif command == "2" or command.startswith("form-team"):
                    await self._form_team_interactive()
                elif command == "3" or command.startswith("assign-task"):
                    await self._assign_task_interactive()
                elif command == "4" or command.startswith("team-status"):
                    self._team_status_interactive()
                elif command == "5" or command.startswith("list-agents"):
                    self._list_agents()
                elif command == "6" or command.startswith("list-teams"):
                    self._list_teams()
                else:
                    print("❌ Unknown command")
                    
            except KeyboardInterrupt:
                print("\n🛑 Exiting...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("👋 Agent Zero shutdown complete")
    
    async def _create_agent_interactive(self):
        """Interactive agent creation"""
        print("\n🤖 CREATE AGENT")
        print("Available specializations:")
        specs = self.factory.list_available_specializations()
        for i, spec in enumerate(specs, 1):
            print(f"{i}. {spec}")
        
        try:
            choice = int(input("Choose specialization (number): ")) - 1
            if 0 <= choice < len(specs):
                specialization = AgentSpecialization(specs[choice])
                agent = await self.factory.create_agent(specialization)
                print(f"✅ Created agent: {agent.agent_id}")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Enter a number")
    
    async def _form_team_interactive(self):
        """Interactive team formation"""
        print("\n👥 FORM TEAM")
        print("Available project templates:")
        templates = self.factory.list_project_templates()
        for i, template in enumerate(templates, 1):
            print(f"{i}. {template}")
        
        try:
            choice = int(input("Choose project template (number): ")) - 1
            if 0 <= choice < len(templates):
                project_type = templates[choice]
                team_id = await self.factory.form_team(project_type)
                print(f"✅ Created team: {team_id}")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Enter a number")
    
    async def _assign_task_interactive(self):
        """Interactive task assignment"""
        print("\n📋 ASSIGN TASK")
        
        if not self.factory.active_agents:
            print("❌ No agents available")
            return
        
        print("Available agents:")
        agents = list(self.factory.active_agents.items())
        for i, (agent_id, agent) in enumerate(agents, 1):
            status_icon = "💤" if agent.status == AgentStatus.IDLE else "🔥"
            print(f"{i}. {status_icon} {agent_id} ({agent.template.specialization.value})")
        
        try:
            choice = int(input("Choose agent (number): ")) - 1
            if 0 <= choice < len(agents):
                agent_id = agents[choice][0]
                task = input("Enter task description: ").strip()
                
                if task:
                    success = await self.factory.assign_task(agent_id, task)
                    if success:
                        print(f"✅ Task assigned to {agent_id}")
                    else:
                        print("❌ Failed to assign task")
                else:
                    print("❌ Task description cannot be empty")
            else:
                print("❌ Invalid choice")
        except ValueError:
            print("❌ Enter a number")
    
    def _team_status_interactive(self):
        """Show team status"""
        print("\n👥 TEAM STATUS")
        
        if not self.factory.teams:
            print("❌ No teams found")
            return
        
        for team_id in self.factory.teams:
            status = self.factory.get_team_status(team_id)
            print(f"\n🏢 Team: {team_id}")
            print(f"   Members: {status['member_count']}")
            
            for member in status['members']:
                status_icon = {"idle": "💤", "working": "🔥", "error": "❌"}.get(member['status'], "❓")
                print(f"   {status_icon} {member['agent_id']} ({member['specialization']})")
                if member['current_task']:
                    print(f"      Task: {member['current_task'][:50]}...")
                print(f"      Completed: {member['tasks_completed']} tasks")
    
    def _list_agents(self):
        """List all agents"""
        print("\n🤖 ALL AGENTS")
        
        if not self.factory.active_agents:
            print("❌ No agents found")
            return
        
        for agent_id, agent in self.factory.active_agents.items():
            status_icon = {"idle": "💤", "working": "🔥", "error": "❌"}.get(agent.status.value, "❓")
            print(f"{status_icon} {agent_id}")
            print(f"   Specialization: {agent.template.specialization.value}")
            print(f"   Status: {agent.status.value}")
            print(f"   Model: {agent.template.preferred_model}")
            print(f"   Tasks completed: {agent.performance_metrics.get('tasks_completed', 0)}")
            if agent.current_task:
                print(f"   Current task: {agent.current_task[:50]}...")
    
    def _list_teams(self):
        """List all teams"""
        print("\n👥 ALL TEAMS")
        
        if not self.factory.teams:
            print("❌ No teams found")
            return
        
        for team_id, members in self.factory.teams.items():
            print(f"🏢 {team_id} ({len(members)} members)")
            for member_id in members:
                if member_id in self.factory.active_agents:
                    agent = self.factory.active_agents[member_id]
                    print(f"   └─ {member_id} ({agent.template.specialization.value})")

# Main execution
async def main():
    """Main entry point"""
    cli = AgentZeroCLI()
    await cli.run_interactive()

if __name__ == "__main__":
    asyncio.run(main())

        
# V2.0 Production Intelligence Integration
try:
    from shared.kaizen import IntelligentModelSelector, TaskContext
    
    # Initialize production intelligence
    model_selector = IntelligentModelSelector()
    
    def select_optimal_ai_model(task_description: str, agent_specialization: str, 
                              priority: str = "balanced") -> str:
        """Select optimal AI model using V2.0 Intelligence Layer"""
        
        context = TaskContext(
            task_type=f"{agent_specialization.value} {task_description}",
            priority=priority,
            project_context=task_description
        )
        
        recommendation = model_selector.select_optimal_model(context)
        return recommendation['recommended_model']
    
    # Enhance agent task execution with intelligent model selection
    async def _execute_agent_task_v2(self, agent_id: str, task: str, context: Optional[Dict] = None):
        """Enhanced task execution with V2.0 Intelligence"""
        
        agent = self.active_agents[agent_id]
        template = agent.template
        
        try:
            print(f"🧠 Agent {agent_id} using V2.0 Intelligence Layer")
            
            # Use intelligent model selection
            optimal_model = select_optimal_ai_model(task, template.specialization, "quality")
            print(f"🎯 Selected optimal model: {optimal_model}")
            
            # Execute with selected model
            full_prompt = f"{template.system_prompt}\n\nTASK: {task}\n\nCONTEXT: {json.dumps(context, indent=2) if context else 'None'}"
            
            response = ollama.generate(
                model=optimal_model,
                prompt=full_prompt,
                stream=False
            )
            
            result = response['response'] if 'response' in response else "No response"
            
            # Update agent state
            agent.status = AgentStatus.IDLE
            agent.current_task = None
            agent.performance_metrics["tasks_completed"] += 1
            agent.last_activity = datetime.now()
            
            print(f"✅ Agent {agent_id} completed task with V2.0 Intelligence")
            return result
            
        except Exception as e:
            agent.status = AgentStatus.ERROR
            print(f"❌ Agent {agent_id} V2.0 execution error: {e}")
            # Fallback to original method
            return await self._execute_agent_task_original(agent_id, task, context)
    
    print("✅ Agent Zero V2.0 Intelligence Layer integrated with Agent Factory")
    V2_INTELLIGENCE_AVAILABLE = True
    
except ImportError as e:
    print(f"⚠️ V2.0 Intelligence Layer not available: {e}")
    V2_INTELLIGENCE_AVAILABLE = False
        