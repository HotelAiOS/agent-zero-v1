# Natural Language Understanding Task Decomposer
# Rozbudowa dla Agent Zero V1 - Week 43 Implementation
# Integracja z istniejącymi komponentami: shared/orchestration/task_decomposer.py

import json
import re
import spacy
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

# Import existing components
from shared.orchestration.task_decomposer import TaskDecomposer, Task, TaskPriority, TaskStatus, TaskType, TaskDependency
from shared.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskIntent:
    """Enhanced task intent classification"""
    primary_intent: str
    confidence: float
    secondary_intents: List[str] = field(default_factory=list)
    domain: str = "general"
    complexity_score: float = 0.5
    
@dataclass
class TaskBreakdown:
    """Enhanced task breakdown structure"""
    main_intent: TaskIntent
    subtasks: List[Task]
    estimated_complexity: float
    confidence_score: float
    risk_factors: List[str] = field(default_factory=list)
    domain_knowledge: Dict[str, Any] = field(default_factory=dict)
    dependencies_graph: Dict[int, List[int]] = field(default_factory=dict)

@dataclass
class DomainContext:
    """Domain-specific context for task understanding"""
    tech_stack: List[str] = field(default_factory=list)
    project_type: str = "general"
    current_phase: str = "development"
    team_skills: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

class TechStackImplications:
    """Knowledge base for technology implications"""
    
    TECH_IMPLICATIONS = {
        "FastAPI": {
            "implies": ["Python", "async/await", "Pydantic", "OpenAPI", "REST API"],
            "common_patterns": ["CRUD operations", "dependency injection", "middleware", "routing"],
            "typical_tasks": ["endpoint creation", "schema definition", "testing setup", "authentication"],
            "complexity_multiplier": 1.2
        },
        "React": {
            "implies": ["JavaScript", "TypeScript", "JSX", "components", "hooks"],
            "common_patterns": ["component hierarchy", "state management", "props", "context API"],
            "typical_tasks": ["component creation", "state logic", "API integration", "routing"],
            "complexity_multiplier": 1.3
        },
        "Neo4j": {
            "implies": ["Cypher queries", "graph modeling", "relationships", "nodes"],
            "common_patterns": ["node creation", "relationship management", "graph traversal"],
            "typical_tasks": ["schema design", "query optimization", "data migration", "indexing"],
            "complexity_multiplier": 1.5
        },
        "Docker": {
            "implies": ["containers", "images", "compose", "networking"],
            "common_patterns": ["containerization", "multi-service", "volume mounting"],
            "typical_tasks": ["Dockerfile creation", "compose setup", "networking", "deployment"],
            "complexity_multiplier": 1.1
        },
        "PostgreSQL": {
            "implies": ["SQL", "ACID", "indexes", "migrations"],
            "common_patterns": ["schema design", "query optimization", "data modeling"],
            "typical_tasks": ["table creation", "migration scripts", "query optimization", "backup"],
            "complexity_multiplier": 1.2
        }
    }
    
    PROJECT_PATTERNS = {
        "fullstack_web_app": {
            "standard_tasks": [
                "architecture_design", "database_design", "backend_api", 
                "frontend_ui", "authentication", "testing", "deployment"
            ],
            "task_dependencies": {
                "database_design": ["architecture_design"],
                "backend_api": ["database_design"],
                "frontend_ui": ["backend_api"],
                "authentication": ["backend_api"],
                "testing": ["frontend_ui", "authentication"],
                "deployment": ["testing"]
            }
        },
        "api_service": {
            "standard_tasks": [
                "api_design", "data_models", "business_logic", 
                "authentication", "documentation", "testing"
            ],
            "task_dependencies": {
                "data_models": ["api_design"],
                "business_logic": ["data_models"],
                "authentication": ["business_logic"],
                "documentation": ["authentication"],
                "testing": ["documentation"]
            }
        }
    }

class NLUTaskDecomposer(TaskDecomposer):
    """Enhanced Task Decomposer with Natural Language Understanding"""
    
    def __init__(self, llm_client=None):
        super().__init__(llm_client)
        self.ollama_client = llm_client or OllamaClient()
        self.spacy_model = None
        self.tech_kb = TechStackImplications()
        self.logger = logging.getLogger("NLUTaskDecomposer")
        self._load_nlp_model()
        
    def _load_nlp_model(self):
        """Load spaCy model for NLP processing"""
        try:
            self.spacy_model = spacy.load("en_core_web_sm")
            self.logger.info("✅ spaCy model loaded successfully")
        except OSError:
            self.logger.warning("⚠️ spaCy model not found - using fallback parsing")
            self.spacy_model = None
    
    async def enhanced_decompose(self, description: str, context: DomainContext) -> TaskBreakdown:
        """
        Main enhanced decomposition method with NLU
        """
        self.logger.info(f"🧠 Enhanced decomposition: {description[:50]}...")
        
        # Step 1: Extract technical entities and concepts
        entities = self._extract_technical_entities(description)
        self.logger.info(f"📊 Extracted entities: {entities}")
        
        # Step 2: Classify intent with LLM
        intent_analysis = await self._classify_intent_with_llm(description, entities, context)
        self.logger.info(f"🎯 Intent: {intent_analysis.primary_intent} ({intent_analysis.confidence:.2f})")
        
        # Step 3: Generate context-aware subtasks
        subtasks = await self._generate_intelligent_subtasks(
            description, intent_analysis, context, entities
        )
        self.logger.info(f"📋 Generated {len(subtasks)} subtasks")
        
        # Step 4: Analyze dependencies
        dependencies = self._analyze_task_dependencies(subtasks, context)
        
        # Step 5: Calculate complexity and risks
        complexity = self._calculate_task_complexity(subtasks, context)
        risks = self._identify_risk_factors(entities, subtasks, context)
        
        return TaskBreakdown(
            main_intent=intent_analysis,
            subtasks=subtasks,
            estimated_complexity=complexity,
            confidence_score=intent_analysis.confidence,
            risk_factors=risks,
            domain_knowledge=self._build_domain_knowledge(entities, context),
            dependencies_graph=dependencies
        )
    
    def _extract_technical_entities(self, description: str) -> Dict[str, List[str]]:
        """Extract technical entities using NLP and keyword matching"""
        entities = {
            "technologies": [],
            "actions": [],
            "components": [],
            "patterns": [],
            "domains": []
        }
        
        # Keyword-based tech detection
        description_lower = description.lower()
        
        for tech, details in self.tech_kb.TECH_IMPLICATIONS.items():
            if tech.lower() in description_lower:
                entities["technologies"].append(tech)
                entities["components"].extend(details["implies"][:3])  # Top 3 implications
        
        # Common action patterns
        action_patterns = [
            r'\b(create|build|develop|implement|design|setup|configure)\b',
            r'\b(test|validate|verify|check)\b',
            r'\b(deploy|install|configure|setup)\b',
            r'\b(integrate|connect|link|combine)\b',
            r'\b(optimize|improve|enhance|refactor)\b'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, description_lower)
            entities["actions"].extend(matches)
        
        # Use spaCy if available
        if self.spacy_model:
            doc = self.spacy_model(description)
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "TECH"]:
                    entities["domains"].append(ent.text)
        
        return entities
    
    async def _classify_intent_with_llm(self, description: str, entities: Dict, context: DomainContext) -> TaskIntent:
        """Use LLM to classify task intent with context"""
        
        prompt = f"""Analyze this task description and classify the primary intent:

TASK: {description}

CONTEXT:
- Tech Stack: {', '.join(context.tech_stack)}
- Project Type: {context.project_type}
- Project Phase: {context.current_phase}
- Detected Entities: {entities}

Classify the primary intent from these categories:
- DEVELOPMENT: Building new features/components
- INTEGRATION: Connecting existing systems  
- ARCHITECTURE: System design and planning
- TESTING: Validation and quality assurance
- DEPLOYMENT: Production setup and configuration
- MAINTENANCE: Updates and fixes
- ANALYSIS: Research and investigation

Respond with JSON:
{{
  "primary_intent": "DEVELOPMENT",
  "confidence": 0.85,
  "secondary_intents": ["TESTING"],
  "domain": "web_development",
  "complexity_score": 0.7,
  "reasoning": "Brief explanation"
}}"""

        try:
            response = await self.ollama_client.generate(prompt, model="llama3.2:3b")
            result = json.loads(response.strip())
            
            return TaskIntent(
                primary_intent=result.get("primary_intent", "DEVELOPMENT"),
                confidence=result.get("confidence", 0.5),
                secondary_intents=result.get("secondary_intents", []),
                domain=result.get("domain", "general"),
                complexity_score=result.get("complexity_score", 0.5)
            )
        except Exception as e:
            self.logger.warning(f"LLM intent classification failed: {e}")
            return TaskIntent(
                primary_intent="DEVELOPMENT",
                confidence=0.3,
                domain="general",
                complexity_score=0.5
            )
    
    async def _generate_intelligent_subtasks(self, description: str, intent: TaskIntent, 
                                          context: DomainContext, entities: Dict) -> List[Task]:
        """Generate context-aware subtasks using AI"""
        
        prompt = f"""Break down this task into specific, actionable subtasks:

MAIN TASK: {description}
INTENT: {intent.primary_intent} (confidence: {intent.confidence:.2f})
DOMAIN: {intent.domain}
TECH STACK: {', '.join(context.tech_stack)}
DETECTED ENTITIES: {entities}

Generate 3-7 specific subtasks. Each should be:
- Concrete and actionable
- Properly sequenced
- Appropriate for the tech stack
- Realistic in scope

Respond with JSON array:
[
  {{
    "title": "Design API endpoints schema",
    "description": "Create OpenAPI specification for user authentication endpoints",
    "task_type": "ARCHITECTURE",
    "priority": "HIGH",
    "estimated_hours": 8,
    "required_skills": ["API design", "OpenAPI"],
    "depends_on": []
  }}
]"""

        try:
            response = await self.ollama_client.generate(prompt, model="llama3.2:3b")
            subtasks_data = json.loads(response.strip())
            
            subtasks = []
            for i, data in enumerate(subtasks_data):
                # Map string enums to actual enums
                task_type = TaskType.BACKEND
                if data.get("task_type") in [t.value.upper() for t in TaskType]:
                    task_type = TaskType(data["task_type"].lower())
                
                priority = TaskPriority.MEDIUM
                if data.get("priority") in [p.value.upper() for p in TaskPriority]:
                    priority = TaskPriority(data["priority"].lower())
                
                subtasks.append(Task(
                    id=i + 1,
                    title=data.get("title", f"Task {i+1}"),
                    description=data.get("description", ""),
                    task_type=task_type,
                    priority=priority,
                    estimated_hours=data.get("estimated_hours", 8),
                    required_agent_type=data.get("required_skills", ["backend"])[0] if data.get("required_skills") else "backend"
                ))
            
            return subtasks
            
        except Exception as e:
            self.logger.warning(f"AI subtask generation failed: {e}, using fallback")
            return self._fallback_subtask_generation(description, intent, context)
    
    def _fallback_subtask_generation(self, description: str, intent: TaskIntent, context: DomainContext) -> List[Task]:
        """Fallback subtask generation without AI"""
        
        if context.project_type in self.tech_kb.PROJECT_PATTERNS:
            pattern = self.tech_kb.PROJECT_PATTERNS[context.project_type]
            subtasks = []
            
            for i, task_name in enumerate(pattern["standard_tasks"]):
                subtasks.append(Task(
                    id=i + 1,
                    title=task_name.replace("_", " ").title(),
                    description=f"{task_name.replace('_', ' ').title()} for {description[:30]}...",
                    task_type=self._map_task_to_type(task_name),
                    priority=TaskPriority.MEDIUM,
                    estimated_hours=8.0
                ))
            
            return subtasks
        
        # Basic fallback
        return [Task(
            id=1,
            title="Analyze Requirements",
            description=f"Detailed analysis of: {description}",
            priority=TaskPriority.HIGH,
            estimated_hours=4.0
        )]
    
    def _analyze_task_dependencies(self, subtasks: List[Task], context: DomainContext) -> Dict[int, List[int]]:
        """Analyze task dependencies"""
        dependencies = {}
        
        # Simple rule-based dependency detection
        task_keywords = {
            "design": ["architecture", "schema", "plan"],
            "implement": ["develop", "build", "create", "code"],
            "test": ["validate", "verify", "check"],
            "deploy": ["production", "server", "hosting"]
        }
        
        design_tasks = [t for t in subtasks if any(kw in t.title.lower() for kw in task_keywords["design"])]
        impl_tasks = [t for t in subtasks if any(kw in t.title.lower() for kw in task_keywords["implement"])]
        test_tasks = [t for t in subtasks if any(kw in t.title.lower() for kw in task_keywords["test"])]
        
        # Implementation depends on design
        for impl_task in impl_tasks:
            deps = [d.id for d in design_tasks if d.id != impl_task.id]
            if deps:
                dependencies[impl_task.id] = deps
        
        # Testing depends on implementation
        for test_task in test_tasks:
            deps = [i.id for i in impl_tasks if i.id != test_task.id]
            if deps:
                dependencies[test_task.id] = deps
        
        return dependencies
    
    def _calculate_task_complexity(self, subtasks: List[Task], context: DomainContext) -> float:
        """Calculate overall task complexity"""
        base_complexity = len(subtasks) * 0.1
        
        # Tech stack complexity multiplier
        tech_multiplier = 1.0
        for tech in context.tech_stack:
            if tech in self.tech_kb.TECH_IMPLICATIONS:
                tech_multiplier *= self.tech_kb.TECH_IMPLICATIONS[tech]["complexity_multiplier"]
        
        # Hours-based complexity
        total_hours = sum(task.estimated_hours for task in subtasks)
        hours_complexity = min(total_hours / 40.0, 2.0)  # Cap at 2.0
        
        return min(base_complexity + hours_complexity * tech_multiplier, 3.0)
    
    def _identify_risk_factors(self, entities: Dict, subtasks: List[Task], context: DomainContext) -> List[str]:
        """Identify potential risk factors"""
        risks = []
        
        # Technology complexity risks
        high_complexity_tech = ["Neo4j", "Kubernetes", "Machine Learning"]
        for tech in entities.get("technologies", []):
            if tech in high_complexity_tech:
                risks.append(f"High complexity technology: {tech}")
        
        # Task volume risk
        if len(subtasks) > 8:
            risks.append("High number of subtasks may indicate scope creep")
        
        # Dependency risks
        high_priority_tasks = [t for t in subtasks if t.priority == TaskPriority.HIGH]
        if len(high_priority_tasks) > len(subtasks) * 0.6:
            risks.append("Too many high-priority tasks - prioritization needed")
        
        # Integration complexity
        if "integration" in entities.get("actions", []):
            risks.append("Integration tasks carry higher failure risk")
        
        return risks
    
    def _build_domain_knowledge(self, entities: Dict, context: DomainContext) -> Dict[str, Any]:
        """Build domain-specific knowledge context"""
        return {
            "detected_technologies": entities.get("technologies", []),
            "recommended_patterns": self._get_recommended_patterns(entities, context),
            "skill_requirements": self._extract_skill_requirements(entities, context),
            "estimated_timeline": self._estimate_timeline(context)
        }
    
    def _get_recommended_patterns(self, entities: Dict, context: DomainContext) -> List[str]:
        """Get recommended patterns based on tech stack"""
        patterns = []
        for tech in entities.get("technologies", []):
            if tech in self.tech_kb.TECH_IMPLICATIONS:
                patterns.extend(self.tech_kb.TECH_IMPLICATIONS[tech]["common_patterns"][:2])
        return list(set(patterns))
    
    def _extract_skill_requirements(self, entities: Dict, context: DomainContext) -> List[str]:
        """Extract required skills"""
        skills = set()
        for tech in entities.get("technologies", []):
            if tech in self.tech_kb.TECH_IMPLICATIONS:
                skills.update(self.tech_kb.TECH_IMPLICATIONS[tech]["implies"][:3])
        return list(skills)
    
    def _estimate_timeline(self, context: DomainContext) -> str:
        """Estimate project timeline"""
        if context.project_type == "fullstack_web_app":
            return "2-4 weeks"
        elif context.project_type == "api_service":
            return "1-2 weeks"
        else:
            return "1-3 weeks"
    
    def _map_task_to_type(self, task_name: str) -> TaskType:
        """Map task name to TaskType enum"""
        mapping = {
            "frontend": TaskType.FRONTEND,
            "backend": TaskType.BACKEND,
            "database": TaskType.DATABASE,
            "deploy": TaskType.DEVOPS,
            "test": TaskType.TESTING,
            "architecture": TaskType.ARCHITECTURE
        }
        
        for key, task_type in mapping.items():
            if key in task_name.lower():
                return task_type
        
        return TaskType.BACKEND


# Demo and testing functionality
async def demo_nlu_task_decomposer():
    """Demo the enhanced task decomposer"""
    print("🚀 NLU Task Decomposer Demo")
    print("=" * 50)
    
    decomposer = NLUTaskDecomposer()
    
    # Test case 1: Full-stack web app
    context1 = DomainContext(
        tech_stack=["FastAPI", "React", "PostgreSQL", "Docker"],
        project_type="fullstack_web_app",
        current_phase="development",
        team_skills=["Python", "JavaScript", "SQL"],
        constraints=["2-week timeline", "3-person team"]
    )
    
    task1 = "Create a user management system with JWT authentication, role-based access control, and a React admin dashboard"
    
    print(f"\n📋 Task: {task1}")
    print(f"🔧 Tech Stack: {', '.join(context1.tech_stack)}")
    
    result1 = await decomposer.enhanced_decompose(task1, context1)
    
    print(f"\n🎯 Intent: {result1.main_intent.primary_intent} ({result1.main_intent.confidence:.1%})")
    print(f"📊 Complexity: {result1.estimated_complexity:.2f}")
    print(f"⚠️ Risk Factors: {len(result1.risk_factors)}")
    
    print(f"\n📋 Subtasks ({len(result1.subtasks)}):")
    for task in result1.subtasks:
        print(f"  {task.id}. {task.title}")
        print(f"     Type: {task.task_type.value} | Priority: {task.priority.value}")
        print(f"     Hours: {task.estimated_hours} | Agent: {task.required_agent_type}")
        if task.id in result1.dependencies_graph:
            print(f"     Depends on: {result1.dependencies_graph[task.id]}")
        print()
    
    if result1.risk_factors:
        print("⚠️ Risk Factors:")
        for risk in result1.risk_factors:
            print(f"  - {risk}")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    asyncio.run(demo_nlu_task_decomposer())