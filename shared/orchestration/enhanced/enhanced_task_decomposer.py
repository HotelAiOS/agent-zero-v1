"""
Enhanced Task Decomposer for Agent Zero V1 - Production Enhancement
Rozbudowa istniejącej klasy TaskDecomposer o real AI reasoning
Zachowuje backward compatibility z obecnym kodem
"""
import json
import re
import logging
import asyncio
import requests
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

# Import istniejące klasy z oryginalnego task_decomposer.py
from .task_decomposer import (
    TaskPriority, TaskStatus, TaskType, TaskDependency, Task, TaskDecomposer
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIReasoningContext:
    """Context dla AI reasoning w task decomposition"""
    project_complexity: str = "medium"  # low, medium, high, enterprise
    tech_stack: List[str] = field(default_factory=list)
    team_size: int = 1
    timeline_weeks: int = 4
    budget_constraints: str = "balanced"  # tight, balanced, flexible
    risk_tolerance: str = "medium"  # low, medium, high
    previous_similar_projects: List[Dict] = field(default_factory=list)

@dataclass 
class TaskReasoning:
    """AI reasoning behind task creation"""
    confidence_score: float = 0.8
    reasoning_path: str = ""
    alternative_approaches: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    optimization_opportunities: List[str] = field(default_factory=list)

@dataclass
class EnhancedTask(Task):
    """Enhanced Task z AI reasoning capabilities"""
    ai_reasoning: TaskReasoning = field(default_factory=TaskReasoning)
    context_tags: List[str] = field(default_factory=list)
    complexity_score: float = 0.5
    automation_potential: float = 0.3
    learning_opportunities: List[str] = field(default_factory=list)

class OllamaClient:
    """Production Ollama client dla AI reasoning"""
    
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.logger = logging.getLogger("OllamaClient")
        
    async def generate_completion(self, prompt: str, model: str = "llama3.2:3b") -> Dict[str, Any]:
        """Generate completion z Ollama model"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2000
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "content": result.get("response", ""),
                    "model_used": model,
                    "tokens": result.get("eval_count", 0)
                }
            else:
                self.logger.warning(f"Ollama request failed: {response.status_code}")
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except Exception as e:
            self.logger.error(f"Ollama client error: {e}")
            return {"success": False, "error": str(e)}

class EnhancedTaskDecomposer(TaskDecomposer):
    """
    Enhanced Task Decomposer z real AI reasoning
    Rozbudowuje istniejący TaskDecomposer zachowując pełną kompatybilność
    """
    
    def __init__(self, llm_client=None, knowledge_graph=None):
        super().__init__(llm_client)
        self.ollama_client = OllamaClient()
        self.knowledge_graph = knowledge_graph
        self.ai_enhanced = True
        self.logger = logging.getLogger("EnhancedTaskDecomposer")
        
        # Intelligence Layer capabilities
        self.available_models = {
            "llama3.2:3b": {"speed": "fast", "cost": "low", "quality": "good"},
            "qwen2.5:14b": {"speed": "medium", "cost": "medium", "quality": "high"},
            "deepseek-coder:33b": {"speed": "slow", "cost": "high", "quality": "excellent"},
            "mixtral:8x7b": {"speed": "medium", "cost": "high", "quality": "excellent"}
        }
        
    def select_optimal_model(self, task_complexity: str, priority: str = "balanced") -> str:
        """Intelligent model selection based na task complexity i priority"""
        
        if priority == "speed":
            return "llama3.2:3b"
        elif priority == "cost":
            return "llama3.2:3b" 
        elif priority == "quality":
            if task_complexity == "high" or task_complexity == "enterprise":
                return "deepseek-coder:33b"
            return "qwen2.5:14b"
        else:  # balanced
            if task_complexity in ["low", "medium"]:
                return "qwen2.5:14b"
            return "mixtral:8x7b"
    
    async def decompose_with_ai_reasoning(
        self, 
        task_description: str, 
        context: AIReasoningContext
    ) -> List[EnhancedTask]:
        """
        Enhanced task decomposition z real AI reasoning
        Rozbudowuje istniejącą funkcjonalność o AI capabilities
        """
        
        self.logger.info(f"Starting AI-enhanced decomposition: {task_description[:50]}...")
        
        # Step 1: Użyj base TaskDecomposer jako foundation
        base_tasks = self.decompose_project(
            self._infer_project_type(task_description, context),
            [task_description] + context.tech_stack
        )
        
        # Step 2: AI Enhancement - jeśli dostępne
        if self.ai_enhanced:
            enhanced_tasks = await self._enhance_tasks_with_ai(
                base_tasks, task_description, context
            )
            return enhanced_tasks
        
        # Fallback: Convert base tasks to enhanced tasks
        return [self._convert_to_enhanced_task(task) for task in base_tasks]
    
    def _infer_project_type(self, description: str, context: AIReasoningContext) -> str:
        """Infer project type z description i context"""
        
        description_lower = description.lower()
        
        # Backend/API focused
        if any(term in description_lower for term in ["api", "backend", "microservice", "database"]):
            return "backend_system"
        
        # Frontend focused  
        elif any(term in description_lower for term in ["frontend", "ui", "react", "vue", "angular"]):
            return "frontend_application"
            
        # Full-stack application
        elif any(term in description_lower for term in ["full-stack", "web app", "application", "platform"]):
            return "fullstack_web_app"
            
        # DevOps/Infrastructure
        elif any(term in description_lower for term in ["deploy", "infrastructure", "docker", "kubernetes"]):
            return "devops_pipeline"
            
        # Default
        return "fullstack_web_app"
    
    async def _enhance_tasks_with_ai(
        self, 
        base_tasks: List[Task], 
        description: str, 
        context: AIReasoningContext
    ) -> List[EnhancedTask]:
        """Enhance base tasks z AI reasoning"""
        
        enhanced_tasks = []
        
        # Select optimal model dla task analysis
        model = self.select_optimal_model(context.project_complexity)
        self.logger.info(f"Using AI model: {model} for complexity: {context.project_complexity}")
        
        for task in base_tasks:
            enhanced_task = await self._analyze_task_with_ai(task, context, model)
            enhanced_tasks.append(enhanced_task)
        
        # AI-driven dependency optimization
        enhanced_tasks = await self._optimize_dependencies_with_ai(enhanced_tasks, context, model)
        
        return enhanced_tasks
    
    async def _analyze_task_with_ai(
        self, 
        task: Task, 
        context: AIReasoningContext, 
        model: str
    ) -> EnhancedTask:
        """Analyze individual task z AI reasoning"""
        
        prompt = f"""
Analyze this software development task and provide intelligent insights:

**Task**: {task.title}
**Description**: {task.description}
**Type**: {task.task_type.value}
**Priority**: {task.priority.value}

**Project Context**:
- Complexity: {context.project_complexity}
- Tech Stack: {', '.join(context.tech_stack)}
- Team Size: {context.team_size}
- Timeline: {context.timeline_weeks} weeks
- Budget: {context.budget_constraints}

**Please analyze and provide JSON response with**:
{{
    "confidence_score": 0.85,
    "complexity_score": 0.7,
    "automation_potential": 0.4,
    "reasoning_path": "This task requires...",
    "risk_factors": ["dependency on external API", "potential performance issues"],
    "optimization_opportunities": ["consider caching", "parallel processing"],
    "learning_opportunities": ["new framework patterns", "performance optimization"],
    "context_tags": ["backend", "database", "performance-critical"],
    "estimated_hours_refined": 12.5
}}

Focus on practical, actionable insights for a {context.project_complexity} complexity project.
"""

        # Get AI analysis
        ai_response = await self.ollama_client.generate_completion(prompt, model)
        
        if ai_response["success"]:
            try:
                analysis = json.loads(ai_response["content"])
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse AI response for task {task.id}")
                analysis = self._generate_fallback_analysis(task, context)
        else:
            analysis = self._generate_fallback_analysis(task, context)
        
        # Create enhanced task
        enhanced_task = EnhancedTask(
            id=task.id,
            title=task.title,
            description=task.description,
            task_type=task.task_type,
            status=task.status,
            priority=task.priority,
            dependencies=task.dependencies,
            estimated_hours=analysis.get("estimated_hours_refined", task.estimated_hours),
            required_agent_type=task.required_agent_type,
            assigned_agent=task.assigned_agent,
            # Enhanced fields
            ai_reasoning=TaskReasoning(
                confidence_score=analysis.get("confidence_score", 0.8),
                reasoning_path=analysis.get("reasoning_path", "Standard task analysis"),
                risk_factors=analysis.get("risk_factors", []),
                optimization_opportunities=analysis.get("optimization_opportunities", [])
            ),
            context_tags=analysis.get("context_tags", []),
            complexity_score=analysis.get("complexity_score", 0.5),
            automation_potential=analysis.get("automation_potential", 0.3),
            learning_opportunities=analysis.get("learning_opportunities", [])
        )
        
        return enhanced_task
    
    def _generate_fallback_analysis(self, task: Task, context: AIReasoningContext) -> Dict[str, Any]:
        """Generate fallback analysis gdy AI nie jest dostępne"""
        
        base_complexity = {
            "low": 0.3,
            "medium": 0.5, 
            "high": 0.7,
            "enterprise": 0.9
        }.get(context.project_complexity, 0.5)
        
        return {
            "confidence_score": 0.7,
            "complexity_score": base_complexity,
            "automation_potential": 0.3,
            "reasoning_path": f"Fallback analysis for {task.task_type.value} task",
            "risk_factors": ["standard implementation risks"],
            "optimization_opportunities": ["code review", "testing optimization"],
            "learning_opportunities": ["best practices"],
            "context_tags": [task.task_type.value],
            "estimated_hours_refined": task.estimated_hours
        }
    
    async def _optimize_dependencies_with_ai(
        self, 
        tasks: List[EnhancedTask], 
        context: AIReasoningContext,
        model: str
    ) -> List[EnhancedTask]:
        """Optimize task dependencies z AI reasoning"""
        
        # Create dependency optimization prompt
        tasks_summary = []
        for task in tasks:
            tasks_summary.append({
                "id": task.id,
                "title": task.title,
                "type": task.task_type.value,
                "complexity": task.complexity_score,
                "current_dependencies": [dep.task_id for dep in task.dependencies]
            })
        
        prompt = f"""
Analyze these software development tasks and optimize their dependencies:

**Tasks**: {json.dumps(tasks_summary, indent=2)}

**Project Context**:
- Complexity: {context.project_complexity}
- Team Size: {context.team_size}
- Timeline: {context.timeline_weeks} weeks

**Provide optimized dependencies as JSON**:
{{
    "optimized_dependencies": [
        {{"task_id": 1, "depends_on": [2, 3], "reasoning": "Task 1 needs database schema from task 2"}},
        {{"task_id": 4, "depends_on": [1], "reasoning": "Frontend needs backend API endpoints"}}
    ],
    "parallel_opportunities": [
        {{"tasks": [2, 5], "reasoning": "Database and DevOps setup can run in parallel"}}
    ],
    "critical_path": [1, 2, 3, 4],
    "optimization_notes": "Consider parallel execution of independent tasks"
}}

Focus on realistic dependencies that maximize parallel work while ensuring proper sequencing.
"""

        ai_response = await self.ollama_client.generate_completion(prompt, model)
        
        if ai_response["success"]:
            try:
                optimization = json.loads(ai_response["content"])
                return self._apply_dependency_optimization(tasks, optimization)
            except json.JSONDecodeError:
                self.logger.warning("Failed to parse dependency optimization")
        
        return tasks
    
    def _apply_dependency_optimization(
        self, 
        tasks: List[EnhancedTask], 
        optimization: Dict[str, Any]
    ) -> List[EnhancedTask]:
        """Apply AI-optimized dependencies do tasks"""
        
        optimized_deps = optimization.get("optimized_dependencies", [])
        
        # Create lookup map
        task_map = {task.id: task for task in tasks}
        
        # Apply optimized dependencies
        for dep_info in optimized_deps:
            task_id = dep_info["task_id"]
            depends_on = dep_info.get("depends_on", [])
            reasoning = dep_info.get("reasoning", "")
            
            if task_id in task_map:
                # Clear existing dependencies
                task_map[task_id].dependencies = []
                
                # Add optimized dependencies
                for dep_id in depends_on:
                    if dep_id in task_map:
                        dependency = TaskDependency(
                            task_id=dep_id,
                            dependency_type="blocks",
                            description=f"AI-optimized: {reasoning}"
                        )
                        task_map[task_id].dependencies.append(dependency)
        
        return list(task_map.values())
    
    def _convert_to_enhanced_task(self, task: Task) -> EnhancedTask:
        """Convert base Task to EnhancedTask"""
        return EnhancedTask(
            id=task.id,
            title=task.title,
            description=task.description,
            task_type=task.task_type,
            status=task.status,
            priority=task.priority,
            dependencies=task.dependencies,
            estimated_hours=task.estimated_hours,
            required_agent_type=task.required_agent_type,
            assigned_agent=task.assigned_agent
        )
    
    # Zachowaj backward compatibility - override base methods
    def decompose_task(self, task_description: str) -> Dict[Any, Any]:
        """Override base method zachowując compatibility"""
        context = AIReasoningContext(project_complexity="medium")
        
        # Dla synchronous compatibility, use base implementation
        return super().decompose_task(task_description)
    
    async def decompose_task_async(self, task_description: str) -> Dict[Any, Any]:
        """Async version of decompose_task z AI enhancement"""
        context = AIReasoningContext(project_complexity="medium")
        
        enhanced_tasks = await self.decompose_with_ai_reasoning(task_description, context)
        
        # Convert to compatible format
        return {
            "subtasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "status": task.status.value,
                    "priority": task.priority.value,
                    "dependencies": [dep.task_id for dep in task.dependencies],
                    "ai_confidence": task.ai_reasoning.confidence_score,
                    "complexity_score": task.complexity_score
                }
                for task in enhanced_tasks
            ]
        }

# Utility functions dla CLI integration
async def create_enhanced_decomposer() -> EnhancedTaskDecomposer:
    """Factory function dla creating EnhancedTaskDecomposer"""
    return EnhancedTaskDecomposer()

def create_reasoning_context(
    complexity: str = "medium",
    tech_stack: List[str] = None,
    team_size: int = 1,
    weeks: int = 4
) -> AIReasoningContext:
    """Helper function dla creating AI reasoning context"""
    return AIReasoningContext(
        project_complexity=complexity,
        tech_stack=tech_stack or [],
        team_size=team_size,
        timeline_weeks=weeks
    )

if __name__ == "__main__":
    async def test_enhanced_decomposer():
        """Test enhanced task decomposer"""
        print("Testing Enhanced Task Decomposer...")
        
        decomposer = EnhancedTaskDecomposer()
        context = create_reasoning_context(
            complexity="high",
            tech_stack=["Python", "FastAPI", "Neo4j", "Docker"],
            team_size=2,
            weeks=6
        )
        
        task_description = """
        Develop an AI-powered task management system with real-time collaboration,
        knowledge graph integration, and intelligent task decomposition capabilities.
        The system should support multiple AI models and provide detailed analytics.
        """
        
        enhanced_tasks = await decomposer.decompose_with_ai_reasoning(task_description, context)
        
        print(f"\n✅ Generated {len(enhanced_tasks)} enhanced tasks:")
        for task in enhanced_tasks:
            print(f"   • {task.title} (confidence: {task.ai_reasoning.confidence_score:.2f})")
            print(f"     Complexity: {task.complexity_score:.2f}, Automation: {task.automation_potential:.2f}")
            if task.dependencies:
                deps = [str(dep.task_id) for dep in task.dependencies]
                print(f"     Dependencies: {', '.join(deps)}")
            print()
        
        print("✅ Enhanced Task Decomposer working correctly!")
    
    # Run test
    asyncio.run(test_enhanced_decomposer())