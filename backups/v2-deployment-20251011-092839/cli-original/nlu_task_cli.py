#!/usr/bin/env python3
"""
Agent Zero V1 - NLU Task Decomposer CLI Integration
Poprawna integracja z istniejącym systemem CLI
Używa python -m cli struktury
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional, Dict, List, Any

# Add paths for proper imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
shared_dir = project_root / "shared"

# Add to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(shared_dir))
sys.path.insert(0, str(shared_dir / "orchestration"))

# Simple console output (no external deps required)
class SimpleConsole:
    def print(self, text, style=None):
        if style == "green":
            print(f"\033[92m{text}\033[0m")
        elif style == "red":
            print(f"\033[91m{text}\033[0m")
        elif style == "yellow":
            print(f"\033[93m{text}\033[0m")
        elif style == "cyan":
            print(f"\033[96m{text}\033[0m")
        else:
            print(text)
    
    def status(self, text):
        return SimpleStatus(text)

class SimpleStatus:
    def __init__(self, text):
        self.text = text
    
    def __enter__(self):
        print(f"🔄 {self.text}")
        return self
    
    def __exit__(self, *args):
        pass

console = SimpleConsole()

# Import NLU components - używamy pliku który już działa!
NLU_AVAILABLE = False
try:
    # Import from the file that's working
    exec(open(shared_dir / "orchestration" / "nlp_enhanced_task_decomposer.py").read())
    
    # Now these classes are available in local namespace
    NLU_AVAILABLE = True
    console.print("✅ NLU Task Decomposer imported successfully", "green")
    
except Exception as e:
    console.print(f"⚠️ NLU components not available: {e}", "yellow")
    
    # Fallback classes
    from enum import Enum
    from dataclasses import dataclass, field
    
    class TaskPriority(Enum):
        LOW = "low"
        MEDIUM = "medium"
        HIGH = "high"
        CRITICAL = "critical"
    
    class TaskType(Enum):
        FRONTEND = "frontend"
        BACKEND = "backend"
        DATABASE = "database"
        DEVOPS = "devops"
        TESTING = "testing"
        ARCHITECTURE = "architecture"
    
    @dataclass
    class Task:
        id: int
        title: str
        description: str
        task_type: TaskType = TaskType.BACKEND
        priority: TaskPriority = TaskPriority.MEDIUM
        estimated_hours: float = 8.0
        required_agent_type: str = "backend"
    
    @dataclass
    class TaskIntent:
        primary_intent: str
        confidence: float
        domain: str = "general"
    
    @dataclass
    class TaskBreakdown:
        main_intent: TaskIntent
        subtasks: List[Task]
        estimated_complexity: float
        confidence_score: float
        risk_factors: List[str] = field(default_factory=list)
        domain_knowledge: Dict[str, Any] = field(default_factory=dict)
        dependencies_graph: Dict[int, List[int]] = field(default_factory=dict)
    
    @dataclass
    class DomainContext:
        tech_stack: List[str] = field(default_factory=list)
        project_type: str = "general"
        current_phase: str = "development"
    
    class NLUTaskDecomposer:
        def __init__(self):
            console.print("⚠️ Using fallback NLU decomposer", "yellow")
        
        async def enhanced_decompose(self, description: str, context: DomainContext) -> TaskBreakdown:
            return TaskBreakdown(
                main_intent=TaskIntent("DEVELOPMENT", 0.5),
                subtasks=[Task(1, "Analyze Task", description, TaskType.BACKEND)],
                estimated_complexity=1.0,
                confidence_score=0.5
            )

def add_nlu_commands(cli_group):
    """
    Dodaje komendy NLU do istniejącego CLI
    To jest główna funkcja do integracji
    """
    
    # Komenda analyze
    def analyze_command():
        """Analyze task with NLU - main command"""
        if len(sys.argv) < 2:
            print_analyze_help()
            return
        
        # Parse arguments manually
        description = None
        tech_stack = []
        project_type = "general"
        output_format = "table"
        
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            
            if arg == "analyze" and i + 1 < len(sys.argv):
                description = sys.argv[i + 1]
                i += 2
            elif arg == "--tech-stack" and i + 1 < len(sys.argv):
                tech_stack = sys.argv[i + 1].split(",")
                tech_stack = [t.strip() for t in tech_stack]
                i += 2
            elif arg == "--project-type" and i + 1 < len(sys.argv):
                project_type = sys.argv[i + 1]
                i += 2
            elif arg == "--output" and i + 1 < len(sys.argv):
                output_format = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        if not description:
            print_analyze_help()
            return
        
        # Run analysis
        asyncio.run(run_analysis(description, tech_stack, project_type, output_format))
    
    # Komenda demo
    def demo_command():
        """Run NLU Task Decomposer demo"""
        asyncio.run(run_demo())
    
    # Komenda export
    def export_command():
        """Export task analysis to file"""
        if len(sys.argv) < 4:
            print_export_help()
            return
        
        description = sys.argv[2] if len(sys.argv) > 2 else ""
        format_type = sys.argv[3] if len(sys.argv) > 3 else "json"
        filename = sys.argv[4] if len(sys.argv) > 4 else "task_analysis.json"
        
        asyncio.run(run_export(description, format_type, filename))
    
    # Register commands - to jest specyficzne dla Agent Zero CLI systemu
    return {
        'analyze': analyze_command,
        'demo': demo_command,
        'export': export_command
    }

async def run_analysis(description: str, tech_stack: List[str], project_type: str, output_format: str):
    """Run task analysis"""
    
    if not NLU_AVAILABLE:
        console.print("❌ NLU decomposer not available", "red")
        return
    
    console.print(f"\n🧠 Analyzing task: {description}", "cyan")
    console.print(f"🔧 Tech Stack: {', '.join(tech_stack) if tech_stack else 'General'}")
    console.print(f"📁 Project Type: {project_type}")
    
    # Create context
    context = DomainContext(
        tech_stack=tech_stack,
        project_type=project_type,
        current_phase="development"
    )
    
    try:
        decomposer = NLUTaskDecomposer()
        
        with console.status("Analyzing with AI..."):
            result = await decomposer.enhanced_decompose(description, context)
        
        display_results(result, output_format)
        
    except Exception as e:
        console.print(f"❌ Analysis failed: {e}", "red")
        raise

def display_results(result: TaskBreakdown, output_format: str):
    """Display analysis results"""
    
    if output_format == "json":
        # JSON output
        data = {
            "intent": {
                "primary": result.main_intent.primary_intent,
                "confidence": result.main_intent.confidence,
                "domain": result.main_intent.domain
            },
            "subtasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "type": task.task_type.value,
                    "priority": task.priority.value,
                    "hours": task.estimated_hours,
                    "agent": task.required_agent_type
                }
                for task in result.subtasks
            ],
            "complexity": result.estimated_complexity,
            "confidence": result.confidence_score,
            "risks": result.risk_factors
        }
        print(json.dumps(data, indent=2))
        return
    
    # Table output
    console.print(f"\n🎯 ANALYSIS RESULTS", "cyan")
    print("=" * 60)
    
    # Intent Analysis
    console.print(f"\n📊 Intent Analysis:")
    print(f"  Primary Intent: {result.main_intent.primary_intent}")
    print(f"  Confidence: {result.main_intent.confidence:.1%}")
    print(f"  Domain: {result.main_intent.domain}")
    
    # Subtasks
    console.print(f"\n📋 Generated Subtasks ({len(result.subtasks)}):")
    print("-" * 60)
    
    for task in result.subtasks:
        deps = result.dependencies_graph.get(task.id, [])
        deps_str = f" (depends on: {', '.join(map(str, deps))})" if deps else ""
        
        print(f"\n{task.id}. {task.title}")
        print(f"   Description: {task.description}")
        print(f"   Type: {task.task_type.value} | Priority: {task.priority.value}")
        print(f"   Hours: {task.estimated_hours}h | Agent: {task.required_agent_type}")
        if deps_str:
            print(f"   Dependencies: {deps_str}")
    
    # Summary
    total_hours = sum(task.estimated_hours for task in result.subtasks)
    high_priority_count = len([t for t in result.subtasks if t.priority == TaskPriority.HIGH])
    
    console.print(f"\n📊 Summary Metrics:")
    print(f"  Total Subtasks: {len(result.subtasks)}")
    print(f"  Total Estimated Hours: {total_hours:.1f}h")
    print(f"  High Priority Tasks: {high_priority_count}")
    print(f"  Overall Complexity: {result.estimated_complexity:.2f}/3.0")
    print(f"  Analysis Confidence: {result.confidence_score:.1%}")
    
    # Risk factors
    if result.risk_factors:
        console.print(f"\n⚠️  Risk Factors:", "yellow")
        for i, risk in enumerate(result.risk_factors, 1):
            print(f"  {i}. {risk}")

async def run_demo():
    """Run demonstration"""
    console.print("🚀 NLU Task Decomposer Demo", "cyan")
    print("=" * 50)
    
    if not NLU_AVAILABLE:
        console.print("❌ Demo requires NLU components", "red")
        return
    
    # Demo tasks
    demo_tasks = [
        {
            "description": "Create user authentication system with JWT and role-based access control",
            "tech_stack": ["FastAPI", "PostgreSQL", "React"],
            "project_type": "fullstack_web_app"
        },
        {
            "description": "Build REST API for e-commerce product catalog with search functionality",
            "tech_stack": ["FastAPI", "PostgreSQL", "Redis"],
            "project_type": "api_service"
        }
    ]
    
    for i, demo in enumerate(demo_tasks, 1):
        console.print(f"\n🎯 Demo {i}/{len(demo_tasks)}", "cyan")
        print("-" * 40)
        
        await run_analysis(
            demo["description"],
            demo["tech_stack"],
            demo["project_type"],
            "table"
        )
        
        if i < len(demo_tasks):
            input("\n⏵ Press Enter to continue...")
    
    console.print("\n✅ Demo completed!", "green")

async def run_export(description: str, format_type: str, filename: str):
    """Export analysis results"""
    console.print(f"📤 Exporting analysis to {filename}", "cyan")
    
    if not NLU_AVAILABLE:
        console.print("❌ Export requires NLU components", "red")
        return
    
    # Run analysis first
    context = DomainContext(project_type="general")
    decomposer = NLUTaskDecomposer()
    
    result = await decomposer.enhanced_decompose(description, context)
    
    # Export to file
    if format_type == "json":
        data = {
            "task": description,
            "intent": result.main_intent.primary_intent,
            "subtasks": [{"title": t.title, "type": t.task_type.value} for t in result.subtasks],
            "complexity": result.estimated_complexity
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    elif format_type == "markdown":
        md_content = f"""# Task Analysis: {description}

## Intent: {result.main_intent.primary_intent}

## Subtasks:
"""
        for task in result.subtasks:
            md_content += f"- **{task.title}** ({task.task_type.value}, {task.estimated_hours}h)\n"
        
        with open(filename, 'w') as f:
            f.write(md_content)
    
    console.print(f"✅ Exported to {filename}", "green")

def print_analyze_help():
    """Print help for analyze command"""
    print("""
🧠 Agent Zero V1 - NLU Task Analyzer

Usage:
  python -m cli analyze "task description" [options]

Options:
  --tech-stack tech1,tech2    Technology stack (comma-separated)
  --project-type type         Project type
  --output format             Output format (table, json)

Examples:
  python -m cli analyze "Create user auth system"
  
  python -m cli analyze "Build e-commerce API" \\
    --tech-stack FastAPI,PostgreSQL --project-type api_service

Tech Stack Options: FastAPI, React, PostgreSQL, Neo4j, Docker, Redis
Project Types: fullstack_web_app, api_service, mobile_app
""")

def print_export_help():
    """Print help for export command"""
    print("""
📤 Export Task Analysis

Usage:
  python -m cli export "task description" format filename

Formats: json, markdown

Example:
  python -m cli export "Create auth system" markdown analysis.md
""")

def print_nlu_help():
    """Print general help for NLU commands"""
    print("""
🧠 Agent Zero V1 - NLU Task Decomposer

Available Commands:
  analyze    Analyze task with AI
  demo       Run demonstration
  export     Export analysis to file
  
Use 'python -m cli COMMAND' for specific command help.

Examples:
  python -m cli analyze "Create user authentication system"
  python -m cli demo
  python -m cli export "Build API" json api_analysis.json
""")

# Main execution when called directly
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print_nlu_help()
    elif sys.argv[1] == "analyze":
        asyncio.run(run_analysis(
            sys.argv[2] if len(sys.argv) > 2 else "",
            [],
            "general",
            "table"
        ))
    elif sys.argv[1] == "demo":
        asyncio.run(run_demo())
    elif sys.argv[1] == "export":
        if len(sys.argv) >= 5:
            asyncio.run(run_export(sys.argv[2], sys.argv[3], sys.argv[4]))
        else:
            print_export_help()
    else:
        print_nlu_help()

# For integration with Agent Zero CLI system
def register_nlu_commands():
    """
    Integration point for Agent Zero CLI
    Zwraca dict z komendami do rejestracji w main CLI
    """
    return add_nlu_commands(None)