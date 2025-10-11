#!/usr/bin/env python3
"""
Agent Zero V1 - Complete Enhanced CLI Commands for NLU Task Decomposer
Week 43 - Production Ready Version with fallbacks
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional
import logging

# Try to import CLI libraries with fallbacks
try:
    import click
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    print("⚠️ Click not available - using basic CLI")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    print("⚠️ Rich not available - using basic output")
    
    class Console:
        def print(self, text, style=None):
            print(text)
        
        def status(self, text):
            return SimpleStatusContext(text)
    
    class SimpleStatusContext:
        def __init__(self, text):
            self.text = text
        
        def __enter__(self):
            print(f"🔄 {self.text}")
            return self
        
        def __exit__(self, *args):
            pass
    
    console = Console()

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "shared" / "orchestration"))

try:
    from complete_nlp_enhanced_task_decomposer import NLUTaskDecomposer, DomainContext, TaskBreakdown, TaskPriority
    NLU_AVAILABLE = True
    print("✅ NLU Task Decomposer available")
except ImportError as e:
    NLU_AVAILABLE = False
    print(f"⚠️ NLU components not available: {e}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleCLI:
    """Simple CLI interface when click is not available"""
    
    def __init__(self):
        self.decomposer = None
        if NLU_AVAILABLE:
            try:
                self.decomposer = NLUTaskDecomposer()
                print("✅ NLU Task Decomposer initialized")
            except Exception as e:
                print(f"⚠️ Failed to initialize NLU: {e}")
    
    async def analyze_task(self, description: str, tech_stack: list = None, project_type: str = "general"):
        """Analyze task description with NLU"""
        
        if not self.decomposer:
            print("❌ NLU decomposer not available")
            return
        
        tech_stack = tech_stack or []
        
        # Create context
        context = DomainContext(
            tech_stack=tech_stack,
            project_type=project_type,
            current_phase="development"
        )
        
        print(f"\n🧠 Analyzing task: {description}")
        print(f"🔧 Tech Stack: {', '.join(tech_stack) if tech_stack else 'General'}")
        print(f"📁 Project Type: {project_type}")
        
        with console.status("Analyzing with AI..."):
            try:
                result = await self.decomposer.enhanced_decompose(description, context)
                self._display_results(result, description)
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
    
    def _display_results(self, result: TaskBreakdown, original_description: str):
        """Display analysis results"""
        
        print(f"\n🎯 ANALYSIS RESULTS")
        print("=" * 60)
        
        # Intent Analysis
        print(f"\n📊 Intent Analysis:")
        print(f"  Primary Intent: {result.main_intent.primary_intent}")
        print(f"  Confidence: {result.main_intent.confidence:.1%}")
        print(f"  Domain: {result.main_intent.domain}")
        print(f"  Complexity Score: {result.main_intent.complexity_score:.2f}")
        
        # Subtasks
        print(f"\n📋 Generated Subtasks ({len(result.subtasks)}):")
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
        
        print(f"\n📊 Summary Metrics:")
        print(f"  Total Subtasks: {len(result.subtasks)}")
        print(f"  Total Estimated Hours: {total_hours:.1f}h")
        print(f"  High Priority Tasks: {high_priority_count}")
        print(f"  Overall Complexity: {result.estimated_complexity:.2f}/3.0")
        print(f"  Analysis Confidence: {result.confidence_score:.1%}")
        
        # Risk factors
        if result.risk_factors:
            print(f"\n⚠️  Risk Factors:")
            for i, risk in enumerate(result.risk_factors, 1):
                print(f"  {i}. {risk}")
        
        # Domain knowledge
        if result.domain_knowledge:
            print(f"\n🧠 Domain Insights:")
            if "detected_technologies" in result.domain_knowledge:
                print(f"  Technologies: {', '.join(result.domain_knowledge['detected_technologies'])}")
            if "recommended_patterns" in result.domain_knowledge:
                print(f"  Patterns: {', '.join(result.domain_knowledge['recommended_patterns'])}")
            if "estimated_timeline" in result.domain_knowledge:
                print(f"  Timeline: {result.domain_knowledge['estimated_timeline']}")
    
    def export_results(self, result: TaskBreakdown, format_type: str, filename: str):
        """Export results to file"""
        
        if format_type == "json":
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
                "risks": result.risk_factors
            }
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        
        elif format_type == "markdown":
            md_content = f"""# Task Analysis Results

## Intent Analysis
- **Primary Intent**: {result.main_intent.primary_intent}
- **Confidence**: {result.main_intent.confidence:.1%}
- **Domain**: {result.main_intent.domain}
- **Complexity**: {result.estimated_complexity:.2f}/3.0

## Subtasks

"""
            for task in result.subtasks:
                deps = result.dependencies_graph.get(task.id, [])
                deps_text = f" (depends on: {', '.join(map(str, deps))})" if deps else ""
                
                md_content += f"""### {task.id}. {task.title}
- **Type**: {task.task_type.value}
- **Priority**: {task.priority.value}
- **Hours**: {task.estimated_hours}h
- **Agent**: {task.required_agent_type}
{deps_text}

{task.description}

"""
            
            if result.risk_factors:
                md_content += "\n## Risk Factors\n\n"
                for risk in result.risk_factors:
                    md_content += f"- {risk}\n"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(md_content)
        
        print(f"✅ Results exported to {filename}")

def parse_args():
    """Parse command line arguments"""
    if len(sys.argv) < 2:
        print_help()
        return None
    
    command = sys.argv[1]
    
    if command == "analyze":
        if len(sys.argv) < 3:
            print("❌ Usage: python enhanced_cli_complete.py analyze \"task description\" [--tech-stack tech1,tech2] [--project-type type]")
            return None
        
        description = sys.argv[2]
        tech_stack = []
        project_type = "general"
        
        # Parse additional arguments
        i = 3
        while i < len(sys.argv):
            if sys.argv[i] == "--tech-stack" and i + 1 < len(sys.argv):
                tech_stack = sys.argv[i + 1].split(",")
                i += 2
            elif sys.argv[i] == "--project-type" and i + 1 < len(sys.argv):
                project_type = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        
        return {
            "command": "analyze",
            "description": description,
            "tech_stack": tech_stack,
            "project_type": project_type
        }
    
    elif command == "demo":
        return {"command": "demo"}
    
    elif command == "help":
        print_help()
        return None
    
    else:
        print(f"❌ Unknown command: {command}")
        print_help()
        return None

def print_help():
    """Print help information"""
    print("""
🚀 Agent Zero V1 - NLU Task Decomposer CLI

Commands:
  analyze "task description" [options]    Analyze task with AI
  demo                                   Run demonstration
  help                                   Show this help

Options for analyze:
  --tech-stack tech1,tech2               Technology stack (comma-separated)
  --project-type type                    Project type (fullstack_web_app, api_service, etc.)

Examples:
  python enhanced_cli_complete.py analyze "Create user auth system"
  
  python enhanced_cli_complete.py analyze "Build e-commerce API" \\
    --tech-stack FastAPI,PostgreSQL,Docker --project-type api_service
  
  python enhanced_cli_complete.py demo

Tech Stack Options:
  FastAPI, React, PostgreSQL, Neo4j, Docker, Redis, etc.

Project Types:
  fullstack_web_app, api_service, mobile_app, data_pipeline, ml_project
""")

async def run_demo():
    """Run demonstration"""
    print("🚀 NLU Task Decomposer Demo")
    print("=" * 50)
    
    cli = SimpleCLI()
    
    if not cli.decomposer:
        print("❌ Demo requires NLU components")
        return
    
    # Demo tasks
    demo_tasks = [
        {
            "description": "Create user authentication system with JWT and role-based access control",
            "tech_stack": ["FastAPI", "PostgreSQL", "React"],
            "project_type": "fullstack_web_app"
        },
        {
            "description": "Build REST API for e-commerce product catalog",
            "tech_stack": ["FastAPI", "PostgreSQL"],
            "project_type": "api_service"
        }
    ]
    
    for i, demo in enumerate(demo_tasks, 1):
        print(f"\n🎯 Demo {i}/{len(demo_tasks)}")
        print("-" * 40)
        await cli.analyze_task(
            demo["description"],
            demo["tech_stack"],
            demo["project_type"]
        )
        
        if i < len(demo_tasks):
            input("\nPress Enter to continue to next demo...")
    
    print("\n✅ Demo completed!")

async def main():
    """Main CLI function"""
    print("🧠 Agent Zero V1 - Enhanced Task Decomposer CLI")
    print("=" * 60)
    
    args = parse_args()
    if not args:
        return
    
    cli = SimpleCLI()
    
    if args["command"] == "analyze":
        await cli.analyze_task(
            args["description"],
            args["tech_stack"],
            args["project_type"]
        )
    
    elif args["command"] == "demo":
        await run_demo()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("CLI error")