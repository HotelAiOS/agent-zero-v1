# CLI Enhancement for NLU Task Decomposer
# Integracja z istniejącym CLI systemem Agent Zero V1
# Lokalizacja: cli/enhanced_commands.py

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add shared directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "shared"))

try:
    from nlp_enhanced_task_decomposer import NLUTaskDecomposer, DomainContext, TaskBreakdown
    from shared.orchestration.task_decomposer import TaskPriority, TaskType
    from shared.ollama_client import OllamaClient
    NLU_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ NLU components not available: {e}")
    NLU_AVAILABLE = False

console = Console()

class CLINLUCommands:
    """Enhanced CLI commands for NLU Task Decomposer"""
    
    def __init__(self):
        self.decomposer = None
        if NLU_AVAILABLE:
            try:
                self.decomposer = NLUTaskDecomposer()
                console.print("✅ NLU Task Decomposer initialized", style="green")
            except Exception as e:
                console.print(f"⚠️ Failed to initialize NLU: {e}", style="yellow")

@click.group(name="task")
def task_cli():
    """Enhanced task management with NLU capabilities"""
    pass

@task_cli.command()
@click.argument('description')
@click.option('--tech-stack', multiple=True, help='Technology stack (e.g., FastAPI, React)')
@click.option('--project-type', default='general', help='Project type (fullstack_web_app, api_service)')
@click.option('--phase', default='development', help='Current project phase')
@click.option('--output', default='table', type=click.Choice(['table', 'json', 'detailed']))
def analyze(description: str, tech_stack: tuple, project_type: str, phase: str, output: str):
    """Analyze task description with NLU and generate intelligent breakdown"""
    
    if not NLU_AVAILABLE:
        console.print("❌ NLU functionality not available", style="red")
        return
    
    # Create context
    context = DomainContext(
        tech_stack=list(tech_stack),
        project_type=project_type,
        current_phase=phase
    )
    
    # Show input analysis
    with console.status("[bold green]Analyzing task with AI..."):
        cli = CLINLUCommands()
        if not cli.decomposer:
            console.print("❌ NLU decomposer not available", style="red")
            return
        
        try:
            result = asyncio.run(cli.decomposer.enhanced_decompose(description, context))
            _display_task_analysis(result, output, description)
        except Exception as e:
            console.print(f"❌ Analysis failed: {e}", style="red")

def _display_task_analysis(result: TaskBreakdown, output_format: str, original_description: str):
    """Display task analysis results in specified format"""
    
    if output_format == "json":
        # JSON output
        output_data = {
            "original_task": original_description,
            "intent": {
                "primary": result.main_intent.primary_intent,
                "confidence": result.main_intent.confidence,
                "domain": result.main_intent.domain,
                "complexity_score": result.main_intent.complexity_score
            },
            "subtasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "type": task.task_type.value,
                    "priority": task.priority.value,
                    "estimated_hours": task.estimated_hours,
                    "required_agent": task.required_agent_type
                }
                for task in result.subtasks
            ],
            "complexity": result.estimated_complexity,
            "confidence": result.confidence_score,
            "risks": result.risk_factors,
            "dependencies": result.dependencies_graph
        }
        console.print(json.dumps(output_data, indent=2))
        return
    
    # Rich visual output
    console.print(Panel.fit(f"[bold blue]Task Analysis Results[/bold blue]", style="blue"))
    
    # Intent Analysis
    intent_table = Table(title="🎯 Intent Analysis", show_header=True)
    intent_table.add_column("Property", style="cyan")
    intent_table.add_column("Value", style="white")
    
    intent_table.add_row("Primary Intent", f"[bold]{result.main_intent.primary_intent}[/bold]")
    intent_table.add_row("Confidence", f"{result.main_intent.confidence:.1%}")
    intent_table.add_row("Domain", result.main_intent.domain)
    intent_table.add_row("Complexity Score", f"{result.main_intent.complexity_score:.2f}")
    
    console.print(intent_table)
    console.print()
    
    # Subtasks Table
    subtasks_table = Table(title="📋 Generated Subtasks", show_header=True)
    subtasks_table.add_column("ID", width=4, justify="center")
    subtasks_table.add_column("Title", style="cyan", width=30)
    subtasks_table.add_column("Type", width=12)
    subtasks_table.add_column("Priority", width=8)
    subtasks_table.add_column("Hours", width=6, justify="right")
    subtasks_table.add_column("Agent", width=12)
    subtasks_table.add_column("Dependencies", width=12)
    
    for task in result.subtasks:
        deps = result.dependencies_graph.get(task.id, [])
        deps_str = ", ".join(map(str, deps)) if deps else "None"
        
        # Color code priority
        priority_color = {
            "high": "red",
            "medium": "yellow",
            "low": "green",
            "critical": "bright_red"
        }.get(task.priority.value, "white")
        
        subtasks_table.add_row(
            str(task.id),
            task.title,
            task.task_type.value,
            f"[{priority_color}]{task.priority.value}[/{priority_color}]",
            f"{task.estimated_hours:.1f}h",
            task.required_agent_type,
            deps_str
        )
    
    console.print(subtasks_table)
    console.print()
    
    # Summary metrics
    summary_table = Table(title="📊 Summary Metrics", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="white")
    
    total_hours = sum(task.estimated_hours for task in result.subtasks)
    high_priority_count = len([t for t in result.subtasks if t.priority == TaskPriority.HIGH])
    
    summary_table.add_row("Total Subtasks", str(len(result.subtasks)))
    summary_table.add_row("Total Est. Hours", f"{total_hours:.1f}h")
    summary_table.add_row("High Priority Tasks", str(high_priority_count))
    summary_table.add_row("Overall Complexity", f"{result.estimated_complexity:.2f}/3.0")
    summary_table.add_row("Analysis Confidence", f"{result.confidence_score:.1%}")
    
    console.print(summary_table)
    
    # Risk factors
    if result.risk_factors:
        console.print("\n⚠️ [bold yellow]Risk Factors:[/bold yellow]")
        for i, risk in enumerate(result.risk_factors, 1):
            console.print(f"  {i}. {risk}", style="yellow")
    
    # Domain knowledge (detailed output only)
    if output_format == "detailed" and result.domain_knowledge:
        console.print("\n🧠 [bold cyan]Domain Knowledge:[/bold cyan]")
        
        if "detected_technologies" in result.domain_knowledge:
            console.print(f"  📱 Technologies: {', '.join(result.domain_knowledge['detected_technologies'])}")
        
        if "recommended_patterns" in result.domain_knowledge:
            console.print(f"  🎨 Patterns: {', '.join(result.domain_knowledge['recommended_patterns'])}")
        
        if "skill_requirements" in result.domain_knowledge:
            console.print(f"  🎯 Skills: {', '.join(result.domain_knowledge['skill_requirements'])}")
        
        if "estimated_timeline" in result.domain_knowledge:
            console.print(f"  📅 Timeline: {result.domain_knowledge['estimated_timeline']}")

@task_cli.command()
@click.argument('description')
@click.option('--tech-stack', multiple=True, help='Technology stack')
@click.option('--project-type', default='general', help='Project type')
@click.option('--format', 'output_format', default='yaml', type=click.Choice(['yaml', 'json', 'markdown']))
def export(description: str, tech_stack: tuple, project_type: str, output_format: str):
    """Export task breakdown for project management tools"""
    
    if not NLU_AVAILABLE:
        console.print("❌ NLU functionality not available", style="red")
        return
    
    context = DomainContext(
        tech_stack=list(tech_stack),
        project_type=project_type
    )
    
    with console.status("[bold green]Generating export..."):
        cli = CLINLUCommands()
        if not cli.decomposer:
            console.print("❌ NLU decomposer not available", style="red")
            return
        
        try:
            result = asyncio.run(cli.decomposer.enhanced_decompose(description, context))
            export_content = _generate_export(result, output_format, description)
            
            # Save to file
            filename = f"task_breakdown.{output_format}"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(export_content)
            
            console.print(f"✅ Exported to {filename}", style="green")
            
        except Exception as e:
            console.print(f"❌ Export failed: {e}", style="red")

def _generate_export(result: TaskBreakdown, format_type: str, original_description: str) -> str:
    """Generate export content in specified format"""
    
    if format_type == "json":
        data = {
            "project": {
                "description": original_description,
                "intent": result.main_intent.primary_intent,
                "complexity": result.estimated_complexity,
                "confidence": result.confidence_score
            },
            "tasks": [
                {
                    "id": task.id,
                    "title": task.title,
                    "description": task.description,
                    "type": task.task_type.value,
                    "priority": task.priority.value,
                    "hours": task.estimated_hours,
                    "agent": task.required_agent_type,
                    "dependencies": result.dependencies_graph.get(task.id, [])
                }
                for task in result.subtasks
            ],
            "risks": result.risk_factors
        }
        return json.dumps(data, indent=2)
    
    elif format_type == "markdown":
        md = f"""# Task Breakdown: {original_description}

## Project Overview
- **Intent**: {result.main_intent.primary_intent}
- **Domain**: {result.main_intent.domain}
- **Complexity**: {result.estimated_complexity:.2f}/3.0
- **Confidence**: {result.confidence_score:.1%}

## Subtasks

"""
        for task in result.subtasks:
            deps = result.dependencies_graph.get(task.id, [])
            deps_text = f" (depends on: {', '.join(map(str, deps))})" if deps else ""
            
            md += f"""### {task.id}. {task.title}
- **Type**: {task.task_type.value}
- **Priority**: {task.priority.value}
- **Estimated Hours**: {task.estimated_hours}h
- **Required Agent**: {task.required_agent_type}
- **Dependencies**: {deps_text}

{task.description}

"""
        
        if result.risk_factors:
            md += "## Risk Factors\n\n"
            for risk in result.risk_factors:
                md += f"- {risk}\n"
        
        return md
    
    elif format_type == "yaml":
        yaml_content = f"""project:
  description: "{original_description}"
  intent: "{result.main_intent.primary_intent}"
  domain: "{result.main_intent.domain}"
  complexity: {result.estimated_complexity:.2f}
  confidence: {result.confidence_score:.2f}

tasks:
"""
        for task in result.subtasks:
            deps = result.dependencies_graph.get(task.id, [])
            yaml_content += f"""  - id: {task.id}
    title: "{task.title}"
    description: "{task.description}"
    type: "{task.task_type.value}"
    priority: "{task.priority.value}"
    hours: {task.estimated_hours}
    agent: "{task.required_agent_type}"
    dependencies: {deps}
"""
        
        if result.risk_factors:
            yaml_content += "\nrisks:\n"
            for risk in result.risk_factors:
                yaml_content += f'  - "{risk}"\n'
        
        return yaml_content

@task_cli.command()
@click.argument('tech_stack', nargs=-1)
def recommend(tech_stack):
    """Get AI recommendations for task decomposition patterns"""
    
    if not tech_stack:
        console.print("❌ Please specify at least one technology", style="red")
        return
    
    console.print(f"🔍 [bold cyan]Recommendations for: {', '.join(tech_stack)}[/bold cyan]")
    
    # Mock recommendations based on tech stack
    recommendations = _get_tech_recommendations(tech_stack)
    
    for category, items in recommendations.items():
        if items:
            console.print(f"\n📋 [bold]{category}[/bold]:")
            for item in items:
                console.print(f"  • {item}")

def _get_tech_recommendations(tech_stack: tuple) -> dict:
    """Get recommendations based on technology stack"""
    
    recommendations = {
        "Typical Task Patterns": [],
        "Common Dependencies": [],
        "Recommended Agents": [],
        "Best Practices": []
    }
    
    stack_lower = [tech.lower() for tech in tech_stack]
    
    if "fastapi" in stack_lower:
        recommendations["Typical Task Patterns"].extend([
            "API endpoint design and implementation",
            "Pydantic model definition",
            "Dependency injection setup",
            "OpenAPI documentation"
        ])
        recommendations["Recommended Agents"].append("backend")
    
    if "react" in stack_lower:
        recommendations["Typical Task Patterns"].extend([
            "Component architecture planning",
            "State management setup",
            "API integration layer",
            "UI/UX implementation"
        ])
        recommendations["Recommended Agents"].append("frontend")
    
    if "postgresql" in stack_lower or "neo4j" in stack_lower:
        recommendations["Typical Task Patterns"].extend([
            "Database schema design",
            "Migration scripts",
            "Query optimization",
            "Data modeling"
        ])
        recommendations["Recommended Agents"].append("database")
    
    if "docker" in stack_lower:
        recommendations["Typical Task Patterns"].extend([
            "Container configuration",
            "Multi-service orchestration",
            "Network setup",
            "Volume management"
        ])
        recommendations["Recommended Agents"].append("devops")
    
    # Common best practices
    recommendations["Best Practices"] = [
        "Start with architecture and design tasks",
        "Define clear interfaces between components",
        "Plan testing strategy early",
        "Consider deployment requirements from the start"
    ]
    
    return recommendations

# Integration with existing CLI
if __name__ == "__main__":
    task_cli()