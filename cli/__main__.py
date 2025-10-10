#!/usr/bin/env python3
"""
Agent Zero V1 - Enhanced CLI with V2.0 Intelligence Layer
Week 43 Implementation
"""

import typer
import uuid
import sys
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path

# Add shared to path
sys.path.append('.')
sys.path.append('./shared')

try:
    from shared.kaizen import (
        get_intelligent_model_recommendation,
        evaluate_task_from_cli,
        get_success_summary,
        generate_kaizen_report_cli,
        get_cost_analysis_cli,
        discover_user_patterns_cli
    )
    from shared.knowledge import sync_tracker_to_graph_cli, get_model_insights_cli
    from shared.utils.simple_tracker import SimpleTracker
    v2_available = True
except ImportError as e:
    print(f"Warning: V2.0 components not available: {e}")
    v2_available = False

app = typer.Typer(help="Agent Zero V1 CLI with V2.0 Intelligence Layer")
console = Console()

@app.command()
def ask(
    question: str,
    model: str = typer.Option(None, help="Specific model to use"),
    priority: str = typer.Option("balanced", help="Priority: cost|quality|speed|balanced"),
    explain: bool = typer.Option(False, help="Show AI reasoning")
):
    """Ask a question with intelligent model selection"""
    
    task_id = str(uuid.uuid4())
    console.print(f"🤖 Processing: [bold]{question}[/bold]")
    
    # V2.0 model selection
    if v2_available and not model:
        recommended_model = get_intelligent_model_recommendation("chat", priority)
        if explain:
            console.print(f"🧠 AI selected: {recommended_model} (priority: {priority})")
    else:
        recommended_model = model or "llama3.2-3b"
    
    # Mock response
    response = f"Mock response using {recommended_model}\n\nThis demonstrates V2.0 Intelligence Layer integration.\nThe system selected {recommended_model} based on {priority} optimization."
    
    console.print(Panel(response, title="Agent Zero Response", border_style="green"))
    
    # V2.0 feedback
    if v2_available:
        tracker = SimpleTracker()
        tracker.track_task(
            task_id=task_id,
            task_type="chat", 
            model_used=recommended_model,
            model_recommended=recommended_model,
            cost=0.0,
            latency=800,
            context={"question": question}
        )
        
        rating = typer.prompt("Rate this response (1-5)", type=int, default=4)
        if 1 <= rating <= 5:
            tracker.record_feedback(task_id, rating)
            console.print("✅ Feedback recorded with V2.0 learning", style="green")

@app.command()
def kaizen_report(
    days: int = typer.Option(1, help="Days to analyze"),
    format: str = typer.Option("summary", help="Format: summary|detailed")
):
    """Generate Kaizen intelligence report"""
    
    if not v2_available:
        console.print("[red]V2.0 components not available[/red]")
        return
    
    console.print(f"📊 Generating Kaizen report for {days} day(s)...")
    
    report = generate_kaizen_report_cli(format)
    
    console.print(Panel(
        f"📅 **Date**: {report['date']}\n"
        f"📈 **Summary**: {report['summary']}\n\n"
        f"🎯 **Key Insights**:\n" +
        "\n".join(f"   • {insight}" for insight in report['key_insights']) +
        f"\n\n🔧 **Action Items**:\n" +
        "\n".join(f"   • {action}" for action in report['top_actions']),
        title="🧠 Kaizen Intelligence Report",
        border_style="cyan"
    ))

@app.command()
def cost_analysis(
    days: int = typer.Option(7, help="Days to analyze"),
    show_optimizations: bool = typer.Option(True, help="Show optimization opportunities")
):
    """Analyze costs and optimization opportunities"""
    
    if not v2_available:
        console.print("[red]V2.0 components not available[/red]")
        return
    
    console.print(f"💰 Analyzing costs for {days} day(s)...")
    
    analysis = get_cost_analysis_cli(days)
    
    console.print(Panel(
        f"💰 **Total Cost**: ${analysis['total_cost']:.4f}\n"
        f"📈 **Avg per Task**: ${analysis['avg_cost_per_task']:.4f}\n"  
        f"🔍 **Total Tasks**: {analysis['total_tasks']}\n"
        f"💡 **Optimization Opportunities**: {analysis['optimization_opportunities']}",
        title="Cost Analysis",
        border_style="yellow"
    ))

@app.command()
def pattern_discovery(days: int = typer.Option(30, help="Days to analyze")):
    """Discover usage patterns and preferences"""
    
    if not v2_available:
        console.print("[red]V2.0 components not available[/red]")
        return
    
    patterns = discover_user_patterns_cli(days)
    
    console.print(Panel(
        f"📈 **Preferences**: {patterns['preferences_count']}\n"
        f"🎯 **Context Patterns**: {patterns['context_patterns_count']}\n"
        f"⏰ **Temporal Patterns**: {patterns['temporal_patterns_count']}",
        title="Pattern Discovery",
        border_style="magenta"
    ))

@app.command()
def model_reasoning(
    task_type: str = typer.Argument(help="Task type: chat|code_generation|analysis"),
    priority: str = typer.Option("balanced", help="Priority: cost|quality|speed|balanced")
):
    """Show AI reasoning behind model selection"""
    
    if not v2_available:
        console.print("[red]V2.0 components not available[/red]")
        return
    
    model = get_intelligent_model_recommendation(task_type, priority)
    
    console.print(Panel(
        f"🤖 **Recommended**: {model}\n"
        f"🎯 **Task Type**: {task_type}\n"
        f"⚖️ **Priority**: {priority}\n\n"
        f"**Reasoning**: Mock V2.0 development mode - intelligent selection based on {priority} optimization for {task_type} tasks.",
        title="AI Model Selection Reasoning",
        border_style="blue"
    ))

@app.command()
def success_breakdown(recent_count: int = typer.Option(10, help="Recent tasks to analyze")):
    """Multi-dimensional success analysis"""
    
    if not v2_available:
        console.print("[red]V2.0 components not available[/red]")
        return
    
    summary = get_success_summary()
    
    console.print(Panel(
        f"📈 **Total Tasks**: {summary['total_tasks']}\n"
        f"✅ **Successful**: {summary['successful_tasks']}\n"
        f"🎯 **Success Rate**: {summary['overall_success_rate']:.1%}",
        title="Success Analysis",
        border_style="green"
    ))

@app.command()
def sync_knowledge_graph(days: int = typer.Option(7, help="Days to sync")):
    """Sync data to Knowledge Graph"""
    
    if not v2_available:
        console.print("[red]V2.0 components not available[/red]")
        return
    
    result = sync_tracker_to_graph_cli(days)
    
    console.print(Panel(
        f"📊 **Total Tasks**: {result['total_tasks']}\n"
        f"✅ **Synced**: {result['synced_tasks']}\n" 
        f"🎯 **Success Rate**: {result['success_rate']:.1%}",
        title="Knowledge Graph Sync",
        border_style="cyan"
    ))

@app.command()
def status():
    """Show system status and V2.0 capabilities"""
    
    console.print("🤖 **Agent Zero V1 + V2.0 Intelligence Layer Status**\n")
    
    # Check components
    table = Table(title="Component Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    
    components = [
        ("SimpleTracker Enhanced", "✅ Available" if v2_available else "❌ Not Available"),
        ("Intelligent Model Selector", "✅ Available" if v2_available else "❌ Not Available"),
        ("Success Evaluator", "✅ Available" if v2_available else "❌ Not Available"),
        ("Metrics Analyzer", "✅ Available" if v2_available else "❌ Not Available"),
        ("Knowledge Graph", "✅ Available" if v2_available else "❌ Not Available")
    ]
    
    for component, status in components:
        color = "green" if "✅" in status else "red"
        table.add_row(component, f"[{color}]{status}[/{color}]")
    
    console.print(table)
    
    if v2_available:
        console.print("\n🚀 **V2.0 Capabilities Available**:")
        console.print("   • Intelligent model selection")
        console.print("   • Multi-dimensional success evaluation")
        console.print("   • Cost optimization analysis")
        console.print("   • Pattern-based learning")
        console.print("   • Daily Kaizen reports")
    else:
        console.print("\n[yellow]⚠️ V2.0 in development mode[/yellow]")

if __name__ == "__main__":
    app()
