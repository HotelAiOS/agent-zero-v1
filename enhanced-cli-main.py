#!/usr/bin/env python3
"""
Agent Zero V1 - Enhanced CLI Commands
V2.0 Intelligence Layer Integration - Week 43 Implementation

Nowe komendy CLI z integracją V2.0 Intelligence Layer:
- a0 kaizen-report         # Daily Kaizen insights
- a0 cost-analysis         # Cost optimization opportunities  
- a0 pattern-discovery     # Manual pattern exploration
- a0 model-reasoning       # AI decision explanations
- a0 success-breakdown     # Multi-dimensional success analysis

Rozszerza istniejący cli/__main__.py o V2.0 capabilities.
"""

import typer
import uuid
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from typing import Optional
from pathlib import Path
import sqlite3
import json

# Import V2.0 Intelligence Layer components
import sys
sys.path.append('.')

# Import all V2.0 components (assuming they're in the same directory)
try:
    exec(open('intelligent-model-selector.py').read(), globals())
    exec(open('success-failure-classifier.py').read(), globals()) 
    exec(open('active-metrics-analyzer.py').read(), globals())
    exec(open('enhanced-feedback-loop.py').read(), globals())
    exec(open('kaizen-knowledge-graph.py').read(), globals())
    v2_components_available = True
except Exception as e:
    print(f"Warning: V2.0 components not fully available: {e}")
    v2_components_available = False

app = typer.Typer(help="Agent Zero V1 CLI with V2.0 Intelligence Layer")
console = Console()

# Initialize V2.0 components if available
if v2_components_available:
    try:
        intelligent_selector = IntelligentModelSelector()
        success_evaluator = SuccessEvaluator()
        metrics_analyzer = ActiveMetricsAnalyzer()
        feedback_engine = EnhancedFeedbackLoopEngine()
        knowledge_graph = KaizenKnowledgeGraph()
    except Exception as e:
        console.print(f"[red]Error initializing V2.0 components: {e}[/red]")
        v2_components_available = False

@app.command()
def ask(
    question: str, 
    provider: Optional[str] = typer.Option(None, help="Specific model to use"),
    priority: str = typer.Option("balanced", help="Priority: cost|quality|speed|balanced"),
    explain: bool = typer.Option(False, help="Show AI reasoning for model choice")
):
    """Ask a question with V2.0 intelligent model selection"""
    
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    console.print(f"🤖 Processing question: [bold]{question}[/bold]")
    
    # V2.0 Intelligence: Get optimal model recommendation
    if v2_components_available and not provider:
        try:
            explanation = explain_model_choice("chat", priority)
            recommended_model = explanation['recommended_model']
            
            if explain:
                console.print(Panel(
                    f"🧠 **AI Reasoning**: {explanation['reasoning']}\n"
                    f"**Confidence**: {explanation['confidence']:.1%}\n"
                    f"**Estimated Cost**: ${explanation['cost_estimate']:.4f}",
                    title="Model Selection Intelligence",
                    border_style="blue"
                ))
        except Exception as e:
            console.print(f"[yellow]Warning: Using fallback model selection: {e}[/yellow]")
            recommended_model = "llama3.2-3b"
    else:
        recommended_model = provider or "llama3.2-3b"
    
    model_used = recommended_model  # In real implementation, this might differ
    
    # Simulate processing with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Querying {model_used}...", total=None)
        
        import time
        time.sleep(1)  # Simulate processing
    
    # Mock response - in real implementation this would call the actual model
    response = f"""Based on your question "{question}", here's a comprehensive analysis:

This is a mock response demonstrating the V2.0 Intelligence Layer integration. The system intelligently selected {model_used} based on your priority setting ({priority}) and the nature of your question.

Key insights:
• The question was classified as a chat-type interaction
• {model_used} was selected for optimal balance of quality and efficiency
• Response generated with {priority} optimization priority

This response would contain the actual AI-generated content in a real deployment."""
    
    end_time = datetime.now()
    latency_ms = int((end_time - start_time).total_seconds() * 1000)
    cost_usd = 0.001 if "gpt" in model_used else 0.0
    
    # Track the task
    if v2_components_available:
        try:
            # Track with SimpleTracker
            intelligent_selector.tracker.track_task(
                task_id=task_id,
                task_type="chat",
                model_used=model_used,
                model_recommended=recommended_model,
                cost=cost_usd,
                latency=latency_ms
            )
            
            # V2.0 Success evaluation
            evaluation = success_evaluator.evaluate_task_success(
                task_id=task_id,
                task_type="chat",
                output=response,
                cost_usd=cost_usd,
                latency_ms=latency_ms
            )
            
            # Real-time alerts
            alerts = metrics_analyzer.analyze_task_completion(
                task_id, model_used, cost_usd, latency_ms
            )
            
            # Show alerts if any
            if alerts:
                for alert in alerts:
                    severity_color = "red" if alert.severity.value == "CRITICAL" else "yellow"
                    console.print(f"[{severity_color}]⚠️ {alert.message}[/{severity_color}]")
        
        except Exception as e:
            console.print(f"[yellow]Warning: V2.0 evaluation failed: {e}[/yellow]")
    
    # Display response
    console.print(Panel(response, title="Agent Zero Response", border_style="green"))
    
    # Enhanced feedback collection
    console.print("\n" + "="*50)
    console.print("🔄 **V2.0 Enhanced Feedback** (optional):", style="bold blue")
    console.print("Rate this result [1-5] or Enter to skip:")
    console.print("="*50)
    
    try:
        rating_input = input("Rating: ").strip()
        if rating_input and rating_input.isdigit() and 1 <= int(rating_input) <= 5:
            rating = int(rating_input)
            comment = input("Comments (optional): ").strip() or None
            
            if v2_components_available:
                # V2.0 Enhanced feedback processing
                feedback_result = feedback_engine.process_feedback_with_learning(
                    task_id=task_id,
                    user_rating=rating,
                    model_used=model_used,
                    model_recommended=recommended_model,
                    task_type="chat",
                    cost=cost_usd,
                    latency=latency_ms,
                    context={'comment': comment}
                )
                
                console.print("✅ Enhanced feedback processed!", style="green")
                
                # Show learning insights
                if feedback_result.get('learning_insights'):
                    console.print("🧠 **Learning Insights**:")
                    for insight in feedback_result['learning_insights']:
                        console.print(f"   • {insight}")
            else:
                # Fallback to simple tracking
                console.print("✅ Feedback recorded", style="green")
        
    except KeyboardInterrupt:
        console.print("\n⏭️ Feedback skipped", style="dim")

@app.command()
def kaizen_report(
    days: int = typer.Option(1, help="Number of days to analyze"),
    format: str = typer.Option("summary", help="Format: summary|detailed"),
    save_to_file: bool = typer.Option(False, help="Save report to file")
):
    """Generate daily Kaizen insights and improvement opportunities"""
    
    if not v2_components_available:
        console.print("[red]❌ V2.0 components not available for Kaizen reporting[/red]")
        return
    
    console.print(f"📊 Generating Kaizen report for last {days} day(s)...")
    
    try:
        # Generate comprehensive Kaizen report
        if format == "detailed":
            report_data = generate_kaizen_report_cli("detailed")
        else:
            report_data = generate_kaizen_report_cli("summary")
        
        # Display report
        console.print(Panel(
            f"📅 **Date**: {report_data['date']}\n"
            f"📈 **Summary**: {report_data.get('summary', 'N/A')}\n\n"
            f"🎯 **Key Insights**:\n" + 
            "\n".join(f"   • {insight}" for insight in report_data.get('key_insights', [])) +
            f"\n\n🔧 **Action Items**:\n" +
            "\n".join(f"   • {action}" for action in report_data.get('top_actions', [])),
            title="🧠 Kaizen Intelligence Report",
            border_style="cyan"
        ))
        
        # Show alerts summary
        critical_alerts = report_data.get('critical_alerts', 0)
        if critical_alerts > 0:
            console.print(f"🚨 **{critical_alerts} Critical Alerts** - Immediate attention required!", style="bold red")
        
        # Save to file if requested
        if save_to_file:
            filename = f"kaizen_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(filename, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            console.print(f"💾 Report saved to: {filename}")
    
    except Exception as e:
        console.print(f"[red]❌ Error generating Kaizen report: {e}[/red]")

@app.command()
def cost_analysis(
    days: int = typer.Option(7, help="Number of days to analyze"),
    show_optimizations: bool = typer.Option(True, help="Show optimization opportunities")
):
    """Analyze costs and identify optimization opportunities"""
    
    if not v2_components_available:
        console.print("[red]❌ V2.0 components not available for cost analysis[/red]")
        return
    
    console.print(f"💰 Analyzing costs for last {days} day(s)...")
    
    try:
        analysis = get_cost_analysis_cli(days)
        
        # Create cost breakdown table
        table = Table(title=f"Cost Analysis ({days} days)")
        table.add_column("Model", style="cyan")
        table.add_column("Tasks", style="white")
        table.add_column("Total Cost", style="yellow")
        table.add_column("Avg Cost/Task", style="green")
        table.add_column("% of Total", style="blue")
        
        total_cost = analysis.get('total_cost', 0)
        model_breakdown = analysis.get('model_breakdown', {})
        
        for model, data in model_breakdown.items():
            table.add_row(
                model,
                str(data['tasks']),
                f"${data['total_cost']:.4f}",
                f"${data['avg_cost']:.4f}",
                f"{data['percentage']:.1f}%"
            )
        
        console.print(table)
        
        # Show summary metrics
        console.print(Panel(
            f"📊 **Total Cost**: ${total_cost:.4f}\n"
            f"📈 **Average per Task**: ${analysis.get('avg_cost_per_task', 0):.4f}\n"
            f"🔍 **Total Tasks**: {analysis.get('total_tasks', 0)}\n"
            f"💡 **Optimization Opportunities**: {analysis.get('optimization_opportunities', 0)}",
            title="Cost Summary",
            border_style="yellow"
        ))
        
        # Show optimizations if enabled
        if show_optimizations and analysis.get('top_optimizations'):
            console.print("\n💡 **Top Cost Optimization Opportunities**:")
            for i, opt in enumerate(analysis['top_optimizations'], 1):
                savings = opt.get('projected_savings', 0)
                console.print(f"{i}. {opt.get('description', 'N/A')} "
                             f"(💰 Save: ${savings:.3f})")
        
        # Show projected savings
        projected_savings = analysis.get('projected_savings', 0)
        if projected_savings > 0:
            savings_pct = analysis.get('savings_percentage', 0)
            console.print(f"\n🎯 **Total Potential Savings**: ${projected_savings:.3f} ({savings_pct:.1f}%)", 
                         style="bold green")
    
    except Exception as e:
        console.print(f"[red]❌ Error in cost analysis: {e}[/red]")

@app.command()
def pattern_discovery(
    days: int = typer.Option(30, help="Number of days to analyze for patterns"),
    pattern_type: str = typer.Option("all", help="Pattern type: user|context|temporal|all")
):
    """Discover usage patterns and preferences"""
    
    if not v2_components_available:
        console.print("[red]❌ V2.0 components not available for pattern discovery[/red]")
        return
    
    console.print(f"🔍 Discovering patterns from last {days} day(s)...")
    
    try:
        patterns_data = discover_user_patterns_cli(days)
        
        # Display patterns summary
        console.print(Panel(
            f"📈 **Preferences Discovered**: {patterns_data['preferences_count']}\n"
            f"🎯 **Context Patterns**: {patterns_data['context_patterns_count']}\n"
            f"⏰ **Temporal Patterns**: {patterns_data['temporal_patterns_count']}",
            title="Pattern Discovery Summary",
            border_style="magenta"
        ))
        
        # Show top preferences
        if patterns_data.get('preferences'):
            console.print("\n👤 **User Preferences Discovered**:")
            for pref in patterns_data['preferences']:
                confidence = pref.get('confidence', 0)
                pref_type = pref.get('type', 'unknown')
                console.print(f"  • {pref_type}: {confidence:.1%} confidence")
        
        # Show top patterns
        if patterns_data.get('top_patterns'):
            console.print("\n🎯 **Context Patterns Identified**:")
            for pattern in patterns_data['top_patterns']:
                success_rate = pattern.get('success_rate', 0)
                conditions = pattern.get('conditions', {})
                console.print(f"  • {conditions}: {success_rate:.1%} success rate")
    
    except Exception as e:
        console.print(f"[red]❌ Error in pattern discovery: {e}[/red]")

@app.command()
def model_reasoning(
    task_type: str = typer.Argument(help="Task type: chat|code_generation|analysis|pipeline"),
    priority: str = typer.Option("balanced", help="Priority: cost|quality|speed|balanced"),
    show_alternatives: bool = typer.Option(True, help="Show alternative models")
):
    """Show AI reasoning behind model selection"""
    
    if not v2_components_available:
        console.print("[red]❌ V2.0 components not available for model reasoning[/red]")
        return
    
    console.print(f"🧠 Analyzing optimal model for {task_type} with {priority} priority...")
    
    try:
        explanation = explain_model_choice(task_type, priority)
        
        # Display main recommendation
        console.print(Panel(
            f"🤖 **Recommended Model**: {explanation['recommended_model']}\n"
            f"🎯 **Confidence**: {explanation['confidence']:.1%}\n"
            f"💰 **Estimated Cost**: ${explanation['cost_estimate']:.4f}\n\n"
            f"**Reasoning**:\n{explanation['reasoning']}",
            title="AI Model Selection Reasoning",
            border_style="blue"
        ))
        
        # Show alternatives if enabled
        if show_alternatives and explanation.get('alternatives'):
            console.print("\n🔄 **Alternative Models**:")
            for alt_model, score in explanation['alternatives'][:3]:
                console.print(f"  • {alt_model}: {score:.2f} score")
    
    except Exception as e:
        console.print(f"[red]❌ Error generating model reasoning: {e}[/red]")

@app.command()
def success_breakdown(
    task_id: Optional[str] = typer.Option(None, help="Specific task ID to analyze"),
    recent_count: int = typer.Option(10, help="Number of recent tasks to analyze")
):
    """Multi-dimensional success analysis"""
    
    if not v2_components_available:
        console.print("[red]❌ V2.0 components not available for success analysis[/red]")
        return
    
    if task_id:
        console.print(f"📊 Analyzing success metrics for task: {task_id}")
        # Implementation would analyze specific task
        console.print("[yellow]Specific task analysis not implemented in demo[/yellow]")
    else:
        console.print(f"📊 Analyzing success metrics for last {recent_count} tasks...")
        
        try:
            summary = get_success_summary()
            
            # Display success metrics
            console.print(Panel(
                f"📈 **Total Tasks**: {summary['total_tasks']}\n"
                f"✅ **Successful Tasks**: {summary['successful_tasks']}\n"
                f"🎯 **Success Rate**: {summary['overall_success_rate']:.1%}\n",
                title="Success Analysis Summary",
                border_style="green"
            ))
            
            # Show breakdown by level
            if summary.get('level_breakdown'):
                console.print("\n📊 **Success Level Breakdown**:")
                for level, data in summary['level_breakdown'].items():
                    count = data.get('count', 0)
                    avg_score = data.get('avg_score', 0)
                    console.print(f"  • {level}: {count} tasks (avg: {avg_score:.2f})")
        
        except Exception as e:
            console.print(f"[red]❌ Error in success analysis: {e}[/red]")

@app.command() 
def sync_knowledge_graph(
    days: int = typer.Option(7, help="Days of data to sync"),
    force: bool = typer.Option(False, help="Force re-sync existing data")
):
    """Sync SimpleTracker data to Neo4j Knowledge Graph"""
    
    if not v2_components_available:
        console.print("[red]❌ V2.0 components not available for knowledge graph sync[/red]")
        return
    
    console.print(f"🔄 Syncing {days} days of data to Knowledge Graph...")
    
    try:
        sync_result = sync_tracker_to_graph_cli(days)
        
        console.print(Panel(
            f"📊 **Total Tasks**: {sync_result.get('total_tasks', 0)}\n"
            f"✅ **Synced Tasks**: {sync_result.get('synced_tasks', 0)}\n"
            f"🎯 **Success Rate**: {sync_result.get('success_rate', 0):.1%}",
            title="Knowledge Graph Sync Complete",
            border_style="cyan"
        ))
        
        if sync_result.get('error'):
            console.print(f"[red]❌ Sync error: {sync_result['error']}[/red]")
    
    except Exception as e:
        console.print(f"[red]❌ Error syncing to knowledge graph: {e}[/red]")

@app.command()
def status():
    """Show V2.0 Intelligence Layer status and capabilities"""
    
    console.print("🤖 **Agent Zero V1 + V2.0 Intelligence Layer Status**\n")
    
    # Check component availability
    components_status = {
        "Intelligent Model Selector": False,
        "Success/Failure Classifier": False, 
        "Active Metrics Analyzer": False,
        "Enhanced Feedback Loop": False,
        "Kaizen Knowledge Graph": False
    }
    
    if v2_components_available:
        try:
            components_status["Intelligent Model Selector"] = True
            components_status["Success/Failure Classifier"] = True
            components_status["Active Metrics Analyzer"] = True
            components_status["Enhanced Feedback Loop"] = True
            components_status["Kaizen Knowledge Graph"] = True
        except:
            pass
    
    # Display status table
    status_table = Table(title="V2.0 Component Status")
    status_table.add_column("Component", style="cyan")
    status_table.add_column("Status", style="green")
    
    for component, available in components_status.items():
        status = "✅ Available" if available else "❌ Not Available"
        color = "green" if available else "red"
        status_table.add_row(component, f"[{color}]{status}[/{color}]")
    
    console.print(status_table)
    
    # Show capabilities
    if v2_components_available:
        console.print("\n🚀 **Available V2.0 Capabilities**:")
        console.print("   • Intelligent model selection with reasoning")
        console.print("   • Multi-dimensional success evaluation") 
        console.print("   • Real-time cost and performance monitoring")
        console.print("   • Pattern-based learning and optimization")
        console.print("   • Cross-project knowledge sharing")
        console.print("   • Daily Kaizen reports with actionable insights")
        
        # Show quick stats if possible
        try:
            learning_status = get_learning_status_cli()
            console.print(f"\n📚 **Learning System Status**:")
            console.print(f"   • Models tracked: {len(learning_status.get('models_tracked', []))}")
            console.print(f"   • Patterns learned: {learning_status.get('context_patterns_count', 0)}")
            console.print(f"   • User preferences: {learning_status.get('user_preferences_count', 0)}")
        except:
            pass
    else:
        console.print(f"\n[yellow]⚠️ V2.0 Intelligence Layer not fully operational[/yellow]")
        console.print("Run with basic CLI functionality only")

if __name__ == "__main__":
    app()