#!/usr/bin/env python3
"""
Agent Zero V1 CLI - Enhanced with Kaizen Feedback Loop
Critical enhancement: After every task, collect user feedback for continuous learning
"""

import typer
import uuid
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional
from pathlib import Path
import sqlite3

# Import existing modules (assuming they exist)
# from services.ai_router.src.router.orchestrator import AIOrchestrator
# from shared.utils.simple_tracker import SimpleTracker

app = typer.Typer()
console = Console()

# Initialize simple tracker for immediate Kaizen feedback
class SimpleTracker:
    """Minimal tracking until full Kaizen system ready"""
    
    def __init__(self, db_path: str = ".agent-zero/tracker.db"):
        self.db_path = Path.home() / db_path
        self.db_path.parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.init_schema()
    
    def init_schema(self):
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                task_type TEXT,
                model_used TEXT,
                model_recommended TEXT,
                cost_usd REAL,
                latency_ms INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        self.conn.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                task_id TEXT,
                rating INTEGER,
                comment TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (task_id) REFERENCES tasks(id)
            )
        ''')
        self.conn.commit()
    
    def track_task(self, task_id: str, task_type: str, model_used: str, 
                   model_recommended: str, cost: float, latency: int):
        self.conn.execute(
            "INSERT INTO tasks (id, task_type, model_used, model_recommended, cost_usd, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
            (task_id, task_type, model_used, model_recommended, cost, latency)
        )
        self.conn.commit()
    
    def record_feedback(self, task_id: str, rating: int, comment: str = None):
        self.conn.execute(
            "INSERT INTO feedback (task_id, rating, comment) VALUES (?, ?, ?)",
            (task_id, rating, comment)
        )
        self.conn.commit()

# Global tracker instance
tracker = SimpleTracker()

def ask_for_quick_feedback(task_id: str, model_used: str) -> None:
    """
    CRITICAL KAIZEN COMPONENT: Simple 1-click feedback after task completion
    This is the foundation of our learning system!
    """
    console.print("\n" + "="*50)
    console.print("🔄 Quick feedback (optional):", style="bold blue")
    console.print("Rate this result: [1-5] or Enter to skip")
    console.print("="*50)
    
    try:
        rating = input("Rating: ").strip()
        if rating and rating.isdigit() and 1 <= int(rating) <= 5:
            # Ask for optional comment if rating is low
            comment = None
            if int(rating) <= 2:
                comment = input("What went wrong? (optional): ").strip() or None
            
            tracker.record_feedback(task_id, int(rating), comment)
            console.print("✅ Thanks! This helps us improve.", style="green")
            
            # Show immediate learning insight
            if int(rating) <= 2:
                console.print(f"⚠️  Low rating noted for {model_used}. We'll learn from this!", style="yellow")
        else:
            console.print("⏭️  Skipped feedback", style="dim")
    except KeyboardInterrupt:
        console.print("\n⏭️  Feedback skipped", style="dim")
    except Exception:
        pass  # Skip silently on any error

@app.command()
def ask(question: str, provider: Optional[str] = None):
    """Ask a question to Agent Zero with Kaizen feedback collection"""
    
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    console.print(f"🤖 Processing question: {question}")
    
    # TODO: Replace with actual AI orchestrator call
    # orchestrator = AIOrchestrator()
    # response = orchestrator.process_question(question, provider)
    
    # MOCK IMPLEMENTATION - replace with actual logic
    model_used = provider or "llama3.2-3b"  # Default model
    model_recommended = "llama3.2-3b"  # Would come from intelligent selector
    
    # Simulate processing
    import time
    time.sleep(1)  # Simulate processing time
    
    response = f"Mock response to: {question} (using {model_used})"
    
    # Calculate metrics
    end_time = datetime.now()
    latency_ms = int((end_time - start_time).total_seconds() * 1000)
    cost_usd = 0.001  # Mock cost - would be calculated from actual API usage
    
    # Track the task
    tracker.track_task(
        task_id=task_id,
        task_type="chat",
        model_used=model_used,
        model_recommended=model_recommended,
        cost=cost_usd,
        latency=latency_ms
    )
    
    # Display response
    console.print(Panel(response, title="Agent Zero Response", border_style="green"))
    
    # CRITICAL: Ask for feedback after every task
    ask_for_quick_feedback(task_id, model_used)

@app.command()
def code(description: str, provider: Optional[str] = None):
    """Generate code with Kaizen feedback collection"""
    
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    console.print(f"💻 Generating code: {description}")
    
    # TODO: Replace with actual code generation logic
    model_used = provider or "qwen2.5-coder:7b"  # Default for code
    model_recommended = "qwen2.5-coder:7b"
    
    # Mock code generation
    import time
    time.sleep(2)  # Simulate processing
    
    code_output = f'''# Generated code for: {description}
def example_function():
    """Mock generated code using {model_used}"""
    return "Hello, World!"
    
# TODO: Implement actual functionality
print(example_function())
'''
    
    # Calculate metrics
    end_time = datetime.now()
    latency_ms = int((end_time - start_time).total_seconds() * 1000)
    cost_usd = 0.005  # Code generation is more expensive
    
    # Track the task
    tracker.track_task(
        task_id=task_id,
        task_type="code_generation",
        model_used=model_used,
        model_recommended=model_recommended,
        cost=cost_usd,
        latency=latency_ms
    )
    
    # Display code
    console.print(Panel(code_output, title="Generated Code", border_style="cyan"))
    
    # CRITICAL: Ask for feedback
    ask_for_quick_feedback(task_id, model_used)

@app.command()
def pipeline(description: str, provider: Optional[str] = None):
    """Execute a complex pipeline with Kaizen feedback"""
    
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    console.print(f"🔧 Executing pipeline: {description}")
    
    # TODO: Replace with actual pipeline logic
    model_used = provider or "llama3.2-3b"
    model_recommended = "llama3.2-3b"
    
    # Mock pipeline execution
    import time
    time.sleep(3)  # Simulate complex processing
    
    pipeline_output = f"Pipeline '{description}' executed successfully using {model_used}"
    
    # Calculate metrics
    end_time = datetime.now()
    latency_ms = int((end_time - start_time).total_seconds() * 1000)
    cost_usd = 0.010  # Pipeline operations can be expensive
    
    # Track the task
    tracker.track_task(
        task_id=task_id,
        task_type="pipeline",
        model_used=model_used,
        model_recommended=model_recommended,
        cost=cost_usd,
        latency=latency_ms
    )
    
    # Display result
    console.print(Panel(pipeline_output, title="Pipeline Result", border_style="magenta"))
    
    # CRITICAL: Ask for feedback
    ask_for_quick_feedback(task_id, model_used)

@app.command()
def compare_models():
    """
    KAIZEN GOLD: Show which model has best quality/cost ratio
    This gives immediate visibility into what's working!
    """
    
    console.print("📊 Model Performance Analysis (Last 7 days)")
    
    # Query performance data
    cursor = tracker.conn.execute('''
        SELECT 
            t.model_used,
            COUNT(*) as usage_count,
            AVG(t.cost_usd) as avg_cost,
            AVG(f.rating) as avg_rating,
            COUNT(f.rating) as feedback_count,
            SUM(CASE WHEN t.model_used != t.model_recommended THEN 1 ELSE 0 END) as overrides
        FROM tasks t 
        LEFT JOIN feedback f ON t.id = f.task_id 
        WHERE t.timestamp >= datetime('now', '-7 days')
        GROUP BY t.model_used 
        ORDER BY avg_rating DESC, avg_cost ASC
    ''')
    
    results = cursor.fetchall()
    
    if not results:
        console.print("❌ No data available. Start using the system to see insights!", style="yellow")
        return
    
    # Create performance table
    table = Table(title="Model Performance (Last 7 days)")
    table.add_column("Model", style="cyan")
    table.add_column("Usage", style="white")
    table.add_column("Avg Cost", style="yellow")
    table.add_column("Avg Rating", style="green")
    table.add_column("Feedback Count", style="blue")
    table.add_column("Overrides", style="red")
    table.add_column("Recommendation", style="bold")
    
    best_score = 0
    best_model = None
    
    for row in results:
        model, usage, avg_cost, avg_rating, feedback_count, overrides = row
        
        # Calculate performance score (higher rating, lower cost = better)
        score = (avg_rating or 0) * 10 - (avg_cost or 0) * 100
        if score > best_score:
            best_score = score
            best_model = model
        
        recommendation = "⭐ BEST" if model == best_model else ""
        
        table.add_row(
            model,
            str(usage),
            f"${avg_cost:.4f}" if avg_cost else "N/A",
            f"{avg_rating:.1f}/5" if avg_rating else "No feedback",
            str(feedback_count),
            str(overrides),
            recommendation
        )
    
    console.print(table)
    
    # Actionable insights
    if best_model:
        console.print(f"\n🎯 **Recommendation**: Use `{best_model}` for best quality/cost ratio", style="bold green")
    
    # Show top issues
    cursor = tracker.conn.execute('''
        SELECT model_used, AVG(rating) as avg_rating 
        FROM tasks t JOIN feedback f ON t.id = f.task_id 
        WHERE f.rating <= 2 AND t.timestamp >= datetime('now', '-7 days')
        GROUP BY model_used
    ''')
    
    poor_performers = cursor.fetchall()
    if poor_performers:
        console.print("\n⚠️  **Issues detected**:", style="bold red")
        for model, rating in poor_performers:
            console.print(f"   • {model}: Low rating ({rating:.1f}/5) - needs attention")

@app.command()
def kaizen_status():
    """Show current Kaizen learning status"""
    
    console.print("🧠 Kaizen Learning Status")
    
    # Basic stats
    cursor = tracker.conn.execute('''
        SELECT 
            COUNT(DISTINCT t.id) as total_tasks,
            COUNT(f.task_id) as feedback_count,
            AVG(f.rating) as avg_rating,
            SUM(t.cost_usd) as total_cost
        FROM tasks t 
        LEFT JOIN feedback f ON t.id = f.task_id 
        WHERE t.timestamp >= datetime('now', '-7 days')
    ''')
    
    stats = cursor.fetchone()
    total_tasks, feedback_count, avg_rating, total_cost = stats
    
    feedback_rate = (feedback_count / total_tasks * 100) if total_tasks > 0 else 0
    
    console.print(f"📈 **Last 7 days:**")
    console.print(f"   • Total tasks: {total_tasks}")
    console.print(f"   • Feedback rate: {feedback_rate:.1f}%")
    console.print(f"   • Average rating: {avg_rating:.1f}/5" if avg_rating else "   • Average rating: No feedback yet")
    console.print(f"   • Total cost: ${total_cost:.4f}" if total_cost else "   • Total cost: $0.0000")
    
    # Learning insights
    if feedback_count > 5:
        console.print("\n🎓 **Learning Insights:**")
        console.print("   • System is collecting feedback ✅")
        console.print("   • Ready for pattern recognition")
        console.print("   • Consider implementing full Kaizen system")
    else:
        console.print("\n🌱 **Early Stage:**")
        console.print("   • Keep using the system to generate insights")
        console.print("   • Provide feedback to help the system learn")

if __name__ == "__main__":
    app()