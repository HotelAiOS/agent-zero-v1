#!/usr/bin/env python3
"""
Agent Zero CLI - Command Line Interface
Usage: a0 [command] [options]
"""

import typer
import requests
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown
from typing import Optional
import os

app = typer.Typer(help="Agent Zero - AI Agent Platform CLI")
console = Console()

API_URL = os.getenv("AGENT_ZERO_API", "http://localhost:18080")
USER_ID = os.getenv("AGENT_ZERO_USER", os.getenv("USER", "cli-user"))

# ==================== CHAT ====================

@app.command()
def ask(
    message: str,
    session: Optional[str] = None,
    model: Optional[str] = None
):
    """
    Zadaj pytanie AI
    
    Examples:
        a0 ask "What is Kubernetes?"
        a0 ask "Continue our conversation" --session=abc-123
    """
    console.print(f"[cyan]🤔 Asking AI...[/cyan]")
    
    payload = {
        "user_id": USER_ID,
        "message": message,
        "task_type": "chat"
    }
    
    if session:
        payload["session_id"] = session
    
    try:
        response = requests.post(f"{API_URL}/chat/", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        
        console.print(Panel(
            Markdown(data['content']),
            title=f"🤖 AI Response ({data.get('model', 'unknown')})",
            border_style="green"
        ))
        
        console.print(f"\n[dim]Session: {data.get('session_id', 'N/A')}[/dim]")
        console.print(f"[dim]Tokens: {data.get('tokens', 0)}[/dim]")
        
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")

@app.command()
def chat(session: Optional[str] = None):
    """
    Rozpocznij interaktywną sesję czatu
    
    Example: a0 chat
    """
    console.print(Panel(
        "[bold green]Agent Zero Interactive Chat[/bold green]\n"
        "Type your messages. Use /exit to quit, /clear for new session.",
        title="🤖 Chat Mode"
    ))
    
    current_session = session
    
    while True:
        try:
            message = console.input("\n[cyan]You:[/cyan] ")
            
            if message.lower() in ["/exit", "/quit", "/q"]:
                console.print("[yellow]Goodbye! 👋[/yellow]")
                break
            
            if message.lower() == "/clear":
                current_session = None
                console.print("[green]✓ New session started[/green]")
                continue
            
            payload = {
                "user_id": USER_ID,
                "message": message,
                "task_type": "chat"
            }
            
            if current_session:
                payload["session_id"] = current_session
            
            response = requests.post(f"{API_URL}/chat/", json=payload, timeout=60)
            data = response.json()
            
            current_session = data.get('session_id')
            
            console.print(f"\n[green]AI:[/green] {data['content']}")
            console.print(f"[dim]({data.get('tokens', 0)} tokens)[/dim]")
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye! 👋[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

@app.command()
def sessions(limit: int = 10):
    """Lista ostatnich sesji czatu"""
    try:
        response = requests.get(f"{API_URL}/chat/sessions/{USER_ID}")
        sessions = response.json()
        
        table = Table(title="Chat Sessions")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="green")
        table.add_column("Created", style="yellow")
        
        for session in sessions[:limit]:
            table.add_row(
                session['id'][:8],
                session.get('title', 'Untitled')[:50],
                session.get('created_at', 'N/A')
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

# ==================== CODE GENERATION ====================

@app.command()
def code(
    description: str,
    language: str = "python",
    save: Optional[str] = None
):
    """
    Generuj kod
    
    Examples:
        a0 code "Function to validate email" --language=python
        a0 code "REST API for users" --save=api.py
    """
    console.print(f"[cyan]⚙️  Generating {language} code...[/cyan]")
    
    payload = {
        "type": "code_generation",
        "description": description,
        "input_data": {
            "language": language
        }
    }
    
    try:
        response = requests.post(f"{API_URL}/agents/task", json=payload, timeout=120)
        data = response.json()
        
        code = data['result']['code']
        
        console.print(Panel(
            code,
            title=f"💻 Generated {language.upper()} Code",
            border_style="blue"
        ))
        
        if save:
            with open(save, 'w') as f:
                f.write(code)
            console.print(f"[green]✓ Saved to {save}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@app.command()
def pipeline(
    description: str,
    language: str = "python",
    output_dir: str = "."
):
    """
    Pełny pipeline: kod + testy + dokumentacja
    
    Example: a0 pipeline "Calculator API" --output-dir=./calculator
    """
    console.print("[cyan]🚀 Running full pipeline...[/cyan]")
    
    try:
        response = requests.post(
            f"{API_URL}/agents/workflow/code-to-production",
            params={
                "description": description,
                "language": language
            },
            timeout=360
        )
        data = response.json()
        
        # Zapisz pliki
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/code.{language}", 'w') as f:
            f.write(data['code'])
        
        with open(f"{output_dir}/test.py", 'w') as f:
            f.write(data['tests'])
        
        with open(f"{output_dir}/README.md", 'w') as f:
            f.write(data['documentation'])
        
        console.print(f"[green]✓ Pipeline completed![/green]")
        console.print(f"[green]📁 Files saved to {output_dir}/[/green]")
        console.print(f"  - code.{language}")
        console.print(f"  - test.py")
        console.print(f"  - README.md")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

# ==================== AGENTS ====================

@app.command()
def agents():
    """Lista dostępnych agentów"""
    try:
        response = requests.get(f"{API_URL}/agents/")
        agents = response.json()
        
        table = Table(title="Available Agents")
        table.add_column("Name", style="cyan")
        table.add_column("Capabilities", style="green")
        table.add_column("Status", style="yellow")
        
        for agent in agents:
            table.add_row(
                agent['name'],
                ", ".join(agent['capabilities']),
                "✓ Active"
            )
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

# ==================== SYSTEM ====================

@app.command()
def status():
    """Status systemu"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        data = response.json()
        
        status_text = f"""
[bold green]✓ System Healthy[/bold green]

Version: {data.get('version', 'unknown')}
Gateway: {data.get('gateway', 'unknown')}
        """
        
        console.print(Panel(status_text, title="🏥 System Status", border_style="green"))
        
    except Exception as e:
        console.print(f"[red]❌ System Offline: {e}[/red]")

@app.command()
def models():
    """Lista dostępnych modeli AI"""
    try:
        response = requests.get(f"{API_URL}/ai/models")
        models = response.json()['models']
        
        table = Table(title="AI Models")
        table.add_column("Name", style="cyan")
        table.add_column("Size", style="yellow")
        
        for model in models:
            table.add_row(model['name'], model.get('size', 'N/A'))
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@app.command()
def config():
    """Pokaż konfigurację CLI"""
    console.print(Panel(
        f"[bold]Agent Zero CLI Configuration[/bold]\n\n"
        f"API URL: {API_URL}\n"
        f"User ID: {USER_ID}\n\n"
        f"[dim]Set via environment variables:[/dim]\n"
        f"[dim]  export AGENT_ZERO_API=http://localhost:18080[/dim]\n"
        f"[dim]  export AGENT_ZERO_USER=your-name[/dim]",
        title="⚙️  Configuration"
    ))

# ==================== MAIN ====================
# ==================== LEARNING & ANALYTICS ====================

@app.command()
def feedback(
    session_id: str,
    message_id: str,
    rating: int = typer.Option(..., min=1, max=5, help="Rating 1-5"),
    comment: Optional[str] = None
):
    """
    Wyślij feedback o odpowiedzi AI
    
    Example: a0 feedback SESSION_ID MSG_ID --rating=5 --comment="Great!"
    """
    feedback_type = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
    
    try:
        response = requests.post(
            f"{API_URL}/chat/feedback",
            json={
                "session_id": session_id,
                "message_id": message_id,
                "feedback_type": feedback_type,
                "rating": rating,
                "comment": comment
            }
        )
        response.raise_for_status()
        
        console.print(f"[green]✓ Feedback recorded![/green]")
        console.print(f"Rating: {'⭐' * rating}")
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@app.command()
def insights(insight_type: Optional[str] = None):
    """
    Zobacz learning insights - co AI nauczyło się
    
    Example: a0 insights
    """
    try:
        params = {}
        if insight_type:
            params['insight_type'] = insight_type
        
        response = requests.get(f"{API_URL}/chat/insights", params=params)
        data = response.json()
        
        console.print(Panel(
            "[bold]Learning Insights[/bold]\n\n" + 
            str(data.get('insights', [])),
            title="🧠 What AI Learned"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

@app.command()
def analytics(user_id: Optional[str] = None):
    """
    Zobacz analytics użytkownika
    
    Example: a0 analytics --user-id=jan
    """
    uid = user_id or USER_ID
    
    try:
        # Get user patterns
        response = requests.get(f"{API_URL}/chat/user/{uid}/patterns")
        patterns = response.json()
        
        # Get sessions
        sessions_response = requests.get(f"{API_URL}/chat/sessions/{uid}")
        sessions = sessions_response.json()
        
        console.print(Panel(
            f"[bold]User Analytics: {uid}[/bold]\n\n"
            f"Total Sessions: {len(sessions)}\n"
            f"Patterns: {patterns.get('patterns', {})}",
            title="📊 Analytics"
        ))
        
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")

def main():
    """Entry point for CLI"""
    app()

if __name__ == "__main__":
    main()
