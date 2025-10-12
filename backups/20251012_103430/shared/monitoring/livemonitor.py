# Agent Zero v1 - Phase 2: Interactive Control
# LiveMonitor - Enhanced with Token Streaming

import asyncio
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field

import aiofiles
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.text import Text
from rich.align import Align


class AgentStatus(Enum):
    """Agent execution status"""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING = "waiting"
    COMPLETED = "completed"
    ERROR = "error"
    PAUSED = "paused"
    STOPPED = "stopped"


class UserCommand(Enum):
    """User intervention commands"""
    STOP = "s"
    PAUSE = "p"
    CONTINUE = "c"
    RETRY = "r"
    SKIP = "k"
    HELP = "h"
    STATUS = "?"


@dataclass
class AgentUpdate:
    """Real-time update from agent execution"""
    agent_id: str
    agent_type: str
    task_id: str
    project_name: str
    status: AgentStatus
    thinking_text: Optional[str] = None
    progress_percent: float = 0.0
    current_step: Optional[str] = None
    total_steps: int = 1
    step_number: int = 1
    tokens_generated: int = 0
    time_elapsed: float = 0.0
    needs_clarification: bool = False
    clarification_question: Optional[str] = None
    error_message: Optional[str] = None
    artifacts_created: List[str] = field(default_factory=list)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0


@dataclass
class ProjectCheckpoint:
    """Project execution checkpoint"""
    project_name: str
    checkpoint_id: str
    timestamp: datetime
    completed_tasks: List[str]
    current_phase: str
    agent_states: Dict[str, Dict]
    execution_context: Dict


class CheckpointManager:
    """Manages project checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        
    async def save_checkpoint(self, checkpoint) -> str:
        """Save checkpoint - accepts dict or ProjectCheckpoint"""
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if isinstance(checkpoint, dict):
            project_name = checkpoint.get("project_name", "unknown")
            checkpoint_id = checkpoint.get("checkpoint_id", "checkpoint")
            checkpoint_data = checkpoint
        else:
            project_name = checkpoint.project_name
            checkpoint_id = checkpoint.checkpoint_id
            checkpoint_data = {
                "project_name": checkpoint.project_name,
                "checkpoint_id": checkpoint.checkpoint_id,
                "timestamp": checkpoint.timestamp.isoformat(),
                "completed_tasks": checkpoint.completed_tasks,
                "current_phase": checkpoint.current_phase,
                "agent_states": checkpoint.agent_states,
                "execution_context": checkpoint.execution_context
            }
        
        checkpoint_file = f"{self.checkpoint_dir}/{project_name}_{checkpoint_id}.json"
        
        async with aiofiles.open(checkpoint_file, "w") as f:
            await f.write(json.dumps(checkpoint_data, indent=2, default=str))
            
        return checkpoint_file
        
    async def load_checkpoint(self, project_name: str, checkpoint_id: str) -> Optional[ProjectCheckpoint]:
        """Load checkpoint from disk"""
        checkpoint_file = f"{self.checkpoint_dir}/{project_name}_{checkpoint_id}.json"
        
        try:
            async with aiofiles.open(checkpoint_file, "r") as f:
                data = json.loads(await f.read())
                
            return ProjectCheckpoint(
                project_name=data["project_name"],
                checkpoint_id=data["checkpoint_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                completed_tasks=data["completed_tasks"],
                current_phase=data["current_phase"],
                agent_states=data["agent_states"],
                execution_context=data["execution_context"]
            )
        except FileNotFoundError:
            return None
            
    async def list_checkpoints(self, project_name: str) -> List[str]:
        """List all checkpoints"""
        import os
        import glob
        
        if not os.path.exists(self.checkpoint_dir):
            return []
            
        pattern = f"{self.checkpoint_dir}/{project_name}_*.json"
        files = glob.glob(pattern)
        
        return [
            os.path.basename(f).replace(f"{project_name}_", "").replace(".json", "")
            for f in files
        ]


class TerminalDashboard:
    """Terminal dashboard for monitoring"""
    
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.agent_updates: Dict[str, AgentUpdate] = {}
        self.start_time = time.time()
        self.live_output = ""  # NEW: Accumulated live output
        
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", size=12),
            Layout(name="live_output", size=8),  # NEW: Live token output
            Layout(name="footer", size=3)
        )
        
    def _create_header(self, project_name: str) -> Panel:
        """Create header"""
        elapsed = time.time() - self.start_time
        elapsed_str = f"{elapsed//60:.0f}m {elapsed%60:.0f}s"
        
        header_text = Text()
        header_text.append("🤖 Agent Zero v1", style="bold cyan")
        header_text.append(f" | {project_name}", style="green")
        header_text.append(f" | {elapsed_str}", style="white")
        
        return Panel(Align.center(header_text), style="cyan")
        
    def _create_main_panel(self) -> Panel:
        """Create main content"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Agent", width=15)
        table.add_column("Status", width=12)
        table.add_column("Progress", width=10)
        
        for agent_id, update in self.agent_updates.items():
            table.add_row(
                agent_id[:13],
                update.status.value,
                f"{update.progress_percent:.0f}%"
            )
            
        return Panel(table, title="Agents", border_style="cyan")
    
    def _create_live_output_panel(self) -> Panel:
        """NEW: Create live token output panel"""
        # Show last 500 characters
        display_text = self.live_output[-500:] if self.live_output else "Waiting for output..."
        return Panel(
            Text(display_text, style="cyan"),
            title="[bold]🔴 Live Output[/bold]",
            subtitle="[dim]Token Stream[/dim]",
            border_style="green"
        )
        
    def _create_footer(self) -> Panel:
        """Create footer"""
        footer_text = Text()
        footer_text.append("Controls: [S]top [P]ause [C]ontinue [H]elp", style="bold white")
        return Panel(Align.center(footer_text), style="white")
        
    def update_display(self, project_name: str):
        """Update dashboard"""
        self.layout["header"].update(self._create_header(project_name))
        self.layout["main"].update(self._create_main_panel())
        self.layout["live_output"].update(self._create_live_output_panel())  # NEW
        self.layout["footer"].update(self._create_footer())
        
    def add_agent_update(self, update: AgentUpdate):
        """Add agent update"""
        self.agent_updates[update.agent_id] = update
    
    def append_live_output(self, text: str):
        """NEW: Append to live output"""
        self.live_output += text


class LiveMonitor:
    """Real-time agent monitoring system with token streaming"""
    
    def __init__(self, checkpoint_manager: Optional[CheckpointManager] = None):
        self.checkpoint_manager = checkpoint_manager or CheckpointManager()
        self.dashboard = TerminalDashboard()
        self.console = Console()
        
        self.is_monitoring = False
        self.is_paused = False
        self.should_stop = False
        self.current_project = ""  # NEW
        self.user_command_queue = asyncio.Queue()
        self.clarification_responses = {}
        self.update_subscribers: List[Callable[[AgentUpdate], None]] = []
        
        self._token_queue = asyncio.Queue()  # NEW: Token streaming queue
        
        self.performance_metrics = {
            'total_tokens': 0,
            'avg_response_time': 0.0,
            'peak_memory_mb': 0.0,
            'errors_count': 0
        }
        
    def subscribe_to_updates(self, callback: Callable[[AgentUpdate], None]):
        """Subscribe to updates"""
        self.update_subscribers.append(callback)
        
    def unsubscribe_from_updates(self, callback: Callable[[AgentUpdate], None]):
        """Unsubscribe from updates"""
        if callback in self.update_subscribers:
            self.update_subscribers.remove(callback)
    
    # NEW METHODS FOR TOKEN STREAMING
    
    async def stream_token(self, token: str):
        """Push token to streaming queue"""
        await self._token_queue.put(token)
        self.dashboard.append_live_output(token)
    
    async def _stream_agent_outputs(self):
        """Generator streamujący tokeny od agentów w czasie rzeczywistym"""
        while not self.should_stop:
            try:
                # Get token with timeout
                chunk = await asyncio.wait_for(self._token_queue.get(), timeout=0.1)
                yield chunk
            except asyncio.TimeoutError:
                continue
    
    async def _send_ws_update(self, chunk: str):
        """Wyślij token do WebSocket dashboardu"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=0.5) as client:
                await client.post(
                    "http://localhost:8000/ws/token",
                    json={"token": chunk, "project": self.current_project}
                )
        except:
            pass  # WebSocket offline - OK
    
    async def start_monitoring_live(self, project_name: str):
        """Monitoring z live streaming tokenów"""
        self.current_project = project_name
        self.should_stop = False
        self.is_monitoring = True
        self.dashboard.start_time = time.time()
        
        self.console.print(f"[bold green]🎬 Live Monitoring: {project_name}[/bold green]")
        self.console.print("[yellow]Naciśnij Ctrl+C aby zatrzymać test[/yellow]\n")
        
        try:
            with Live(self.dashboard.layout, console=self.console, refresh_per_second=10) as live:
                while self.is_monitoring and not self.should_stop:
                    self.dashboard.update_display(project_name)
                    
                    # Process token queue
                    while not self._token_queue.empty():
                        try:
                            chunk = self._token_queue.get_nowait()
                            self.dashboard.append_live_output(chunk)
                            await self._send_ws_update(chunk)
                        except asyncio.QueueEmpty:
                            break
                    
                    await asyncio.sleep(0.1)
                    
        except KeyboardInterrupt:
            self.console.print("\n[red bold]⏹️ Test przerwany przez użytkownika[/red bold]")
            self.should_stop = True
        
        self.console.print(f"\n[green]✅ Monitoring zakończony[/green]")
            
    async def start_monitoring(self, project_name: str) -> None:
        """Start monitoring (standard version)"""
        self.is_monitoring = True
        self.dashboard.start_time = time.time()
        
        try:
            with Live(self.dashboard.layout, console=self.console, refresh_per_second=2) as live:
                while self.is_monitoring and not self.should_stop:
                    self.dashboard.update_display(project_name)
                    await asyncio.sleep(0.5)
        except KeyboardInterrupt:
            self.should_stop = True
                
    async def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.is_monitoring = False
        
    async def send_agent_update(self, update: AgentUpdate) -> None:
        """Send agent update"""
        self.dashboard.add_agent_update(update)
        
        self.performance_metrics['total_tokens'] += update.tokens_generated
        if update.memory_usage_mb > self.performance_metrics['peak_memory_mb']:
            self.performance_metrics['peak_memory_mb'] = update.memory_usage_mb
            
        if update.status == AgentStatus.ERROR:
            self.performance_metrics['errors_count'] += 1
            
        for callback in self.update_subscribers:
            try:
                callback(update)
            except Exception as e:
                self.console.print(f"[red]Error in subscriber: {e}[/red]")
                
    async def prompt_user(self, question: str) -> str:
        """Prompt user for input"""
        self.console.clear()
        self.console.print(Panel(
            f"🤔 Agent needs clarification:\n\n{question}",
            title="User Input Required",
            border_style="yellow"
        ))
        
        response = input("\n💬 Your response: ").strip()
        self.console.print(f"[green]✓ Response recorded[/green]")
        await asyncio.sleep(1)
        
        return response
        
    async def create_checkpoint(self, project_name: str, completed_tasks: List[str], 
                              current_phase: str, agent_states: Dict[str, Dict],
                              execution_context: Dict) -> str:
        """Create checkpoint"""
        checkpoint_id = f"checkpoint_{int(time.time())}"
        
        checkpoint = ProjectCheckpoint(
            project_name=project_name,
            checkpoint_id=checkpoint_id,
            timestamp=datetime.now(),
            completed_tasks=completed_tasks,
            current_phase=current_phase,
            agent_states=agent_states,
            execution_context=execution_context
        )
        
        checkpoint_file = await self.checkpoint_manager.save_checkpoint(checkpoint)
        self.console.print(f"[green]✓ Checkpoint saved: {checkpoint_file}[/green]")
        return checkpoint_id
        
    async def restore_from_checkpoint(self, project_name: str, checkpoint_id: str) -> Optional[ProjectCheckpoint]:
        """Restore from checkpoint"""
        checkpoint = await self.checkpoint_manager.load_checkpoint(project_name, checkpoint_id)
        
        if checkpoint:
            self.console.print(f"[green]✓ Checkpoint restored[/green]")
        else:
            self.console.print(f"[red]✗ Checkpoint not found[/red]")
            
        return checkpoint
        
    def get_performance_metrics(self) -> Dict:
        """Get metrics"""
        return self.performance_metrics.copy()
