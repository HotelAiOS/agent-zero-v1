# Enhanced LiveMonitor with Keyboard Controls
# Add this to the end of shared/monitoring/livemonitor.py

import sys
import termios
import tty
import select
from threading import Thread

class KeyboardController:
    """Handles keyboard input for user controls"""
    
    def __init__(self, live_monitor):
        self.live_monitor = live_monitor
        self.running = False
        self.thread = None
        
    def start(self):
        """Start keyboard listener"""
        self.running = True
        self.thread = Thread(target=self._listen_keyboard, daemon=True)
        self.thread.start()
        
    def stop(self):
        """Stop keyboard listener"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
    def _listen_keyboard(self):
        """Listen for keyboard input"""
        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)
        
        try:
            tty.setcbreak(sys.stdin.fileno())
            
            while self.running:
                # Check if input is available
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1).lower()
                    
                    if key == '''s''':  # Stop
                        self.live_monitor.should_stop = True
                        self.live_monitor.console.print("
[yellow]⏹️  User requested STOP[/yellow]")
                        
                    elif key == '''p''':  # Pause
                        self.live_monitor.is_paused = True
                        self.live_monitor.console.print("
[yellow]⏸️  Execution PAUSED[/yellow]")
                        
                    elif key == '''c''':  # Continue
                        self.live_monitor.is_paused = False
                        self.live_monitor.console.print("
[green]▶️  Execution RESUMED[/green]")
                        
                    elif key == '''r''':  # Retry
                        self.live_monitor.console.print("
[blue]🔄 Retry requested[/blue]")
                        # Will be handled by orchestrator
                        
                    elif key == '''h''':  # Help
                        self._show_help()
                        
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            
    def _show_help(self):
        """Display help message"""
        from rich.panel import Panel
        
        help_text = """
🎮 Interactive Controls:

[S] - Stop execution immediately
[P] - Pause execution (can resume)
[C] - Continue after pause
[R] - Retry current task
[H] - Show this help
[Q] - Quit monitoring (execution continues)

💡 Tip: Press any key to interact
        """
        
        self.live_monitor.console.print(Panel(
            help_text,
            title="Controls Help",
            border_style="blue"
        ))
