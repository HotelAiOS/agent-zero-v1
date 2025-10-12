import asyncio
import signal
import sys
import os
from ai_streaming import streaming_ai_brain, ThinkingStage

class LiveDashboard:
    def __init__(self):
        self.running = False
        self.current_task = None
        
    async def run_dashboard(self):
        """Main interactive dashboard"""
        
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🎛️  LIVE AI MONITORING DASHBOARD")
        print("=" * 60)
        print("✨ Real-time AI thinking visualization")
        print("🎮 Full operator control and monitoring")
        print("=" * 60)
        print()
        print("📋 COMMANDS:")
        print("   • Type any task to start AI")
        print("   • Type 'cancel' to stop current task")
        print("   • Type 'status' to see system status")
        print("   • Type 'quit' to exit dashboard")
        print("   • Press Ctrl+C for emergency stop")
        print("=" * 60)
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        while True:
            try:
                print()
                command = input("🎛️ OPERATOR > ").strip()
                
                if not command:
                    continue
                    
                if command.lower() == 'quit':
                    print("\n👋 Shutting down dashboard...")
                    break
                    
                elif command.lower() == 'cancel':
                    print("🛑 CANCELLATION REQUESTED...")
                    streaming_ai_brain.cancel_current_task()
                    
                elif command.lower() == 'status':
                    status = "ACTIVE" if self.current_task else "IDLE"
                    print(f"📊 SYSTEM STATUS: {status}")
                    if self.current_task:
                        print(f"📋 Current task: {self.current_task[:50]}...")
                        
                elif command.lower().startswith('help'):
                    self._show_help()
                    
                else:
                    # Start new AI task
                    await self._run_live_task(command)
                    
            except KeyboardInterrupt:
                print("\n🛑 EMERGENCY STOP REQUESTED")
                streaming_ai_brain.cancel_current_task()
                break
                
            except Exception as e:
                print(f"❌ Dashboard error: {e}")
        
        print("\n🚪 Dashboard closed. Goodbye!")
    
    async def _run_live_task(self, task: str):
        """Run AI task with live monitoring"""
        
        print(f"\n🚀 STARTING AI TASK")
        print("─" * 60)
        print(f"📋 Task: {task}")
        print("─" * 60)
        
        self.current_task = task
        last_progress = 0
        
        try:
            async for update in streaming_ai_brain.think_with_stream(task):
                
                # Clear previous line and show status
                print(f"\r{' ' * 100}", end="")
                print(f"\r{update.message}", end="", flush=True)
                
                # Show progress bar if progress changed significantly
                if update.progress - last_progress > 0.05 or update.progress == 1.0:
                    print()  # New line
                    self._show_progress_bar(update)
                    last_progress = update.progress
                
                # Show metadata for important stages
                if update.stage in [ThinkingStage.MODEL_SELECTION, ThinkingStage.GENERATING]:
                    if update.metadata:
                        key_info = []
                        if "selected_model" in update.metadata:
                            key_info.append(f"Model: {update.metadata['selected_model']}")
                        if "complexity" in update.metadata:
                            key_info.append(f"Type: {update.metadata['complexity']}")
                        if "model_size" in update.metadata:
                            key_info.append(f"Size: {update.metadata['model_size']}")
                        
                        if key_info:
                            print(f"\n💡 {' | '.join(key_info)}")
                
                # Show partial output for generation stage
                if update.stage == ThinkingStage.GENERATING and update.current_output and len(update.current_output) > 20:
                    preview = update.current_output[:150].replace('\n', ' ')
                    print(f"\n📝 Preview: {preview}...")
                
                # Handle completion states
                if update.stage == ThinkingStage.COMPLETED:
                    print(f"\n\n🎉 SUCCESS! Task completed")
                    print("=" * 60)
                    print(f"⏱️  Total time: {update.elapsed_time:.1f} seconds")
                    print(f"📊 Confidence: {update.confidence:.2f}")
                    print(f"🤖 Model used: {update.metadata.get('model_used', 'Unknown')}")
                    print(f"📝 Response length: {update.metadata.get('response_length', 0)} characters")
                    print("=" * 60)
                    
                    # Show response preview
                    if update.current_output:
                        print("📄 RESPONSE PREVIEW:")
                        print("-" * 40)
                        preview_lines = update.current_output[:500].split('\n')[:15]
                        for line in preview_lines:
                            print(line)
                        if len(update.current_output) > 500:
                            print("... (output truncated)")
                        print("-" * 40)
                    break
                    
                elif update.stage == ThinkingStage.CANCELLED:
                    print(f"\n\n🛑 CANCELLED after {update.elapsed_time:.1f} seconds")
                    print("Task was stopped by operator")
                    break
                    
                elif update.stage == ThinkingStage.ERROR:
                    print(f"\n\n❌ ERROR occurred")
                    print(f"🔍 Details: {update.metadata.get('error', 'Unknown error')}")
                    print(f"⏱️  Failed after: {update.elapsed_time:.1f} seconds")
                    break
                
                # Small delay for readability
                await asyncio.sleep(0.2)
                
        except Exception as e:
            print(f"\n❌ Dashboard error during task: {e}")
        finally:
            self.current_task = None
    
    def _show_progress_bar(self, update):
        """Show visual progress bar"""
        bar_length = 40
        filled = int(update.progress * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        minutes = int(update.elapsed_time // 60)
        seconds = int(update.elapsed_time % 60)
        
        print(f"[{bar}] {update.progress*100:.0f}% | {minutes:02d}:{seconds:02d} | Confidence: {update.confidence:.2f}")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 HELP - Available Commands:")
        print("   🎯 [any text] - Start AI task with that text")
        print("   🛑 cancel - Cancel current running task")
        print("   📊 status - Show system status")
        print("   ❓ help - Show this help")
        print("   👋 quit - Exit dashboard")
        print("\n💡 Tips:")
        print("   • AI tasks can take 5-20 minutes depending on complexity")
        print("   • You can cancel anytime if AI goes in wrong direction")
        print("   • First model load may take longer (downloading)")
        print("   • Watch for confidence score - higher is better")
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C"""
        print(f"\n🚨 EMERGENCY STOP - Operator interrupt")
        streaming_ai_brain.cancel_current_task()
        print("💡 Type 'quit' to exit dashboard")

async def main():
    """Main entry point"""
    print("🎛️ Starting Live AI Dashboard...")
    dashboard = LiveDashboard()
    await dashboard.run_dashboard()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Dashboard terminated by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
