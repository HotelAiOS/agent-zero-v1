import asyncio
import signal
import sys
import os
from token_streaming_client import token_streaming_ollama
from simple_classifier import classifier

class TokenDashboard:
    def __init__(self):
        self.running = False
        self.current_task = None
        self.cancel_requested = False
        
    async def run_token_dashboard(self):
        """Real-time token-by-token AI dashboard"""
        
        # Clear screen
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🔥 LIVE TOKEN STREAMING DASHBOARD")
        print("=" * 60)
        print("✨ Watch AI think word-by-word in real-time!")
        print("🧠 Every token appears as AI generates it")
        print("=" * 60)
        print()
        print("📋 COMMANDS:")
        print("   • Type any task to start live AI streaming")
        print("   • Press Ctrl+C to cancel current generation")
        print("   • Type 'quit' to exit dashboard")
        print("=" * 60)
        
        # Setup signal handler
        signal.signal(signal.SIGINT, self._handle_interrupt)
        
        while True:
            try:
                print()
                command = input("🔥 TOKEN STREAM > ").strip()
                
                if not command:
                    continue
                    
                if command.lower() == 'quit':
                    print("\n👋 Closing token dashboard...")
                    break
                    
                else:
                    # Start live token streaming
                    await self._run_live_token_stream(command)
                    
            except KeyboardInterrupt:
                print("\n🛑 Generation cancelled by user")
                self.cancel_requested = True
                continue
                
            except Exception as e:
                print(f"❌ Dashboard error: {e}")
        
        print("\n🚪 Token dashboard closed!")
    
    async def _run_live_token_stream(self, task: str):
        """Stream AI response token by token live"""
        
        print(f"\n🚀 STARTING LIVE TOKEN STREAM")
        print("=" * 60)
        print(f"📋 Task: {task}")
        print("=" * 60)
        
        self.current_task = task
        self.cancel_requested = False
        
        # Step 1: Task Classification
        print("🎯 Analyzing task and selecting model...")
        classification = classifier.classify_with_ai(task)
        selected_model = classification['model']
        
        print(f"🤖 Selected model: {selected_model}")
        print(f"📊 Task complexity: {classification['complexity']}")
        print()
        print("🔥 LIVE AI GENERATION (token by token):")
        print("-" * 60)
        
        messages = [{"role": "user", "content": task}]
        
        try:
            current_response = ""
            token_count = 0
            
            async for chunk in token_streaming_ollama.chat_stream(selected_model, messages):
                if self.cancel_requested:
                    print("\n\n🛑 GENERATION CANCELLED")
                    break
                
                if "error" in chunk:
                    print(f"\n❌ Error: {chunk['error']}")
                    break
                
                # Extract token from chunk
                if "message" in chunk and "content" in chunk["message"]:
                    token = chunk["message"]["content"]
                    
                    if token:
                        # Print token without newline (live streaming effect)
                        print(token, end="", flush=True)
                        current_response += token
                        token_count += 1
                        
                        # Small delay for visual effect (optional)
                        await asyncio.sleep(0.01)  # 10ms delay for readability
                
                # Check if generation is complete
                if chunk.get("done", False):
                    print("\n\n" + "=" * 60)
                    print("🎉 GENERATION COMPLETE!")
                    print(f"📊 Total tokens generated: {token_count}")
                    print(f"📝 Total characters: {len(current_response)}")
                    print(f"🤖 Model used: {selected_model}")
                    
                    # Show full response in formatted way
                    print("\n📄 COMPLETE RESPONSE:")
                    print("-" * 40)
                    print(current_response)
                    print("-" * 40)
                    break
                    
        except Exception as e:
            print(f"\n❌ Streaming error: {e}")
        finally:
            self.current_task = None
            self.cancel_requested = False
    
    def _handle_interrupt(self, signum, frame):
        """Handle Ctrl+C during generation"""
        print(f"\n🛑 INTERRUPTING TOKEN STREAM...")
        self.cancel_requested = True

async def main():
    """Main entry point"""
    print("🔥 Starting Token Streaming Dashboard...")
    dashboard = TokenDashboard()
    await dashboard.run_token_dashboard()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Token dashboard terminated")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
