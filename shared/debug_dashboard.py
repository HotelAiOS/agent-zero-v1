import asyncio
import os
import sys
import signal
from datetime import datetime

class DebugDashboard:
    def __init__(self):
        self.running = True
        self.agents = {}
        
    def signal_handler(self, signum, frame):
        print("\n🛑 Zamykanie...")
        self.running = False
        sys.exit(0)
        
    async def run(self):
        signal.signal(signal.SIGINT, self.signal_handler)
        
        print("🚀 DEBUG Dashboard uruchomiony!")
        print("🛑 Użyj Ctrl+C aby zatrzymać")
        
        counter = 0
        while self.running:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Header
            print("🎛️" + "="*50 + "🎛️")
            print("🐛  DEBUG LIVE DASHBOARD  🐛")
            print(f"⏰  {datetime.now().strftime('%H:%M:%S')}")
            print(f"🔄  Counter: {counter}")
            print("🎛️" + "="*50 + "🎛️")
            
            # Menu
            print("\n🎮 OPCJE:")
            print("1️⃣  Dodaj test agenta")
            print("2️⃣  Usuń agentów")
            print("3️⃣  Pokaż agentów")
            print("0️⃣  Wyjście")
            
            # Pokaż agentów
            if self.agents:
                print(f"\n🤖 AKTYWNI AGENCI ({len(self.agents)}):")
                for agent_id, data in self.agents.items():
                    print(f"   └─ {agent_id}: {data['status']}")
            else:
                print("\n💤 Brak aktywnych agentów")
            
            print(f"\n👆 Wybierz opcję (auto-refresh za 3s): ", end="", flush=True)
            
            try:
                # Wait for input with timeout
                choice = await self.get_input_with_timeout(3)
                
                if choice == '1':
                    agent_id = f"agent-{len(self.agents)+1}"
                    self.agents[agent_id] = {
                        'status': 'active',
                        'created': datetime.now()
                    }
                    print(f"✅ Dodano {agent_id}")
                    await asyncio.sleep(1)
                    
                elif choice == '2':
                    self.agents.clear()
                    print("🗑️ Usunięto wszystkich agentów")
                    await asyncio.sleep(1)
                    
                elif choice == '3':
                    print("\n📋 SZCZEGÓŁY AGENTÓW:")
                    for agent_id, data in self.agents.items():
                        print(f"🤖 {agent_id}")
                        print(f"   Status: {data['status']}")
                        print(f"   Utworzony: {data['created']}")
                    
                    input("⏎ Naciśnij Enter...")
                    
                elif choice == '0':
                    print("👋 Zamykanie...")
                    self.running = False
                    break
                    
            except asyncio.TimeoutError:
                # Auto refresh
                pass
            except Exception as e:
                print(f"❌ Błąd: {e}")
                await asyncio.sleep(1)
            
            counter += 1
        
        print("✅ Dashboard zakończony!")
    
    async def get_input_with_timeout(self, timeout):
        """Get input with timeout for auto-refresh"""
        try:
            # Create a task for input
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(input)
                return await asyncio.wait_for(
                    asyncio.wrap_future(future), 
                    timeout=timeout
                )
        except asyncio.TimeoutError:
            print("\n🔄 Auto-refresh...")
            raise

async def main():
    dashboard = DebugDashboard()
    try:
        await dashboard.run()
    except KeyboardInterrupt:
        print("\n🛑 Ctrl+C received!")
    except Exception as e:
        print(f"❌ Błąd: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🚀 Uruchamianie Debug Dashboard...")
    asyncio.run(main())
