import asyncio
import signal
import sys
import os
import time
import json
from datetime import datetime
from ai_streaming import streaming_ai_brain, ThinkingStage

class EnhancedLiveDashboard:
    def __init__(self):
        self.running = False
        self.current_task = None
        self.agents = {}
        self.models_status = {}
        self.emergency_stop = False
        
        # Obsługa sygnałów
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        """Obsługa Ctrl+C"""
        print("\n🛑 Otrzymano sygnał zatrzymania...")
        self.emergency_stop = True
        self.running = False
        sys.exit(0)
        
    async def run_dashboard(self):
        """Main interactive dashboard z rozszerzeniami"""
        self.running = True
        
        while self.running and not self.emergency_stop:
            # Clear screen
            os.system('clear' if os.name == 'posix' else 'cls')
            
            # Header
            print("🎛️" + "="*60 + "🎛️")
            print("🚀  ENHANCED LIVE AI MONITORING DASHBOARD  🚀")
            print("🎛️" + "="*60 + "🎛️")
            print("✨ Real-time AI thinking visualization")
            print("🔥 Token streaming with emergency controls")
            print("🛑 Użyj Ctrl+C aby zatrzymać")
            print("🎛️" + "="*60 + "🎛️")
            
            # Show current status
            await self.display_status()
            
            # Interactive menu
            await self.show_menu()
            
            # Refresh every 2 seconds
            await asyncio.sleep(2)
    
    async def display_status(self):
        """Wyświetl aktualny status systemu"""
        current_time = datetime.now().strftime("%H:%M:%S")
        
        print(f"\n📊 STATUS SYSTEMU - {current_time}")
        print("-" * 60)
        
        if not self.agents:
            print("💤 Brak aktywnych agentów")
        else:
            for agent_id, agent_data in self.agents.items():
                status_icon = self.get_status_icon(agent_data.get('stage', 'idle'))
                print(f"{status_icon} Agent {agent_id}: {agent_data.get('stage', 'Unknown')}")
                if agent_data.get('current_prompt'):
                    prompt_preview = agent_data['current_prompt'][:50] + "..."
                    print(f"   └─ Prompt: {prompt_preview}")
        
        print(f"\n🔌 Aktywnych agentów: {len(self.agents)}")
        print(f"🧠 Emergency mode: {'🔴 TAK' if self.emergency_stop else '🟢 NIE'}")
    
    def get_status_icon(self, stage):
        """Ikona dla stanu agenta"""
        icons = {
            'analyzing': '🔍',
            'model_selection': '🎯',
            'model_loading': '⏳',
            'thinking': '🤔',
            'generating': '🔥',
            'completed': '✅',
            'cancelled': '🛑',
            'error': '❌',
            'idle': '💤'
        }
        return icons.get(stage, '❓')
    
    async def show_menu(self):
        """Pokaż interaktywne menu"""
        print("\n🎮 OPCJE:")
        print("1️⃣  Uruchom test generacji")
        print("2️⃣  Zatrzymaj wszystkie agenty")
        print("3️⃣  Pokaż szczegóły agentów")
        print("4️⃣  Test AI streaming")
        print("5️⃣  Monitor w tle (auto-refresh)")
        print("0️⃣  Wyjście")
        
        # W trybie auto-refresh nie czekamy na input
        if hasattr(self, 'auto_mode') and self.auto_mode:
            return
            
        print("\n👆 Wybierz opcję (Enter = refresh): ", end="", flush=True)
    
    async def start_test_generation(self):
        """Uruchom test generacji AI"""
        agent_id = f"test-{int(time.time())}"
        test_prompt = "Napisz prostą funkcję hello world w Pythonie z komentarzami"
        
        print(f"\n🚀 Uruchamianie test generacji...")
        print(f"🤖 Agent ID: {agent_id}")
        print(f"📝 Prompt: {test_prompt}")
        
        # Dodaj agenta do śledzenia
        self.agents[agent_id] = {
            'stage': 'analyzing',
            'current_prompt': test_prompt,
            'start_time': datetime.now(),
            'tokens_generated': 0
        }
        
        try:
            # Uruchom streaming z ai_streaming.py
            print("\n🔥 Rozpoczynanie streaming AI...")
            
            async for update in streaming_ai_brain(
                prompt=test_prompt,
                model="llama3.2:3b"  # Domyślny model
            ):
                # Aktualizuj status agenta
                if hasattr(update, 'stage'):
                    self.agents[agent_id]['stage'] = update.stage.value
                if hasattr(update, 'token') and update.token:
                    self.agents[agent_id]['tokens_generated'] += 1
                    print(f"🔥 Token: '{update.token}'", end="", flush=True)
                if hasattr(update, 'message'):
                    print(f"\n💭 {update.message}")
                
                # Odśwież display co kilka tokenów
                if self.agents[agent_id]['tokens_generated'] % 5 == 0:
                    await self.refresh_display()
                
                # Check emergency stop
                if self.emergency_stop:
                    print("\n🛑 Emergency stop aktywowany!")
                    break
            
            # Zakończ generację
            self.agents[agent_id]['stage'] = 'completed'
            print(f"\n✅ Generacja zakończona! Tokeny: {self.agents[agent_id]['tokens_generated']}")
            
        except Exception as e:
            print(f"\n❌ Błąd generacji: {e}")
            self.agents[agent_id]['stage'] = 'error'
        
        input("\n⏎ Naciśnij Enter aby kontynuować...")
    
    async def stop_all_agents(self):
        """Zatrzymaj wszystkich agentów"""
        print("\n🚨 EMERGENCY STOP AKTYWOWANY!")
        self.emergency_stop = True
        
        for agent_id in self.agents:
            self.agents[agent_id]['stage'] = 'cancelled'
        
        print("🛑 Wszystkie agenty zatrzymane!")
        input("\n⏎ Naciśnij Enter aby kontynuować...")
        
        # Reset emergency
        self.emergency_stop = False
    
    async def show_agent_details(self):
        """Pokaż szczegółowe informacje o agentach"""
        print("\n📋 SZCZEGÓŁY AGENTÓW:")
        print("=" * 60)
        
        if not self.agents:
            print("💤 Brak agentów do wyświetlenia")
        else:
            for agent_id, data in self.agents.items():
                print(f"\n🤖 Agent: {agent_id}")
                print(f"   Status: {data.get('stage', 'unknown')}")
                print(f"   Start: {data.get('start_time', 'N/A')}")
                print(f"   Tokeny: {data.get('tokens_generated', 0)}")
                if data.get('current_prompt'):
                    print(f"   Prompt: {data['current_prompt'][:100]}...")
        
        input("\n⏎ Naciśnij Enter aby kontynuować...")
    
    async def test_ai_streaming(self):
        """Test podstawowego AI streaming"""
        print("\n🧪 TEST AI STREAMING")
        print("-" * 30)
        
        try:
            test_prompt = "Co to jest Python?"
            print(f"📝 Test prompt: {test_prompt}")
            print("🔥 Rozpoczynanie streaming...\n")
            
            # Prosty test streaming
            async for update in streaming_ai_brain(test_prompt):
                if hasattr(update, 'stage'):
                    stage_icon = self.get_status_icon(update.stage.value)
                    print(f"{stage_icon} Stage: {update.stage.value}")
                
                if hasattr(update, 'token') and update.token:
                    print(f"Token: '{update.token}'", end=" ", flush=True)
                
                if hasattr(update, 'message'):
                    print(f"\n💭 {update.message}")
                
                # Możliwość przerwania
                if self.emergency_stop:
                    print("\n🛑 Test przerwany!")
                    break
            
            print("\n✅ Test zakończony!")
            
        except Exception as e:
            print(f"❌ Błąd testu: {e}")
        
        input("\n⏎ Naciśnij Enter aby kontynuować...")
    
    async def auto_monitor_mode(self):
        """Tryb automatycznego monitoringu"""
        print("\n🔄 TRYB AUTO-MONITOR")
        print("🛑 Ctrl+C aby zatrzymać")
        print("-" * 30)
        
        self.auto_mode = True
        
        try:
            while not self.emergency_stop:
                await self.refresh_display()
                await asyncio.sleep(1)  # Refresh co sekundę
        except KeyboardInterrupt:
            print("\n🛑 Auto-monitor zatrzymany!")
        finally:
            self.auto_mode = False
    
    async def refresh_display(self):
        """Odśwież display (pomocnicza funkcja)"""
        os.system('clear' if os.name == 'posix' else 'cls')
        await self.display_status()
    
    async def interactive_mode(self):
        """Tryb interaktywny z menu"""
        while self.running and not self.emergency_stop:
            await self.run_dashboard()
            
            # Czekaj na wybór użytkownika
            try:
                choice = input().strip()
                
                if choice == '1':
                    await self.start_test_generation()
                elif choice == '2':
                    await self.stop_all_agents()
                elif choice == '3':
                    await self.show_agent_details()
                elif choice == '4':
                    await self.test_ai_streaming()
                elif choice == '5':
                    await self.auto_monitor_mode()
                elif choice == '0':
                    print("👋 Zamykanie dashboardu...")
                    self.running = False
                    break
                elif choice == '':
                    # Enter = refresh
                    continue
                else:
                    print("❌ Nieprawidłowy wybór!")
                    await asyncio.sleep(1)
                    
            except (EOFError, KeyboardInterrupt):
                print("\n🛑 Zamykanie...")
                self.running = False
                break

# Main function
async def main():
    """Uruchom Enhanced Live Dashboard"""
    dashboard = EnhancedLiveDashboard()
    
    print("🚀 Uruchamianie Enhanced Live Dashboard...")
    
    try:
        await dashboard.interactive_mode()
    except Exception as e:
        print(f"❌ Błąd krytyczny: {e}")
    finally:
        print("👋 Dashboard zakończony!")

if __name__ == "__main__":
    asyncio.run(main())
