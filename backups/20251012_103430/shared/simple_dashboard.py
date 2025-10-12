import os
import sys
import signal
import time
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

class AgentState(Enum):
    IDLE = "💤 idle"
    ANALYZING = "🔍 analyzing"
    LOADING = "⏳ loading"
    THINKING = "🤔 thinking"
    GENERATING = "🔥 generating"
    COMPLETED = "✅ completed"
    ERROR = "❌ error"
    STOPPED = "🛑 stopped"

@dataclass
class Agent:
    agent_id: str
    state: AgentState
    prompt: str = ""
    tokens: int = 0
    start_time: float = 0
    model: str = "llama3.2:3b"

class SimpleDashboard:
    def __init__(self):
        self.running = True
        self.agents: Dict[str, Agent] = {}
        signal.signal(signal.SIGINT, self.signal_handler)
        
    def signal_handler(self, signum, frame):
        print("\n🛑 Zamykanie dashboard...")
        self.running = False
        sys.exit(0)
    
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def display_header(self):
        current_time = datetime.now().strftime('%H:%M:%S')
        print("🎛️" + "="*60 + "🎛️")
        print("🚀  SIMPLE LIVE DASHBOARD - AGENT ZERO  🚀")
        print(f"⏰  {current_time} | Agentów: {len(self.agents)}")
        print("🎛️" + "="*60 + "🎛️")
    
    def display_agents(self):
        print(f"\n🤖 AKTYWNI AGENCI ({len(self.agents)}):")
        print("-" * 60)
        
        if not self.agents:
            print("💤 Brak aktywnych agentów")
        else:
            for agent_id, agent in self.agents.items():
                runtime = int(time.time() - agent.start_time) if agent.start_time else 0
                print(f"{agent.state.value} {agent_id}")
                print(f"   └─ Tokeny: {agent.tokens} | Czas: {runtime}s | Model: {agent.model}")
                if agent.prompt:
                    prompt_preview = agent.prompt[:50] + "..." if len(agent.prompt) > 50 else agent.prompt
                    print(f"   └─ Prompt: {prompt_preview}")
    
    def display_menu(self):
        print("\n🎮 MENU:")
        print("1️⃣  🚀 Dodaj test agenta")
        print("2️⃣  🔥 Symuluj generację")
        print("3️⃣  📊 Szczegóły agentów")
        print("4️⃣  🛑 Zatrzymaj agenta")
        print("5️⃣  🗑️  Usuń wszystkich agentów")
        print("6️⃣  🧪 Test Ollama (symulacja)")
        print("0️⃣  👋 Wyjście")
        print("\n👆 Wybierz opcję: ", end="")
    
    def add_test_agent(self):
        agent_id = f"agent-{len(self.agents)+1:03d}"
        
        print(f"\n🚀 Tworzenie agenta: {agent_id}")
        prompt = input("📝 Wpisz prompt (Enter = domyślny): ").strip()
        
        if not prompt:
            prompt = "Napisz funkcję hello world w Pythonie"
        
        agent = Agent(
            agent_id=agent_id,
            state=AgentState.IDLE,
            prompt=prompt,
            start_time=time.time()
        )
        
        self.agents[agent_id] = agent
        print(f"✅ Agent {agent_id} utworzony!")
        input("⏎ Naciśnij Enter...")
    
    def simulate_generation(self):
        if not self.agents:
            print("❌ Brak agentów do generacji!")
            input("⏎ Naciśnij Enter...")
            return
        
        print("\n🔥 SYMULACJA GENERACJI")
        print("Dostępni agenci:")
        for i, agent_id in enumerate(self.agents.keys(), 1):
            print(f"{i}. {agent_id}")
        
        try:
            choice = int(input("Wybierz agenta (numer): ")) - 1
            agent_ids = list(self.agents.keys())
            
            if 0 <= choice < len(agent_ids):
                selected_agent = agent_ids[choice]
                self.run_simulation(selected_agent)
            else:
                print("❌ Nieprawidłowy wybór!")
        except ValueError:
            print("❌ Wpisz numer!")
        
        input("⏎ Naciśnij Enter...")
    
    def run_simulation(self, agent_id: str):
        agent = self.agents[agent_id]
        
        print(f"\n🚀 Rozpoczynanie generacji dla {agent_id}")
        
        stages = [
            (AgentState.ANALYZING, "Analizuję prompt..."),
            (AgentState.LOADING, "Ładuję model..."),
            (AgentState.THINKING, "Myślę nad odpowiedzią..."),
            (AgentState.GENERATING, "Generuję kod...")
        ]
        
        for state, message in stages:
            agent.state = state
            print(f"{state.value} {message}")
            time.sleep(1)
        
        # Symulacja tokenów
        print("\n🔥 Generowanie tokenów:")
        sample_tokens = ["def", " hello_world", "()", ":", "\n", "    ", "print", "(", "'Hello, World!'", ")"]
        
        for token in sample_tokens:
            agent.tokens += 1
            print(f"Token {agent.tokens}: '{token}'")
            time.sleep(0.3)
        
        agent.state = AgentState.COMPLETED
        print(f"\n✅ Generacja zakończona! Wygenerowano {agent.tokens} tokenów")
    
    def show_details(self):
        if not self.agents:
            print("💤 Brak agentów")
            input("⏎ Naciśnij Enter...")
            return
        
        print("\n📊 SZCZEGÓŁY AGENTÓW:")
        print("=" * 60)
        
        for agent_id, agent in self.agents.items():
            runtime = int(time.time() - agent.start_time) if agent.start_time else 0
            
            print(f"\n🤖 Agent: {agent_id}")
            print(f"   Status: {agent.state.value}")
            print(f"   Model: {agent.model}")
            print(f"   Tokeny: {agent.tokens}")
            print(f"   Czas działania: {runtime}s")
            print(f"   Prompt: {agent.prompt}")
        
        input("\n⏎ Naciśnij Enter...")
    
    def stop_agent(self):
        if not self.agents:
            print("❌ Brak agentów!")
            input("⏎ Naciśnij Enter...")
            return
        
        print("\n🛑 ZATRZYMYWANIE AGENTA")
        print("Dostępni agenci:")
        for i, agent_id in enumerate(self.agents.keys(), 1):
            print(f"{i}. {agent_id}")
        
        try:
            choice = int(input("Wybierz agenta (numer): ")) - 1
            agent_ids = list(self.agents.keys())
            
            if 0 <= choice < len(agent_ids):
                selected_agent = agent_ids[choice]
                self.agents[selected_agent].state = AgentState.STOPPED
                print(f"🛑 Agent {selected_agent} zatrzymany!")
            else:
                print("❌ Nieprawidłowy wybór!")
        except ValueError:
            print("❌ Wpisz numer!")
        
        input("⏎ Naciśnij Enter...")
    
    def clear_agents(self):
        count = len(self.agents)
        self.agents.clear()
        print(f"🗑️ Usunięto {count} agentów!")
        input("⏎ Naciśnij Enter...")
    
    def test_ollama(self):
        print("\n🧪 TEST OLLAMA (SYMULACJA)")
        print("-" * 30)
        print("📋 Sprawdzanie dostępnych modeli...")
        time.sleep(1)
        
        models = ["llama3.2:3b", "llama3.2:1b", "codellama:7b", "mistral:7b"]
        
        print("✅ Dostępne modele:")
        for model in models:
            print(f"   🧠 {model}")
        
        print("\n🔥 Test generacji:")
        test_response = "Hello! This is a test response from Ollama simulation."
        
        for i, char in enumerate(test_response, 1):
            print(char, end="", flush=True)
            time.sleep(0.05)
        
        print(f"\n\n✅ Test zakończony! Wygenerowano {len(test_response)} znaków")
        input("⏎ Naciśnij Enter...")
    
    def run(self):
        print("🚀 Uruchamianie Simple Dashboard...")
        print("🛑 Użyj Ctrl+C aby zatrzymać\n")
        
        while self.running:
            try:
                self.clear_screen()
                self.display_header()
                self.display_agents()
                self.display_menu()
                
                choice = input().strip()
                
                if choice == '1':
                    self.add_test_agent()
                elif choice == '2':
                    self.simulate_generation()
                elif choice == '3':
                    self.show_details()
                elif choice == '4':
                    self.stop_agent()
                elif choice == '5':
                    self.clear_agents()
                elif choice == '6':
                    self.test_ollama()
                elif choice == '0':
                    print("👋 Zamykanie dashboard...")
                    break
                else:
                    print("❌ Nieprawidłowy wybór!")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n🛑 Ctrl+C otrzymany!")
                break
            except Exception as e:
                print(f"❌ Błąd: {e}")
                input("⏎ Naciśnij Enter...")
        
        print("✅ Dashboard zakończony!")

def main():
    dashboard = SimpleDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
