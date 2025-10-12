import os
import sys
import signal
import time
from datetime import datetime
from typing import Dict, Any
from dataclasses import dataclass
from enum import Enum

# Try importing ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("✅ Ollama client zaimportowany")
except ImportError:
    OLLAMA_AVAILABLE = False
    print("⚠️ Ollama client niedostępny - będzie symulacja")

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
    response: str = ""
    error_message: str = ""

class OllamaDashboard:
    def __init__(self):
        self.running = True
        self.agents: Dict[str, Agent] = {}
        self.available_models = []
        signal.signal(signal.SIGINT, self.signal_handler)
        
        # Check Ollama on startup
        self.check_ollama_status()
        
    def signal_handler(self, signum, frame):
        print("\n🛑 Emergency stop wszystkich agentów...")
        for agent in self.agents.values():
            if agent.state == AgentState.GENERATING:
                agent.state = AgentState.STOPPED
        self.running = False
        sys.exit(0)
    
    def check_ollama_status(self):
        """Sprawdź status Ollama przy starcie"""
        if OLLAMA_AVAILABLE:
            try:
                models = ollama.list()
                if models and 'models' in models:
                    self.available_models = [m['name'] for m in models['models']]
                    print(f"✅ Ollama OK - {len(self.available_models)} modeli dostępnych")
                else:
                    print("⚠️ Ollama połączony ale brak modeli")
            except Exception as e:
                print(f"❌ Ollama error: {e}")
        else:
            print("❌ Ollama client niedostępny")
    
    def clear_screen(self):
        os.system('clear' if os.name == 'posix' else 'cls')
    
    def display_header(self):
        current_time = datetime.now().strftime('%H:%M:%S')
        ollama_status = "🟢 Online" if OLLAMA_AVAILABLE else "🔴 Offline"
        
        print("🎛️" + "="*60 + "🎛️")
        print("🧠  OLLAMA LIVE DASHBOARD - AGENT ZERO  🧠")
        print(f"⏰  {current_time} | Agentów: {len(self.agents)} | Ollama: {ollama_status}")
        print(f"🤖  Modeli dostępnych: {len(self.available_models)}")
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
                if agent.error_message:
                    print(f"   └─ ❌ Błąd: {agent.error_message}")
    
    def display_menu(self):
        print("\n🎮 MENU:")
        print("1️⃣  🚀 Dodaj agenta")
        print("2️⃣  🔥 Generuj z Ollama" + (" (REAL)" if OLLAMA_AVAILABLE else " (SYM)"))
        print("3️⃣  📊 Szczegóły agentów")
        print("4️⃣  🛑 Zatrzymaj agenta")
        print("5️⃣  🗑️  Usuń wszystkich")
        print("6️⃣  🧪 Test połączenia Ollama")
        print("7️⃣  📋 Lista modeli")
        print("8️⃣  🔄 Sprawdź status Ollama")
        print("0️⃣  👋 Wyjście")
        print("\n👆 Wybierz opcję: ", end="")
    
    def add_agent(self):
        agent_id = f"agent-{len(self.agents)+1:03d}"
        
        print(f"\n🚀 Tworzenie agenta: {agent_id}")
        prompt = input("📝 Wpisz prompt: ").strip()
        
        if not prompt:
            prompt = "Napisz prostą funkcję hello world w Pythonie z komentarzami"
            print(f"📝 Użyto domyślnego promptu")
        
        # Wybór modelu
        if self.available_models:
            print("\n🧠 Dostępne modele:")
            for i, model in enumerate(self.available_models, 1):
                print(f"{i}. {model}")
            
            try:
                choice = input("Wybierz model (Enter = default): ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(self.available_models):
                    selected_model = self.available_models[int(choice)-1]
                else:
                    selected_model = self.available_models[0] if self.available_models else "llama3.2:3b"
            except:
                selected_model = "llama3.2:3b"
        else:
            selected_model = "llama3.2:3b"
        
        agent = Agent(
            agent_id=agent_id,
            state=AgentState.IDLE,
            prompt=prompt,
            start_time=time.time(),
            model=selected_model
        )
        
        self.agents[agent_id] = agent
        print(f"✅ Agent {agent_id} utworzony z modelem {selected_model}!")
        input("⏎ Naciśnij Enter...")
    
    def generate_with_ollama(self):
        if not self.agents:
            print("❌ Brak agentów!")
            input("⏎ Naciśnij Enter...")
            return
        
        print("\n🔥 GENERACJA Z OLLAMA")
        print("Dostępni agenci:")
        for i, agent_id in enumerate(self.agents.keys(), 1):
            agent = self.agents[agent_id]
            print(f"{i}. {agent_id} ({agent.state.value}) - {agent.model}")
        
        try:
            choice = int(input("Wybierz agenta: ")) - 1
            agent_ids = list(self.agents.keys())
            
            if 0 <= choice < len(agent_ids):
                selected_agent = agent_ids[choice]
                self.run_ollama_generation(selected_agent)
            else:
                print("❌ Nieprawidłowy wybór!")
        except ValueError:
            print("❌ Wpisz numer!")
        
        input("⏎ Naciśnij Enter...")
    
    def run_ollama_generation(self, agent_id: str):
        agent = self.agents[agent_id]
        
        print(f"\n🚀 Rozpoczynanie generacji dla {agent_id}")
        print(f"🧠 Model: {agent.model}")
        print(f"📝 Prompt: {agent.prompt}")
        print("-" * 40)
        
        try:
            # Stage 1: Analyzing
            agent.state = AgentState.ANALYZING
            print("🔍 Analizuję prompt...")
            time.sleep(0.5)
            
            # Stage 2: Loading
            agent.state = AgentState.LOADING
            print("⏳ Ładuję model...")
            time.sleep(1)
            
            # Stage 3: Generating
            agent.state = AgentState.GENERATING
            print("🔥 Rozpoczynam generację...\n")
            
            if OLLAMA_AVAILABLE:
                # Prawdziwa generacja z Ollama
                response_text = ""
                
                try:
                    response = ollama.generate(
                        model=agent.model,
                        prompt=agent.prompt,
                        stream=True
                    )
                    
                    print("📝 Odpowiedź:")
                    print("-" * 20)
                    
                    for chunk in response:
                        if 'response' in chunk:
                            token = chunk['response']
                            print(token, end='', flush=True)
                            response_text += token
                            agent.tokens += 1
                            
                            # Pokazuj progress co 10 tokenów
                            if agent.tokens % 10 == 0:
                                print(f" [{agent.tokens}]", end='', flush=True)
                    
                    agent.response = response_text
                    agent.state = AgentState.COMPLETED
                    print(f"\n\n✅ Generacja zakończona!")
                    print(f"📊 Wygenerowano {agent.tokens} tokenów")
                    
                except Exception as e:
                    agent.state = AgentState.ERROR
                    agent.error_message = str(e)
                    print(f"\n❌ Błąd Ollama: {e}")
            else:
                # Symulacja
                print("⚠️ Symulacja (Ollama niedostępny):")
                sample_response = f"""
def hello_world():
    \"\"\"
    Prosta funkcja wyświetlająca powitanie.
    \"\"\"
    print("Hello, World!")
    return "Hello, World!"

# Wywołanie funkcji
if __name__ == "__main__":
    hello_world()
"""
                
                for char in sample_response:
                    print(char, end='', flush=True)
                    agent.tokens += 1
                    if char in [' ', '\n']:
                        time.sleep(0.05)
                
                agent.response = sample_response
                agent.state = AgentState.COMPLETED
                print(f"\n✅ Symulacja zakończona! Tokeny: {agent.tokens}")
                
        except KeyboardInterrupt:
            agent.state = AgentState.STOPPED
            print(f"\n🛑 Generacja przerwana przez użytkownika!")
        except Exception as e:
            agent.state = AgentState.ERROR
            agent.error_message = str(e)
            print(f"\n❌ Błąd: {e}")
    
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
            print(f"   Czas: {runtime}s")
            print(f"   Prompt: {agent.prompt}")
            
            if agent.response:
                response_preview = agent.response[:100] + "..." if len(agent.response) > 100 else agent.response
                print(f"   Odpowiedź: {response_preview}")
            
            if agent.error_message:
                print(f"   Błąd: {agent.error_message}")
        
        input("\n⏎ Naciśnij Enter...")
    
    def test_ollama_connection(self):
        print("\n🧪 TEST POŁĄCZENIA OLLAMA")
        print("-" * 30)
        
        if not OLLAMA_AVAILABLE:
            print("❌ Ollama client niedostępny")
            input("⏎ Naciśnij Enter...")
            return
        
        try:
            print("📡 Sprawdzanie połączenia...")
            models = ollama.list()
            
            if models and 'models' in models:
                print("✅ Połączenie OK!")
                print(f"📋 Dostępne modele ({len(models['models'])}):")
                for model in models['models']:
                    print(f"   🧠 {model['name']}")
                    
                # Quick test generation
                print("\n🧪 Test szybkiej generacji...")
                test_response = ollama.generate(
                    model="llama3.2:3b",  # Podstawowy model
                    prompt="Odpowiedz jednym słowem: 'Test'",
                    stream=False
                )
                
                if test_response and 'response' in test_response:
                    print(f"✅ Test response: {test_response['response'][:50]}")
                else:
                    print("⚠️ Brak odpowiedzi w teście")
                    
            else:
                print("⚠️ Połączenie OK ale brak modeli")
                
        except Exception as e:
            print(f"❌ Błąd połączenia: {e}")
        
        input("\n⏎ Naciśnij Enter...")
    
    def list_models(self):
        print("\n📋 LISTA MODELI OLLAMA")
        print("-" * 30)
        
        if not self.available_models:
            if OLLAMA_AVAILABLE:
                print("⚠️ Brak dostępnych modeli")
                print("💡 Spróbuj: ollama pull llama3.2:3b")
            else:
                print("❌ Ollama client niedostępny")
        else:
            print(f"✅ Dostępne modele ({len(self.available_models)}):")
            for i, model in enumerate(self.available_models, 1):
                print(f"{i}. 🧠 {model}")
        
        input("\n⏎ Naciśnij Enter...")
    
    def refresh_ollama_status(self):
        print("\n🔄 ODŚWIEŻANIE STATUSU OLLAMA")
        print("-" * 30)
        self.check_ollama_status()
        input("⏎ Naciśnij Enter...")
    
    def stop_agent(self):
        if not self.agents:
            print("❌ Brak agentów!")
            input("⏎ Naciśnij Enter...")
            return
        
        print("\n🛑 ZATRZYMYWANIE AGENTA")
        for i, agent_id in enumerate(self.agents.keys(), 1):
            agent = self.agents[agent_id]
            print(f"{i}. {agent_id} ({agent.state.value})")
        
        try:
            choice = int(input("Wybierz agenta: ")) - 1
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
    
    def run(self):
        print("🚀 Uruchamianie Ollama Dashboard...")
        print("🛑 Użyj Ctrl+C dla emergency stop\n")
        
        while self.running:
            try:
                self.clear_screen()
                self.display_header()
                self.display_agents()
                self.display_menu()
                
                choice = input().strip()
                
                if choice == '1':
                    self.add_agent()
                elif choice == '2':
                    self.generate_with_ollama()
                elif choice == '3':
                    self.show_details()
                elif choice == '4':
                    self.stop_agent()
                elif choice == '5':
                    self.clear_agents()
                elif choice == '6':
                    self.test_ollama_connection()
                elif choice == '7':
                    self.list_models()
                elif choice == '8':
                    self.refresh_ollama_status()
                elif choice == '0':
                    print("👋 Zamykanie dashboard...")
                    break
                else:
                    print("❌ Nieprawidłowy wybór!")
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n🛑 Emergency stop aktywowany!")
                break
            except Exception as e:
                print(f"❌ Błąd: {e}")
                input("⏎ Naciśnij Enter...")
        
        print("✅ Dashboard zakończony!")

def main():
    dashboard = OllamaDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
