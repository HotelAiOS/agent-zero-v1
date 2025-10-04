import asyncio
import signal
import sys
import os
from token_streaming_client import token_streaming_ollama
from simple_classifier import classifier
import time

class PolskiTokenDashboard:
    def __init__(self):
        self.running = False
        self.current_task = None
        self.cancel_requested = False
        
    async def uruchom_dashboard(self):
        """Dashboard z polskim interfejsem i live token streaming"""
        
        # Wyczyść ekran
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print("🎛️ POLSKI AI DASHBOARD Z LIVE STREAMING")
        print("=" * 70)
        print("✨ Oglądaj jak AI myśli słowo po słowie w czasie rzeczywistym!")
        print("🧠 Każdy token pojawia się gdy AI go generuje")
        print("🇵🇱 Polski interfejs + angielska generacja AI")
        print("=" * 70)
        print()
        print("📋 KOMENDY:")
        print("   • Wpisz dowolne zadanie aby uruchomić AI")
        print("   • Naciśnij Ctrl+C aby przerwać generację")
        print("   • Wpisz 'status' aby sprawdzić system")
        print("   • Wpisz 'pomoc' aby pokazać pomoc")
        print("   • Wpisz 'koniec' aby wyjść")
        print("=" * 70)
        
        # Obsługa Ctrl+C
        signal.signal(signal.SIGINT, self._obsluz_przerwanie)
        
        while True:
            try:
                print()
                komenda = input("🎛️ OPERATOR > ").strip()
                
                if not komenda:
                    continue
                    
                if komenda.lower() in ['koniec', 'quit', 'wyjście', 'exit']:
                    print("\n👋 Zamykam dashboard...")
                    break
                    
                elif komenda.lower() in ['status', 'stan']:
                    self._pokaz_status()
                    
                elif komenda.lower() in ['pomoc', 'help', 'h']:
                    self._pokaz_pomoc()
                    
                else:
                    # Uruchom live streaming AI
                    await self._uruchom_live_streaming(komenda)
                    
            except KeyboardInterrupt:
                print("\n🛑 Generacja przerwana przez użytkownika")
                self.cancel_requested = True
                continue
                
            except Exception as e:
                print(f"❌ Błąd dashboard: {e}")
        
        print("\n🚪 Dashboard zamknięty! Do widzenia!")
    
    async def _uruchom_live_streaming(self, zadanie: str):
        """Uruchom AI z live token streaming po polsku"""
        
        print(f"\n🚀 ROZPOCZYNAM LIVE STREAMING AI")
        print("=" * 70)
        print(f"📋 Zadanie: {zadanie}")
        print("=" * 70)
        
        self.current_task = zadanie
        self.cancel_requested = False
        start_time = time.time()
        
        # Krok 1: Analiza zadania
        print("🎯 Analizuję zadanie i wybieram model AI...")
        classification = classifier.classify_with_ai(zadanie)
        selected_model = classification['model']
        
        # Model info po polsku
        model_info = {
            "phi3:mini": "szybki model (2.2GB) - proste zadania", 
            "qwen2.5:14b": "średni model (9GB) - ogólne zadania",
            "deepseek-coder:33b": "kod specialist (18GB) - programowanie",
            "deepseek-r1:32b": "myśliciel (19GB) - złożone analizy", 
            "mixtral:8x7b": "kreatywny (26GB) - pisanie i dokumentacja"
        }
        
        print(f"🤖 Wybrany model: {selected_model}")
        print(f"📊 Opis: {model_info.get(selected_model, 'nieznany model')}")
        print(f"🎯 Złożoność zadania: {classification['complexity']}")
        print()
        print("🔥 LIVE GENERACJA AI (token po token):")
        print("-" * 70)
        print("💡 Obserwuj jak AI pisze odpowiedź na żywo...")
        print("-" * 70)
        
        messages = [{"role": "user", "content": zadanie}]
        
        try:
            current_response = ""
            token_count = 0
            start_generation = time.time()
            
            async for chunk in token_streaming_ollama.chat_stream(selected_model, messages):
                if self.cancel_requested:
                    print("\n\n🛑 GENERACJA PRZERWANA PRZEZ OPERATORA")
                    print(f"⏱️ Przerwane po {time.time() - start_time:.1f} sekundach")
                    break
                
                if "error" in chunk:
                    print(f"\n❌ Błąd: {chunk['error']}")
                    if "timeout" in str(chunk['error']).lower():
                        print("💡 Spróbuj z prostszym zadaniem lub poczekaj na model")
                    break
                
                # Wyciągnij token z odpowiedzi
                if "message" in chunk and "content" in chunk["message"]:
                    token = chunk["message"]["content"]
                    
                    if token:
                        # Wyświetl token bez nowej linii (efekt live streaming)
                        print(token, end="", flush=True)
                        current_response += token
                        token_count += 1
                        
                        # Małe opóźnienie dla efektu wizualnego
                        await asyncio.sleep(0.02)  # 20ms dla lepszej czytelności
                
                # Sprawdź czy generacja zakończona
                if chunk.get("done", False):
                    generation_time = time.time() - start_generation
                    total_time = time.time() - start_time
                    
                    print("\n\n" + "=" * 70)
                    print("🎉 GENERACJA ZAKOŃCZONA POMYŚLNIE!")
                    print("=" * 70)
                    print(f"📊 Statystyki:")
                    print(f"   • Wygenerowane tokeny: {token_count}")
                    print(f"   • Całkowite znaki: {len(current_response)}")
                    print(f"   • Użyty model: {selected_model}")
                    print(f"   • Czas generacji: {generation_time:.1f}s")
                    print(f"   • Czas całkowity: {total_time:.1f}s")
                    
                    if token_count > 0:
                        print(f"   • Prędkość: {token_count/generation_time:.1f} tokenów/s")
                    
                    # Pokaż pełną odpowiedź w sformatowany sposób
                    print("\n📄 KOMPLETNA ODPOWIEDŹ:")
                    print("-" * 50)
                    print(current_response)
                    print("-" * 50)
                    
                    # Ocena jakości
                    if len(current_response) > 100:
                        print("✅ Jakość: Obszerna odpowiedź")
                    elif len(current_response) > 50:
                        print("✅ Jakość: Dobra odpowiedź")
                    else:
                        print("⚠️ Jakość: Krótka odpowiedź - może spróbować ponownie")
                    
                    break
                    
        except Exception as e:
            print(f"\n❌ Błąd podczas streamingu: {e}")
            print("💡 Sprawdź czy Ollama działa: ollama ps")
        finally:
            self.current_task = None
            self.cancel_requested = False
    
    def _pokaz_status(self):
        """Pokaż status systemu po polsku"""
        print("\n📊 STATUS SYSTEMU:")
        print("-" * 30)
        status = "ZAJĘTY" if self.current_task else "GOTOWY"
        print(f"🎛️ Dashboard: AKTYWNY")
        print(f"🤖 AI System: {status}")
        
        if self.current_task:
            print(f"📋 Obecne zadanie: {self.current_task[:40]}...")
        
        # Sprawdź Ollama
        import subprocess
        try:
            result = subprocess.run(['ollama', 'ps'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0 and result.stdout.strip():
                print("✅ Ollama: POŁĄCZONY")
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                if lines and lines[0].strip():
                    print(f"🤖 Załadowane modele: {len(lines)}")
            else:
                print("⚠️ Ollama: BRAK MODELI")
        except:
            print("❌ Ollama: NIEDOSTĘPNY")
    
    def _pokaz_pomoc(self):
        """Pokaż pomoc po polsku"""
        print("\n📚 POMOC - Dostępne Komendy:")
        print("-" * 40)
        print("🎯 [dowolny tekst] - Uruchom zadanie AI")
        print("🛑 Ctrl+C - Przerwij obecną generację")
        print("📊 status/stan - Sprawdź status systemu")
        print("❓ pomoc/help - Pokaż tę pomoc")
        print("👋 koniec/quit - Wyjdź z dashboard")
        print()
        print("💡 Przykłady zadań:")
        print('   • "napisz funkcję hello world w pythonie"')
        print('   • "wyjaśnij co to jest API"')
        print('   • "stwórz prosty HTML dla strony"')
        print('   • "napisz wiersz o kotach"')
        print()
        print("⚡ Wskazówki:")
        print("   • Pierwsze uruchomienie modelu może trwać 5-10 minut")
        print("   • Możesz przerwać w każdej chwili przez Ctrl+C")
        print("   • Obserwuj jak AI pisze odpowiedź słowo po słowie")
        print("   • Proste zadania są szybsze (phi3:mini)")
        print("   • Zadania kodu używają deepseek-coder (wolniejszy)")
    
    def _obsluz_przerwanie(self, signum, frame):
        """Obsłuż Ctrl+C po polsku"""
        print(f"\n🚨 PRZERWANIE - Operator zatrzymał operację")
        self.cancel_requested = True
        print("💡 Wpisz 'koniec' aby wyjść z dashboard")

async def main():
    """Główna funkcja"""
    print("🎛️ Uruchamiam Polski Dashboard z Live Streaming...")
    dashboard = PolskiTokenDashboard()
    await dashboard.uruchom_dashboard()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Dashboard zamknięty przez użytkownika")
    except Exception as e:
        print(f"\n❌ Krytyczny błąd: {e}")
