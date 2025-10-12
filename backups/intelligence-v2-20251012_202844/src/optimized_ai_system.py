#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Interactive AI System z RZECZYWISTĄ HIERARCHIĄ
Dostosowane do aktualnej listy modeli użytkownika z inteligentnym doborem specjalistów
"""

import requests
import time
import threading
import os
import json
import subprocess
import signal
import sys
from datetime import datetime
from queue import Queue, Empty

# Katalog na logi
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Globalne zmienne kontroli
STOP_REQUESTED = False
CURRENT_MODEL = None

class Colors:
    """ANSI color codes dla ładnego terminala"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

class InteractiveLogger:
    def __init__(self, model_name):
        self.model_name = model_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOG_DIR, f"interactive_{model_name.replace(':', '_').replace('/', '_')}_{timestamp}.log")
        self.stats_file = os.path.join(LOG_DIR, f"stats_{model_name.replace(':', '_').replace('/', '_')}_{timestamp}.json")
        
    def log(self, level, message, silent=False):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        # Zapisz zawsze do pliku
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            if not silent:
                print(f"{Colors.RED}Błąd zapisu logu: {e}{Colors.RESET}")
        
        # Pokaż tylko ważne wiadomości (nie DEBUG)
        if not silent and level in ["INFO", "WARNING", "ERROR", "SUCCESS"]:
            color = Colors.CYAN if level == "INFO" else Colors.YELLOW if level == "WARNING" else Colors.RED if level == "ERROR" else Colors.GREEN
            print(f"{Colors.DIM}[{timestamp}] {color}{level}{Colors.RESET}{Colors.DIM}: {message}{Colors.RESET}")
    
    def save_stats(self, stats_data):
        try:
            with open(self.stats_file, "w", encoding="utf-8") as f:
                json.dump(stats_data, f, indent=2, default=str)
        except Exception as e:
            self.log("ERROR", f"Błąd zapisu stats: {e}", silent=True)

class InteractiveMonitor:
    def __init__(self, model_name, ollama_url):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.logger = InteractiveLogger(model_name)
        
        # Statystyki
        self.token_count = 0
        self.last_token_time = None
        self.start_time = None
        self.silence_warnings = 0
        
        # Konfiguracja
        self.CHECK_INTERVAL = 30  # Status co 30s
        self.SILENCE_WARNING = 120  # Ostrzeżenie po 2min bez tokenów
        self.MAX_SILENCE = 300    # Fail po 5min bez tokenów
        
        global CURRENT_MODEL
        CURRENT_MODEL = model_name
        
        self.logger.log("INFO", f"Interaktywny monitor rozpoczęty dla {model_name}")
    
    def check_model_status(self):
        """Sprawdź status modelu przez API"""
        try:
            response = requests.get(f"{self.ollama_url}/api/ps", timeout=5)
            if response.status_code == 200:
                data = response.json()
                models = data.get('models', [])
                
                for model in models:
                    if model.get('name', '').startswith(self.model_name):
                        size_mb = model.get('size', 0) / 1024 / 1024
                        return True, f"Aktywny ({size_mb:.0f}MB)"
                
                return False, f"Nieaktywny ({len(models)} modeli załadowanych)"
            else:
                return False, f"API error: {response.status_code}"
        except Exception as e:
            return False, f"Błąd: {str(e)}"
    
    def check_system_load(self):
        """Sprawdź load systemu"""
        try:
            result = subprocess.run(['uptime'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                uptime_output = result.stdout.strip()
                if 'load average:' in uptime_output:
                    load_part = uptime_output.split('load average:')[1]
                    load_1min = load_part.split(',')[0].strip()
                    return True, f"Load: {load_1min}"
            return False, "Load: niedostępny"
        except Exception:
            return False, "Load: błąd"
    
    def start_streaming_with_display(self, prompt):
        """Rozpocznij streaming z live display"""
        try:
            self.logger.log("INFO", "Rozpoczynam streaming z live display")
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"Agent Zero: {prompt}\n\nOdpowiedź:",
                    "stream": True
                },
                stream=True,
                timeout=60
            )
            
            if response.status_code != 200:
                return None, f"HTTP Error: {response.status_code}"
                
            self.logger.log("INFO", "Streaming połączenie nawiązane")
            return response, "OK"
            
        except Exception as e:
            return None, f"Błąd: {str(e)}"
    
    def display_streaming_response(self, response, result_queue):
        """Wyświetl streaming response w czasie rzeczywistym + monitoring"""
        global STOP_REQUESTED
        
        self.start_time = time.time()
        self.last_token_time = self.start_time
        collected_text = ""
        
        # Monitoring w tle
        monitoring_active = True
        
        def background_monitor():
            while monitoring_active and not STOP_REQUESTED:
                time.sleep(self.CHECK_INTERVAL)
                
                if STOP_REQUESTED:
                    break
                
                current_time = time.time()
                silence_duration = current_time - self.last_token_time
                elapsed_min = (current_time - self.start_time) / 60
                
                # Sprawdź status
                model_active, model_status = self.check_model_status()
                load_ok, load_status = self.check_system_load()
                
                # Ostrzeżenia o ciszy
                if silence_duration > self.SILENCE_WARNING:
                    if silence_duration > self.MAX_SILENCE:
                        self.logger.log("ERROR", f"Model nie odpowiada od {silence_duration/60:.1f} minut")
                        result_queue.put(("timeout", "Model przestał odpowiadać"))
                        return
                    elif silence_duration > self.SILENCE_WARNING and self.silence_warnings < 3:
                        self.silence_warnings += 1
                        self.logger.log("WARNING", f"Brak nowych tokenów od {silence_duration/60:.1f} minut")
                
                # Status raport (tylko jeśli są tokeny lub długo czekamy)
                if self.token_count > 0 or elapsed_min > 2:
                    tokens_per_min = self.token_count / elapsed_min if elapsed_min > 0 else 0
                    self.logger.log("INFO", f"Tokeny: {self.token_count}, Tempo: {tokens_per_min:.1f}/min, {model_status}, {load_status}", silent=True)
        
        # Uruchom monitoring w tle
        monitor_thread = threading.Thread(target=background_monitor)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}🤖 {self.model_name}{Colors.RESET}")
        print(f"{Colors.DIM}{'─' * 60}{Colors.RESET}")
        
        try:
            # Główna pętla wyświetlania - DOKŁADNIE JAK OLLAMA!
            for line in response.iter_lines(decode_unicode=True):
                if STOP_REQUESTED:
                    break
                    
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Wyświetl token na żywo
                        if 'response' in data:
                            token = data['response']
                            print(token, end='', flush=True)  # LIVE DISPLAY!
                            
                            collected_text += token
                            self.token_count += 1
                            self.last_token_time = time.time()
                            self.silence_warnings = 0  # Reset warnings
                            
                            # Log co 50 tokenów (bez wyświetlania)
                            if self.token_count % 50 == 0:
                                self.logger.log("DEBUG", f"Token #{self.token_count}", silent=True)
                        
                        # Sprawdź czy skończone
                        if data.get('done', False):
                            monitoring_active = False
                            total_time = time.time() - self.start_time
                            
                            print(f"\n{Colors.DIM}{'─' * 60}{Colors.RESET}")
                            
                            # Statystyki końcowe
                            tokens_per_min = self.token_count / (total_time / 60) if total_time > 60 else self.token_count
                            
                            print(f"{Colors.GREEN}✅ Ukończone!{Colors.RESET} {Colors.DIM}{self.token_count} tokenów w {total_time/60:.1f} minut ({tokens_per_min:.1f} tok/min){Colors.RESET}")
                            
                            # Zapisz statystyki
                            stats = {
                                "model": self.model_name,
                                "tokens": self.token_count,
                                "time_minutes": total_time / 60,
                                "tokens_per_minute": tokens_per_min,
                                "completed": True,
                                "timestamp": datetime.now().isoformat()
                            }
                            self.logger.save_stats(stats)
                            self.logger.log("SUCCESS", f"Zadanie ukończone: {self.token_count} tokenów")
                            
                            result_queue.put(("success", {
                                "response": collected_text,
                                "token_count": self.token_count,
                                "total_time": total_time,
                                "tokens_per_minute": tokens_per_min
                            }))
                            return
                            
                    except json.JSONDecodeError:
                        continue
                        
        except KeyboardInterrupt:
            monitoring_active = False
            print(f"\n\n{Colors.YELLOW}⚠️ Przerwane przez użytkownika{Colors.RESET}")
            self.logger.log("WARNING", f"Przerwane przez użytkownika po {self.token_count} tokenach")
            result_queue.put(("interrupted", "Przerwane przez użytkownika"))
            
        except Exception as e:
            monitoring_active = False
            print(f"\n\n{Colors.RED}❌ Błąd: {e}{Colors.RESET}")
            self.logger.log("ERROR", f"Błąd streaming: {e}")
            result_queue.put(("error", str(e)))
        
        monitoring_active = False

class OptimizedAISystem:
    def __init__(self, ollama_host="localhost", ollama_port=11434):
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        
        # HIERARCHIA DOSTOSOWANA DO RZECZYWISTYCH MODELI
        self.model_hierarchies = {
            "coding": [
                # Tier 1 - Specjaliści kodowania
                {"name": "deepseek-coder:33b", "type": "🏆 Elite Coder", "desc": "Najlepszy ekspert (18GB)", "tier": 1, "size_gb": 18},
                {"name": "codellama:13b", "type": "🎯 Code Specialist", "desc": "Sprawdzony specjalista (7.4GB)", "tier": 1, "size_gb": 7.4},
                
                # Tier 2 - Zaawansowani
                {"name": "deepseek-r1:32b", "type": "🧠 Reasoning+Code", "desc": "Myślenie + kod (19GB)", "tier": 2, "size_gb": 19},
                {"name": "qwen2.5:14b", "type": "⚡ Smart Coder", "desc": "Inteligentny i wszechstronny (9GB)", "tier": 2, "size_gb": 9},
                
                # Tier 3 - Sprawdzone standardowe
                {"name": "llama3.1:8b", "type": "🔧 Reliable", "desc": "Niezawodny standard (4.9GB)", "tier": 3, "size_gb": 4.9},
                {"name": "hermes3:8b", "type": "🎭 Creative Coder", "desc": "Kreatywny i wszechstronny (4.7GB)", "tier": 3, "size_gb": 4.7},
                
                # Tier 4 - Szybkie fallback
                {"name": "llama3.2:3b", "type": "🚀 Fast", "desc": "Ultra szybki (2GB)", "tier": 4, "size_gb": 2},
                {"name": "phi3:mini", "type": "⚡ Micro", "desc": "Mały ale sprytny (2.2GB)", "tier": 4, "size_gb": 2.2}
            ],
            
            "reasoning": [
                # Tier 1 - Eksperci myślenia
                {"name": "qwen3:30b", "type": "🏆 Mega Brain", "desc": "Potężny umysł (18GB)", "tier": 1, "size_gb": 18},
                {"name": "mixtral:8x7b", "type": "🧠 Reasoning Master", "desc": "Mistrz analizy (26GB)", "tier": 1, "size_gb": 26},
                {"name": "dolphin-mixtral:8x7b", "type": "🐬 Uncensored Expert", "desc": "Bez ograniczeń (26GB)", "tier": 1, "size_gb": 26},
                
                # Tier 2 - Zaawansowani 
                {"name": "qwen2.5:14b", "type": "💡 Advanced Thinker", "desc": "Zaawansowane myślenie (9GB)", "tier": 2, "size_gb": 9},
                {"name": "solar:10.7b", "type": "☀️ Solar Power", "desc": "Mocny performancer (6.1GB)", "tier": 2, "size_gb": 6.1},
                {"name": "deepseek-r1:14b", "type": "🔬 Deep Reasoner", "desc": "Głębokie analizy (9GB)", "tier": 2, "size_gb": 9},
                
                # Tier 3 - Sprawdzone
                {"name": "llama3.1:8b", "type": "⚖️ Balanced", "desc": "Doskonały balans (4.9GB)", "tier": 3, "size_gb": 4.9},
                {"name": "mistral:7b", "type": "🎯 Focused", "desc": "Skupiony i precyzyjny (4.4GB)", "tier": 3, "size_gb": 4.4},
                
                # Tier 4 - Szybkie
                {"name": "llama3.2:3b", "type": "🚀 Quick Think", "desc": "Szybkie myślenie (2GB)", "tier": 4, "size_gb": 2}
            ],
            
            "general": [
                # Uniwersalne modele dla zadań ogólnych
                {"name": "llama3.1:8b", "type": "🎯 Universal", "desc": "Najlepszy ogólny wybór (4.9GB)", "tier": 1, "size_gb": 4.9},
                {"name": "llama3.2:3b", "type": "⚡ Quick Response", "desc": "Szybkie odpowiedzi (2GB)", "tier": 1, "size_gb": 2},
                {"name": "mistral:7b", "type": "🔧 Workhorse", "desc": "Solidny koń roboczy (4.4GB)", "tier": 2, "size_gb": 4.4},
                {"name": "hermes3:8b", "type": "🎭 Versatile", "desc": "Wszechstronny asystent (4.7GB)", "tier": 2, "size_gb": 4.7}
            ],
            
            "polish": [
                # Specjalnie dla zadań po polsku
                {"name": "SpeakLeash/bielik-11b-v2.3-instruct", "type": "🇵🇱 Polish Expert", "desc": "Model dla języka polskiego (6.7GB)", "tier": 1, "size_gb": 6.7},
                {"name": "llama3.1:8b", "type": "🌍 Multilingual", "desc": "Dobry w wielu językach (4.9GB)", "tier": 2, "size_gb": 4.9},
                {"name": "qwen2.5:14b", "type": "🗣️ Polyglot", "desc": "Zaawansowany multilingualny (9GB)", "tier": 2, "size_gb": 9}
            ]
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Obsługa Ctrl+C"""
        global STOP_REQUESTED
        STOP_REQUESTED = True
        print(f"\n{Colors.YELLOW}🛑 Otrzymano sygnał przerwania... Kończę bezpiecznie...{Colors.RESET}")
    
    def analyze_task_advanced(self, task_description):
        """Zaawansowana analiza zadania z wieloma kategoriami"""
        task_lower = task_description.lower()
        
        # Sprawdź język
        polish_indicators = ["napisz", "stwórz", "zrób", "jak", "dlaczego", "gdzie", "kiedy", "czy", "funkcja", "kod", "python"]
        is_polish = any(word in task_lower for word in polish_indicators) and not any(word in task_lower for word in ["write", "create", "how", "why", "function"])
        
        # Kategorie zadań
        coding_keywords = ["kod", "code", "python", "javascript", "function", "program", "bug", "debug", "algorithm", "script", "api", "class", "method"]
        reasoning_keywords = ["analiza", "analysis", "strategia", "strategy", "plan", "zaprojektuj", "design", "architektura", "optimization", "compare", "evaluate"]
        
        categories = []
        
        if is_polish:
            categories.append("polish")
        
        if any(word in task_lower for word in coding_keywords):
            categories.append("coding")
        elif any(word in task_lower for word in reasoning_keywords):
            categories.append("reasoning")
        else:
            categories.append("general")
        
        return categories
    
    def select_best_models(self, categories, max_models=4):
        """Wybierz najlepsze modele dla danych kategorii"""
        selected_models = []
        
        # Dla każdej kategorii wybierz modele
        for category in categories:
            if category in self.model_hierarchies:
                hierarchy = self.model_hierarchies[category]
                
                # Wybierz po 1-2 modele z każdego tier
                for tier in [1, 2, 3, 4]:
                    tier_models = [m for m in hierarchy if m["tier"] == tier]
                    if tier_models:
                        # Wybierz najlepszy z tego tier (pierwszy)
                        if tier_models[0] not in selected_models:
                            selected_models.append(tier_models[0])
                        
                        # Dla tier 1-2, dodaj drugi model jako backup
                        if tier <= 2 and len(tier_models) > 1:
                            if tier_models[1] not in selected_models:
                                selected_models.append(tier_models[1])
                
                # Nie więcej niż max_models na kategorię
                if len(selected_models) >= max_models:
                    break
        
        # Jeśli nie ma wystarczająco modeli, dodaj uniwersalne
        if len(selected_models) < max_models:
            general_models = self.model_hierarchies["general"]
            for model in general_models:
                if model not in selected_models and len(selected_models) < max_models:
                    selected_models.append(model)
        
        return selected_models[:max_models]
    
    def try_model_interactive(self, model_config, prompt):
        """Próbuj model z interaktywnym wyświetlaniem"""
        global STOP_REQUESTED
        
        if STOP_REQUESTED:
            return {"success": False, "error": "Przerwane przez użytkownika"}
        
        model_name = model_config["name"]
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}🎯 Próbuję: {model_name}{Colors.RESET}")
        print(f"{Colors.DIM}   Typ: {model_config['type']} - {model_config['desc']}{Colors.RESET}")
        print(f"{Colors.DIM}   Tier: {model_config['tier']}, Rozmiar: {model_config['size_gb']}GB{Colors.RESET}")
        print(f"{Colors.DIM}   ⏰ Unlimited time - Ctrl+C aby przerwać{Colors.RESET}")
        
        # Monitor
        monitor = InteractiveMonitor(model_name, self.ollama_url)
        
        # Sprawdź status przed startem
        model_active, status = monitor.check_model_status()
        load_ok, load_status = monitor.check_system_load()
        print(f"{Colors.DIM}   💻 Status: {status}, {load_status}{Colors.RESET}")
        
        # Rozpocznij streaming
        response, error = monitor.start_streaming_with_display(prompt)
        if not response:
            return {"success": False, "error": error}
        
        # Live display w osobnym wątku
        result_queue = Queue()
        display_thread = threading.Thread(
            target=monitor.display_streaming_response,
            args=(response, result_queue)
        )
        
        start_time = time.time()
        display_thread.start()
        
        # Czekaj na rezultat lub przerwanie
        try:
            while display_thread.is_alive() and not STOP_REQUESTED:
                try:
                    status, result = result_queue.get(timeout=1)
                    break
                except Empty:
                    continue
            else:
                if STOP_REQUESTED:
                    return {"success": False, "error": "Przerwane przez użytkownika"}
                status, result = "error", "Thread zakończony bez wyniku"
            
            display_thread.join(timeout=5)
            response_time = time.time() - start_time
            
            if status == "success":
                return {
                    "success": True,
                    "response": result["response"],
                    "model_used": model_name,
                    "model_type": model_config["type"],
                    "model_tier": model_config["tier"],
                    "model_size_gb": model_config["size_gb"],
                    "response_time": round(response_time, 2),
                    "token_count": result["token_count"],
                    "tokens_per_minute": round(result["tokens_per_minute"], 1),
                    "interactive_display": True
                }
            else:
                return {
                    "success": False,
                    "error": result,
                    "model_attempted": model_name,
                    "response_time": round(response_time, 2),
                    "reason": status
                }
                
        except KeyboardInterrupt:
            STOP_REQUESTED = True
            return {"success": False, "error": "Przerwane przez użytkownika"}
    
    def interactive_session(self, task_description):
        """Interaktywna sesja z inteligentną selekcją modeli"""
        global STOP_REQUESTED
        
        print(f"{Colors.BOLD}{Colors.HEADER}🚀 Agent Zero Phase 4 - Optimized AI System{Colors.RESET}")
        print(f"{Colors.DIM}{'=' * 70}{Colors.RESET}")
        print(f"{Colors.CYAN}Czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}")
        print(f"{Colors.DIM}💡 Inteligentna selekcja z Twojej kolekcji 26 modeli{Colors.RESET}")
        
        print(f"\n{Colors.BOLD}📝 Zadanie:{Colors.RESET}")
        print(f"{Colors.BLUE}{task_description}{Colors.RESET}")
        
        # Inteligentna analiza zadania
        categories = self.analyze_task_advanced(task_description)
        print(f"\n{Colors.BOLD}🧠 Analiza zadania:{Colors.RESET}")
        print(f"{Colors.DIM}   Kategorie: {', '.join(categories)}{Colors.RESET}")
        
        # Dobierz najlepsze modele
        selected_models = self.select_best_models(categories)
        
        print(f"\n{Colors.BOLD}🎯 Inteligentnie dobrane modele:{Colors.RESET}")
        for i, model in enumerate(selected_models, 1):
            tier_color = Colors.GREEN if model["tier"] == 1 else Colors.YELLOW if model["tier"] == 2 else Colors.CYAN
            print(f"{Colors.DIM}   {i}. {model['name']} ({tier_color}Tier {model['tier']}{Colors.RESET}{Colors.DIM}, {model['size_gb']}GB) - {model['desc']}{Colors.RESET}")
        
        print(f"\n{Colors.DIM}📁 Logi zapisywane w: {LOG_DIR}/{Colors.RESET}")
        print(f"{Colors.YELLOW}🔧 Naciśnij Ctrl+C aby przerwać w dowolnym momencie{Colors.RESET}")
        
        # Próbuj modele w inteligentnej kolejności
        for i, model_config in enumerate(selected_models, 1):
            if STOP_REQUESTED:
                break
                
            tier_emoji = "🏆" if model_config["tier"] == 1 else "⭐" if model_config["tier"] == 2 else "🔧" if model_config["tier"] == 3 else "⚡"
            print(f"\n{Colors.BOLD}{Colors.BLUE}{tier_emoji * 3} PRÓBA {i}/{len(selected_models)} {tier_emoji * 3}{Colors.RESET}")
            
            result = self.try_model_interactive(model_config, task_description)
            
            if result["success"]:
                print(f"\n{Colors.BOLD}{Colors.GREEN}🎊 OPTIMAL SUCCESS!{Colors.RESET}")
                print(f"{Colors.DIM}📊 Szczegóły:{Colors.RESET}")
                print(f"{Colors.DIM}   • Model: {result['model_used']} (Tier {result['model_tier']}, {result['model_size_gb']}GB){Colors.RESET}")
                print(f"{Colors.DIM}   • Typ: {result['model_type']}{Colors.RESET}")
                print(f"{Colors.DIM}   • Czas: {result['response_time']}s{Colors.RESET}")
                print(f"{Colors.DIM}   • Tokeny: {result['token_count']}{Colors.RESET}")
                print(f"{Colors.DIM}   • Tempo: {result['tokens_per_minute']} tok/min{Colors.RESET}")
                return result
            else:
                print(f"\n{Colors.RED}❌ Model zawiódł: {result.get('reason', 'nieznany błąd')}{Colors.RESET}")
                print(f"{Colors.DIM}   Błąd: {result.get('error', 'brak szczegółów')}{Colors.RESET}")
                
                if i < len(selected_models) and not STOP_REQUESTED:
                    print(f"{Colors.CYAN}🔄 Przechodzę do następnego modelu w hierarchii...{Colors.RESET}")
                    time.sleep(1)
        
        if STOP_REQUESTED:
            print(f"\n{Colors.YELLOW}🛑 Sesja przerwana przez użytkownika{Colors.RESET}")
        else:
            print(f"\n{Colors.RED}💥 Wszystkie wybrane modele zawiodły{Colors.RESET}")
        
        return {"success": False, "error": "Sesja zakończona"}

def main():
    """Główna funkcja z optimized experience"""
    try:
        ai_system = OptimizedAISystem()
        
        # Test zadania
        task = input(f"\n{Colors.BOLD}💭 Wpisz swoje zadanie (lub Enter dla testu): {Colors.RESET}").strip()
        if not task:
            task = "Napisz kompletną funkcję sortującą w Python z komentarzami i przykładami użycia"
        
        result = ai_system.interactive_session(task)
        
        if result["success"]:
            print(f"\n{Colors.BOLD}{Colors.GREEN}✅ Sesja zakończona sukcesem!{Colors.RESET}")
        else:
            print(f"\n{Colors.BOLD}{Colors.RED}❌ Sesja zakończona niepowodzeniem{Colors.RESET}")
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}🛑 Program przerwany przez użytkownika{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}💥 Nieoczekiwany błąd: {e}{Colors.RESET}")
    
    print(f"\n{Colors.DIM}📁 Logi i statystyki dostępne w: {LOG_DIR}/{Colors.RESET}")
    print(f"{Colors.CYAN}👋 Do widzenia!{Colors.RESET}")

if __name__ == "__main__":
    main()