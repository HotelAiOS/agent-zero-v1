#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Unlimited Time AI System z Logowaniem
Jeśli model generuje tokeny = bez limitu czasu
Sprawdzanie co 30s, 5min bez reakcji = fail + logi + przełączenie
"""

import requests
import time
import threading
import psutil
import os
import json
from datetime import datetime
from queue import Queue, Empty

# Katalog na logi
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

class ModelLogger:
    def __init__(self, model_name):
        self.model_name = model_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(LOG_DIR, f"model_{model_name.replace(':', '_')}_{timestamp}.log")
        self.debug_file = os.path.join(LOG_DIR, f"debug_{model_name.replace(':', '_')}_{timestamp}.json")
        
    def log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_entry + "\n")
        except Exception as e:
            print(f"Błąd zapisu logu: {e}")
    
    def log_debug_data(self, data):
        try:
            with open(self.debug_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Błąd zapisu debug: {e}")

class UnlimitedModelMonitor:
    def __init__(self, model_name, ollama_url):
        self.model_name = model_name
        self.ollama_url = ollama_url
        self.logger = ModelLogger(model_name)
        
        # Statystyki
        self.token_count = 0
        self.last_token_time = None
        self.start_time = None
        self.cpu_usage = 0
        self.memory_usage = 0
        
        # Konfiguracja monitoringu
        self.CHECK_INTERVAL = 30  # Sprawdzaj co 30s
        self.MAX_SILENCE = 300    # 5min bez reakcji = fail
        
        self.logger.log("INFO", f"Monitor zainicjowany dla {model_name}")
        
    def find_ollama_processes(self):
        """Znajdź procesy Ollama"""
        processes = []
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'cmdline']):
                try:
                    if 'ollama' in proc.info['name'].lower():
                        processes.append(proc.info)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except Exception as e:
            self.logger.log("ERROR", f"Błąd wyszukiwania procesów: {e}")
        return processes
    
    def check_system_resources(self):
        """Sprawdź zużycie zasobów systemowych"""
        try:
            processes = self.find_ollama_processes()
            
            if not processes:
                return False, "Brak procesów Ollama", {}
            
            total_cpu = 0
            total_memory = 0
            active_processes = 0
            
            for proc in processes:
                cpu = proc.get('cpu_percent', 0)
                memory = proc.get('memory_percent', 0)
                
                total_cpu += cpu
                total_memory += memory
                
                if cpu > 0.5:  # Aktywny jeśli używa więcej niż 0.5% CPU
                    active_processes += 1
            
            self.cpu_usage = total_cpu
            self.memory_usage = total_memory
            
            resources = {
                "cpu_percent": total_cpu,
                "memory_percent": total_memory,
                "active_processes": active_processes,
                "total_processes": len(processes)
            }
            
            is_working = active_processes > 0
            status = f"CPU: {total_cpu:.1f}%, RAM: {total_memory:.1f}%, Aktywne: {active_processes}/{len(processes)}"
            
            return is_working, status, resources
            
        except Exception as e:
            self.logger.log("ERROR", f"Błąd sprawdzania zasobów: {e}")
            return False, f"Błąd: {e}", {}
    
    def start_streaming_request(self, prompt):
        """Rozpocznij streaming request"""
        try:
            self.logger.log("INFO", f"Rozpoczynam streaming request")
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"Agent Zero: {prompt}\n\nOdpowiedź:",
                    "stream": True
                },
                stream=True,
                timeout=60  # Timeout tylko na nawiązanie połączenia
            )
            self.logger.log("INFO", f"Streaming request uruchomiony, status: {response.status_code}")
            return response
        except Exception as e:
            self.logger.log("ERROR", f"Błąd uruchamiania streaming: {e}")
            return None
    
    def monitor_streaming_unlimited(self, response, result_queue):
        """Monitor streaming z unlimited time - tylko sprawdza czy model żyje"""
        self.start_time = time.time()
        self.last_token_time = self.start_time
        collected_text = ""
        
        self.logger.log("INFO", "Rozpoczynam monitoring bez limitu czasu")
        
        # Wątek monitoringu zasobów
        monitoring_active = True
        
        def resource_monitor():
            last_check = time.time()
            silence_start = None
            
            while monitoring_active:
                time.sleep(self.CHECK_INTERVAL)  # Sprawdzaj co 30s
                
                current_time = time.time()
                time_since_last_token = current_time - self.last_token_time
                
                # Sprawdź zasoby systemowe
                working, status, resources = self.check_system_resources()
                
                # Sprawdź czy model przestał generować tokeny
                if time_since_last_token > self.MAX_SILENCE:
                    if silence_start is None:
                        silence_start = current_time - self.MAX_SILENCE
                        
                    silence_duration = current_time - silence_start
                    
                    self.logger.log("WARNING", f"Cisza przez {silence_duration/60:.1f}min, {status}")
                    
                    # Log debug data
                    debug_data = {
                        "model": self.model_name,
                        "silence_duration_minutes": silence_duration / 60,
                        "tokens_generated": self.token_count,
                        "total_time_minutes": (current_time - self.start_time) / 60,
                        "system_resources": resources,
                        "last_token_time": datetime.fromtimestamp(self.last_token_time).isoformat(),
                        "failure_reason": "max_silence_exceeded"
                    }
                    self.logger.log_debug_data(debug_data)
                    
                    result_queue.put(("timeout", f"Brak tokenów przez {silence_duration/60:.1f} minut"))
                    return
                else:
                    silence_start = None  # Reset licznika ciszy
                
                # Raport postępu co 30s
                elapsed_min = (current_time - self.start_time) / 60
                if self.token_count > 0:
                    tokens_per_min = self.token_count / elapsed_min
                    self.logger.log("INFO", f"Postęp: {self.token_count} tokenów, {tokens_per_min:.1f} tok/min, czas: {elapsed_min:.1f}min, {status}")
                else:
                    self.logger.log("INFO", f"Oczekiwanie na pierwsze tokeny... czas: {elapsed_min:.1f}min, {status}")
                
                last_check = current_time
        
        # Uruchom monitoring zasobów w osobnym wątku
        resource_thread = threading.Thread(target=resource_monitor)
        resource_thread.daemon = True
        resource_thread.start()
        
        try:
            # Główna pętla odczytu tokenów
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        data = json.loads(line)
                        
                        # Sprawdź czy otrzymaliśmy token
                        if 'response' in data:
                            token = data['response']
                            collected_text += token
                            self.token_count += 1
                            self.last_token_time = time.time()
                            
                            # Log każdy 10 token żeby nie spamować
                            if self.token_count % 10 == 0:
                                self.logger.log("DEBUG", f"Token #{self.token_count}: {repr(token[:20])}")
                            
                        # Sprawdź czy model skończył
                        if data.get('done', False):
                            monitoring_active = False
                            total_time = time.time() - self.start_time
                            
                            self.logger.log("SUCCESS", f"Model ukończył zadanie: {self.token_count} tokenów w {total_time/60:.1f} minut")
                            
                            result_queue.put(("success", {
                                "response": collected_text,
                                "token_count": self.token_count,
                                "total_time": total_time
                            }))
                            return
                            
                    except json.JSONDecodeError:
                        continue  # Zignoruj niepoprawne JSON
                        
        except Exception as e:
            monitoring_active = False
            self.logger.log("ERROR", f"Błąd w głównej pętli streaming: {e}")
            result_queue.put(("error", str(e)))
        
        monitoring_active = False

class UnlimitedAISystem:
    def __init__(self, ollama_host="localhost", ollama_port=11434):
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        
        # Hierarchia modeli - od najlepszych do fallback
        self.model_hierarchy = [
            {
                "name": "codellama:13b",
                "type": "specialist",
                "description": "Specjalista kodowania - unlimited time",
                "quality_score": 10
            },
            {
                "name": "deepseek-coder:33b", 
                "type": "expert",
                "description": "Ekspert kodowania - unlimited time",
                "quality_score": 10
            },
            {
                "name": "llama3.1:8b",
                "type": "generalist", 
                "description": "Generalista z kodowaniem",
                "quality_score": 8
            },
            {
                "name": "llama3.2:3b",
                "type": "fallback",
                "description": "Szybki fallback",
                "quality_score": 6
            }
        ]
    
    def try_model_unlimited(self, model_config, prompt):
        """Próbuj model bez limitu czasu"""
        model_name = model_config["name"]
        
        print(f"\n🎯 Model: {model_name}")
        print(f"   Typ: {model_config['type']}")  
        print(f"   Opis: {model_config['description']}")
        print(f"   ⏰ UNLIMITED TIME - czeka dopóki generuje tokeny!")
        print(f"   📊 Sprawdzanie co 30s, fail po 5min bez tokenów")
        
        # Utwórz monitor
        monitor = UnlimitedModelMonitor(model_name, self.ollama_url)
        
        # Sprawdź zasoby przed startem
        working, status, resources = monitor.check_system_resources()
        print(f"💻 Status systemu: {status}")
        
        # Rozpocznij streaming
        response = monitor.start_streaming_request(prompt)
        if not response:
            return {"success": False, "error": "Nie udało się rozpocząć streaming"}
        
        # Monitor w osobnym wątku  
        result_queue = Queue()
        monitor_thread = threading.Thread(
            target=monitor.monitor_streaming_unlimited,
            args=(response, result_queue)
        )
        
        start_time = time.time()
        monitor_thread.start()
        
        print(f"🚀 Monitoring uruchomiony - czekam na tokeny...")
        
        # Czekaj na rezultat (bez limitu czasu!)
        try:
            status, result = result_queue.get()  # Bez timeout!
            monitor_thread.join(timeout=30)
            
            response_time = time.time() - start_time
            
            if status == "success":
                tokens_per_min = result["token_count"] / (result["total_time"] / 60)
                return {
                    "success": True,
                    "response": result["response"],
                    "model_used": model_name,
                    "model_type": model_config["type"],
                    "response_time": round(response_time, 2),
                    "token_count": result["token_count"],
                    "tokens_per_minute": round(tokens_per_min, 1),
                    "quality_score": model_config["quality_score"],
                    "unlimited_time_used": True
                }
            else:
                return {
                    "success": False,
                    "error": result,
                    "model_attempted": model_name,
                    "response_time": round(response_time, 2),
                    "reason": status,
                    "token_count": monitor.token_count
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Błąd monitoringu: {str(e)}",
                "model_attempted": model_name
            }
    
    def generate_response(self, task_description):
        """Główna metoda z unlimited time"""
        print(f"\n🎯 ZADANIE: {task_description}")
        print(f"⏰ UNLIMITED TIME MODE - modele mają czas ile potrzebują!")
        
        # Wybierz modele do testowania
        if any(word in task_description.lower() for word in ["kod", "code", "python", "function", "program"]):
            models_to_try = [m for m in self.model_hierarchy if m["type"] in ["specialist", "expert", "generalist", "fallback"]]
        else:
            models_to_try = [m for m in self.model_hierarchy if m["type"] in ["generalist", "fallback"]]
        
        print(f"📋 Hierarchia modeli: {[m['name'] for m in models_to_try]}")
        
        # Próbuj modele w hierarchii
        for i, model_config in enumerate(models_to_try, 1):
            print(f"\n{'='*25} PRÓBA {i}/{len(models_to_try)} {'='*25}")
            
            result = self.try_model_unlimited(model_config, task_description)
            
            if result["success"]:
                print(f"\n🏆 UNLIMITED SUCCESS!")
                if result.get("token_count", 0) > 0:
                    print(f"📝 Wygenerowano {result['token_count']} tokenów!")
                    print(f"⚡ Tempo: {result['tokens_per_minute']} tokenów/min")
                return result
            else:
                print(f"\n❌ FAILED: {result.get('reason', 'nieznany')}")
                print(f"   Błąd: {result.get('error', 'brak')}")
                if result.get("token_count", 0) > 0:
                    print(f"   🔍 Model wygenerował {result['token_count']} tokenów przed fail")
                    
                print(f"📊 Logi zapisane w: logs/")
        
        return {"success": False, "error": "Wszystkie modele w hierarchii zawiodły"}

def main():
    print("🚀 Agent Zero Phase 4 - UNLIMITED TIME System")
    print("=" * 60)
    print(f"Czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 Jeśli model generuje tokeny = UNLIMITED TIME!")
    print("🔍 Sprawdzanie co 30s, 5min bez reakcji = fail + logs + next model")
    
    ai_system = UnlimitedAISystem()
    
    # Test zadania
    task = "Napisz kompletną funkcję sortującą w Python z komentarzami i przykładami użycia"
    
    print(f"\n🧪 Test unlimited time system:")
    
    result = ai_system.generate_response(task)
    
    if result["success"]:
        print(f"\n🎊 UNLIMITED SUCCESS!")
        print(f"   Model: {result['model_used']} ({result['model_type']})")
        print(f"   Czas: {result['response_time']}s")
        print(f"   Tokeny: {result['token_count']}")
        print(f"   Tempo: {result['tokens_per_minute']} tok/min")
        print(f"   Jakość: {result['quality_score']}/10")
        print(f"   Odpowiedź: {result['response'][:300]}...")
    else:
        print(f"\n💥 ULTIMATE FAIL: {result.get('error')}")
    
    print(f"\n📁 Logi dostępne w katalogu: {LOG_DIR}/")
    print(f"🎯 Unlimited time + smart monitoring = najlepsza jakość!")

if __name__ == "__main__":
    main()