#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Cierpliwy System AI z Kontrolą Życia
Daje specjalistom czas (do 1h), ale z kontrolą czy pracują (5min bez znaku życia = przełączenie)
"""

import requests
import time
import threading
from datetime import datetime
from queue import Queue, Empty

class ModelTimeout(Exception):
    pass

class ModelUnresponsive(Exception):
    pass

def heartbeat_monitor(req_func, result_queue, heartbeat_interval=300, max_total_time=3600):
    """
    Monitoruje model przez max 1h z heartbeat co 5min
    Jeśli 5min bez odpowiedzi = model uznany za martwy
    """
    start_time = time.time()
    last_heartbeat = start_time
    
    while True:
        try:
            # Próba wywołania modelu z krótkim timeoutem
            response = req_func(timeout=20)
            
            if response.status_code == 200:
                # Model odpowiedział - sukces!
                result_queue.put(("success", response))
                return
            else:
                # Model odpowiedział błędem, ale żyje
                current_time = time.time()
                last_heartbeat = current_time
                
                # Sprawdź czy nie przekroczono max czasu
                if current_time - start_time > max_total_time:
                    result_queue.put(("timeout", f"Model nie ukończył zadania w {max_total_time/60:.0f} minut"))
                    return
                    
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 💓 Model żyje, ale wciąż pracuje...")
                time.sleep(30)  # Sprawdzaj częściej jeśli model daje odpowiedzi błędne
                
        except requests.Timeout:
            current_time = time.time()
            
            # Sprawdź czy to pierwszy timeout czy kolejny
            time_since_last_heartbeat = current_time - last_heartbeat
            
            if time_since_last_heartbeat > heartbeat_interval:
                # Za długo bez znaku życia
                result_queue.put(("unresponsive", f"Brak znaku życia przez {time_since_last_heartbeat/60:.1f} minut"))
                return
            elif current_time - start_time > max_total_time:
                # Przekroczony maksymalny czas
                result_queue.put(("timeout", f"Limit czasu {max_total_time/60:.0f} minut przekroczony"))
                return
            else:
                # Model może pracować - daj mu więcej czasu
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Specjalista myśli... ({(current_time-start_time)/60:.1f}min)")
                time.sleep(30)  # Sprawdzaj co 30s podczas timeout
                
        except requests.RequestException as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️ Problem z połączeniem: {e}")
            time.sleep(10)
            continue

class PatientAISystem:
    def __init__(self, ollama_host="localhost", ollama_port=11434):
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        
        # Modele uporządkowane według specjalizacji i reliability
        self.model_hierarchy = {
            "coding": [
                {
                    "name": "codellama:13b",
                    "type": "specialist", 
                    "description": "Specjalista od kodowania",
                    "max_wait_time": 3600,  # 1h dla specjalisty
                    "heartbeat_interval": 300,  # 5min bez znaku życia = fail
                    "quality_score": 10
                },
                {
                    "name": "deepseek-coder:33b",
                    "type": "expert", 
                    "description": "Ekspert kodowania (duży model)",
                    "max_wait_time": 3600,  # 1h dla eksperta
                    "heartbeat_interval": 300,
                    "quality_score": 10
                },
                {
                    "name": "llama3.1:8b",
                    "type": "generalist", 
                    "description": "Generalista z dobrym kodowaniem",
                    "max_wait_time": 300,  # 5min dla generalisty
                    "heartbeat_interval": 120,
                    "quality_score": 8
                },
                {
                    "name": "llama3.2:3b",
                    "type": "fallback", 
                    "description": "Szybki fallback",
                    "max_wait_time": 120,
                    "heartbeat_interval": 60,
                    "quality_score": 6
                }
            ],
            "reasoning": [
                {
                    "name": "qwen2.5:14b",
                    "type": "specialist", 
                    "description": "Specjalista od zaawansowanego myślenia",
                    "max_wait_time": 3600,
                    "heartbeat_interval": 300,
                    "quality_score": 10
                },
                {
                    "name": "mixtral:8x7b",
                    "type": "expert", 
                    "description": "Ekspert złożonego myślenia",
                    "max_wait_time": 3600,
                    "heartbeat_interval": 300,
                    "quality_score": 10
                },
                {
                    "name": "mistral:7b",
                    "type": "generalist", 
                    "description": "Dobry w analizie i myśleniu",
                    "max_wait_time": 300,
                    "heartbeat_interval": 120,
                    "quality_score": 9
                },
                {
                    "name": "llama3.1:8b",
                    "type": "fallback", 
                    "description": "Niezawodny fallback",
                    "max_wait_time": 180,
                    "heartbeat_interval": 60,
                    "quality_score": 8
                }
            ],
            "general": [
                {
                    "name": "llama3.1:8b",
                    "type": "primary", 
                    "description": "Główny model ogólny",
                    "max_wait_time": 300,
                    "heartbeat_interval": 120,
                    "quality_score": 8
                },
                {
                    "name": "llama3.2:3b",
                    "type": "fast", 
                    "description": "Szybkie odpowiedzi",
                    "max_wait_time": 120,
                    "heartbeat_interval": 60,
                    "quality_score": 7
                },
                {
                    "name": "mistral:7b",
                    "type": "alternative", 
                    "description": "Alternatywny wybór",
                    "max_wait_time": 300,
                    "heartbeat_interval": 120,
                    "quality_score": 8
                }
            ]
        }
    
    def analyze_task(self, task_description):
        """Analiza zadania dla wyboru odpowiedniej hierarchii modeli"""
        task_lower = task_description.lower()
        
        coding_keywords = [
            "kod", "code", "python", "javascript", "function", "program", 
            "bug", "debug", "algorithm", "implementuj", "napisz program", "script"
        ]
        
        reasoning_keywords = [
            "analiza", "analysis", "strategia", "strategy", "zaprojektuj", "design",
            "architektura", "optimization", "skalowanie", "scaling", "plan", "solve"
        ]
        
        if any(word in task_lower for word in coding_keywords):
            return "coding"
        elif any(word in task_lower for word in reasoning_keywords):
            return "reasoning" 
        else:
            return "general"
    
    def try_model_with_patience(self, model_config, task_description):
        """Próbuje model z cierpliwością ale kontrolą życia"""
        
        model_name = model_config["name"]
        max_wait = model_config["max_wait_time"]
        heartbeat = model_config["heartbeat_interval"]
        
        print(f"\n🎯 Próbuję: {model_name}")
        print(f"   Typ: {model_config['type']}")
        print(f"   Opis: {model_config['description']}")
        print(f"   Max czas: {max_wait//60}min, kontrola życia co: {heartbeat//60}min")
        
        def make_request(timeout=30):
            return requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": f"Agent Zero AI: {task_description}\n\nPodaj konkretną, praktyczną odpowiedź:",
                    "stream": False,
                    "options": {"temperature": 0.2, "max_tokens": 300}
                },
                timeout=timeout
            )
        
        # Uruchom monitoring w osobnym wątku
        result_queue = Queue()
        monitor_thread = threading.Thread(
            target=heartbeat_monitor,
            args=(make_request, result_queue, heartbeat, max_wait)
        )
        
        start_time = time.time()
        monitor_thread.start()
        
        # Czekaj na rezultat
        try:
            status, result = result_queue.get(timeout=max_wait + 60)  # Extra buffer
            monitor_thread.join(timeout=10)  # Daj wątkowi czas na zakończenie
            
            response_time = time.time() - start_time
            
            if status == "success":
                response_data = result.json()
                return {
                    "success": True,
                    "response": response_data.get("response", "").strip(),
                    "model_used": model_name,
                    "model_type": model_config["type"],
                    "response_time": round(response_time, 2),
                    "quality_score": model_config["quality_score"],
                    "waited_patiently": response_time > 300  # Czy czekaliśmy cierpliwie
                }
            else:
                return {
                    "success": False,
                    "error": result,
                    "model_attempted": model_name,
                    "response_time": round(response_time, 2),
                    "reason": status
                }
                
        except Empty:
            return {
                "success": False,
                "error": "Monitor thread nie odpowiedział",
                "model_attempted": model_name,
                "response_time": max_wait
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_attempted": model_name,
                "response_time": time.time() - start_time
            }
    
    def generate_response(self, task_description):
        """Główna metoda z cierpliwym podejściem"""
        print(f"\n🎯 ZADANIE: {task_description}")
        
        # Wybierz hierarchię modeli
        task_type = self.analyze_task(task_description)
        models = self.model_hierarchy[task_type]
        
        print(f"📊 Typ zadania: {task_type}")
        print(f"🔄 Hierarchia modeli: {len(models)} kandydatów")
        
        # Wypróbuj modele w kolejności
        for i, model_config in enumerate(models, 1):
            print(f"\n{'='*20} PRÓBA {i}/{len(models)} {'='*20}")
            
            result = self.try_model_with_patience(model_config, task_description)
            
            if result["success"]:
                print(f"\n✅ SUKCES!")
                if result.get("waited_patiently"):
                    print(f"🕰️ Cierpliwie poczekaliśmy na specjalistę!")
                return result
            else:
                print(f"\n❌ NIEPOWODZENIE:")
                print(f"   Powód: {result.get('reason', 'nieznany')}")
                print(f"   Błąd: {result.get('error', 'brak szczegółów')}")
                print(f"   Czas: {result.get('response_time', 0):.1f}s")
                
                if i < len(models):
                    print(f"🔄 Przechodzę do następnego modelu...")
                else:
                    print(f"😞 Wyczerpano wszystkie opcje")
        
        return {
            "success": False,
            "error": "Żaden model z hierarchii nie był dostępny",
            "attempts": len(models)
        }

def main():
    print("🚀 Agent Zero Phase 4 - CIERPLIWY System AI z Kontrolą Życia")
    print("=" * 70)
    print(f"Czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("💡 Specjalistom dajemy czas (do 1h), ale kontrolujemy czy żyją (5min)!")
    
    ai_system = PatientAISystem()
    
    # Zadania testowe - od prostych do złożonych
    test_tasks = [
        "Napisz funkcję sortującą w Python",
        "Jak zoptymalizować PostgreSQL dla wysokiego ruchu?", 
        "Zaprojektuj architekturę mikrousług dla e-commerce",
        "Co to jest REST API?",
        "Strategia skalowania systemu AI dla miliona użytkowników dziennie"
    ]
    
    print(f"\n🧪 Testowanie z cierpliwym podejściem:")
    
    successful_tests = 0
    specialist_successes = 0
    total_time = 0
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n" + "🎯" + "="*60)
        print(f"TEST {i}/{len(test_tasks)}")
        
        start = time.time()
        result = ai_system.generate_response(task)
        test_time = time.time() - start
        
        if result["success"]:
            successful_tests += 1
            total_time += result["response_time"]
            
            if result["model_type"] in ["specialist", "expert"]:
                specialist_successes += 1
                print(f"🏆 SPECJALISTA DOSTARCZYŁ!")
            
            print(f"\n🎉 SUKCES w {test_time:.1f}s!")
            print(f"   Model: {result['model_used']} ({result['model_type']})")
            print(f"   Czas odpowiedzi: {result['response_time']}s")
            print(f"   Jakość: {result['quality_score']}/10")
            print(f"   Odpowiedź: {result['response'][:150]}...")
        else:
            print(f"\n💥 NIEPOWODZENIE w {test_time:.1f}s")
            print(f"   Błąd: {result.get('error', 'nieznany')}")
    
    print(f"\n🎊 CIERPLIWY TEST ZAKOŃCZONY!")
    print(f"✅ Udane: {successful_tests}/{len(test_tasks)}")
    print(f"🏆 Specjaliści: {specialist_successes}/{successful_tests} sukcesów")
    if successful_tests > 0:
        print(f"⏱️ Średni czas: {total_time/successful_tests:.1f}s")
    print(f"🧠 Cierpliwość + kontrola życia = najlepsza jakość odpowiedzi!")

if __name__ == "__main__":
    main()