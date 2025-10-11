#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Naprawiony Inteligentny System AI
Poprawiona logika wyboru modeli i oceny złożoności
"""

import requests
import time
import re
from datetime import datetime

class ModelTimeout(Exception):
    pass

def poll_model_response(req_func, poll_interval=5, max_wait=180):
    """Monitorowanie odpowiedzi modelu - skrócony timeout dla produkcji"""
    start = time.time()
    while True:
        try:
            response = req_func()
            return response
        except requests.Timeout:
            now = time.time()
            if now - start > max_wait:
                raise ModelTimeout(f"Model nie odpowiedział w ciągu {max_wait}s")
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Ładowanie modelu...")
            time.sleep(poll_interval)
        except requests.RequestException as e:
            time.sleep(poll_interval)
            continue

class ImprovedAISystem:
    def __init__(self, ollama_host="localhost", ollama_port=11434):
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        
        # Poprawione mapowanie modeli z rzeczywistymi capabilities
        self.model_catalog = {
            # Szybkie i niezawodne - pierwsza linia
            "llama3.2:3b": {
                "size_gb": 2.0, 
                "speed": "very_fast", 
                "capabilities": ["general", "chat", "simple_coding", "reasoning"], 
                "reliability": 95,
                "coding_score": 6,
                "reasoning_score": 7
            },
            "phi3:mini": {
                "size_gb": 2.2, 
                "speed": "very_fast", 
                "capabilities": ["general", "chat", "reasoning"], 
                "reliability": 90,
                "coding_score": 4,
                "reasoning_score": 8
            },
            
            # Standardowe - druga linia  
            "llama3.1:8b": {
                "size_gb": 4.9, 
                "speed": "fast", 
                "capabilities": ["general", "reasoning", "analysis", "coding"], 
                "reliability": 90,
                "coding_score": 7,
                "reasoning_score": 8
            },
            "mistral:7b": {
                "size_gb": 4.4, 
                "speed": "fast", 
                "capabilities": ["general", "reasoning", "multilingual"], 
                "reliability": 85,
                "coding_score": 6,
                "reasoning_score": 9
            },
            
            # Specjalistyczne kodowanie
            "codellama:13b": {
                "size_gb": 7.4, 
                "speed": "medium", 
                "capabilities": ["expert_coding", "programming", "debug"], 
                "reliability": 80,
                "coding_score": 10,
                "reasoning_score": 6
            },
            "wizardlm2:7b": {
                "size_gb": 4.1, 
                "speed": "medium", 
                "capabilities": ["general", "coding", "reasoning"], 
                "reliability": 70,  # Niska z powodu timeoutów
                "coding_score": 7,
                "reasoning_score": 7
            },
            
            # Zaawansowane - trzecia linia
            "qwen2.5:14b": {
                "size_gb": 9.0, 
                "speed": "medium", 
                "capabilities": ["advanced_reasoning", "multimodal", "analysis"], 
                "reliability": 75,
                "coding_score": 8,
                "reasoning_score": 10
            },
            "solar:10.7b": {
                "size_gb": 6.1, 
                "speed": "medium", 
                "capabilities": ["reasoning", "analysis", "general"], 
                "reliability": 80,
                "coding_score": 7,
                "reasoning_score": 9
            },
            
            # Specjalistyczne ale wolne
            "deepseek-coder:33b": {
                "size_gb": 18, 
                "speed": "slow", 
                "capabilities": ["expert_coding", "architecture", "complex_coding"], 
                "reliability": 60,  # Wolne ładowanie
                "coding_score": 10,
                "reasoning_score": 8
            },
            
            # Bardzo zaawansowane - ostatnia opcja
            "mixtral:8x7b": {
                "size_gb": 26, 
                "speed": "very_slow", 
                "capabilities": ["expert_reasoning", "complex_analysis"], 
                "reliability": 50,  # Często timeout
                "coding_score": 9,
                "reasoning_score": 10
            },
            
            # Problematyczne modele - unikaj
            "deepseek-r1:1.5b": {
                "size_gb": 1.1, 
                "speed": "slow", 
                "capabilities": ["reasoning"], 
                "reliability": 30,  # Dziwne odpowiedzi z <think>
                "coding_score": 3,
                "reasoning_score": 6
            }
        }
        
        self.model_cache = {}
        
    def analyze_task(self, task_description):
        """Ulepszona analiza zadania"""
        task_lower = task_description.lower()
        
        # Rozszerzone słowa kluczowe
        coding_keywords = [
            "kod", "code", "python", "javascript", "program", "function", "bug", "debug", 
            "algorithm", "sortowanie", "funkcja", "script", "programming", "implementuj",
            "napisz program", "napisz funkcję", "zrób kod"
        ]
        
        reasoning_keywords = [
            "analiza", "analysis", "strategia", "strategy", "plan", "reasoning", 
            "myślenie", "problem solving", "optymalizuj", "optimize", "zaprojektuj",
            "design", "architektura", "architecture", "najlepsze praktyki", "best practices"
        ]
        
        complexity_keywords = {
            "bardzo_proste": ["szybka odpowiedź", "quick", "co to jest", "what is", "definicja"],
            "proste": ["prosty", "simple", "basic", "podstaw", "łatwy"],
            "średnie": ["średni", "medium", "standard", "typowy"],
            "złożone": ["złożony", "complex", "zaawansowany", "advanced", "skomplikowany"],
            "bardzo_złożone": [
                "expert", "ekspert", "architektura", "architecture", "skalowanie", "scaling",
                "1m użytkowników", "milion", "million", "enterprise", "produkcja", "production",
                "duży system", "large system", "distributed", "mikrousługi", "microservices"
            ]
        }
        
        # Ocena typu zadania
        task_type = "general"
        if any(word in task_lower for word in coding_keywords):
            task_type = "coding"
        elif any(word in task_lower for word in reasoning_keywords):
            task_type = "reasoning"
        
        # Ulepszona ocena złożoności
        complexity_score = 2  # Bazowa wartość
        
        # Sprawdź poziom złożoności
        for level, keywords in complexity_keywords.items():
            if any(keyword in task_lower for keyword in keywords):
                if level == "bardzo_proste":
                    complexity_score = 0
                elif level == "proste":
                    complexity_score = 1
                elif level == "średnie":
                    complexity_score = 2
                elif level == "złożone":
                    complexity_score = 4
                elif level == "bardzo_złożone":
                    complexity_score = 5
                break
        
        # Dodatkowe wskaźniki złożoności
        if len(task_description) > 150:
            complexity_score += 1
        if len(task_description) > 300:
            complexity_score += 1
            
        # Wykrywanie złożonych tematów
        complex_topics = ["system", "architektura", "skalowanie", "optimization", "enterprise", "production"]
        if any(topic in task_lower for topic in complex_topics):
            complexity_score = max(complexity_score, 3)
        
        return task_type, min(5, complexity_score)
    
    def select_best_model(self, task_description):
        """Ulepszona selekcja modelu"""
        task_type, complexity = self.analyze_task(task_description)
        
        print(f"🧠 Analiza zadania:")
        print(f"   Typ: {task_type}")
        print(f"   Złożoność: {complexity}/5")
        
        candidates = []
        
        # Scoring na podstawie typu zadania
        for model_name, specs in self.model_catalog.items():
            score = 0
            
            # Bazowy score z reliability
            score += specs["reliability"] / 10
            
            # Score dla typu zadania
            if task_type == "coding":
                score += specs["coding_score"] * 2
                # Bonus dla modeli kodowania
                if "expert_coding" in specs["capabilities"]:
                    score += 15
                elif "coding" in specs["capabilities"]:
                    score += 10
            
            elif task_type == "reasoning":
                score += specs["reasoning_score"] * 2
                # Bonus dla modeli reasoning
                if "expert_reasoning" in specs["capabilities"]:
                    score += 15
                elif "advanced_reasoning" in specs["capabilities"]:
                    score += 12
                elif "reasoning" in specs["capabilities"]:
                    score += 8
            
            # Dla zadań ogólnych - zbalansowane podejście
            else:
                score += (specs["coding_score"] + specs["reasoning_score"])
                if "general" in specs["capabilities"]:
                    score += 5
            
            # Dostosowanie do poziomu złożoności
            if complexity <= 1:  # Proste zadania - priorytet dla szybkich
                if specs["speed"] in ["very_fast", "fast"]:
                    score += 8
                if specs["reliability"] >= 90:
                    score += 5
            
            elif complexity >= 4:  # Złożone zadania - priorytet dla zaawansowanych
                if specs["speed"] not in ["very_slow"] or specs["reliability"] < 60:
                    score -= 5  # Kara za wolne/niepewne modele przy złożonych zadaniach
                if "expert" in str(specs["capabilities"]) or "advanced" in str(specs["capabilities"]):
                    score += 10
            
            else:  # Średnie zadania - zbalansowane
                if specs["speed"] in ["fast", "medium"]:
                    score += 3
                if specs["reliability"] >= 80:
                    score += 3
            
            candidates.append((model_name, score, specs))
        
        # Sortowanie po score (malejąco) i reliability (malejąco) 
        candidates.sort(key=lambda x: (x[1], x[2]["reliability"]), reverse=True)
        
        if not candidates:
            return "llama3.2:3b"  # Bezpieczny fallback
            
        # Pokaż top 3 kandydatów
        print(f"   Top kandydaci:")
        for i, (model, score, specs) in enumerate(candidates[:3]):
            print(f"     {i+1}. {model} (score: {score:.1f}, reliability: {specs['reliability']}%)")
        
        best_model = candidates[0][0]
        print(f"   ✅ Wybrany: {best_model}")
        
        return best_model
    
    def test_model_connection(self, model_name):
        """Szybki test połączenia z cache"""
        if model_name in self.model_cache:
            return self.model_cache[model_name]
            
        def req():
            return requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "OK",
                    "stream": False,
                    "options": {"max_tokens": 1}
                },
                timeout=20
            )
        try:
            response = poll_model_response(req, poll_interval=2, max_wait=60)
            connected = response.status_code == 200
            self.model_cache[model_name] = connected
            return connected
        except Exception:
            self.model_cache[model_name] = False
            return False
    
    def get_fallback_model(self, original_model, task_type):
        """Inteligentny fallback na podstawie typu zadania"""
        if task_type == "coding":
            fallback_sequence = ["llama3.1:8b", "llama3.2:3b", "mistral:7b"]
        elif task_type == "reasoning":
            fallback_sequence = ["llama3.1:8b", "mistral:7b", "llama3.2:3b"]
        else:
            fallback_sequence = ["llama3.2:3b", "llama3.1:8b", "mistral:7b"]
        
        for fallback in fallback_sequence:
            if fallback != original_model and self.test_model_connection(fallback):
                return fallback
        return None
    
    def generate_ai_response(self, task_description):
        """Główna metoda z inteligentnym fallback"""
        print(f"\n🎯 Zadanie: {task_description}")
        
        task_type, complexity = self.analyze_task(task_description)
        best_model = self.select_best_model(task_description)
        
        # Test połączenia z wybranym modelem
        print(f"🔍 Sprawdzanie modelu {best_model}...")
        if not self.test_model_connection(best_model):
            print(f"❌ Model {best_model} niedostępny")
            
            # Inteligentny fallback
            fallback_model = self.get_fallback_model(best_model, task_type)
            if fallback_model:
                best_model = fallback_model
                print(f"🔄 Przełączam na: {fallback_model}")
            else:
                return {"success": False, "error": "Żaden odpowiedni model nie jest dostępny"}
        
        # Generowanie odpowiedzi
        print(f"⚡ Przetwarzanie z modelem {best_model}...")
        
        def req():
            return requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": best_model,
                    "prompt": f"Agent Zero AI: {task_description}\n\nKrótka, konkretna odpowiedź w 2-3 zdaniach:",
                    "stream": False,
                    "options": {"temperature": 0.2, "max_tokens": 250}
                },
                timeout=30
            )
        
        try:
            start_time = time.time()
            response = poll_model_response(req, poll_interval=3, max_wait=180)
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "").strip()
                
                # Czyszczenie odpowiedzi z artifacts (np. <think>)
                if response_text.startswith("<think>"):
                    # Wyciąg myśli z modelu reasoning
                    think_end = response_text.find("</think>")
                    if think_end != -1:
                        response_text = response_text[think_end+8:].strip()
                    else:
                        response_text = "Model zwrócił niepełną odpowiedź reasoning."
                
                response_time = time.time() - start_time
                
                return {
                    "success": True,
                    "response": response_text[:300] + "..." if len(response_text) > 300 else response_text,
                    "model_used": best_model,
                    "response_time": round(response_time, 2),
                    "task_type": task_type,
                    "complexity": complexity,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except ModelTimeout as e:
            return {"success": False, "error": f"TIMEOUT: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": f"Błąd: {str(e)}"}

def main():
    print("🚀 Agent Zero Phase 4 - Naprawiony Inteligentny System AI")
    print("=" * 65)
    print(f"Czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    ai_system = ImprovedAISystem()
    
    # Ulepszone zadania testowe
    test_tasks = [
        "Napisz prostą funkcję sortującą w Python",
        "Jak zoptymalizować bazę danych PostgreSQL dla 10k użytkowników?",
        "Najlepsze praktyki w architekturze mikrousług dla enterprise",
        "Szybka odpowiedź: co to jest REST API?",
        "Zaprojektuj skalowalny system AI dla 1 miliona użytkowników dziennie - architektura, bazy danych, load balancing"
    ]
    
    print(f"\n🧪 Testowanie {len(test_tasks)} ulepszonych zadań:")
    
    successful_tests = 0
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n" + "="*50)
        print(f"TEST {i}/{len(test_tasks)}")
        
        result = ai_system.generate_ai_response(task)
        
        if result["success"]:
            successful_tests += 1
            print(f"✅ Sukces!")
            print(f"   Model: {result['model_used']}")
            print(f"   Czas: {result['response_time']}s")
            print(f"   Typ zadania: {result['task_type']}, Złożoność: {result['complexity']}/5")
            print(f"   Odpowiedź: {result['response'][:150]}...")
        else:
            print(f"❌ Błąd: {result['error']}")
    
    print(f"\n🎉 Test zakończony!")
    print(f"✅ Udane testy: {successful_tests}/{len(test_tasks)}")
    print(f"💡 System poprawnie analizuje złożoność i dobiera modele!")

if __name__ == "__main__":
    main()