#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - AI-First Inteligentny System
Automatyczny wybór najlepszego modelu dla każdego zadania
"""

import requests
import time
import re
from datetime import datetime

class ModelTimeout(Exception):
    pass

def poll_model_response(req_func, poll_interval=5, max_wait=300):
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

class IntelligentAISystem:
    def __init__(self, ollama_host="localhost", ollama_port=11434):
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        
        # Mapowanie dostępnych modeli (z Twojej listy)
        self.model_catalog = {
            # Szybkie i lekkie
            "llama3.2:3b": {"size_gb": 2.0, "speed": "very_fast", "capabilities": ["general", "chat"], "memory_efficient": True},
            "phi3:mini": {"size_gb": 2.2, "speed": "very_fast", "capabilities": ["general", "chat"], "memory_efficient": True},
            "deepseek-r1:1.5b": {"size_gb": 1.1, "speed": "ultra_fast", "capabilities": ["reasoning"], "memory_efficient": True},
            
            # Standardowe
            "llama3.1:8b": {"size_gb": 4.9, "speed": "fast", "capabilities": ["general", "reasoning", "analysis"], "memory_efficient": True},
            "mistral:7b": {"size_gb": 4.4, "speed": "fast", "capabilities": ["general", "reasoning"], "memory_efficient": True},
            "wizardlm2:7b": {"size_gb": 4.1, "speed": "fast", "capabilities": ["general", "coding"], "memory_efficient": True},
            
            # Specjalistyczne - kod
            "codellama:13b": {"size_gb": 7.4, "speed": "medium", "capabilities": ["coding", "programming", "debug"], "memory_efficient": False},
            "deepseek-coder:33b": {"size_gb": 18, "speed": "slow", "capabilities": ["expert_coding", "architecture"], "memory_efficient": False},
            
            # Zaawansowane
            "qwen2.5:14b": {"size_gb": 9.0, "speed": "medium", "capabilities": ["advanced_reasoning", "multimodal"], "memory_efficient": False},
            "solar:10.7b": {"size_gb": 6.1, "speed": "medium", "capabilities": ["reasoning", "analysis"], "memory_efficient": False},
            
            # Ciężkie - tylko dla złożonych zadań
            "mixtral:8x7b": {"size_gb": 26, "speed": "very_slow", "capabilities": ["expert_reasoning", "complex_analysis"], "memory_efficient": False},
            "qwen3:30b": {"size_gb": 18, "speed": "very_slow", "capabilities": ["expert_analysis"], "memory_efficient": False},
            "deepseek-r1:32b": {"size_gb": 19, "speed": "very_slow", "capabilities": ["expert_reasoning"], "memory_efficient": False}
        }
        
        self.model_cache = {}  # Cache sprawdzonych modeli
        
    def analyze_task(self, task_description):
        """Inteligentna analiza zadania i wybór najlepszego modelu"""
        task_lower = task_description.lower()
        
        # Analiza słów kluczowych
        coding_keywords = ["kod", "code", "python", "javascript", "program", "function", "bug", "debug", "algorithm", "sortowanie", "funkcja"]
        reasoning_keywords = ["analiza", "analysis", "strategia", "strategy", "plan", "reasoning", "myślenie", "problem solving"]
        complex_keywords = ["złożony", "complex", "advanced", "expert", "architekt", "architecture", "skalowanie", "scaling"]
        simple_keywords = ["szybko", "quick", "fast", "simple", "prosty", "podstaw", "basic"]
        
        # Punktacja zadania
        complexity_score = 0
        task_type = "general"
        
        # Sprawdzenie typu zadania
        if any(word in task_lower for word in coding_keywords):
            task_type = "coding"
            complexity_score += 2
            
        if any(word in task_lower for word in reasoning_keywords):
            task_type = "reasoning"
            complexity_score += 1
            
        if any(word in task_lower for word in complex_keywords):
            complexity_score += 3
            
        if any(word in task_lower for word in simple_keywords):
            complexity_score -= 2
            
        # Długość zadania jako wskaźnik złożoności
        if len(task_description) > 100:
            complexity_score += 1
        if len(task_description) > 200:
            complexity_score += 1
            
        return task_type, max(0, complexity_score)
    
    def select_best_model(self, task_description):
        """Wybiera najlepszy model dla zadania"""
        task_type, complexity = self.analyze_task(task_description)
        
        print(f"🧠 Analiza zadania:")
        print(f"   Typ: {task_type}")
        print(f"   Złożoność: {complexity}/5")
        
        candidates = []
        
        # Wybór kandydatów na podstawie typu zadania
        for model_name, specs in self.model_catalog.items():
            score = 0
            
            # Dopasowanie do typu zadania
            if task_type == "coding":
                if "expert_coding" in specs["capabilities"]:
                    score += 10
                elif "coding" in specs["capabilities"]:
                    score += 8
                elif "programming" in specs["capabilities"]:
                    score += 6
            
            elif task_type == "reasoning":
                if "expert_reasoning" in specs["capabilities"]:
                    score += 10
                elif "advanced_reasoning" in specs["capabilities"]:
                    score += 8
                elif "reasoning" in specs["capabilities"]:
                    score += 6
            
            # Bonus dla zadań ogólnych
            if "general" in specs["capabilities"]:
                score += 3
                
            # Dostosowanie do złożoności
            if complexity <= 1:  # Proste zadania
                if specs["speed"] in ["ultra_fast", "very_fast"]:
                    score += 5
                if specs["memory_efficient"]:
                    score += 3
            elif complexity >= 4:  # Złożone zadania
                if specs["speed"] in ["slow", "very_slow"] and score > 5:  # Tylko jeśli już pasuje do zadania
                    score += 2
            else:  # Średnie zadania
                if specs["speed"] in ["fast", "medium"]:
                    score += 3
                    
            if score > 0:
                candidates.append((model_name, score, specs))
        
        # Sortowanie po wyniku (malejąco)
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if not candidates:
            # Fallback na najszybszy model
            return "llama3.2:3b"
            
        best_model = candidates[0][0]
        print(f"   Wybrany model: {best_model} (score: {candidates[0][1]})")
        
        return best_model
    
    def test_model_connection(self, model_name):
        """Szybki test połączenia z modelem"""
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
                timeout=30
            )
        try:
            response = poll_model_response(req, poll_interval=3, max_wait=120)
            connected = response.status_code == 200
            self.model_cache[model_name] = connected
            return connected
        except Exception:
            self.model_cache[model_name] = False
            return False
    
    def generate_ai_response(self, task_description):
        """Główna metoda - inteligentne przetwarzanie zadania"""
        print(f"\n🎯 Zadanie: {task_description}")
        
        # 1. Wybór najlepszego modelu
        best_model = self.select_best_model(task_description)
        
        # 2. Test połączenia z modelem
        print(f"🔍 Sprawdzanie modelu {best_model}...")
        if not self.test_model_connection(best_model):
            print(f"❌ Model {best_model} niedostępny, szukam alternatywy...")
            
            # Fallback na dostępne modele
            fallback_models = ["llama3.2:3b", "llama3.1:8b", "mistral:7b", "phi3:mini"]
            for fallback in fallback_models:
                if self.test_model_connection(fallback):
                    best_model = fallback
                    print(f"✅ Użyję modelu zastępczego: {fallback}")
                    break
            else:
                return {"success": False, "error": "Żaden model nie jest dostępny"}
        
        # 3. Generowanie odpowiedzi
        print(f"⚡ Przetwarzanie z modelem {best_model}...")
        
        def req():
            return requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": best_model,
                    "prompt": f"Agent Zero AI: {task_description}\n\nOdpowiedź w 2-3 zdaniach:",
                    "stream": False,
                    "options": {"temperature": 0.3, "max_tokens": 300}
                },
                timeout=60
            )
        
        try:
            start_time = time.time()
            response = poll_model_response(req, poll_interval=5, max_wait=300)
            
            if response.status_code == 200:
                result = response.json()
                response_time = time.time() - start_time
                
                return {
                    "success": True,
                    "response": result.get("response", "").strip(),
                    "model_used": best_model,
                    "response_time": round(response_time, 2),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
                
        except ModelTimeout as e:
            return {"success": False, "error": str(e)}
        except Exception as e:
            return {"success": False, "error": f"Błąd: {str(e)}"}

def main():
    print("🚀 Agent Zero Phase 4 - Inteligentny System AI")
    print("=" * 60)
    print(f"Czas: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    ai_system = IntelligentAISystem()
    
    # Test zadań różnego typu i złożoności
    test_tasks = [
        "Napisz prostą funkcję sortującą w Python",
        "Jak zoptymalizować bazę danych dla dużego ruchu?",
        "Jakie są najlepsze praktyki w architekturze mikrousług?",
        "Szybka odpowiedź: co to jest AI?",
        "Zaawansowana analiza: jak zaprojektować skalowalny system AI dla 1M użytkowników?"
    ]
    
    print(f"\n🧪 Testowanie {len(test_tasks)} różnych zadań:")
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n" + "="*50)
        print(f"TEST {i}/{len(test_tasks)}")
        
        result = ai_system.generate_ai_response(task)
        
        if result["success"]:
            print(f"✅ Sukces!")
            print(f"   Model: {result['model_used']}")
            print(f"   Czas: {result['response_time']}s")
            print(f"   Odpowiedź: {result['response'][:200]}...")
        else:
            print(f"❌ Błąd: {result['error']}")
    
    print(f"\n🎉 Test zakończony!")
    print(f"💡 System inteligentnie dobiera model do każdego zadania!")

if __name__ == "__main__":
    main()