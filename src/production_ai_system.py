#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - AI-First Production System
Polska wersja, monitoring postępu, timeout aż 1h i wykrywanie zwisu modelu.
"""

import requests
import time
from datetime import datetime

class ModelTimeout(Exception):
    pass

def poll_model_response(req_func, poll_interval=5, max_wait=3600):
    """
    Monitorowanie odpowiedzi modelu: pyta co 5s o postęp, max. 1h
    """
    start = time.time()
    last_error = None
    while True:
        try:
            response = req_func()
            return response
        except requests.Timeout:
            now = time.time()
            if now - start > max_wait:
                raise ModelTimeout(f"Model nie odpowiedział w ciągu {max_wait // 60} minut!")
            print(f"[{datetime.now().isoformat()}] ⏳ Wciąż czekam na odpowiedź modelu...")
            time.sleep(poll_interval)
        except requests.RequestException as e:
            last_error = e
            time.sleep(poll_interval)
            continue
    if last_error:
        raise last_error

class ProductionAISystem:
    def __init__(self, ollama_host="localhost", ollama_port=11434):
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        self.available_models = {
            "fast": "llama3.2:3b",
            "standard": "llama3.1:8b",
            "advanced": "qwen2.5:14b",
            "code": "codellama:13b",
            "expert": "deepseek-coder:33b",
            "complex": "mixtral:8x7b"
        }
        self.performance_stats = {}

    def test_model_connection(self, model_name):
        def req():
            return requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "Test connection - respond with 'OK'",
                    "stream": False,
                    "options": {"max_tokens": 5}
                },
                timeout=60
            )
        try:
            response = poll_model_response(req, poll_interval=5, max_wait=600)
            if response.status_code == 200:
                return True, "Connected"
            return False, f"HTTP {response.status_code}"
        except Exception as e:
            return False, str(e)

    def generate_ai_reasoning(self, prompt, model_type="standard"):
        model_name = self.available_models.get(model_type, "llama3.2:3b")
        def req():
            return requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": f"Agent Zero AI System: {prompt}",
                    "stream": False,
                    "options": {"temperature": 0.2, "max_tokens": 400}
                },
                timeout=120
            )
        try:
            response = poll_model_response(req, poll_interval=10, max_wait=3600)
            if response.status_code == 200:
                result = response.json()
                return {"success": True,
                        "reasoning": result.get("response", "").strip(),
                        "model_used": model_name,
                        "timestamp": datetime.now().isoformat()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}"}
        except ModelTimeout as e:
            return {"success": False, "error": f"TIMEOUT: {str(e)}"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def system_health_check(self):
        print("🏥 Agent Zero Phase 4 - System Health Check")
        print("=" * 50)
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "healthy_models": 0,
            "total_models": len(self.available_models),
            "model_status": {}
        }
        for model_type, model_name in self.available_models.items():
            print(f"\n🔍 Testing {model_type} model ({model_name})...")
            connected, status = self.test_model_connection(model_name)
            if connected:
                print(f"  ✅ {model_name}: Connected")
                health_status["healthy_models"] += 1
            else:
                print(f"  ❌ {model_name}: {status}")
            health_status["model_status"][model_name] = status
        print(f"\n📊 System Summary:")
        print(f"  Healthy Models: {health_status['healthy_models']}/{health_status['total_models']}")
        print(f"  System Health: {'✅ Excellent' if health_status['healthy_models'] >= 3 else '⚠️ Partial' if health_status['healthy_models'] >= 1 else '❌ Critical'}")
        return health_status

    def demo_ai_capabilities(self):
        print("\n🤖 Agent Zero Phase 4 - AI Capabilities Demo")
        print("=" * 50)
        tests = [
            ("Optymalizacja bazy SQL", "standard"),
            ("Napisz sortowanie w Python", "code"),
            ("Jak skalować AI?", "advanced")
        ]
        for prompt, model_type in tests:
            print(f"\n🧠 Test {model_type} (model: {self.available_models.get(model_type)}):")
            result = self.generate_ai_reasoning(prompt, model_type)
            if result["success"]:
                print(f"   Odpowiedź: {result['reasoning'][:200]}...")
            else:
                print(f"   Błąd/timeout: {result['error']}")

def main():
    print("🚀 Agent Zero Phase 4 - Production AI System")
    print("=" * 60)
    print(datetime.now().isoformat())
    ai_system = ProductionAISystem()
    health = ai_system.system_health_check()
    if health["healthy_models"] > 0:
        ai_system.demo_ai_capabilities()
    else:
        print("⚠️ No healthy models detected. Please check Ollama service.")
    print("\n🎉 System gotowy! 🚀")

if __name__ == "__main__":
    main()
