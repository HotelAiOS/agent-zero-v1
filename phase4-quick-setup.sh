# Agent Zero V2.0 Phase 4 - Quick Setup Commands
# Saturday, October 11, 2025 @ 12:23 CEST

echo "🚀 Setting up Phase 4 Production Environment..."

# 1. Create proper directory structure
mkdir -p {src,tests,config,logs,scripts}
echo "✅ Directory structure created"

# 2. Create the production AI system
cat > src/production_ai_system.py << 'EOF'
#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 4 - Production AI System
32GB System with Multi-Model Support
"""

import requests
import json
import time
from datetime import datetime

class ProductionAISystem:
    """Production AI System for Agent Zero Phase 4"""
    
    def __init__(self, ollama_host="localhost", ollama_port=11434):
        self.ollama_url = f"http://{ollama_host}:{ollama_port}"
        
        # Based on your actual ollama list output
        self.available_models = {
            "fast": "llama3.2:3b",           # 2.0GB - Fast decisions
            "standard": "llama3.1:8b",       # 4.9GB - Standard reasoning  
            "advanced": "qwen2.5:14b",       # 9.0GB - Advanced reasoning
            "code": "codellama:13b",         # 7.4GB - Code analysis
            "expert": "deepseek-coder:33b",  # 18GB - Expert coding
            "complex": "mixtral:8x7b"        # 26GB - Complex reasoning
        }
        
        self.performance_stats = {}
        
    def test_model_connection(self, model_name):
        """Test connection to a specific model"""
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json={
                "model": model_name,
                "prompt": "Test connection - respond with 'OK'",
                "stream": False,
                "options": {"max_tokens": 5}
            }, timeout=10)
            
            if response.status_code == 200:
                return True, "Connected"
            else:
                return False, f"HTTP {response.status_code}"
        except Exception as e:
            return False, str(e)
    
    def generate_ai_reasoning(self, prompt, model_type="standard"):
        """Generate AI reasoning with your models"""
        
        model_name = self.available_models.get(model_type, "llama3.2:3b")
        start_time = time.time()
        
        try:
            response = requests.post(f"{self.ollama_url}/api/generate", json={
                "model": model_name,
                "prompt": f"Agent Zero AI System: {prompt}",
                "stream": False,
                "options": {"temperature": 0.2, "max_tokens": 200}
            }, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                response_time = time.time() - start_time
                
                return {
                    "success": True,
                    "reasoning": result.get("response", "").strip(),
                    "model_used": model_name,
                    "response_time": response_time,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_attempted": model_name,
                "timestamp": datetime.now().isoformat()
            }
    
    def system_health_check(self):
        """Check health of your AI system"""
        
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
                health_status["model_status"][model_name] = "healthy"
            else:
                print(f"  ❌ {model_name}: {status}")
                health_status["model_status"][model_name] = f"error: {status}"
        
        print(f"\n📊 System Summary:")
        print(f"  Healthy Models: {health_status['healthy_models']}/{health_status['total_models']}")
        print(f"  System Health: {'✅ Excellent' if health_status['healthy_models'] >= 3 else '⚠️ Partial' if health_status['healthy_models'] >= 1 else '❌ Critical'}")
        
        return health_status
    
    def demo_ai_capabilities(self):
        """Demonstrate AI capabilities with your models"""
        
        print("\n🤖 Agent Zero Phase 4 - AI Capabilities Demo")
        print("=" * 50)
        
        test_scenarios = [
            ("How should Agent Zero handle database optimization?", "standard"),
            ("Analyze this Python function for efficiency improvements", "code"),
            ("What are the strategic considerations for AI system scaling?", "advanced")
        ]
        
        for prompt, model_type in test_scenarios:
            print(f"\n🧠 Testing {model_type} reasoning:")
            print(f"   Prompt: {prompt}")
            
            result = self.generate_ai_reasoning(prompt, model_type)
            
            if result["success"]:
                print(f"   Model: {result['model_used']}")
                print(f"   Response: {result['reasoning'][:150]}...")
                print(f"   Time: {result['response_time']:.2f}s")
            else:
                print(f"   Error: {result['error']}")
        
        print(f"\n✅ AI Capabilities Demo Complete!")

def main():
    """Main execution for Phase 4 testing"""
    
    print("🚀 Agent Zero Phase 4 - Production AI System")
    print("=" * 60)
    print("Saturday, October 11, 2025 @ 12:23 CEST")
    print("32GB System with Multi-Model Support")
    
    ai_system = ProductionAISystem()
    
    # Run health check
    health = ai_system.system_health_check()
    
    # If we have healthy models, demo capabilities
    if health["healthy_models"] > 0:
        ai_system.demo_ai_capabilities()
    else:
        print("\n⚠️ No healthy models detected. Please check Ollama service.")
    
    print(f"\n🎉 Phase 4 Production AI System Ready!")
    print(f"🏆 Your 32GB system with {len(ai_system.available_models)} models is LEGENDARY!")

if __name__ == "__main__":
    main()
EOF

echo "✅ Production AI system created"

# 3. Create quick test script
cat > test_ai_system.py << 'EOF'
#!/usr/bin/env python3
import sys
sys.path.append('src')
from production_ai_system import ProductionAISystem

# Quick test
ai = ProductionAISystem()
result = ai.generate_ai_reasoning("Test Agent Zero Phase 4 AI reasoning")

if result["success"]:
    print(f"✅ AI Test Successful!")
    print(f"Model: {result['model_used']}")
    print(f"Response: {result['reasoning'][:100]}...")
else:
    print(f"❌ AI Test Failed: {result['error']}")
EOF

echo "✅ Test script created"

# 4. Make scripts executable
chmod +x src/production_ai_system.py test_ai_system.py

echo ""
echo "🎯 Phase 4 Setup Complete! Run with:"
echo "  python3 src/production_ai_system.py"
echo "  python3 test_ai_system.py"
echo ""
echo "📊 Your directory structure:"
ls -la