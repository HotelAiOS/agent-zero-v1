import logging
from ollama_client import ollama

logger = logging.getLogger(__name__)

class SimpleClassifier:
    def __init__(self):
        self.models = {
            "simple": "phi3:mini",
            "code": "deepseek-coder:33b", 
            "complex": "deepseek-r1:32b",
            "creative": "mixtral:8x7b"
        }
    
    def classify_task(self, task):
        """Simple task classification"""
        task_lower = task.lower()
        
        # Simple rules
        if "hello" in task_lower or "simple" in task_lower:
            return {
                "complexity": "simple",
                "model": self.models["simple"],
                "reasoning": "Simple task detected"
            }
        elif "code" in task_lower or "debug" in task_lower:
            return {
                "complexity": "code", 
                "model": self.models["code"],
                "reasoning": "Code task detected"
            }
        elif "complex" in task_lower or "architecture" in task_lower:
            return {
                "complexity": "complex",
                "model": self.models["complex"], 
                "reasoning": "Complex task detected"
            }
        else:
            return {
                "complexity": "general",
                "model": "qwen2.5:14b",
                "reasoning": "General task"
            }
    def classify_with_ai(self, task):
        """AI-powered classification"""
        prompt = f"""Classify this software task in one word:

Task: {task}

Respond with ONLY one of these words:
- simple (basic tasks like hello world)
- code (programming, debugging, algorithms)  
- complex (architecture, system design)
- creative (documentation, design, writing)

One word only:"""

        try:
            response = ollama.chat("phi3:mini", [
                {"role": "user", "content": prompt}
            ])
            
            if 'error' not in response:
                ai_result = response['message']['content'].strip().lower()
                
                # Map AI result to our models
                if ai_result in self.models:
                    return {
                        "complexity": ai_result,
                        "model": self.models[ai_result],
                        "reasoning": f"AI classified as {ai_result}"
                    }
            
            # Fallback to rule-based
            return self.classify_task(task)
            
        except Exception as e:
            logger.error(f"AI classification failed: {e}")
            return self.classify_task(task)

    def classify_with_ai(self, task):
        """AI-powered classification"""
        prompt = f"""Classify this software task in one word:

Task: {task}

Respond with ONLY one of these words:
- simple (basic tasks like hello world)
- code (programming, debugging, algorithms)  
- complex (architecture, system design)
- creative (documentation, design, writing)

One word only:"""

        try:
            response = ollama.chat("phi3:mini", [
                {"role": "user", "content": prompt}
            ])
            
            if 'error' not in response:
                ai_result = response['message']['content'].strip().lower()
                
                # Map AI result to our models
                if ai_result in self.models:
                    return {
                        "complexity": ai_result,
                        "model": self.models[ai_result],
                        "reasoning": f"AI classified as {ai_result}"
                    }
            
            # Fallback to rule-based
            return self.classify_task(task)
            
        except Exception as e:
            logger.error(f"AI classification failed: {e}")
            return self.classify_task(task)


# Global instance
classifier = SimpleClassifier()
