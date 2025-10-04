import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from ollama_client import ollama
from simple_classifier import classifier

logger = logging.getLogger(__name__)

@dataclass
class ThinkingResult:
    response: str
    model_used: str
    processing_time: float
    confidence: float
    reasoning_steps: List[str]
    classification: Dict[str, Any]

class AIBrain:
    def __init__(self):
        self.classifier = classifier
        self.thinking_history = []

    async def think(self, task: str) -> ThinkingResult:
        classification = self.classifier.classify_with_ai(task)
        selected_model = classification["model"]
        start_time = asyncio.get_event_loop().time()

        try:
            logger.info(f"🧠 Thinking with {selected_model}...")
            response = ollama.chat(selected_model, [{"role": "user", "content": task}])
            end_time = asyncio.get_event_loop().time()
            processing_time = end_time - start_time

            if "error" in response:
                response = ollama.chat("phi3:mini", [{"role": "user", "content": task}])
                selected_model = "phi3:mini"

            response_text = response["message"]["content"]
            confidence = 0.8
            reasoning_steps = []

            result = ThinkingResult(
                response=response_text,
                model_used=selected_model,
                processing_time=processing_time,
                confidence=confidence,
                reasoning_steps=reasoning_steps,
                classification=classification
            )

            self.thinking_history.append(result)
            return result

        except Exception as e:
            logger.error(f"❌ Error: {e}")
            return ThinkingResult("Error", "fallback", 0.0, 0.1, [], classification)

    def get_stats(self):
        return {"total_thoughts": len(self.thinking_history)}

ai_brain = AIBrain()
