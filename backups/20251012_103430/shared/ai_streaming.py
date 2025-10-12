import asyncio
import time
from typing import AsyncGenerator, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from simple_classifier import classifier
from ollama_client import ollama

class ThinkingStage(Enum):
    ANALYZING = "analyzing"
    MODEL_SELECTION = "model_selection"
    MODEL_LOADING = "model_loading"
    THINKING = "thinking"
    GENERATING = "generating"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"

@dataclass
class ThinkingUpdate:
    stage: ThinkingStage
    message: str
    progress: float  # 0.0 to 1.0
    elapsed_time: float
    current_output: str
    confidence: float
    can_cancel: bool
    metadata: Dict[str, Any]

class StreamingAIBrain:
    def __init__(self):
        self.classifier = classifier
        self.current_task_id = None
        self.cancel_requested = False
        
    async def think_with_stream(self, task: str) -> AsyncGenerator[ThinkingUpdate, None]:
        """AI thinking with real-time streaming updates"""
        
        task_id = f"task_{int(time.time())}"
        self.current_task_id = task_id
        self.cancel_requested = False
        start_time = time.time()
        
        try:
            # Stage 1: Analyzing
            yield ThinkingUpdate(
                stage=ThinkingStage.ANALYZING,
                message="🎯 Analyzing task requirements...",
                progress=0.1,
                elapsed_time=time.time() - start_time,
                current_output="",
                confidence=0.0,
                can_cancel=True,
                metadata={"task_preview": task[:50] + "..."}
            )
            
            await asyncio.sleep(1)
            if self.cancel_requested:
                yield self._create_cancel_update(start_time)
                return
            
            # Stage 2: Classification
            classification = self.classifier.classify_with_ai(task)
            selected_model = classification['model']
            
            yield ThinkingUpdate(
                stage=ThinkingStage.MODEL_SELECTION,
                message=f"🤖 Selected model: {selected_model}",
                progress=0.2,
                elapsed_time=time.time() - start_time,
                current_output="",
                confidence=0.3,
                can_cancel=True,
                metadata={"selected_model": selected_model, "complexity": classification['complexity']}
            )
            
            await asyncio.sleep(1)
            if self.cancel_requested:
                yield self._create_cancel_update(start_time)
                return
            
            # Stage 3: Model Loading
            yield ThinkingUpdate(
                stage=ThinkingStage.MODEL_LOADING,
                message=f"⚡ Loading {selected_model} (please be patient)...",
                progress=0.3,
                elapsed_time=time.time() - start_time,
                current_output="",
                confidence=0.4,
                can_cancel=True,
                metadata={"model_size": self._get_model_size(selected_model)}
            )
            
            # Simulate model loading time
            for i in range(3):
                if self.cancel_requested:
                    yield self._create_cancel_update(start_time)
                    return
                await asyncio.sleep(2)
                
                yield ThinkingUpdate(
                    stage=ThinkingStage.MODEL_LOADING,
                    message=f"⚡ Loading {selected_model}... {(i+1)*33:.0f}%",
                    progress=0.3 + (i * 0.05),
                    elapsed_time=time.time() - start_time,
                    current_output="",
                    confidence=0.4,
                    can_cancel=True,
                    metadata={"loading_progress": (i+1)*33}
                )
            
            # Stage 4: Thinking
            thinking_steps = [
                "🧠 Understanding requirements...",
                "🎯 Planning solution approach...",
                "📋 Designing architecture...",
                "🔧 Considering best practices..."
            ]
            
            for i, step in enumerate(thinking_steps):
                if self.cancel_requested:
                    yield self._create_cancel_update(start_time)
                    return
                
                yield ThinkingUpdate(
                    stage=ThinkingStage.THINKING,
                    message=step,
                    progress=0.45 + (i * 0.1),
                    elapsed_time=time.time() - start_time,
                    current_output="",
                    confidence=0.5 + (i * 0.05),
                    can_cancel=True,
                    metadata={"thinking_step": i+1, "total_steps": len(thinking_steps)}
                )
                await asyncio.sleep(3)
            
            # Stage 5: Actual AI Call
            yield ThinkingUpdate(
                stage=ThinkingStage.GENERATING,
                message="💻 AI generating solution...",
                progress=0.85,
                elapsed_time=time.time() - start_time,
                current_output="Starting generation...",
                confidence=0.7,
                can_cancel=True,
                metadata={"generating": True}
            )
            
            # Real AI call
            try:
                response = ollama.chat(selected_model, [{"role": "user", "content": task}], timeout=None)
                
                if 'error' in response:
                    # Try backup model
                    yield ThinkingUpdate(
                        stage=ThinkingStage.GENERATING,
                        message="🔄 Primary model failed, trying backup...",
                        progress=0.90,
                        elapsed_time=time.time() - start_time,
                        current_output="Switching to backup model...",
                        confidence=0.6,
                        can_cancel=True,
                        metadata={"backup_needed": True}
                    )
                    
                    response = ollama.chat("phi3:mini", [{"role": "user", "content": task}], timeout=None)
                    selected_model = "phi3:mini (backup)"
                
                response_text = response['message']['content']
                
                # Stage 6: Completion
                yield ThinkingUpdate(
                    stage=ThinkingStage.COMPLETED,
                    message="🎉 AI task completed successfully!",
                    progress=1.0,
                    elapsed_time=time.time() - start_time,
                    current_output=response_text,
                    confidence=0.85,
                    can_cancel=False,
                    metadata={"model_used": selected_model, "response_length": len(response_text)}
                )
                
            except Exception as e:
                yield ThinkingUpdate(
                    stage=ThinkingStage.ERROR,
                    message=f"❌ Error: {str(e)}",
                    progress=0.0,
                    elapsed_time=time.time() - start_time,
                    current_output="",
                    confidence=0.0,
                    can_cancel=False,
                    metadata={"error": str(e)}
                )
                
        except Exception as e:
            yield ThinkingUpdate(
                stage=ThinkingStage.ERROR,
                message=f"❌ System error: {str(e)}",
                progress=0.0,
                elapsed_time=time.time() - start_time,
                current_output="",
                confidence=0.0,
                can_cancel=False,
                metadata={"error": str(e)}
            )
    
    def cancel_current_task(self):
        """Cancel currently running task"""
        self.cancel_requested = True
    
    def _create_cancel_update(self, start_time: float) -> ThinkingUpdate:
        return ThinkingUpdate(
            stage=ThinkingStage.CANCELLED,
            message="🛑 Task cancelled by user",
            progress=0.0,
            elapsed_time=time.time() - start_time,
            current_output="Task was cancelled",
            confidence=0.0,
            can_cancel=False,
            metadata={"cancelled_at": time.time()}
        )
    
    def _get_model_size(self, model: str) -> str:
        sizes = {
            "phi3:mini": "2.2GB",
            "qwen2.5:14b": "9.0GB", 
            "deepseek-coder:33b": "18GB",
            "deepseek-r1:32b": "19GB",
            "mixtral:8x7b": "26GB"
        }
        return sizes.get(model, "Unknown")

# Global instance
streaming_ai_brain = StreamingAIBrain()
