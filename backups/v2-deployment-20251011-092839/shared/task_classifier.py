import ollama
import tiktoken
import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
import json
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class TaskProfile:
    """Comprehensive task analysis profile"""
    complexity: str         # simple|medium|complex|critical
    domain: str            # code|analysis|creative|reasoning|general  
    context_size: str      # small|medium|large|xlarge
    token_count: int       # Actual token count
    requires_reasoning: bool
    requires_creativity: bool
    estimated_time: int    # seconds
    recommended_model: str
    confidence: float      # 0.0-1.0
    reasoning: str         # Why this classification

class IntelligentTaskClassifier:
    """Sophisticated task classification using AI + ML"""
    
    def __init__(self):
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.classification_model = "phi3:mini"  # Fast classifier
        self.performance_history: Dict[str, List[float]] = {}
        
        # Model capability matrix
        self.model_matrix = {
            "phi3:mini": {
                "max_tokens": 4096,
                "strengths": ["speed", "simple_tasks"],
                "cost": 0.1,  # Relative cost
                "ram_usage": 2.2
            },
            "qwen2.5:14b": {
                "max_tokens": 8192, 
                "strengths": ["reasoning", "general", "analysis"],
                "cost": 0.4,
                "ram_usage": 9.0
            },
            "deepseek-coder:33b": {
                "max_tokens": 16384,
                "strengths": ["coding", "technical", "debugging"],
                "cost": 0.8,
                "ram_usage": 18.0
            },
            "deepseek-r1:32b": {
                "max_tokens": 32768,
                "strengths": ["complex_reasoning", "analysis", "critical"],
                "cost": 0.9,
                "ram_usage": 19.0
            },
            "mixtral:8x7b": {
                "max_tokens": 32768,
                "strengths": ["creative", "multi_domain", "complex"],
                "cost": 1.0,
                "ram_usage": 26.0
            }
        }
    
    async def classify_task(self, task: str, context: List[Dict] = None) -> TaskProfile:
        """Advanced AI-powered task classification"""
        
        # 1. Token analysis
        tokens = self._count_tokens(task, context)
        context_size = self._classify_context_size(tokens)
        
        # 2. AI classification
        classification_prompt = f"""Analyze this software development task:
        
        Task: {task}
        
        Classify with this JSON format:
        {{
            "complexity": "simple|medium|complex|critical",
            "domain": "code|analysis|creative|reasoning|general",
            "requires_reasoning": true|false,
            "requires_creativity": true|false,
            "estimated_time_seconds": number,
            "reasoning": "brief explanation"
        }}
        
        Consider:
        - Code tasks = complex algorithms, debugging, architecture
        - Analysis tasks = data analysis, performance optimization
        - Reasoning tasks = planning, decision making, problem solving
        - Creative tasks = documentation, naming, design patterns
        """
        
        try:
            response = ollama.chat(
                model=self.classification_model,
                messages=[{"role": "user", "content": classification_prompt}]
            )
            
            # Parse AI response
            ai_analysis = self._parse_json_response(response['message']['content'])
            
            # 3. Intelligent model selection
            recommended_model = self._select_optimal_model(
                domain=ai_analysis.get('domain', 'general'),
                complexity=ai_analysis.get('complexity', 'medium'),
                token_count=tokens,
                requires_reasoning=ai_analysis.get('requires_reasoning', False)
            )
            
            # 4. Confidence scoring
            confidence = self._calculate_confidence(ai_analysis, tokens, recommended_model)
            
            return TaskProfile(
                complexity=ai_analysis.get('complexity', 'medium'),
                domain=ai_analysis.get('domain', 'general'), 
                context_size=context_size,
                token_count=tokens,
                requires_reasoning=ai_analysis.get('requires_reasoning', False),
                requires_creativity=ai_analysis.get('requires_creativity', False),
                estimated_time=ai_analysis.get('estimated_time_seconds', 30),
                recommended_model=recommended_model,
                confidence=confidence,
                reasoning=ai_analysis.get('reasoning', 'Standard classification')
            )
            
        except Exception as e:
            logger.error(f"❌ Classification error: {e}")
            # Fallback to safe defaults
            return self._fallback_classification(task, tokens)
    
    def _count_tokens(self, task: str, context: List[Dict] = None) -> int:
        """Accurate token counting with tiktoken"""
        tokens = len(self.tokenizer.encode(task))
        
        if context:
            for msg in context:
                tokens += len(self.tokenizer.encode(msg.get('content', '')))
        
        return tokens
    
    def _classify_context_size(self, tokens: int) -> str:
        """Context size classification"""
        if tokens < 1000: return "small"
        elif tokens < 4000: return "medium"  
        elif tokens < 8000: return "large"
        else: return "xlarge"
    
    def _select_optimal_model(self, domain: str, complexity: str, 
                             token_count: int, requires_reasoning: bool) -> str:
        """Sophisticated model selection algorithm"""
        
        # Critical/complex reasoning tasks
        if complexity == "critical" or (complexity == "complex" and requires_reasoning):
            if token_count > 16000:
                return "claude"  # Cloud fallback for huge context
            else:
                return "deepseek-r1:32b"  # Best local reasoning
        
        # Code domain specialization
        elif domain == "code":
            if token_count > 8000:
                return "claude"  # Better for large codebases
            else:
                return "deepseek-coder:33b"  # Best local coding
        
        # Creative tasks
        elif domain == "creative":
            return "mixtral:8x7b"  # Most creative local model
        
        # Analysis tasks
        elif domain == "analysis" or requires_reasoning:
            return "deepseek-r1:32b" if token_count < 16000 else "claude"
        
        # Simple/fast tasks
        elif complexity == "simple":
            return "phi3:mini"  # Fastest
        
        # Default balanced choice
        else:
            return "qwen2.5:14b"  # Good all-rounder
    
    def _calculate_confidence(self, analysis: Dict, tokens: int, model: str) -> float:
        """Calculate confidence in our decisions"""
        confidence = 0.8  # Base confidence
        
        # Increase confidence for clear domain match
        if analysis.get('domain') == 'code' and 'coder' in model:
            confidence += 0.1
        elif analysis.get('domain') == 'reasoning' and 'r1' in model:
            confidence += 0.1
        
        # Decrease confidence for edge cases
        if tokens > 20000:  # Very large context
            confidence -= 0.2
        elif analysis.get('complexity') == 'critical':
            confidence -= 0.1  # Conservative for critical
        
        return min(1.0, max(0.3, confidence))
    
    def _parse_json_response(self, response: str) -> Dict[str, Any]:
        """Parse JSON from AI response"""
        try:
            # Find JSON in response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            json_str = response[json_start:json_end]
            return json.loads(json_str)
        except:
            return {
                "complexity": "medium",
                "domain": "general", 
                "requires_reasoning": True,
                "requires_creativity": False,
                "estimated_time_seconds": 30,
                "reasoning": "Fallback classification"
            }
    
    def _fallback_classification(self, task: str, tokens: int) -> TaskProfile:
        """Safe fallback when classification fails"""
        return TaskProfile(
            complexity="medium",
            domain="general",
            context_size=self._classify_context_size(tokens),
            token_count=tokens,
            requires_reasoning=True,
            requires_creativity=False,
            estimated_time=30,
            recommended_model="qwen2.5:14b",  # Safe default
            confidence=0.5,
            reasoning="Fallback classification due to analysis error"
        )

# Global classifier instance
task_classifier = IntelligentTaskClassifier()
