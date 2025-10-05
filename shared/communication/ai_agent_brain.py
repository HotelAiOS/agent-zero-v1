"""
AI Agent Brain - Async wrapper dla AI Brain z code generation
"""
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys
import os

# Import AI Brain
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from ai_brain import ai_brain

logger = logging.getLogger(__name__)


class AIAgentBrain:
    """Async wrapper dla AI Brain - umożliwia agentom używanie AI"""
    
    def __init__(self):
        self.brain = ai_brain
        logger.info("🧠 AI Agent Brain initialized")
    
    async def generate_code(
        self, 
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generuj kod używając AI Brain.
        
        Args:
            task_description: Co agent ma zrobić
            context: Dodatkowy kontekst (requirements, tech stack)
            
        Returns:
            Dict z kodem, wyjaśnieniem, modelem użytym
        """
        # Build prompt
        prompt = f"""Generate production-ready code for:
{task_description}

Provide:
1. Complete working code
2. Inline comments
3. Error handling
4. Usage example

Format code in proper code blocks.
"""
        
        # AI Brain.think() jest JUŻ ASYNC!
        # Wywołaj bezpośrednio z await
        result = await self.brain.think(prompt)
        
        return {
            "code": result.response,
            "model_used": result.model_used,
            "processing_time": result.processing_time,
            "confidence": result.confidence,
            "timestamp": datetime.now().isoformat()
        }


# Singleton
ai_agent_brain = AIAgentBrain()
