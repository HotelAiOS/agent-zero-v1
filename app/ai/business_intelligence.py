# app/ai/business_intelligence.py

import logging
from typing import Dict
from .intelligence_core import AgentZeroIntelligence  # lokalny import

class BusinessIntelligence:
    """
    Agent do analizy i przekładu wymagań biznesowych na strukturę techniczną,
    integruje się z AgentZeroIntelligence ("mózg AI").
    """

    def __init__(self):
        self.logger = logging.getLogger("BusinessIntelligence")
        self.brain = AgentZeroIntelligence()

    def parse_business_text(self, text: str) -> Dict:
        """
        Parsuje tekst biznesowy, wydziela intencje, cele i ograniczenia.
        Prosta heurystyka (możliwe rozszerzenie w fazie 2).
        """
        intent = "feature" if "feature" in text.lower() else "request"
        objectives = []
        constraints = []

        if "zoptymalizuj" in text.lower():
            objectives.append("optimize performance")
        if "koszt" in text.lower():
            constraints.append("cost sensitive")
        if "czas" in text.lower():
            constraints.append("time critical")

        parsed = {
            "intent": intent,
            "objectives": objectives,
            "constraints": constraints,
            "raw_text": text
        }
        self.logger.info(f"[BI] Parsed: {parsed}")
        return parsed

    async def analyze_and_generate(self, business_text: str) -> Dict:
        """
        Główna funkcja: przetwarzanie tekstu oraz wywołanie agentowego AI.
        """
        # 1) Parsujemy wymagania
        parsed = self.parse_business_text(business_text)
        # 2) Przekazujemy do mózgu AI (asynchronicznie)
        ai_result = await self.brain.think_and_execute(business_text)
        # 3) Zwracamy pełen wynik (parsowanie + odpowiedź AI)
        return {
            "input_parsed": parsed,
            "ai_result": ai_result,
            "model_used": self.brain.last_model
        }
