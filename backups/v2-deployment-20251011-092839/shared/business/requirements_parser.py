from __future__ import annotations
import re, asyncio, time, json, logging
from typing import Dict, List, Optional, Any, Tuple
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/business", tags=["business"])
parser: BusinessRequirementsParser  # ustawione niżej

class BusinessRequest(BaseModel):
    request: str = Field(..., min_length=1)
    priority: Optional[str] = Field("medium")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TechnicalSpec(BaseModel):
    intent: str
    entities: List[str]
    complexity: str
    agents_needed: List[str]
    estimated_cost: float
    estimated_time_minutes: int
    confidence_score: float
    technical_requirements: Dict[str, Any]
    validation_issues: List[Dict[str, Any]] = []

class ValidationResponse(BaseModel):
    is_valid: bool
    confidence: float
    errors: List[str] = []
    suggestions: List[str] = []

class BusinessRequirementsParser:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def parse_intent(self, text: str) -> str:
        t = text.lower()
        if any(k in t for k in ["create","build","develop","make","implement"]):
            return "CREATE"
        if any(k in t for k in ["update","modify","change","enhance","improve"]):
            return "UPDATE"
        return "UNKNOWN"

    def extract_entities(self, text: str) -> List[str]:
        tl = text.lower()
        ents: List[str] = []
        keywords = ["api","database","user","payment",
                    "inventory","analytics","mobile","dashboard"]
        for ent in keywords:
            if re.search(rf"\b{ent}\b", tl):
                ents.append(ent)
        if re.search(r"\breport(ing|s)?\b", tl):
            ents.append("report")
        if "authentication" in tl or "auth" in tl:
            ents.append("auth")
        return list(dict.fromkeys(ents))

    def assess_complexity(self, text: str, entities: List[str]) -> str:
        tl = text.lower()
        if "enterprise" in tl or len(entities) >= 5:
            return "Enterprise"
        if "complex" in tl or "scalable" in tl or len(entities) >= 4:
            return "Complex"
        if "simple" in tl or len(entities) <= 2:
            return "Simple"
        return "Moderate"

    def select_agents(self, intent: str, entities: List[str], complexity: str) -> List[str]:
        agents = ["orchestrator","code_generator"]
        if "api" in entities:
            agents.append("api_specialist")
        if complexity in ["Complex","Enterprise"] or "database" in entities:
            agents.append("database_specialist")
        agents.append("solution_architect")
        return agents

    def estimate_cost_and_time(
        self,
        complexity: str,
        agents: List[str],
        entities: List[str]
    ) -> Tuple[float,int]:
        if complexity == "Simple":
            return (0.05, 10)
        if complexity == "Moderate":
            return (0.10, 30)
        if complexity == "Complex":
            return (0.20, 60)
        # Enterprise
        return (0.50, 180)

    def generate_technical_spec(
        self,
        intent: str,
        entities: List[str],
        complexity: str,
        business_request: str
    ) -> dict:
        tech_reqs: Dict[str,Any] = {}
        if "api" in entities:
            tech_reqs["api"] = {
                "type":"REST",
                "authentication":"JWT",
                "documentation":"OpenAPI/Swagger"
            }
        # always include database for Complex/Enterprise
        if complexity in ["Complex","Enterprise"] or "database" in entities:
            tech_reqs["database"] = {"type":"PostgreSQL"}
        tech_reqs["security"] = {"encryption":"TLS"}
        cost, time_min = self.estimate_cost_and_time(complexity, [], entities)
        return {
            "intent": intent,
            "entities": entities,
            "complexity": complexity,
            "agents_needed": self.select_agents(intent, entities, complexity),
            "estimated_cost": cost,
            "estimated_time_minutes": time_min,
            "confidence_score": 0.8,
            "technical_requirements": tech_reqs,
            "validation_issues": []
        }

    def validate_request(self, request: str) -> dict:
        errors: List[str] = []
        suggestions: List[str] = []
        conf = 0.8
        if len(request.strip()) < 10:
            errors.append("Request too short")
            conf = 0.2
        if re.search(r"password|auth", request, re.IGNORECASE):
            suggestions.append("Include security considerations")
        return {
            "is_valid": not errors,
            "errors": errors,
            "suggestions": suggestions,
            "confidence": conf
        }

    def sanitize_input(self, text: str) -> str:
        out = re.sub(r"<script.*?>.*?</script>", "", text,
                     flags=re.IGNORECASE|re.DOTALL)
        out = re.sub(r"alert\([^)]*\)", "", out, flags=re.IGNORECASE)
        out = re.sub(r"<.*?>", "", out)
        out = out.replace('"', "")
        return out.strip()

parser = BusinessRequirementsParser()

@router.post("/parse", response_model=TechnicalSpec)
def parse_endpoint(req: BusinessRequest):
    v = parser.validate_request(req.request)
    if not v["is_valid"]:
        raise HTTPException(
            status_code=400,
            detail={"message":"Invalid business request","errors":v["errors"]}
        )
    intent = parser.parse_intent(req.request)
    ents = parser.extract_entities(req.request)
    comp = parser.assess_complexity(req.request, ents)
    if req.priority == "critical" or req.context.get("budget") == "enterprise":
        comp = "Enterprise"
    return parser.generate_technical_spec(intent, ents, comp, req.request)

@router.post("/validate", response_model=ValidationResponse)
def validate_endpoint(req: BusinessRequest):
    return ValidationResponse(**parser.validate_request(req.request))

@router.get("/health")
def health_check():
    return {
        "status":"healthy",
        "service":"business_requirements_parser",
        "version":"1.0.0",
        "components":{"parser":True}
    }

@router.get("/capabilities")
def get_capabilities():
    return {
        "supported_intents":["CREATE","UPDATE"],
        "supported_entities":["api","database","user","report","auth"],
        "complexity_levels":["Simple","Moderate","Complex","Enterprise"],
        "features":{"validation":True}
    }
