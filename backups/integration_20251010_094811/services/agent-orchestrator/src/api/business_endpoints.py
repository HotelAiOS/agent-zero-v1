# services/agent-orchestrator/src/api/business_endpoints.py - NOWY PLIK
"""
Business Requirements Parser API - Final 15% implementation
CZAS: 30 minut | STATUS: Missing → Complete
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, validator
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime

# Import existing parser (już 85% gotowy)
from business.requirements_parser import BusinessRequirementsParser

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/business", tags=["business"])
parser = BusinessRequirementsParser()

# Request/Response Models
class BusinessRequest(BaseModel):
    request: str
    context: Optional[Dict[str, Any]] = {}
    priority: Optional[str] = "medium"
    
    @validator('request')
    def validate_request(cls, v):
        if len(v.strip()) < 10:
            raise ValueError('Request too short')
        return v.strip()

class TechnicalSpec(BaseModel):
    intent: str
    entities: List[str]
    complexity: str
    agents_needed: List[str]
    estimated_cost: float
    estimated_time_minutes: int
    confidence_score: float

# API Endpoints
@router.post("/parse", response_model=TechnicalSpec)
async def parse_business_request(request: BusinessRequest):
    """Convert business requirements to technical specifications"""
    try:
        spec = parser.generate_technical_spec(
            intent=parser.parse_intent(request.request),
            entities=parser.extract_entities(request.request),
            complexity=parser.assess_complexity(request.request, []),
            business_request=request.request,
            context=request.context
        )
        
        logger.info(f"Parsed: {spec['intent']} - {spec['complexity']}")
        return TechnicalSpec(**spec)
        
    except Exception as e:
        logger.error(f"Parse error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/validate")
async def validate_request(request: BusinessRequest):
    """Validate business request quality"""
    try:
        validation = parser.validate_request(request.request)
        return validation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "business_parser", 
        "timestamp": datetime.utcnow().isoformat()
    }
