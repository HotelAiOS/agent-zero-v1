from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import logging
import os

from .models.schemas import GenerateRequest, GenerateResponse, HealthResponse
from .router.orchestrator import AIOrchestrator

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Agent Zero AI Router",
    version="1.0.0",
    description="Inteligentny router dla wielomodelowej orkiestracji AI"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
REQUEST_COUNT = Counter(
    'ai_router_requests_total',
    'Total requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'ai_router_request_duration_seconds',
    'Request latency'
)
TOKEN_COUNT = Counter(
    'ai_router_tokens_total',
    'Total tokens used',
    ['model', 'provider']
)
GENERATION_COUNT = Counter(
    'ai_router_generations_total',
    'Total AI generations',
    ['provider', 'task_type']
)

# Orkiestrator
orchestrator = AIOrchestrator()

@app.get("/")
async def root():
    """Root endpoint"""
    REQUEST_COUNT.labels(method='GET', endpoint='/', status='200').inc()
    return {"message": "Agent Zero AI Router v1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check"""
    REQUEST_COUNT.labels(method='GET', endpoint='/health', status='200').inc()
    providers = await orchestrator.health_check()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        providers=providers
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generuj odpowiedź AI"""
    try:
        logger.info(f"Received request: task={request.task_type}, provider={request.provider}")
        
        with REQUEST_LATENCY.time():
            response = await orchestrator.route(request)
        
        # Inkrementuj metryki
        REQUEST_COUNT.labels(method='POST', endpoint='/generate', status='200').inc()
        TOKEN_COUNT.labels(model=response.model, provider=response.provider.value).inc(response.tokens)
        GENERATION_COUNT.labels(provider=response.provider.value, task_type=request.task_type.value).inc()
        
        return response
        
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        REQUEST_COUNT.labels(method='POST', endpoint='/generate', status='500').inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """Lista dostępnych modeli"""
    REQUEST_COUNT.labels(method='GET', endpoint='/models', status='200').inc()
    models = await orchestrator.ollama_client.list_models()
    return {"models": models}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
