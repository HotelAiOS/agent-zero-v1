from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
import uuid
import time
import httpx

from .models.schemas import (
    ChatRequest, ChatResponse, SessionListResponse, 
    SessionHistoryResponse, ChatSession, Message, MessageRole
)
from .storage.redis_store import RedisStore
from .context.manager import ContextManager

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Agent Zero Chat Service",
    version="1.0.0",
    description="Zarządzanie sesjami czatu z pamięcią kontekstu"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage & Context
redis_store = RedisStore(
    redis_url=os.getenv("REDIS_URL", "redis://redis-cluster:6379/0"),
    password=os.getenv("REDIS_PASSWORD", "redis-dev-secure-password")
)
context_manager = ContextManager(max_tokens=8000)

# AI Router URL
AI_ROUTER_URL = os.getenv("AI_ROUTER_URL", "http://ai-router-service:8000")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agent Zero Chat Service v1.0.0"}

@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Wyślij wiadomość w czacie"""
    start_time = time.time()
    
    try:
        # Pobierz lub utwórz sesję
        if request.session_id:
            session = redis_store.get_session(request.session_id)
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
        else:
            session_id = str(uuid.uuid4())
            session = ChatSession(
                id=session_id,
                user_id=request.user_id,
                title=request.message[:50] + "..." if len(request.message) > 50 else request.message
            )
            redis_store.save_session(session)
        
        # Zapisz wiadomość użytkownika
        user_message = Message(
            id=str(uuid.uuid4()),
            session_id=session.id,
            role=MessageRole.USER,
            content=request.message,
            tokens=context_manager.count_tokens(request.message)
        )
        redis_store.save_message(user_message)
        
        # Pobierz historię
        history = redis_store.get_messages(session.id, limit=20)
        
        # Zbuduj kontekst
        context = context_manager.build_context(history)
        
        # Wywołaj AI Router
        async with httpx.AsyncClient(timeout=120.0) as client:
            ai_response = await client.post(
                f"{AI_ROUTER_URL}/generate",
                json={
                    "prompt": context + f"\n\nUser: {request.message}\n\nAssistant:",
                    "task_type": request.task_type,
                    "model": request.model_preference,
                    "max_tokens": 2048,
                    "temperature": 0.7
                }
            )
            ai_response.raise_for_status()
            ai_data = ai_response.json()
        
        # Zapisz odpowiedź asystenta
        assistant_message = Message(
            id=str(uuid.uuid4()),
            session_id=session.id,
            role=MessageRole.ASSISTANT,
            content=ai_data["response"],
            tokens=ai_data["tokens"]
        )
        redis_store.save_message(assistant_message)
        
        # Aktualizuj sesję
        session.message_count += 2
        session.total_tokens += user_message.tokens + assistant_message.tokens
        redis_store.save_session(session)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return ChatResponse(
            session_id=session.id,
            message_id=assistant_message.id,
            content=ai_data["response"],
            tokens=ai_data["tokens"],
            latency_ms=latency_ms,
            model_used=ai_data["model"]
        )
    
    except Exception as e:
        logger.error(f"Chat failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{user_id}", response_model=SessionListResponse)
async def get_user_sessions(user_id: str):
    """Pobierz sesje użytkownika"""
    session_ids = redis_store.get_user_sessions(user_id)
    sessions = [redis_store.get_session(sid) for sid in session_ids]
    sessions = [s for s in sessions if s]  # Filtruj None
    return SessionListResponse(sessions=sessions, total=len(sessions))

@app.get("/sessions/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(session_id: str):
    """Pobierz historię sesji"""
    session = redis_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = redis_store.get_messages(session_id, limit=100)
    return SessionHistoryResponse(session=session, messages=messages)

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, user_id: str):
    """Usuń sesję"""
    success = redis_store.delete_session(session_id, user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete session")
    return {"status": "deleted"}
@app.post("/feedback")
async def submit_feedback(
    session_id: str,
    message_id: str,
    feedback_type: str,
    rating: int = None,
    comment: str = None
):
    """Zapisz feedback użytkownika"""
    try:
        # Zapisz do bazy
        # TO DO: Implementacja zapisu do PostgreSQL
        
        return {
            "status": "feedback_recorded",
            "session_id": session_id,
            "message_id": message_id
        }
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/insights")
async def get_learning_insights(insight_type: str = None, limit: int = 10):
    """Pobierz learning insights"""
    try:
        # TO DO: Query PostgreSQL learning_insights
        return {
            "insights": [],
            "message": "Learning insights endpoint - implementation pending"
        }
    except Exception as e:
        logger.error(f"Get insights failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/user/{user_id}/patterns")
async def get_user_patterns(user_id: str):
    """Analizuj wzorce użytkownika"""
    try:
        # TO DO: Call analyze_user_behavior function
        return {
            "user_id": user_id,
            "patterns": {},
            "message": "User patterns endpoint - implementation pending"
        }
    except Exception as e:
        logger.error(f"Get user patterns failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
