from fastapi import FastAPI
from datetime import datetime

app = FastAPI(title="Agent Zero V2.0", version="2.0.0")

@app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}

@app.get("/")
async def root():
    return {"message": "Agent Zero V2.0 Intelligence Layer", "docs": "/docs"}
