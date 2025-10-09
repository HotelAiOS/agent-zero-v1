from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

from app.ai.business_intelligence import BusinessIntelligence

app = FastAPI()
bi = BusinessIntelligence()

class BIRequest(BaseModel):
    text: str

@app.post("/api/ai/business")
async def business_intelligence_endpoint(data: BIRequest):
    result = await bi.analyze_and_generate(data.text)
    return result

# Manualny test/dev-server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api.main:app", host="127.0.0.1", port=8080, reload=True)
