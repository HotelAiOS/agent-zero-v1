from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

# Uproszczona aplikacja FastAPI bez EmailStr
app = FastAPI(title="API Agent Zero", description="Uproszczone API")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str

@app.post("/logowanie", response_model=Token)
async def logowanie_token(dane_formularza: OAuth2PasswordRequestForm = Depends()):
    """Uproszczony endpoint logowania"""
    if dane_formularza.username == "admin" and dane_formularza.password == "tajne":
        return {"access_token": "test-token-agent-zero", "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nieprawidlowa nazwa uzytkownika lub haslo"
        )

@app.get("/")
async def root():
    return {"message": "Agent Zero API działa!", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy", "agent": "zero"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

class DaneLogowania(BaseModel):
    nazwa_uzytkownika: str
    haslo: str

class Token(BaseModel):
    access_token: str
    token_type: str

@app.post("/logowanie", response_model=Token)
async def logowanie_token(dane_formularza: OAuth2PasswordRequestForm = Depends()):
    """Endpoint logowania z tokenem JWT"""
    if dane_formularza.username == "admin" and dane_formularza.password == "tajne":
        return {"access_token": "przykladowy-jwt-token", "token_type": "bearer"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Nieprawidlowa nazwa uzytkownika lub haslo",
            headers={"WWW-Authenticate": "Bearer"},
        )

@app.get("/chroniony")
async def chroniona_sciezka(token: str = Depends(oauth2_scheme)):
    """Chroniony endpoint wymagający autoryzacji"""
    return {"wiadomosc": "To jest chroniona sciezka", "token": token}

@app.get("/")
async def root():
    """Główny endpoint"""
    return {"wiadomosc": "API Agent Zero - System Autoryzacji", "status": "online"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
