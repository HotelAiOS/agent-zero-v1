# Modele danych wygenerowane przez Agent Zero
from pydantic import BaseModel, EmailStr
from typing import Optional
from datetime import datetime

class UzytkownikBaza(BaseModel):
    """Podstawowy model użytkownika"""
    nazwa_uzytkownika: str
    email: EmailStr
    imie: Optional[str] = None
    nazwisko: Optional[str] = None

class UzytkownikTworzenie(UzytkownikBaza):
    """Model tworzenia użytkownika"""
    haslo: str

class Uzytkownik(UzytkownikBaza):
    """Kompletny model użytkownika"""
    id: int
    jest_aktywny: bool = True
    jest_adminem: bool = False
    utworzony: datetime
    ostatnie_logowanie: Optional[datetime] = None
    
    class Config:
        from_attributes = True

class TokenOdpowiedz(BaseModel):
    """Model odpowiedzi z tokenem"""
    access_token: str
    token_type: str
    expires_in: int
    user_id: int
    username: str

class StatusOdpowiedz(BaseModel):
    """Model odpowiedzi statusu"""
    status: str
    wiadomosc: str
    czas: datetime
    wersja: str = "1.0.0"
