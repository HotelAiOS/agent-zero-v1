"""
Workspace Extension PL - kompletny system po polsku w jednym pliku
Rozszerza TWÓJ istniejący system Agent Zero o zarządzanie przestrzeniami roboczymi
"""
import os
import json
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class PlikRobocza:
    id_pliku: str
    nazwa_pliku: str
    zawartosc: str
    id_agenta: str
    id_zadania: str
    utworzono: datetime
    sciezka_pliku: Path

class RozszerzenieWorkspace:
    """Rozszerza TWÓJ system o zarządzanie przestrzeniami roboczymi"""
    
    def __init__(self, katalog_bazowy: str = "./agent_zero_przestrzenie"):
        self.sciezka_bazowa = Path(katalog_bazowy)
        self.sciezka_bazowa.mkdir(exist_ok=True)
        self.id_workspace = None
        self.sciezka_workspace = None
        self.pliki: Dict[str, PlikRobocza] = {}
        
    async def inicjalizuj_workspace(self, nazwa_projektu: str = "Projekt Agent Zero"):
        """Inicjalizuj przestrzeń roboczą"""
        self.id_workspace = f"pr-{uuid.uuid4().hex[:8]}"
        self.sciezka_workspace = self.sciezka_bazowa / self.id_workspace
        self.sciezka_workspace.mkdir(exist_ok=True)
        
        # Utwórz strukturę
        (self.sciezka_workspace / "src").mkdir(exist_ok=True)
        (self.sciezka_workspace / "testy").mkdir(exist_ok=True)
        (self.sciezka_workspace / "dokumentacja").mkdir(exist_ok=True)
        
        # Konfiguracja
        config = {
            "id_workspace": self.id_workspace,
            "nazwa_projektu": nazwa_projektu,
            "utworzono": datetime.now().isoformat(),
            "wygenerowane_przez": "agent_zero"
        }
        
        with open(self.sciezka_workspace / "konfiguracja_workspace.json", 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        # README po polsku
        czas_utworzenia = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        readme = f"""# {nazwa_projektu}

Wygenerowane przez system Agent Zero AI.

## Informacje o przestrzeni roboczej
- ID: {self.id_workspace}  
- Utworzono: {czas_utworzenia}
- Ścieżka: {self.sciezka_workspace}

## Wygenerowane pliki
Ten katalog zawiera kod wygenerowany przez autonomicznych agentów AI.

## Użycie
Sprawdź katalog `src/` dla wygenerowanego kodu Python.

## Instalacja i uruchomienie
```
# Zainstaluj zależności
pip install -r requirements.txt

# Uruchom serwer FastAPI
uvicorn src.endpointy_autoryzacji:app --reload
```

## Testy
```
# Uruchom testy
pytest testy/ -v
```

---
*Wygenerowane automatycznie przez Agent Zero*
"""
        
        with open(self.sciezka_workspace / "README.md", 'w', encoding='utf-8') as f:
            f.write(readme)
        
        print(f"📁 Przestrzeń robocza zainicjalizowana: {self.id_workspace}")
        print(f"   Ścieżka: {self.sciezka_workspace}")
        
        return self.id_workspace
    
    async def zapisz_wygenerowany_kod(self, kod: str, id_agenta: str, id_zadania: str, nazwa_pliku: str = None):
        """Zapisz wygenerowany kod"""
        if not self.sciezka_workspace:
            await self.inicjalizuj_workspace()
        
        if not nazwa_pliku:
            nazwa_pliku = f"wygenerowany_kod_{int(datetime.now().timestamp())}.py"
        
        # Określ katalog na podstawie typu pliku
        if "test" in nazwa_pliku.lower():
            sciezka_pliku = self.sciezka_workspace / "testy" / nazwa_pliku
        else:
            sciezka_pliku = self.sciezka_workspace / "src" / nazwa_pliku
        
        # Utwórz katalog jeśli nie istnieje
        sciezka_pliku.parent.mkdir(parents=True, exist_ok=True)
        
        # Zapisz plik
        with open(sciezka_pliku, 'w', encoding='utf-8') as f:
            f.write(kod)
        
        # Zapisz metadane
        id_pliku = f"plik-{uuid.uuid4().hex[:8]}"
        plik_roboczy = PlikRobocza(
            id_pliku=id_pliku,
            nazwa_pliku=nazwa_pliku,
            zawartosc=kod,
            id_agenta=id_agenta,
            id_zadania=id_zadania,
            utworzono=datetime.now(),
            sciezka_pliku=sciezka_pliku
        )
        
        self.pliki[id_pliku] = plik_roboczy
        
        print(f"💾 Zapisano do workspace: {nazwa_pliku} ({len(kod)} znaków)")
        print(f"   Agent: {id_agenta}")
        print(f"   Ścieżka: {sciezka_pliku}")
        
        return id_pliku
    
    def pobierz_podsumowanie_workspace(self):
        """Podsumowanie przestrzeni roboczej"""
        if not self.sciezka_workspace:
            return "Brak zainicjalizowanej przestrzeni roboczej"
        
        liczba_plikow = len(self.pliki)
        suma_znakow_kodu = sum(len(f.zawartosc) for f in self.pliki.values())
        
        return {
            "id_workspace": self.id_workspace,
            "sciezka_workspace": str(self.sciezka_workspace),
            "liczba_plikow": liczba_plikow,
            "suma_znakow_kodu": suma_znakow_kodu,
            "pliki": [
                {
                    "nazwa_pliku": f.nazwa_pliku,
                    "id_agenta": f.id_agenta,
                    "rozmiar": len(f.zawartosc),
                    "utworzono": f.utworzono.isoformat()
                }
                for f in self.pliki.values()
            ]
        }

async def test_workspace_po_polsku():
    """Test przestrzeni roboczej po polsku"""
    
    print("="*60)
    print("🧪 TEST PRZESTRZENI ROBOCZEJ AGENT ZERO PO POLSKU")  
    print("="*60)
    
    # Inicjalizuj rozszerzenie workspace
    workspace = RozszerzenieWorkspace()
    await workspace.inicjalizuj_workspace("Platforma SaaS wygenerowana przez AI")
    
    print("\n📝 Testowanie tworzenia plików...")
    
    # Przykładowe kody (jak TWÓJ system generuje)
    przykladowe_kody = [
        {
            "kod": '''from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

# Aplikacja FastAPI wygenerowana przez Agent Zero
app = FastAPI(title="API Autoryzacji Agent Zero", description="Wygenerowane automatycznie")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

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
''',
            "id_agenta": "backend_ai_001",
            "nazwa_pliku": "endpointy_autoryzacji.py"
        },
        {
            "kod": '''import pytest
from fastapi.testclient import TestClient
from endpointy_autoryzacji import app

# Klient testowy
client = TestClient(app)

def test_logowanie_sukces():
    """Test pomyślnego logowania"""
    response = client.post(
        "/logowanie",
        data={"username": "admin", "password": "tajne"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()

def test_glowny_endpoint():
    """Test głównego endpointu"""
    response = client.get("/")
    assert response.status_code == 200
    dane = response.json()
    assert "wiadomosc" in dane

def test_endpoint_chroniony_bez_tokena():
    """Test chronionego endpointu bez tokena"""
    response = client.get("/chroniony")
    assert response.status_code == 401

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
''',
            "id_agenta": "test_ai_001", 
            "nazwa_pliku": "test_autoryzacji.py"
        },
        {
            "kod": '''# Modele danych wygenerowane przez Agent Zero
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
''',
            "id_agenta": "model_ai_001", 
            "nazwa_pliku": "modele.py"
        }
    ]
    
    # Zapisz kody
    for i, przyklad in enumerate(przykladowe_kody, 1):
        print(f"\n📝 Krok {i}: Zapisywanie {przyklad['nazwa_pliku']}...")
        
        id_pliku = await workspace.zapisz_wygenerowany_kod(
            kod=przyklad['kod'],
            id_agenta=przyklad['id_agenta'],
            id_zadania=f"zadanie_{i:03d}",
            nazwa_pliku=przyklad['nazwa_pliku']
        )
        
        print(f"   ✅ Zapisano jako: {id_pliku}")
    
    # Dodaj requirements.txt
    print(f"\n📝 Krok 4: Tworzenie requirements.txt...")
    wymagania = '''# Wymagania dla platformy Agent Zero
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
pytest==7.4.3
httpx==0.25.2
python-multipart==0.0.6
email-validator==2.1.0
'''
    
    plik_req = workspace.sciezka_workspace / "requirements.txt"
    with open(plik_req, 'w', encoding='utf-8') as f:
        f.write(wymagania)
    
    print(f"   ✅ Utworzono requirements.txt")
    
    # Dodaj skrypt uruchomienia
    print(f"\n📝 Krok 5: Tworzenie skryptu uruchomienia...")
    skrypt = '''#!/bin/bash
# Skrypt uruchomienia Agent Zero

echo "🚀 Uruchamianie Agent Zero Platform..."
echo "======================================"

# Zainstaluj zależności
echo "📦 Instalowanie zależności..."
pip install -r requirements.txt

echo "🌐 Uruchamianie serwera FastAPI..."
echo "📡 Aplikacja dostępna na: http://localhost:8000"
echo "📋 Dokumentacja API: http://localhost:8000/docs"
echo ""

uvicorn src.endpointy_autoryzacji:app --reload --host 0.0.0.0 --port 8000
'''
    
    skrypt_path = workspace.sciezka_workspace / "uruchom.sh"
    with open(skrypt_path, 'w', encoding='utf-8') as f:
        f.write(skrypt)
    
    # Ustaw uprawnienia executable
    skrypt_path.chmod(0o755)
    print(f"   ✅ Utworzono uruchom.sh (executable)")
    
    # Pokaż podsumowanie
    print(f"\n📊 PODSUMOWANIE:")
    print("="*40)
    
    podsumowanie = workspace.pobierz_podsumowanie_workspace()
    print(f"ID Workspace: {podsumowanie['id_workspace']}")
    print(f"Ścieżka: {podsumowanie['sciezka_workspace']}")
    print(f"Pliki: {podsumowanie['liczba_plikow']}")
    print(f"Kod: {podsumowanie['suma_znakow_kodu']} znaków")
    
    print(f"\n📁 Wygenerowane pliki:")
    for info_pliku in podsumowanie['pliki']:
        print(f"   🐍 {info_pliku['nazwa_pliku']} ({info_pliku['rozmiar']} znaków)")
        print(f"      Agent: {info_pliku['id_agenta']}")
    
    # Struktura katalogów
    print(f"\n📂 Struktura projektu:")
    for element in sorted(workspace.sciezka_workspace.rglob("*")):
        if element.is_file():
            wzgledna_sciezka = element.relative_to(workspace.sciezka_workspace)
            rozmiar = element.stat().st_size
            print(f"   📄 {wzgledna_sciezka} ({rozmiar} bajtów)")
    
    print(f"\n✅ TEST ZAKOŃCZONY SUKCESEM!")
    print("="*60)
    print(f"🗂️  Projekt gotowy: {workspace.sciezka_workspace}")
    print(f"🚀  Aby uruchomić:")
    print(f"     cd {workspace.sciezka_workspace}")
    print(f"     ./uruchom.sh")
    print(f"🌐  Następnie otwórz: http://localhost:8000")
    print(f"📋  Dokumentacja: http://localhost:8000/docs")
    
    return workspace

if __name__ == "__main__":
    asyncio.run(test_workspace_po_polsku())
