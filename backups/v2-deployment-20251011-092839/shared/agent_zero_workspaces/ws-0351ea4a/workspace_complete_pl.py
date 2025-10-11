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
  
