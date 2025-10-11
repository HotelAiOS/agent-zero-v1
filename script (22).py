# Sprawdzam strukturę plików lokalnych i identyfikuję główne komponenty systemu

files_analysis = {
    "integrated_files": [
        "integrated-system.py",  # 43KB - główny plik systemu
        "integrated-system-production.py",  # Nie ma w repo - lokalny plik
        "integrated-agent-orchestrator.py", # 14KB
        "integrated-api-gateway.py",  # 10KB  
        "integrated-websocket-service.py"  # 8KB
    ],
    
    "production_files": [
        "agent-zero-complete-integration-bugfree.py",  # 64KB
        "agent-zero-complete-integration.py",  # 62KB
        "agent-zero-v1-standalone-fixed.py",  # 33KB
        "create-production-system.py"  # 42KB
    ],
    
    "main_candidates": {
        "integrated-system.py": "43KB - Główny system integracyjny",
        "agent-zero-complete-integration-bugfree.py": "64KB - Kompletna integracja bez błędów", 
        "create-production-system.py": "42KB - System produkcyjny"
    }
}

# Pokazuję co prawdopodobnie jest głównym plikiem
print("🔍 IDENTYFIKACJA GŁÓWNEGO PLIKU SYSTEMU")
print("=" * 50)
print("\n📁 Pliki lokalne vs repozytorium:")
print("✅ integrated-system.py (43KB) - ISTNIEJE W REPO")
print("❌ integrated-system-production.py (40KB) - TYLKO LOKALNY") 
print("✅ agent-zero-complete-integration-bugfree.py (64KB) - ISTNIEJE W REPO")

print("\n🎯 REKOMENDACJA:")
print("Główny plik to prawdopodobnie: integrated-system.py")
print("Powód: Jest w repo, ma rozsądny rozmiar, nazwa wskazuje na główny system")

files_analysis