# 🔍 DIAGNOZA SYSTEMU AGENT ZERO V1

## 🎯 IDENTYFIKACJA GŁÓWNEGO PLIKU

**Główny plik systemu: `integrated-system.py` (43KB)**

### ✅ Co to jest:
- **Pełny system integracyjny** Agent Zero V1 z V2.0 Intelligence Layer
- **FastAPI serwer** działający na porcie **8000**
- **AI Reasoning Engine** z integracją z Ollama
- **Komponenty**: Neo4j, Redis, RabbitMQ, WebSocket
- **Baza danych**: SQLite (agent_zero_integrated.db)

## 🚀 JAK URUCHOMIĆ SYSTEM

### 1. **Sprawdź czy system już działa:**
```bash
# Sprawdź port 8000
sudo netstat -tlnp | grep :8000
lsof -ti:8000

# Sprawdź procesy Python
ps aux | grep python | grep integrated
```

### 2. **Zatrzymaj istniejące procesy:**
```bash
# Zabij procesy na porcie 8000
lsof -ti:8000 | xargs kill -9

# Zabij wszystkie Python procesy related
pkill -f "integrated-system"
pkill -f "python.*integrated"
```

### 3. **Uruchom system główny:**
```bash
# Opcja 1: Tryb demo (testowy)
python integrated-system.py --mode demo

# Opcja 2: Tryb produkcyjny (serwer)
python integrated-system.py --mode production

# Opcja 3: Domyślny (demo)
python integrated-system.py
```

## 🔧 ENDPOINTY I TESTOWANIE

### **Główne endpointy:**
- **Health check**: `GET http://localhost:8000/api/v1/health`
- **Dokumentacja**: `http://localhost:8000/docs`
- **Decompose project**: `POST http://localhost:8000/api/v1/decompose`
- **Get tasks**: `GET http://localhost:8000/api/v1/tasks`
- **WebSocket**: `ws://localhost:8000/ws/tasks`

### **Test podstawowy:**
```bash
# Test połączenia
curl http://localhost:8000/api/v1/health

# Test z JSONem
curl -X POST "http://localhost:8000/api/v1/decompose" \
     -H "Content-Type: application/json" \
     -d '{"description": "Create AI system", "complexity": "medium"}'
```

## 🛠️ KOMPONENTY SYSTEMU

### **Główne moduły:**
1. **IntegratedAIReasoningEngine** - AI analysis with Ollama
2. **IntegratedEnhancedTaskDecomposer** - Task decomposition
3. **ProductionIntegratedSystem** - Main orchestrator
4. **FastAPI endpoints** - REST API interface

### **Zewnętrzne zależności:**
- **Ollama**: http://localhost:11434 (AI models)
- **Neo4j**: bolt://localhost:7687 (knowledge graph)
- **Redis**: localhost:6379 (caching)
- **RabbitMQ**: localhost:5672 (messaging)

## 🐛 ROZWIĄZYWANIE PROBLEMÓW

### **Problem 1: Port zajęty**
```bash
# Identyfikuj proces
sudo netstat -tlnp | grep :8000
# Zabij proces
kill -9 <PID>
```

### **Problem 2: Python 3.13 + Pydantic**
```bash
# Zainstaluj pre-compiled wheels
pip install --upgrade pip
pip install --only-binary=all fastapi uvicorn pydantic
```

### **Problem 3: Ollama niedostępny**
- System działa bez Ollama (fallback mode)
- Sprawdź: `curl http://localhost:11434/api/tags`

### **Problem 4: 404 Not Found**
- Sprawdź czy używasz właściwych endpointów `/api/v1/`
- Upewnij się że serwer wystartował poprawnie

## 📊 MONITORING I LOGI

### **Pliki logów:**
- `agent_zero_integrated.log` - główne logi systemu
- Console output - real-time monitoring

### **Baza danych:**
- `agent_zero_integrated.db` - SQLite database
- Tabele: `enhanced_tasks`, `decomposition_sessions`

## 🎉 SUKCES - GOTOWY DO DZIAŁANIA!

System jest **kompletny i gotowy**. Główny plik `integrated-system.py` zawiera:
- ✅ Pełną implementację AI reasoning
- ✅ Task decomposition z AI enhancement
- ✅ Production-ready FastAPI endpoints
- ✅ Database persistence
- ✅ WebSocket real-time updates
- ✅ Full error handling i fallbacks

**Następne kroki:**
1. Uruchom system: `python integrated-system.py --mode production`
2. Sprawdź health: `curl http://localhost:8000/api/v1/health`
3. Otwórz docs: `http://localhost:8000/docs`
4. Test API endpoints
5. Ready for development! 🚀