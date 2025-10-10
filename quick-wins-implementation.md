# Agent Zero V1 - Quick Wins Kaizen Implementation

## 🎯 INSTRUKCJE INSTALACJI I UŻYCIA

### Priorytet 1: Natychmiastowe wdrożenie (DZIŚ - 3.5h)

#### 1. Instalacja podstawowych plików

```bash
# 1. Zastąp istniejący cli/__main__.py
cp cli-main-enhanced.py cli/__main__.py

# 2. Utwórz katalog dla nowych komponentów
mkdir -p shared/utils
mkdir -p shared/kaizen

# 3. Dodaj nowe pliki
cp simple-tracker.py shared/utils/simple_tracker.py
cp feedback-loop-engine.py shared/kaizen/feedback_loop.py

# 4. Zainstaluj wymagane zależności
pip install typer rich sqlite3 pathlib
```

#### 2. Pierwsza konfiguracja

```bash
# Testuj podstawowe działanie
python cli/__main__.py kaizen-status

# Wykonaj pierwsze zadanie z feedbackiem
python cli/__main__.py ask "Jak działa system Agent Zero?"

# Sprawdź statystyki modeli
python cli/__main__.py compare-models
```

#### 3. Natychmiastowe korzyści

Po pierwszym uruchomieniu system:
- ✅ Zbiera feedback po każdym zadaniu (1-5 gwiazdek)
- ✅ Przechowuje metryki w SQLite (~/.agent-zero/tracker.db)
- ✅ Pokazuje porównanie wydajności modeli
- ✅ Identyfikuje problematyczne modele
- ✅ Generuje actionable insights

### Przykład użycia

```bash
# Wykonaj zadanie kodowania
python cli/__main__.py code "Stwórz funkcję do sortowania listy"
# System zapyta o rating 1-5 ⭐

# Sprawdź co się zmieniło
python cli/__main__.py compare-models
# Zobaczysz statystyki: koszt, jakość, użycie

# Status uczenia się
python cli/__main__.py kaizen-status
# 📈 7 zadań, 85% feedback rate, avg 4.2/5
```

## 🔧 INTEGRACJA Z ISTNIEJĄCYM KODEM

### Gdzie wstawić feedback collection

W każdej funkcji `@app.command()` w cli/__main__.py:

```python
# PRZED (istniejący kod):
@app.command()
def ask(question: str):
    response = process_question(question)
    console.print(response)
    # KONIEC

# PO (z Kaizen):
@app.command() 
def ask(question: str):
    task_id = str(uuid.uuid4())
    start_time = datetime.now()
    
    response = process_question(question)
    console.print(response)
    
    # DODAJ TE LINIE:
    tracker.track_task(task_id, "chat", model_used, model_recommended, cost, latency)
    ask_for_quick_feedback(task_id, model_used)  # 🔑 KLUCZOWE!
```

### Integracja z AI Router

W `services/ai-router/src/router/orchestrator.py`:

```python
# PRZED (statyczne mapowanie):
self.task_model_map = {
    TaskType.CODE: "qwen2.5-coder:7b",
    TaskType.CHAT: "llama3.2-3b",
    # ...
}

# PO (z trackingiem do przyszłego uczenia):
from shared.utils.simple_tracker import SimpleTracker

class AIOrchestrator:
    def __init__(self):
        self.tracker = SimpleTracker()
        self.task_model_map = {...}  # Tymczasowo zostaw
    
    def select_model(self, task_type: str) -> str:
        # Tymczasowo używaj mapy, ale trackuj decyzje
        recommended = self.task_model_map.get(task_type, "llama3.2-3b")
        
        # TODO Week 44: Zastąp IntelligentModelSelector
        # recommended = await self.intelligent_selector.recommend_model(task_type)
        
        return recommended
```

## 📊 CO OSIĄGNIESZ PO 3.5h

### Natychmiastowe korzyści:
1. **Feedback Loop** - System zaczyna się uczyć z każdej interakcji ✅
2. **Model Insights** - Widzisz który model działa najlepiej ✅  
3. **Cost Tracking** - Śledzisz koszty w czasie rzeczywistym ✅
4. **Quality Metrics** - Mierzysz jakość output'ów ✅
5. **Foundation** - Masz bazę dla pełnego systemu Kaizen ✅

### Przykład danych po tygodniu:

```
📊 Model Performance (Last 7 days)
┌─────────────────┬───────┬──────────┬────────────┬──────────────┬───────────┬──────────────┐
│ Model           │ Usage │ Avg Cost │ Avg Rating │ Feedback Count│ Overrides │ Recommendation│
├─────────────────┼───────┼──────────┼────────────┼──────────────┼───────────┼──────────────┤
│ claude-sonnet   │ 15    │ $0.0234  │ 4.8/5      │ 12           │ 1         │ ⭐ BEST       │
│ llama3.2-3b     │ 43    │ $0.0012  │ 4.1/5      │ 38           │ 8         │              │
│ qwen2.5-coder   │ 28    │ $0.0023  │ 4.5/5      │ 24           │ 3         │              │
└─────────────────┴───────┴──────────┴────────────┴──────────────┴───────────┴──────────────┘

🎯 Recommendation: Use claude-sonnet for best quality/cost ratio
⚠️ Issues detected:
   • llama3.2-3b: High override rate (18%) - users prefer other models
```

## 🚀 NASTĘPNE KROKI (Week 44-45)

Po wdrożeniu Quick Wins:

### Week 44: Inteligentny System (21 SP)
1. **Neo4j Knowledge Graph** - Pełna analiza wzorców
2. **Pattern Recognition** - Automatyczne wykrywanie problemów  
3. **Adaptive Learning** - Wagi decyzyjne się uczą
4. **Real-time Alerts** - Natychmiastowe ostrzeżenia

### Week 45: Dojrzały Kaizen (28 SP)
1. **Cross-Project Learning** - Uczenie między projektami
2. **Auto-improvements** - Automatyczne optymalizacje
3. **Kaizen Dashboard** - Wizualizacja postępów
4. **Advanced Analytics** - Predykcje i rekomendacje

## ⚡ DZIAŁAJ TERAZ!

**Czas: 3.5h | Priorytet: KRYTYCZNY | Impact: Fundamentalny**

1. **14:00-15:00** → Wdróż cli/__main__.py z feedbackiem
2. **15:00-15:30** → Dodaj SimpleTracker 
3. **15:30-16:00** → Test i command `compare-models`

**Po 16:00 masz działający system uczący się! 🎉**

Bez tego Kaizen nie istnieje. Z tym masz foundation dla AI-first learning platform.

## 🔥 SUCCESS METRICS

Po pierwszym dniu użytkowania:
- [ ] Feedback rate > 50%
- [ ] Tracked tasks > 10  
- [ ] Model comparison działa
- [ ] Cost tracking aktywny
- [ ] Pattern detection ready

**To fundamentalna różnica między AI tool a AI-first learning system!**