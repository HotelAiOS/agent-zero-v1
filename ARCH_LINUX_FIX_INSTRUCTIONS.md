## 🔧 INSTRUKCJA NAPRAWY ARCH LINUX - Agent Zero V2.0

### Szybka Naprawa (5 minut)

Wykonaj te komendy w katalogu projektu:

```bash
# 1. Pobierz i uruchom szybką naprawę
chmod +x quick-arch-fix.sh
./quick-arch-fix.sh

# 2. Uruchom system
./start_v2.sh

# 3. Testuj (w nowym terminalu)
./test_with_venv.sh
```

---

### Co zostanie naprawione:

✅ **Python Virtual Environment** - rozwiązuje externally-managed-environment  
✅ **ML Dependencies** - instaluje joblib, scikit-learn, pandas  
✅ **Neo4j przez Docker** - łatwiejsze niż systemowa instalacja  
✅ **API Entry Point** - tworzy brakujący api/main.py  
✅ **Docker Configuration** - naprawia docker-compose.yml  
✅ **Dockerfile** - tworzy brakujący Dockerfile.ai-intelligence  

---

### Po naprawie:

🌐 **API Docs**: http://localhost:8000/docs  
🔍 **Neo4j Browser**: http://localhost:7474  
📊 **Health Check**: http://localhost:8000/health  

---

### Komendy kontrolne:

```bash
# Sprawdź status
docker ps
curl http://localhost:7474
curl http://localhost:8000/health

# Logi
docker logs neo4j-agent-zero

# Restart Neo4j
docker restart neo4j-agent-zero

# Aktywuj venv
source venv/bin/activate
```

---

### Jeśli nadal problemy:

1. **Neo4j nie startuje:**
   ```bash
   docker stop neo4j-agent-zero
   docker rm neo4j-agent-zero
   docker run -d --name neo4j-agent-zero -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=none neo4j:5.15-community
   ```

2. **Brakuje dependencies:**
   ```bash
   source venv/bin/activate
   pip install joblib scikit-learn pandas numpy neo4j
   ```

3. **Test nie działa:**
   ```bash
   source venv/bin/activate
   python test-complete-implementation.py
   ```

---

### Debug Commands:

```bash
# Sprawdź venv
ls -la venv/bin/

# Test Python packages
source venv/bin/activate
python -c "import joblib, sklearn, pandas; print('✅ All ML packages OK')"

# Test Neo4j connection
python -c "from neo4j import GraphDatabase; print('✅ Neo4j driver OK')"

# Test API import
python -c "from shared.experience.enhanced_tracker import V2ExperienceTracker; print('✅ V2 Components OK')"
```

---

**Czas wykonania**: ~5 minut  
**Wymagania**: Docker, Python 3.8+, git  
**Status**: Ready to deploy! 🚀