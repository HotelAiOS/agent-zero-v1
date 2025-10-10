# Agent Zero V1 - Integrated Backend Services Deployment

## 🎯 Cel: Integracja z Istniejącym Systemem

Ten deployment **rozszerza** Twój działający system Agent Zero V1, używając:
- ✅ **SimpleTracker** jako backend dla API endpoints
- ✅ **FeedbackLoopEngine** dla real-time monitoring  
- ✅ **BusinessParser** (A0-19) jako REST API
- ✅ **Neo4jClient** dla knowledge persistence
- ✅ **AgentExecutor** dla task execution

**NIE zastępuje** istniejącej funkcjonalności - **dodaje** API layer dla Developer B Frontend.

## 📦 Pliki Integracyjne

### **1. integrated-api-gateway.py** 
API Gateway który importuje i używa `simple-tracker.py` jako backend:
- `/api/v1/agents/status` → SimpleTracker.get_daily_stats()  
- `/api/v1/tasks/current` → SimpleTracker.export_for_analysis()
- `/api/v1/business/parse` → business-requirements-parser.py
- Zachowuje compatibility z CLI system

### **2. integrated-websocket-service.py**
WebSocket service dla real-time monitoring:
- Używa SimpleTracker dla live metrics
- Integracja z FeedbackLoopEngine
- Streaming updates dla Developer B Frontend
- endpoint: ws://localhost:8080/ws/agents/live-monitor

### **3. integrated-agent-orchestrator.py** 
Orchestrator używający existing components:
- agent_executor.py dla execution
- task_decomposer.py dla task breakdown
- neo4j_client.py dla persistence  
- SimpleTracker dla tracking

### **4. enhanced-docker-compose.yml**
Rozszerza istniejący docker-compose.yml o:
- Application services (API Gateway, WebSocket, Orchestrator)
- Volume mounts dla component imports
- Service dependencies i health checks
- **Zachowuje** istniejące Neo4j, RabbitMQ, Redis

## 🚀 Deployment Steps

### **Step 1: Backup Current System**
```bash
mkdir -p backups/integration_$(date +%s)
cp docker-compose.yml backups/integration_$(date +%s)/
cp -r services backups/integration_$(date +%s)/ 2>/dev/null || true
```

### **Step 2: Copy Integrated Files**
```bash
# Replace Docker configuration
cp enhanced-docker-compose.yml docker-compose.yml

# Copy service implementations (overwrites empty services)
cp integrated-api-gateway.py services/api-gateway/src/main.py
cp integrated-websocket-service.py services/chat-service/src/main.py  
cp integrated-agent-orchestrator.py services/agent-orchestrator/src/main.py

# Copy setup script
cp services-integration-setup.py ./
chmod +x services-integration-setup.py
```

### **Step 3: Install Dependencies**
```bash
# Install FastAPI and WebSocket dependencies
pip install fastapi uvicorn[standard] websockets python-multipart

# Or use integrated setup
python services-integration-setup.py
```

### **Step 4: Deploy Integrated System**
```bash
# Stop existing services
docker-compose down

# Start integrated system
docker-compose up -d

# Check service health  
docker-compose ps
```

### **Step 5: Verify Integration**
```bash
# Test API Gateway with SimpleTracker integration
curl http://localhost:8000/api/v1/agents/status

# Test WebSocket service
curl http://localhost:8080/health

# Test Agent Orchestrator
curl http://localhost:8002/health

# Test integration status
curl http://localhost:8000/api/v1/system/integration-status
```

## 🔍 Integration Verification

### **Expected API Response (uses real SimpleTracker data):**
```json
{
  "agents": {
    "active_count": 3,
    "total_tasks_today": 12,
    "success_rate": 85.5,
    "avg_cost": 0.003
  },
  "performance": {
    "best_model": "qwen2.5-coder:7b", 
    "total_feedback": 8,
    "avg_rating": 4.2
  },
  "source": "SimpleTracker_integration"
}
```

### **WebSocket Connection Test:**
```bash
# Connect to WebSocket (real-time updates from FeedbackLoopEngine)
wscat -c ws://localhost:8080/ws/agents/live-monitor
```

### **Business Parser API Test:**
```bash
curl -X POST http://localhost:8000/api/v1/business/parse \
  -H "Content-Type: application/json" \
  -d '{"requirement_text": "Create user authentication system"}'
```

## ✅ Success Criteria - Developer B Ready

Po successful deployment:

✅ **API Gateway** zwraca real data z SimpleTracker (nie mock)  
✅ **WebSocket** streamuje live updates z FeedbackLoopEngine  
✅ **Agent Orchestrator** używa existing agent_executor.py  
✅ **Business Parser** API dostępny (A0-19 functionality)  
✅ **CLI system** dalej działa identical jak wcześniej  
✅ **Docker services** wszystkie healthy  
✅ **Integration verification** wszystkie testy pass

## 🎯 Developer B Frontend Endpoints

### **Agents Status (GET):**
```
GET http://localhost:8000/api/v1/agents/status
→ Real data from SimpleTracker
```

### **Current Tasks (GET):**  
```
GET http://localhost:8000/api/v1/tasks/current
→ Recent tasks from SimpleTracker export
```

### **Live Monitoring (WebSocket):**
```
WS ws://localhost:8080/ws/agents/live-monitor  
→ Real-time updates from FeedbackLoopEngine
```

### **Task Orchestration (POST):**
```
POST http://localhost:8002/api/v1/orchestration/plan
→ Multi-agent coordination using existing components
```

## 🔧 Troubleshooting

### **Service nie startuje:**
```bash
# Check logs
docker-compose logs api-gateway

# Verify imports
docker exec -it agent-zero-api-gateway python -c "import sys; print(sys.path)"
```

### **SimpleTracker nie found:**
```bash
# Verify project mount
docker exec -it agent-zero-api-gateway ls -la /app/project/

# Check SimpleTracker import
docker exec -it agent-zero-api-gateway python -c "
import sys; sys.path.insert(0, '/app/project');
exec(open('/app/project/simple-tracker.py').read(), globals());
tracker = SimpleTracker(); print('SimpleTracker OK')
"
```

### **Integration check fails:**
```bash
# Full system status
curl http://localhost:8000/api/v1/system/integration-status
```

## 🌟 Integration Benefits

**For Developer B Frontend:**
- Real agent data (not mocked)
- Compatible with existing CLI workflow  
- Business requirements API ready (A0-19)
- Real-time monitoring capabilities

**For System Continuity:**  
- CLI system unchanged
- SimpleTracker preserved and extended
- All existing data maintained
- Kaizen feedback loop enhanced with API access

**For Future Development:**
- Scalable service architecture  
- Component reusability maintained
- Easy to add new services
- Integration patterns established

---

**🚀 Result: Developer B gets production-ready backend services that extend (not replace) your working Agent Zero V1 system!**