# Agent Zero V1 - Critical Fixes Package

**Generated:** 2025-10-08  
**Version:** 1.0.0  
**Status:** Production Ready ✅

## 📋 Overview

This package contains fixes for 3 critical issues in Agent Zero V1:

| Issue ID | Component | Severity | Fix Time |
|----------|-----------|----------|----------|
| **A0-5** | Neo4j Connection | CRITICAL | 30 min |
| **A0-6** | AgentExecutor Signature | HIGH | 45 min |
| **TECH-001** | Task Decomposer JSON | HIGH | 60 min |

**Total Fix Time:** ~2 hours  
**ROI Impact:** Unblocks 870,600 PLN project value

---

## 📦 Package Contents

```
agent-zero-fixes/
├── neo4j_client.py          # Fixed Neo4j client with retry logic
├── agent_executor.py        # Standardized executor interface
├── task_decomposer.py       # Robust JSON parser for LLM
├── docker-compose.yml       # Enhanced Docker configuration
├── apply_fixes.py           # Automated fix application script
├── .env.example             # Environment configuration template
└── README.md                # This file
```

---

## 🚀 Quick Start

### Step 1: Prepare

```bash
# Download all files to a temporary directory
cd /tmp
mkdir agent-zero-fixes
cd agent-zero-fixes

# Place all 5 fix files here:
ls -la
# Should show:
# - neo4j_client.py
# - agent_executor.py
# - task_decomposer.py
# - docker-compose.yml
# - apply_fixes.py
```

### Step 2: Apply Fixes

```bash
# Make script executable
chmod +x apply_fixes.py

# Run automated fix (recommended)
python apply_fixes.py --project-root /path/to/agent-zero-v1

# Or apply without Docker restart
python apply_fixes.py --project-root /path/to/agent-zero-v1 --skip-docker
```

### Step 3: Verify

```bash
# Check Docker services
docker-compose ps

# Test Neo4j connection
docker exec -it agent-zero-neo4j cypher-shell -u neo4j -p agent-pass

# View logs
docker-compose logs -f neo4j

# Run tests
cd /path/to/agent-zero-v1
pytest tests/test_full_integration.py -v
```

---

## 🔧 Manual Installation (Alternative)

If you prefer manual installation:

### 1. Backup Existing Files

```bash
cd /path/to/agent-zero-v1
mkdir -p backups/$(date +%Y%m%d_%H%M%S)

# Backup files
cp shared/knowledge/neo4j_client.py backups/$(date +%Y%m%d_%H%M%S)/
cp src/core/agent_executor.py backups/$(date +%Y%m%d_%H%M%S)/
cp shared/orchestration/task_decomposer.py backups/$(date +%Y%m%d_%H%M%S)/
cp docker-compose.yml backups/$(date +%Y%m%d_%H%M%S)/
```

### 2. Copy Fixed Files

```bash
# Copy from fixes directory
cp /tmp/agent-zero-fixes/neo4j_client.py shared/knowledge/
cp /tmp/agent-zero-fixes/agent_executor.py src/core/
cp /tmp/agent-zero-fixes/task_decomposer.py shared/orchestration/
cp /tmp/agent-zero-fixes/docker-compose.yml .
```

### 3. Restart Services

```bash
# Restart Docker
docker-compose down
docker-compose up -d

# Wait for services to be healthy
sleep 30

# Check status
docker-compose ps
```

---

## 🔍 What Each Fix Does

### Neo4j Client (`neo4j_client.py`)

**Problem:** Connection timeouts, no retry logic  
**Solution:**
- ✅ Exponential backoff retry (5 attempts)
- ✅ Connection pooling (max 50 connections)
- ✅ Health check mechanism
- ✅ Automatic reconnection on failures

**Key Changes:**
```python
# Before
self.driver = GraphDatabase.driver(uri, auth=(user, pass))

# After
def _connect_with_retry(self):
    for attempt in range(1, self.max_retries + 1):
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password),
                max_connection_pool_size=50,
                connection_timeout=30.0
            )
            self.driver.verify_connectivity()
            return
        except ServiceUnavailable:
            delay = self.retry_delay * (2 ** (attempt - 1))
            time.sleep(delay)
```

### AgentExecutor (`agent_executor.py`)

**Problem:** Method signature mismatch, incompatible with new AI interface  
**Solution:**
- ✅ Standardized `execute_task(context, callback)` signature
- ✅ Type-safe `ExecutionContext` and `ExecutionResult` dataclasses
- ✅ Async/await support with timeout handling
- ✅ Backward compatibility wrapper

**Key Changes:**
```python
# Before
async def execute_agent_task(self, agent, task, output_dir):
    return agent.execute(task, output_dir)

# After
async def execute_task(
    self,
    context: ExecutionContext,
    callback: Optional[Callable] = None
) -> ExecutionResult:
    # Standardized execution with timeout, error handling, callbacks
```

### Task Decomposer (`task_decomposer.py`)

**Problem:** LLM JSON parsing fails frequently  
**Solution:**
- ✅ 5 parsing strategies (direct, code block, regex, first object, fixes)
- ✅ Handles markdown, single quotes, trailing commas
- ✅ Retry logic with prompt refinement (3 attempts)
- ✅ Comprehensive validation

**Key Changes:**
```python
# Before (line ~220)
result = json.loads(llm_response)  # Often fails

# After
class RobustJSONParser:
    @staticmethod
    def extract_json(text: str) -> Optional[Dict]:
        strategies = [
            _parse_direct,
            _parse_code_block,
            _parse_with_regex,
            _parse_first_json_object,
            _parse_with_fixes
        ]
        for strategy in strategies:
            try:
                result = strategy(text)
                if result:
                    return result
            except:
                continue
```

### Docker Compose (`docker-compose.yml`)

**Problem:** Missing health checks, suboptimal configuration  
**Solution:**
- ✅ Health checks for all services
- ✅ Optimized Neo4j memory settings
- ✅ Persistent volumes for data
- ✅ Network isolation

---

## 📊 Verification Checklist

After applying fixes, verify:

- [ ] **Neo4j connection works**
  ```bash
  docker exec -it agent-zero-neo4j cypher-shell -u neo4j -p agent-pass
  # Should connect without errors
  ```

- [ ] **All Docker services healthy**
  ```bash
  docker-compose ps
  # All services should show "Up" and "healthy"
  ```

- [ ] **No errors in logs**
  ```bash
  docker-compose logs --tail=50
  # Should show no errors
  ```

- [ ] **Integration tests pass**
  ```bash
  pytest tests/test_full_integration.py -v
  # All tests should pass
  ```

---

## 🐛 Troubleshooting

### Neo4j won't start

```bash
# Check logs
docker-compose logs neo4j

# Common issues:
# 1. Port already in use
sudo lsof -i :7687
# Kill process if needed

# 2. Insufficient memory
# Edit docker-compose.yml, reduce heap size

# 3. Data corruption
docker-compose down -v  # WARNING: Deletes data
docker-compose up -d
```

### AgentExecutor signature errors

```bash
# Check all call sites updated
grep -r "execute_agent_task" src/
# Should return no results (or only in legacy wrapper)

# Verify imports
python -c "from src.core.agent_executor import ExecutionContext; print('OK')"
```

### Task decomposer still failing

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Test parser directly
python task_decomposer.py
# Should show parsing tests passing
```

---

## 📞 Support

If issues persist:

1. **Check backup files** in `backups_YYYYMMDD_HHMMSS/`
2. **Review Linear issues:**
   - [A0-5: Neo4j Connection](https://linear.app/biesbit-a0/issue/A0-5)
   - [A0-6: AgentExecutor](https://linear.app/biesbit-a0/issue/A0-6)
3. **Check Notion documentation:**
   - [Master Dashboard](https://notion.so/2865661115808152bcdcedfa6f01f2e2)

---

## ✅ Success Criteria

Fixes are successful when:

✅ Neo4j connects on first try  
✅ All 17 integration tests pass  
✅ Docker services show "healthy" status  
✅ Agent tasks execute without signature errors  
✅ LLM responses parse successfully (>95% rate)

---

**Last Updated:** 2025-10-08 20:53 CEST  
**Package Version:** 1.0.0  
**Project:** Agent Zero V1 → V2.0+ Transition
