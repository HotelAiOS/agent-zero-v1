# Agent Zero V1 - Arch Linux Setup Guide

## 🚨 Problem Resolution

This guide resolves the following Arch Linux specific issues:
- `error: externally-managed-environment` (PEP 668)
- `ModuleNotFoundError: No module named 'docker'`
- `ModuleNotFoundError: No module named 'aiohttp_cors'`
- WebSocket server connection issues
- Fish Shell compatibility problems

## ✅ Solution: Virtual Environment Approach

### 📦 Files in this package:

1. `scripts/one_click_setup.sh` - Complete automated setup
2. `scripts/create_venv_and_install.sh` - Virtual environment + dependencies  
3. `scripts/run_tests_venv.sh` - Test runner in venv
4. `scripts/start_websocket_venv.sh` - WebSocket monitor starter
5. `scripts/agent_zero_system_test_venv.py` - Comprehensive system tests
6. `scripts/websocket_monitor_minimal.py` - Advanced WebSocket monitor
7. `scripts/venv_requirements.txt` - All dependencies with versions
8. `scripts/fish_setup.fish` - Fish Shell specific setup

## 🚀 Quick Setup (3 steps)

### Method 1: One-Click Setup (RECOMMENDED)
```bash
cd /path/to/agent-zero-v1
chmod +x scripts/*.sh scripts/*.fish
./scripts/one_click_setup.sh
```

### Method 2: Fish Shell
```fish
cd /path/to/agent-zero-v1
fish scripts/fish_setup.fish
python scripts/agent_zero_system_test_venv.py
```

### Method 3: Manual Steps
```bash
cd /path/to/agent-zero-v1

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r scripts/venv_requirements.txt

# Run tests
python scripts/agent_zero_system_test_venv.py

# Start WebSocket monitor
python scripts/websocket_monitor_minimal.py
```

## 🎯 What gets installed:

### Core Framework
- `fastapi==0.104.1` - Web framework
- `uvicorn==0.24.0` - ASGI server
- `websockets==12.0` - WebSocket support
- `aiohttp==3.9.1` + `aiohttp-cors==0.7.0` - HTTP client/server

### Database & Message Queue
- `neo4j==5.14.1` - Graph database
- `redis==5.0.1` - Cache and pub/sub
- `pika==1.3.2` - RabbitMQ client

### Development & Testing
- `docker==6.1.3` - Container management
- `pytest==7.4.3` - Testing framework
- `pytest-asyncio==0.21.1` - Async testing
- `psutil==5.9.6` - System monitoring

## 🌐 WebSocket Monitor Features

**URL:** http://localhost:8000

**Features:**
- 📊 Real-time system monitoring
- 🧪 Interactive agent testing
- 📡 Ping/Pong connectivity tests
- 📈 Live statistics (clients, messages, uptime)
- 🎨 Dark theme UI with animations
- 🔄 Auto-reconnection on connection loss
- 🤖 Agent Zero V1 specific message handling

**Message Types:**
- `ping` - Connectivity testing
- `agent_test` - Multi-agent communication testing
- `system_status` - Platform health monitoring
- `keepalive` - Connection maintenance

## 🐧 Arch Linux Specific Features

### PEP 668 Compliance
- ✅ Uses virtual environment (no --break-system-packages)
- ✅ Protects system Python packages
- ✅ Compatible with pacman package management
- ✅ Follows Arch Linux best practices

### Fish Shell Support
- ✅ Native Fish script (`fish_setup.fish`)
- ✅ Proper environment variable handling
- ✅ Fish-compatible activation commands
- ✅ Interactive shell integration

## 🧪 Testing & Validation

### System Test Coverage
- **Dependencies:** 8/8 Python modules
- **Services:** 4/4 backend services (Neo4j, RabbitMQ, Redis, WebSocket)
- **Core Components:** AgentExecutor signature validation
- **File Structure:** Critical project files verification

### Expected Results
- **17/17 tests passed (100% success rate)**
- **All services RUNNING**
- **Virtual environment ACTIVE**
- **WebSocket monitor OPERATIONAL**

## 🔧 Troubleshooting

### Problem: "Virtual environment not found"
```bash
./scripts/create_venv_and_install.sh
```

### Problem: "Permission denied"
```bash
chmod +x scripts/*.sh scripts/*.fish
```

### Problem: "Port 8000 already in use"
```bash
sudo fuser -k 8000/tcp
./scripts/start_websocket_venv.sh
```

### Problem: Fish Shell activation issues
```fish
# Manual Fish activation:
set -gx PATH "$PWD/venv/bin" $PATH
set -gx VIRTUAL_ENV "$PWD/venv"
```

### Problem: Import errors
```bash
# Verify venv activation:
source venv/bin/activate  # bash
# or
set -gx PATH "$PWD/venv/bin" $PATH  # fish

# Test imports:
python -c "import docker, aiohttp_cors, neo4j; print('All OK!')"
```

## 📊 Success Metrics

After successful setup, you should see:
- ✅ **System Status:** 100% operational
- ✅ **Test Results:** 17/17 passed
- ✅ **Services:** Neo4j (7474/7687), RabbitMQ (5672), Redis (6379), WebSocket (8000)
- ✅ **Environment:** Python 3.13+ in virtual environment
- ✅ **Platform:** Agent Zero V1 Multi-Agent ready for development

---

**🎯 Agent Zero V1 Development Team**  
**📅 Updated: October 8, 2025**  
**🐧 Arch Linux + Fish Shell Compatible**  
**🤖 Multi-Agent Enterprise Platform**