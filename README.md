# Agent Zero v1 - Autonomous AI Agent System

**Status:** 99% Complete | **Build:** Passing | **Tests:** 8/8 Passing

Multi-agent AI system with LLM integration, RabbitMQ messaging, and full orchestration capabilities.

---

## 🚀 Features

### ✅ Core Components
- **Agent Factory** - Dynamic agent creation from YAML templates (8 agent types)
- **Lifecycle Manager** - State management, metrics, and health monitoring
- **LLM Integration** - Ollama with DeepSeek Coder for code generation
- **RabbitMQ Messaging** - Real-time agent-to-agent communication
- **Capability System** - Smart agent matching based on skills and tech stack
- **Protocol Validation** - Code review, security audit, testing protocols
- **Learning System** - Feedback collection and optimization
- **Persistence** - MongoDB storage for agents and tasks
- **REST API** - HTTP endpoints for system control
- **GitHub Integration** - Automated commits and issue tracking

### 🤖 Agent Types
1. **Architect** - System design and architecture
2. **Backend** - API and server-side development
3. **Frontend** - UI/UX and client-side development
4. **Database** - Schema design and optimization
5. **DevOps** - Infrastructure and deployment
6. **QA Tester** - Testing and quality assurance
7. **Security** - Security audits and pentesting
8. **Performance** - Optimization and profiling

---

## 📦 Installation

### Prerequisites
System requirements
Python 3.11+

Docker (for RabbitMQ)

Ollama (for LLM)

text

### Setup
1. Clone repository
git clone https://github.com/HotelAiOS/agent-zero-v1.git
cd agent-zero-v1

2. Create virtual environment
python -m venv venv
source venv/bin/activate # Linux/Mac

venv\Scripts\activate # Windows
3. Install dependencies
pip install -r requirements.txt

4. Start RabbitMQ
docker run -d
--name rabbitmq
-p 5672:5672
-p 15672:15672
rabbitmq:3-management

5. Install Ollama and pull model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull deepseek-coder:6.7b

6. Configure (optional)
cp shared/llm/config.yaml.example shared/llm/config.yaml

Edit config.yaml if needed
text

---

## 🎯 Quick Start

### Create and Run Agent Team

from agent_factory.factory import AgentFactory

Initialize factory
factory = AgentFactory()

Create agents
backend = factory.create_agent("backend", agent_id="backend_1")
database = factory.create_agent("database", agent_id="database_1")

Start messaging
backend.start_listening()
database.start_listening()

Execute task
task = {
'description': 'Create REST API for user management',
'tech_stack': ['python', 'fastapi'],
'requirements': ['JWT auth', 'CRUD operations']
}
result = backend.execute_task(task)
print(result['output']) # Generated code

Send message between agents
backend.send_message(
recipient_id="database_1",
subject="Schema Request",
content="Need users table schema"
)

Cleanup
factory.lifecycle_manager.terminate_agent("backend_1")
factory.lifecycle_manager.terminate_agent("database_1")

text

---

## 📊 Architecture

┌─────────────────────────────────────────────────────┐
│ Agent Factory │
│ ┌───────────┐ ┌───────────┐ ┌────────────┐ │
│ │ Templates │ │ Lifecycle │ │ Capability │ │
│ │ (YAML) │ │ Manager │ │ Matcher │ │
│ └───────────┘ └───────────┘ └────────────┘ │
└────────────┬────────────────────────────────────────┘
│
┌────▼─────┐
│ Agents │
└────┬─────┘
│
┌────────┴────────┐
│ │
┌───▼────┐ ┌────▼─────┐
│ LLM │ │ RabbitMQ │
│(Ollama)│ │Messaging │
└────────┘ └──────────┘

text

---

## 🧪 Testing

### Run All Tests
LLM Integration Test
PYTHONPATH=./shared venv/bin/python shared/llm/test_agent_integration.py

RabbitMQ Communication Test
PYTHONPATH=./shared venv/bin/python shared/messaging/test_agent_communication.py

Team Communication Test
PYTHONPATH=./shared venv/bin/python shared/agent_factory/test_team_communication.py

text

### Test Results
✅ LLM Integration: Backend agent generated email validation code
✅ Agent-to-Agent: Backend ↔ Database communication successful
✅ Team Communication: 3 agents, 8 messages exchanged

text

---

## 📚 API Reference

### Factory
Create agent
agent = factory.create_agent(agent_type, agent_id=None)

Get available agents
agents = factory.lifecycle_manager.get_available_agents(agent_type="backend")

System health
health = factory.lifecycle_manager.get_system_health()

text

### Agent Instance
Execute task with LLM
result = agent.execute_task(task_dict)

Send message
agent.send_message(recipient_id, subject, content, payload={})

Broadcast
agent.broadcast(subject, content, project_id=None)

Listen for messages
agent.start_listening()
agent.stop_listening()

text

### Messaging
from messaging import AgentCommunicator, MessageType, MessagePriority

Create communicator
comm = AgentCommunicator("agent_id", auto_connect=True)

Direct message
comm.send_direct(recipient_id, subject, content)

Task request
comm.request_task(recipient_id, task_description, task_id, project_id)

Code review request
comm.request_code_review(recipient_id, code, task_id, project_id)

text

---

## 🛠️ Configuration

### LLM Config (`shared/llm/config.yaml`)
ollama:
base_url: "http://localhost:11434"
timeout: 120

agent_models:
backend:
model: "deepseek-coder:6.7b"
temperature: 0.7
num_ctx: 8192

text

### RabbitMQ Config
from messaging import BusConfig

config = BusConfig(
host="localhost",
port=5672,
username="guest",
password="guest"
)

text

---

## 📈 Metrics & Monitoring

### Agent Metrics
metrics = lifecycle_manager.get_agent_metrics(agent_id)
print(f"Tasks completed: {metrics.tasks_completed}")
print(f"Messages sent: {metrics.messages_sent}")
print(f"Avg response time: {metrics.average_response_time:.2f}s")

text

### System Health
health = lifecycle_manager.get_system_health()

Returns:
{
"status": "healthy",
"total_agents": 3,
"state_distribution": {...},
"total_tasks_completed": 10,
"total_messages": 25,
"error_rate": 0.0
}

text

---

## 🔧 Development

### Project Structure
agent-zero-v1/
├── shared/
│ ├── agent_factory/ # Agent creation & lifecycle
│ │ ├── templates/ # YAML agent definitions
│ │ ├── factory.py
│ │ ├── lifecycle.py
│ │ └── capabilities.py
│ ├── llm/ # LLM integration
│ │ ├── ollama_client.py
│ │ ├── prompt_builder.py
│ │ └── response_parser.py
│ ├── messaging/ # RabbitMQ messaging
│ │ ├── message.py
│ │ ├── bus.py
│ │ ├── publisher.py
│ │ ├── consumer.py
│ │ └── agent_comm.py
│ ├── protocols/ # Validation protocols
│ ├── learning/ # Feedback system
│ ├── persistence/ # MongoDB storage
│ └── core_integration/ # Core orchestration
└── README.md

text

### Running in Development
Enable debug logging
export LOG_LEVEL=DEBUG

Run with custom config
PYTHONPATH=./shared python your_script.py

text

---

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📝 License

MIT License - see LICENSE file for details

---

## 🎯 Roadmap

### Current: v1.0 (99% Complete)
- ✅ Core agent system
- ✅ LLM integration
- ✅ RabbitMQ messaging
- 🔄 Documentation finalization

### Future: v1.1
- Neo4j knowledge graph
- Advanced learning algorithms
- Web dashboard
- Multi-LLM support

---

## 📞 Support

- **Issues:** GitHub Issues
- **Documentation:** [Wiki](https://github.com/HotelAiOS/agent-zero-v1/wiki)
- **Email:** support@hotelaios.com

---

**Built with ❤️ by the Agent Zero Team**

Last Updated: October 5, 2025
