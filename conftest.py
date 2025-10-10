"""
Pytest configuration for Agent Zero V1
Automatically configures Python path for microservices architecture
"""
import sys
from pathlib import Path

# Add shared modules to Python path
project_root = Path(__file__).parent.parent
shared_path = project_root / "shared"
orchestrator_path = project_root / "services" / "agent-orchestrator" / "src"

if shared_path.exists():
    sys.path.insert(0, str(shared_path))

if orchestrator_path.exists():
    sys.path.insert(0, str(orchestrator_path))

print(f"✓ Pytest configured for Agent Zero V1")
print(f"  - Shared path: {shared_path}")
print(f"  - Orchestrator path: {orchestrator_path}")
