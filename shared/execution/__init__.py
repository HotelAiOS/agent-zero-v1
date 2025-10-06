"""
Execution Engine - Phase 1 Components

Components:
- ProjectOrchestrator: Entry point for project execution
- AgentExecutor: Task execution by individual agents
- CodeGenerator: Code generation and file writing
"""

from .project_orchestrator import (
    ProjectOrchestrator,
    ProjectResult,
    PhaseResult,
    TaskResult,
    PhaseStatus,
    TaskStatus
)

from .agent_executor import (
    AgentExecutor,
    ToolCall
)

from .code_generator import (
    CodeGenerator,
    CodeBlock,
    Language
)

__all__ = [
    'ProjectOrchestrator',
    'ProjectResult',
    'PhaseResult',
    'TaskResult',
    'PhaseStatus',
    'TaskStatus',
    'AgentExecutor',
    'ToolCall',
    'CodeGenerator',
    'CodeBlock',
    'Language'
]
