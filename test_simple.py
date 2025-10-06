import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'shared'))

print('Testing imports...')

try:
    from execution.project_orchestrator import ProjectOrchestrator
    print('OK: ProjectOrchestrator')
    
    from execution.agent_executor import AgentExecutor
    print('OK: AgentExecutor')
    
    from execution.code_generator import CodeGenerator
    print('OK: CodeGenerator')
    
    from llm.llm_factory import LLMFactory
    print('OK: LLMFactory')
    
    from agent_factory.factory import AgentFactory
    print('OK: AgentFactory')
    
    print('\nSUCCESS: All Phase 1 components working!')
    
except Exception as e:
    print(f'ERROR: {e}')
    import traceback
    traceback.print_exc()
