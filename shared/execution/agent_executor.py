"""
Agent Executor - Executes individual tasks using agents and LLM.

Responsibilities:
- Execute single task by agent
- Stream LLM output in real-time
- Handle tool calls (code generation, bash commands)
- Validate output quality
- Save code artifacts to files
"""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, AsyncIterator
from dataclasses import dataclass

from agent_factory.factory import AgentInstance
from orchestration.task_decomposer import Task
from llm.llm_factory import LLMFactory


@dataclass
class ToolCall:
    """Reprezentacja wywołania narzędzia przez LLM."""
    tool_name: str
    parameters: Dict
    output: Optional[str] = None
    success: bool = False


class AgentExecutor:
    """
    Wykonuje pojedyncze zadania za pomocą agentów AI.

    Proces wykonania:
    1. Przygotuj prompt dla agenta z kontekstem zadania
    2. Streamuj odpowiedź LLM token po tokenie
    3. Wykryj i wykonaj tool calls (generowanie kodu, bash)
    4. Waliduj output
    5. Zapisz artefakty do plików
    """

    def __init__(self, llm_factory: LLMFactory):
        """
        Args:
            llm_factory: Factory do tworzenia klientów LLM
        """
        self.llm_factory = llm_factory
        self.tool_calls_executed: List[ToolCall] = []

    async def execute_task(
        self,
        agent: AgentInstance,
        task: Task,
        output_dir: Path
    ):
        """
        Wykonaj zadanie używając agenta.

        Args:
            agent: Instancja agenta AI
            task: Zadanie do wykonania
            output_dir: Katalog do zapisu wyników

        Returns:
            TaskResult z wynikami wykonania
        """
        from execution.project_orchestrator import TaskResult, TaskStatus

        start_time = datetime.now()

        print(f"    🤖 Agent {agent.agent_type} rozpoczyna: {task.name}")

        try:
            # Przygotuj prompt
            prompt = self._build_task_prompt(agent, task)

            # Utwórz katalog dla tego zadania
            task_dir = output_dir / self._sanitize_filename(task.name)
            task_dir.mkdir(exist_ok=True, parents=True)

            # Wykonaj zadanie z LLM
            full_output = ""
            artifacts = []

            print(f"    💭 Thinking...")

            async for chunk in self._stream_llm_response(agent, prompt):
                full_output += chunk

            print(f"    📝 Output received: {len(full_output)} chars")

            # Wykryj i wykonaj tool calls
            tool_calls = self._extract_tool_calls(full_output)

            for tool_call in tool_calls:
                print(f"    🔧 Tool call: {tool_call.tool_name}")
                await self._execute_tool_call(tool_call, task_dir)

                if tool_call.success and tool_call.output:
                    artifacts.append(tool_call.output)

            # Jeśli nie ma tool calls, wyciągnij kod z markdown
            if not tool_calls:
                print(f"    📄 Extracting code blocks from output...")
                code_artifacts = await self._extract_and_save_code(
                    full_output,
                    task_dir,
                    task
                )
                artifacts.extend(code_artifacts)

            # Walidacja outputu
            is_valid = self._validate_output(full_output, task)

            if not is_valid:
                print(f"    ⚠️  Output validation warning")

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            result = TaskResult(
                task_id=task.id,
                task_name=task.name,
                status=TaskStatus.COMPLETED,
                agent_id=agent.id,
                agent_type=agent.agent_type,
                output=full_output,
                artifacts=artifacts,
                duration_seconds=duration
            )

            # Zapisz output do pliku
            output_file = task_dir / "output.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_output)

            print(f"    ✅ Completed in {duration:.2f}s - {len(artifacts)} artifacts")

            return result

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            print(f"    ❌ Failed: {str(e)}")

            return TaskResult(
                task_id=task.id,
                task_name=task.name,
                status=TaskStatus.FAILED,
                agent_id=agent.id,
                agent_type=agent.agent_type,
                error=str(e),
                duration_seconds=duration
            )

    def _build_task_prompt(self, agent: AgentInstance, task: Task) -> str:
        """
        Zbuduj prompt dla agenta z kontekstem zadania.
        """
        prompt = f"""You are a {agent.agent_type} agent working on a software project.

YOUR ROLE:
{agent.capabilities}

TASK TO COMPLETE:
{task.name}

TASK DESCRIPTION:
{task.description}

REQUIREMENTS:
"""

        for req in task.required_capabilities:
            prompt += f"- {req}\n"

        prompt += f"""

ESTIMATED DURATION: {task.estimated_duration}

INSTRUCTIONS:
1. Analyze the task requirements carefully
2. Generate high-quality, production-ready code
3. Include comments explaining your implementation
4. Follow best practices for {agent.agent_type}
5. Wrap code in markdown code blocks with language specification

IMPORTANT:
- Use markdown code blocks: ```
- Be specific and detailed in your implementation
- Include error handling where appropriate
- Write clean, maintainable code

Now, complete this task:
"""

        return prompt

    async def _stream_llm_response(
        self,
        agent: AgentInstance,
        prompt: str
    ) -> AsyncIterator[str]:
        """
        Streamuj odpowiedź z LLM.

        UWAGA: Obecna implementacja używa synchronicznego chat(),
        w przyszłości można dodać prawdziwy streaming.
        """
        # Pobierz LLM client dla tego typu agenta
        llm_client = self.llm_factory.get_client(agent.agent_type)

        # Przygotuj messages
        messages = [
            {
                "role": "system",
                "content": f"You are an expert {agent.agent_type} AI assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Wywołaj LLM synchronicznie
        response = llm_client.chat(messages, agent.agent_type)

        # Yield całą odpowiedź
        yield response

    def _extract_tool_calls(self, llm_output: str) -> List[ToolCall]:
        """
        Wykryj tool calls w output LLM.

        Obecnie wspierane:
        - <code language="...">...</code>
        - <bash>...</bash>
        """
        tool_calls = []

        # Wykryj code blocks
        code_pattern = r'<code\s+language="(\w+)">(.*?)</code>'
        for match in re.finditer(code_pattern, llm_output, re.DOTALL):
            language = match.group(1)
            code = match.group(2).strip()

            tool_calls.append(ToolCall(
                tool_name="generate_code",
                parameters={
                    "language": language,
                    "code": code
                }
            ))

        # Wykryj bash commands
        bash_pattern = r'<bash>(.*?)</bash>'
        for match in re.finditer(bash_pattern, llm_output, re.DOTALL):
            command = match.group(1).strip()

            tool_calls.append(ToolCall(
                tool_name="execute_bash",
                parameters={
                    "command": command
                }
            ))

        return tool_calls

    async def _execute_tool_call(
        self,
        tool_call: ToolCall,
        output_dir: Path
    ):
        """Wykonaj tool call."""
        if tool_call.tool_name == "generate_code":
            # Generuj kod do pliku
            language = tool_call.parameters.get("language", "txt")
            code = tool_call.parameters.get("code", "")

            # Określ rozszerzenie pliku
            extensions = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
                "html": ".html",
                "css": ".css",
                "sql": ".sql",
                "bash": ".sh",
                "yaml": ".yaml",
                "json": ".json"
            }
            ext = extensions.get(language.lower(), ".txt")

            # Zapisz kod
            filename = f"generated_{language}_{len(list(output_dir.glob(f'*{ext}')))+1}{ext}"
            filepath = output_dir / filename

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)

            tool_call.output = str(filepath)
            tool_call.success = True

            print(f"      💾 Saved: {filename}")

        elif tool_call.tool_name == "execute_bash":
            # Wykonaj komendę bash (WYŁĄCZONE dla bezpieczeństwa)
            command = tool_call.parameters.get("command", "")

            print(f"      🔒 Bash execution disabled for security")

            tool_call.success = False

    async def _extract_and_save_code(
        self,
        llm_output: str,
        output_dir: Path,
        task: Task
    ) -> List[str]:
        """
        Wyciągnij bloki kodu z markdown i zapisz do plików.
        """
        artifacts = []

        # Wykryj markdown code blocks
        code_block_pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(code_block_pattern, llm_output, re.DOTALL)

        for idx, match in enumerate(matches, 1):
            language = match.group(1) or "txt"
            code = match.group(2).strip()

            if not code:
                continue

            # Określ rozszerzenie
            extensions = {
                "python": ".py",
                "javascript": ".js",
                "typescript": ".ts",
                "html": ".html",
                "css": ".css",
                "sql": ".sql",
                "bash": ".sh",
                "shell": ".sh",
                "yaml": ".yaml",
                "json": ".json",
                "dockerfile": ".Dockerfile",
                "txt": ".txt"
            }
            ext = extensions.get(language.lower(), ".txt")

            # Generuj nazwę pliku
            task_name_sanitized = self._sanitize_filename(task.name)
            filename = f"{task_name_sanitized}_{idx}{ext}"
            filepath = output_dir / filename

            # Zapisz kod
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(code)

            artifacts.append(str(filepath))

            print(f"      📄 Extracted: {filename} ({len(code)} chars)")

        return artifacts

    def _validate_output(self, output: str, task: Task) -> bool:
        """Waliduj output z LLM."""
        # Podstawowa walidacja
        if not output or len(output) < 50:
            return False

        # Sprawdź czy output zawiera kod
        has_code = '```' in output or '<code' in output

        # Sprawdź czy output odnosi się do zadania
        task_words = set(task.name.lower().split())
        output_words = set(output.lower().split())
        overlap = len(task_words & output_words)

        # Przynajmniej 30% słów z nazwy zadania powinno być w output
        relevance = overlap / len(task_words) if task_words else 0

        return has_code or relevance > 0.3

    def _sanitize_filename(self, name: str) -> str:
        """Sanityzuj nazwę dla użycia w nazwie pliku."""
        sanitized = re.sub(r'[^\w\s-]', '', name)
        sanitized = re.sub(r'[-\s]+', '_', sanitized)
        return sanitized.lower()[:50]
