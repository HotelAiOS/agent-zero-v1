"""
Task Decomposer - FIXED VERSION
Agent Zero V1 - Critical Fix TECH-001

Fixes:
- Robust JSON extraction from LLM responses
- Multiple parsing strategies with fallbacks
- Comprehensive error handling
- Retry logic with prompt refinement
"""

import json
import re
import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TaskDecomposition:
    """Decomposed task structure"""
    task_id: str
    subtasks: List[Dict[str, Any]]
    dependencies: List[Dict[str, str]]
    metadata: Dict[str, Any]

    @property
    def is_valid(self) -> bool:
        return len(self.subtasks) > 0


class RobustJSONParser:
    """Robust JSON parser for LLM responses"""

    @staticmethod
    def extract_json(text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON from text using multiple strategies"""
        strategies = [
            RobustJSONParser._parse_direct,
            RobustJSONParser._parse_code_block,
            RobustJSONParser._parse_with_regex,
            RobustJSONParser._parse_first_json_object,
            RobustJSONParser._parse_with_fixes
        ]

        for strategy in strategies:
            try:
                result = strategy(text)
                if result:
                    logger.info(f"✅ JSON extracted using {strategy.__name__}")
                    return result
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")
                continue

        logger.error("❌ All JSON extraction strategies failed")
        return None

    @staticmethod
    def _parse_direct(text: str) -> Optional[Dict]:
        """Try direct JSON parsing"""
        return json.loads(text.strip())

    @staticmethod
    def _parse_code_block(text: str) -> Optional[Dict]:
        """Extract JSON from markdown code blocks"""
        pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(pattern, text)

        if matches:
            for match in matches:
                try:
                    return json.loads(match.strip())
                except:
                    continue
        return None

    @staticmethod
    def _parse_with_regex(text: str) -> Optional[Dict]:
        """Find JSON-like structure with regex"""
        pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(pattern, text, re.DOTALL)

        if matches:
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        return None

    @staticmethod
    def _parse_first_json_object(text: str) -> Optional[Dict]:
        """Find and parse first JSON object in text"""
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        brace_count = 0
        for i, char in enumerate(text[start_idx:], start=start_idx):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(text[start_idx:i+1])
                    except:
                        return None
        return None

    @staticmethod
    def _parse_with_fixes(text: str) -> Optional[Dict]:
        """Apply common fixes and retry parsing"""
        fixes = [
            lambda s: s.replace("'", '"'),
            lambda s: re.sub(r',\s*}', '}', s),
            lambda s: re.sub(r',\s*]', ']', s),
            lambda s: s.replace('True', 'true').replace('False', 'false'),
        ]

        current = text
        for fix in fixes:
            try:
                current = fix(current)
                return json.loads(current)
            except:
                continue

        return None


class TaskDecomposer:
    """Enhanced Task Decomposer with robust LLM response handling"""

    def __init__(self, llm_client: Any, max_retries: int = 3):
        self.llm = llm_client
        self.max_retries = max_retries
        self.parser = RobustJSONParser()
        logger.info("TaskDecomposer initialized")

    async def decompose_task(
        self,
        task_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> TaskDecomposition:
        """Decompose high-level task into subtasks"""
        logger.info(f"Decomposing task: {task_description[:100]}...")

        for attempt in range(1, self.max_retries + 1):
            try:
                prompt = self._build_prompt(task_description, context, attempt)
                llm_response = await self._call_llm(prompt)

                parsed_json = self.parser.extract_json(llm_response)

                if parsed_json is None:
                    raise ValueError("Failed to extract valid JSON from LLM response")

                decomposition = self._validate_and_construct(parsed_json)

                if decomposition.is_valid:
                    logger.info(f"✅ Task decomposed into {len(decomposition.subtasks)} subtasks")
                    return decomposition
                else:
                    raise ValueError("Decomposition validation failed")

            except Exception as e:
                logger.warning(f"Decomposition attempt {attempt}/{self.max_retries} failed: {e}")

                if attempt >= self.max_retries:
                    logger.error("❌ Task decomposition failed after all retries")
                    raise ValueError(
                        f"Failed to decompose task after {self.max_retries} attempts: {str(e)}"
                    )

                await asyncio.sleep(1 * attempt)

        raise ValueError("Unexpected decomposition failure")

    def _build_prompt(
        self,
        task_description: str,
        context: Optional[Dict],
        attempt: int
    ) -> str:
        """Build LLM prompt"""

        base_prompt = f"""Decompose the following task into subtasks.

Task: {task_description}

Context: {json.dumps(context or {}, indent=2)}

Respond with valid JSON in this format:
{{
    "task_id": "unique_id",
    "subtasks": [
        {{
            "id": "subtask_1",
            "description": "Subtask description",
            "agent_type": "code_agent",
            "inputs": {{}},
            "estimated_effort": "medium"
        }}
    ],
    "dependencies": [
        {{"from": "subtask_1", "to": "subtask_2"}}
    ],
    "metadata": {{}}
}}
"""

        if attempt > 1:
            base_prompt += f"""
IMPORTANT (Attempt {attempt}):
- Respond ONLY with valid JSON
- No markdown code blocks
- Use double quotes, not single quotes
- No trailing commas
"""

        return base_prompt

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        if hasattr(self.llm, 'generate'):
            if asyncio.iscoroutinefunction(self.llm.generate):
                return await self.llm.generate(prompt)
            else:
                return self.llm.generate(prompt)
        else:
            raise NotImplementedError("LLM client must implement generate()")

    def _validate_and_construct(
        self,
        parsed_json: Dict[str, Any]
    ) -> TaskDecomposition:
        """Validate parsed JSON and construct TaskDecomposition"""

        required_fields = ["task_id", "subtasks"]
        for field in required_fields:
            if field not in parsed_json:
                raise ValueError(f"Missing required field: {field}")

        subtasks = parsed_json["subtasks"]
        if not isinstance(subtasks, list) or len(subtasks) == 0:
            raise ValueError("Subtasks must be a non-empty list")

        for i, subtask in enumerate(subtasks):
            if not isinstance(subtask, dict):
                raise ValueError(f"Subtask {i} must be a dictionary")
            if "id" not in subtask or "description" not in subtask:
                raise ValueError(f"Subtask {i} missing required fields")

        return TaskDecomposition(
            task_id=parsed_json["task_id"],
            subtasks=subtasks,
            dependencies=parsed_json.get("dependencies", []),
            metadata=parsed_json.get("metadata", {})
        )
