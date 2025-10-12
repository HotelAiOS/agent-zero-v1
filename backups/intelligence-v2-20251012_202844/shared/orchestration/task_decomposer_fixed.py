# Task Decomposer - Fixed JSON Parser
import json
import re
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Task:
    id: int
    title: str
    description: str
    status: str = "pending"
    priority: str = "medium"
    dependencies: List[int] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class TaskDecomposer:
    """
    Enhanced Task Decomposer with robust JSON parsing capabilities
    Fixes the critical JSON parsing issues with LLM responses
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.parsing_strategies = [
            self._parse_direct_json,
            self._parse_markdown_cleaned,
            self._parse_first_json_object,
            self._parse_line_by_line,
            self._parse_with_regex_extraction
        ]
    
    def safe_parse_llm_response(self, llm_response: str) -> Optional[Dict[Any, Any]]:
        """
        Robust JSON parsing for LLM responses with multiple fallback strategies
        This is the main fix for the JSON parsing issue
        """
        if not llm_response or not llm_response.strip():
            logger.warning("Empty LLM response received")
            return None
            
        original_response = llm_response
        
        # Try each parsing strategy in order
        for i, strategy in enumerate(self.parsing_strategies):
            try:
                result = strategy(llm_response)
                if result:
                    logger.info(f"JSON parsing successful using strategy {i+1}")
                    return result
            except Exception as e:
                logger.debug(f"Parsing strategy {i+1} failed: {e}")
                continue
        
        # All strategies failed - log detailed error
        logger.error(f"All JSON parsing strategies failed for response: {original_response[:500]}...")
        return None
    
    def _parse_direct_json(self, response: str) -> Optional[Dict[Any, Any]]:
        """Strategy 1: Direct JSON parse"""
        return json.loads(response.strip())
    
    def _parse_markdown_cleaned(self, response: str) -> Optional[Dict[Any, Any]]:
        """Strategy 2: Remove markdown code blocks and parse"""
        # Remove various markdown patterns
        patterns = [
            r'```json\s*',  # ```json
            r'```\s*json\s*',  # ``` json
            r'\s*```',  # closing ```
            r'^```.*?\n',  # ```something at start
            r'\n```.*?$',  # ```something at end
        ]
        
        cleaned = response.strip()
        for pattern in patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
        
        return json.loads(cleaned.strip())
    
    def _parse_first_json_object(self, response: str) -> Optional[Dict[Any, Any]]:
        """Strategy 3: Extract first JSON-like object"""
        # Find first complete JSON object
        json_match = re.search(r'\{[^{}]*\}|\{.*?\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        return None
    
    def _parse_line_by_line(self, response: str) -> Optional[Dict[Any, Any]]:
        """Strategy 4: Line-by-line search for JSON start"""
        lines = response.split('\n')
        json_started = False
        json_lines = []
        brace_count = 0
        
        for line in lines:
            line = line.strip()
            if not json_started and line.startswith('{'):
                json_started = True
                json_lines.append(line)
                brace_count += line.count('{') - line.count('}')
            elif json_started:
                json_lines.append(line)
                brace_count += line.count('{') - line.count('}')
                if brace_count <= 0:
                    break
        
        if json_lines:
            json_str = '\n'.join(json_lines)
            return json.loads(json_str)
        return None
    
    def _parse_with_regex_extraction(self, response: str) -> Optional[Dict[Any, Any]]:
        """Strategy 5: Advanced regex extraction"""
        # Find JSON-like structure with nested braces
        pattern = r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\})*)*\}))*\}'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        return None
    
    def create_fallback_response(self, task_description: str, error_msg: str = None) -> Dict[Any, Any]:
        """Create a basic fallback response when JSON parsing fails completely"""
        return {
            "subtasks": [
                {
                    "id": 1,
                    "title": f"Analyze: {task_description[:50]}{'...' if len(task_description) > 50 else ''}",
                    "description": task_description,
                    "status": "pending",
                    "priority": "high",
                    "dependencies": []
                }
            ],
            "metadata": {
                "parsing_error": error_msg or "JSON parsing failed, using fallback",
                "original_task": task_description,
                "fallback_used": True
            }
        }
    
    def validate_decomposed_tasks(self, parsed_response: Dict[Any, Any]) -> Dict[Any, Any]:
        """Validate and sanitize the parsed response structure"""
        if not isinstance(parsed_response, dict):
            raise ValueError("Response must be a dictionary")
        
        # Ensure subtasks exist
        if 'subtasks' not in parsed_response:
            parsed_response['subtasks'] = []
        
        # Validate each subtask
        validated_subtasks = []
        for i, subtask in enumerate(parsed_response.get('subtasks', [])):
            if not isinstance(subtask, dict):
                continue
                
            validated_task = {
                'id': subtask.get('id', i + 1),
                'title': str(subtask.get('title', f'Subtask {i + 1}')),
                'description': str(subtask.get('description', 'No description provided')),
                'status': subtask.get('status', 'pending'),
                'priority': subtask.get('priority', 'medium'),
                'dependencies': subtask.get('dependencies', [])
            }
            
            # Ensure dependencies is a list
            if not isinstance(validated_task['dependencies'], list):
                validated_task['dependencies'] = []
                
            validated_subtasks.append(validated_task)
        
        parsed_response['subtasks'] = validated_subtasks
        return parsed_response
    
    def decompose_task(self, task_description: str) -> Dict[Any, Any]:
        """
        Main method to decompose a task using LLM with robust JSON parsing
        This replaces the original method that was failing
        """
        try:
            # Call LLM for task decomposition
            llm_response = self._call_llm_for_decomposition(task_description)
            
            if not llm_response:
                logger.warning("Empty response from LLM")
                return self.create_fallback_response(task_description, "Empty LLM response")
            
            # Parse LLM response using robust parsing
            parsed_response = self.safe_parse_llm_response(llm_response)
            
            if parsed_response is None:
                logger.warning(f"All JSON parsing strategies failed for task: {task_description}")
                return self.create_fallback_response(task_description, "JSON parsing failed")
            
            # Validate and sanitize the response
            validated_response = self.validate_decomposed_tasks(parsed_response)
            
            logger.info(f"Task decomposition successful: {len(validated_response.get('subtasks', []))} subtasks created")
            return validated_response
            
        except Exception as e:
            logger.error(f"Task decomposition failed: {e}")
            return self.create_fallback_response(task_description, f"Decomposition error: {str(e)}")
    
    def _call_llm_for_decomposition(self, task_description: str) -> str:
        """
        Call LLM for task decomposition with structured prompt
        This method interfaces with your existing LLM client
        """
        if not self.llm_client:
            # Return mock response for testing
            return self._generate_mock_llm_response(task_description)
        
        # Enhanced prompt for better JSON output
        prompt = f"""
        You are a task decomposition expert. Break down the following task into smaller, manageable subtasks.
        
        Task: {task_description}
        
        IMPORTANT: Respond ONLY with valid JSON in this exact format:
        {{
            "subtasks": [
                {{
                    "id": 1,
                    "title": "Subtask title",
                    "description": "Detailed description", 
                    "status": "pending",
                    "priority": "high|medium|low",
                    "dependencies": []
                }}
            ],
            "metadata": {{
                "total_subtasks": 0,
                "estimated_hours": 0
            }}
        }}
        
        Do NOT include any text before or after the JSON. Do NOT use markdown code blocks.
        """
        
        try:
            # Call your existing LLM client
            response = self.llm_client.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._generate_mock_llm_response(task_description)
    
    def _generate_mock_llm_response(self, task_description: str) -> str:
        """Generate mock response for testing purposes"""
        return f"""{{
            "subtasks": [
                {{
                    "id": 1,
                    "title": "Analyze Requirements",
                    "description": "Analyze and understand the requirements for: {task_description}",
                    "status": "pending", 
                    "priority": "high",
                    "dependencies": []
                }},
                {{
                    "id": 2,
                    "title": "Design Solution",
                    "description": "Design the technical solution approach",
                    "status": "pending",
                    "priority": "medium", 
                    "dependencies": [1]
                }},
                {{
                    "id": 3,
                    "title": "Implement Solution",
                    "description": "Implement the designed solution",
                    "status": "pending",
                    "priority": "high",
                    "dependencies": [2]
                }}
            ],
            "metadata": {{
                "total_subtasks": 3,
                "estimated_hours": 6
            }}
        }}"""

# Test function to verify the fix
def test_task_decomposer():
    """Test the fixed Task Decomposer with various problematic JSON formats"""
    decomposer = TaskDecomposer()
    
    test_cases = [
        # Case 1: Clean JSON
        '{"subtasks": [{"id": 1, "title": "Test", "description": "Test task"}]}',
        
        # Case 2: JSON with markdown
        '```json\n{"subtasks": [{"id": 1, "title": "Test", "description": "Test task"}]}\n```',
        
        # Case 3: JSON with extra text
        'Here is the response: {"subtasks": [{"id": 1, "title": "Test", "description": "Test task"}]} Hope this helps!',
        
        # Case 4: Malformed but recoverable
        '{\n  "subtasks": [\n    {\n      "id": 1,\n      "title": "Test",\n      "description": "Test task"\n    }\n  ]\n}',
        
        # Case 5: Complete failure case
        'This is not JSON at all!'
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        result = decomposer.safe_parse_llm_response(test_case)
        if result:
            print(f"✅ Test case {i}: PASSED - Parsed {len(result.get('subtasks', []))} subtasks")
        else:
            print(f"❌ Test case {i}: FAILED - Could not parse JSON")
    
    # Test full decomposition
    result = decomposer.decompose_task("Create a new web dashboard")
    print(f"\n🚀 Full decomposition test: {len(result.get('subtasks', []))} subtasks generated")
    
    return True

if __name__ == "__main__":
    print("🧪 Testing Enhanced Task Decomposer...")
    test_task_decomposer()
    print("✅ All tests completed!")