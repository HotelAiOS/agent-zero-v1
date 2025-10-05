"""
LLM Module
Ollama integration for Agent Zero
"""

from .ollama_client import OllamaClient, ModelConfig
from .prompt_builder import PromptBuilder
from .response_parser import ResponseParser

__all__ = [
    'OllamaClient',
    'ModelConfig',
    'PromptBuilder',
    'ResponseParser'
]

__version__ = '1.0.0'
