"""
Code Generator - Extracts, validates and writes code from LLM output.

Responsibilities:
- Generate code from LLM output
- Extract code blocks from markdown
- Validate syntax (Python, JS, etc.)
- Apply to file system
- Run code formatters (black, prettier)
"""

import re
import ast
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class Language(Enum):
    """Wspierane języki programowania."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    HTML = "html"
    CSS = "css"
    SQL = "sql"
    BASH = "bash"
    YAML = "yaml"
    JSON = "json"
    DOCKERFILE = "dockerfile"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


@dataclass
class CodeBlock:
    """Reprezentacja bloku kodu."""
    language: Language
    code: str
    line_start: int
    line_end: int
    filename: Optional[str] = None
    is_valid: bool = False
    validation_errors: List[str] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []


class CodeGenerator:
    """
    Generator kodu - wyciąga, waliduje i zapisuje kod z outputu LLM.
    
    Proces:
    1. Ekstraktuj bloki kodu z markdown
    2. Określ język programowania
    3. Waliduj składnię
    4. Wygeneruj nazwę pliku
    5. Zapisz do systemu plików
    6. (Opcjonalnie) Formatuj kod
    """
    
    def __init__(self, format_code: bool = False):
        """
        Args:
            format_code: Czy formatować kod (wymaga black, prettier, etc.)
        """
        self.format_code = format_code
        
        # Mapowanie rozszerzeń plików
        self.extensions = {
            Language.PYTHON: ".py",
            Language.JAVASCRIPT: ".js",
            Language.TYPESCRIPT: ".ts",
            Language.HTML: ".html",
            Language.CSS: ".css",
            Language.SQL: ".sql",
            Language.BASH: ".sh",
            Language.YAML: ".yaml",
            Language.JSON: ".json",
            Language.DOCKERFILE: "",
            Language.MARKDOWN: ".md",
            Language.UNKNOWN: ".txt"
        }
    
    def generate_code(
        self,
        llm_output: str,
        output_dir: Path,
        base_filename: str = "generated"
    ) -> List[str]:
        """
        Generuj pliki kodu z outputu LLM.
        
        Args:
            llm_output: Output z LLM zawierający kod
            output_dir: Katalog wyjściowy
            base_filename: Bazowa nazwa pliku
            
        Returns:
            Lista ścieżek do wygenerowanych plików
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ekstraktuj bloki kodu
        code_blocks = self.extract_code_blocks(llm_output)
        
        if not code_blocks:
            print("      ⚠️  No code blocks found in output")
            return []
        
        print(f"      📦 Found {len(code_blocks)} code blocks")
        
        generated_files = []
        
        for idx, block in enumerate(code_blocks, 1):
            # Waliduj kod
            self.validate_syntax(block)
            
            if block.validation_errors:
                print(f"      ⚠️  Block {idx} validation errors: {len(block.validation_errors)}")
                for error in block.validation_errors[:3]:  # Pokaż pierwsze 3
                    print(f"         - {error}")
            
            # Generuj nazwę pliku
            filename = self._generate_filename(
                base_filename,
                block.language,
                idx,
                output_dir
            )
            
            filepath = output_dir / filename
            
            # Zapisz do pliku
            self.write_to_file(block.code, filepath)
            
            # Formatuj jeśli włączone
            if self.format_code and block.is_valid:
                self.format_code_file(filepath, block.language)
            
            generated_files.append(str(filepath))
            
            print(f"      ✅ {filename} ({len(block.code)} chars)")
        
        return generated_files
    
    def extract_code_blocks(self, markdown: str) -> List[CodeBlock]:
        """
        Ekstraktuj bloki kodu z markdown.
        
        Args:
            markdown: Tekst zawierający markdown
            
        Returns:
            Lista wykrytych bloków kodu
        """
        blocks = []
        
        # Pattern dla markdown code blocks: ``````
        pattern = r'``````'
        
        for match in re.finditer(pattern, markdown, re.DOTALL):
            language_str = match.group(1) or "unknown"
            code = match.group(2).strip()
            
            if not code:
                continue
            
            # Określ język
            language = self._detect_language(language_str, code)
            
            # Policz linie
            start_pos = match.start()
            line_start = markdown[:start_pos].count('\n') + 1
            line_end = line_start + code.count('\n')
            
            blocks.append(CodeBlock(
                language=language,
                code=code,
                line_start=line_start,
                line_end=line_end
            ))
        
        return blocks
    
    def validate_syntax(self, code_block: CodeBlock) -> bool:
        """
        Waliduj składnię kodu.
        
        Args:
            code_block: Blok kodu do walidacji
            
        Returns:
            True jeśli kod jest poprawny
        """
        code_block.validation_errors = []
        
        if code_block.language == Language.PYTHON:
            try:
                ast.parse(code_block.code)
                code_block.is_valid = True
                return True
            except SyntaxError as e:
                code_block.validation_errors.append(f"Python syntax error: {e}")
                code_block.is_valid = False
                return False
        
        elif code_block.language == Language.JSON:
            try:
                json.loads(code_block.code)
                code_block.is_valid = True
                return True
            except json.JSONDecodeError as e:
                code_block.validation_errors.append(f"JSON parse error: {e}")
                code_block.is_valid = False
                return False
        
        elif code_block.language == Language.JAVASCRIPT:
            # Podstawowa walidacja JS - sprawdź czy nie ma oczywistych błędów
            if self._has_basic_js_errors(code_block.code):
                code_block.validation_errors.append("JavaScript syntax issues detected")
                code_block.is_valid = False
                return False
        
        # Dla innych języków - zakładamy poprawność
        code_block.is_valid = True
        return True
    
    def write_to_file(self, code: str, filepath: Path) -> None:
        """
        Zapisz kod do pliku.
        
        Args:
            code: Kod do zapisania
            filepath: Ścieżka do pliku
        """
        # Upewnij się że katalog istnieje
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Zapisz z kodowaniem UTF-8
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(code)
    
    def format_code_file(self, filepath: Path, language: Language) -> bool:
        """
        Formatuj plik kodu używając odpowiedniego formattera.
        
        Args:
            filepath: Ścieżka do pliku
            language: Język programowania
            
        Returns:
            True jeśli formatowanie się powiodło
        """
        try:
            if language == Language.PYTHON:
                # Użyj black do formatowania Python
                result = subprocess.run(
                    ['black', '--quiet', str(filepath)],
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
            
            elif language in [Language.JAVASCRIPT, Language.TYPESCRIPT]:
                # Użyj prettier do formatowania JS/TS
                result = subprocess.run(
                    ['prettier', '--write', str(filepath)],
                    capture_output=True,
                    timeout=10
                )
                return result.returncode == 0
            
            # Inne języki - bez formatowania
            return True
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # Formatter nie dostępny lub timeout
            return False
    
    def _detect_language(self, language_hint: str, code: str) -> Language:
        """
        Wykryj język programowania.
        
        Args:
            language_hint: Wskazówka z markdown (np. 'python')
            code: Kod do analizy
            
        Returns:
            Wykryty język
        """
        # Normalizuj hint
        hint_lower = language_hint.lower()
        
        # Mapowanie znanych nazw
        language_map = {
            "python": Language.PYTHON,
            "py": Language.PYTHON,
            "javascript": Language.JAVASCRIPT,
            "js": Language.JAVASCRIPT,
            "typescript": Language.TYPESCRIPT,
            "ts": Language.TYPESCRIPT,
            "html": Language.HTML,
            "css": Language.CSS,
            "sql": Language.SQL,
            "bash": Language.BASH,
            "sh": Language.BASH,
            "shell": Language.BASH,
            "yaml": Language.YAML,
            "yml": Language.YAML,
            "json": Language.JSON,
            "dockerfile": Language.DOCKERFILE,
            "markdown": Language.MARKDOWN,
            "md": Language.MARKDOWN
        }
        
        if hint_lower in language_map:
            return language_map[hint_lower]
        
        # Wykryj na podstawie zawartości
        if self._looks_like_python(code):
            return Language.PYTHON
        elif self._looks_like_javascript(code):
            return Language.JAVASCRIPT
        elif self._looks_like_html(code):
            return Language.HTML
        elif self._looks_like_json(code):
            return Language.JSON
        
        return Language.UNKNOWN
    
    def _generate_filename(
        self,
        base: str,
        language: Language,
        index: int,
        output_dir: Path
    ) -> str:
        """
        Generuj unikalną nazwę pliku.
        
        Args:
            base: Bazowa nazwa
            language: Język programowania
            index: Indeks bloku kodu
            output_dir: Katalog wyjściowy
            
        Returns:
            Nazwa pliku
        """
        ext = self.extensions[language]
        
        if language == Language.DOCKERFILE:
            return "Dockerfile"
        
        # Bazowa nazwa
        filename = f"{base}_{index}{ext}"
        
        # Sprawdź czy plik już istnieje
        counter = 1
        while (output_dir / filename).exists():
            filename = f"{base}_{index}_{counter}{ext}"
            counter += 1
        
        return filename
    
    def _looks_like_python(self, code: str) -> bool:
        """Sprawdź czy kod wygląda jak Python."""
        python_keywords = ['def ', 'class ', 'import ', 'from ', 'print(']
        return any(keyword in code for keyword in python_keywords)
    
    def _looks_like_javascript(self, code: str) -> bool:
        """Sprawdź czy kod wygląda jak JavaScript."""
        js_keywords = ['function ', 'const ', 'let ', 'var ', '=>', 'console.log']
        return any(keyword in code for keyword in js_keywords)
    
    def _looks_like_html(self, code: str) -> bool:
        """Sprawdź czy kod wygląda jak HTML."""
        return bool(re.search(r'<\w+[^>]*>.*</\w+>', code, re.DOTALL))
    
    def _looks_like_json(self, code: str) -> bool:
        """Sprawdź czy kod wygląda jak JSON."""
        try:
            json.loads(code)
            return True
        except:
            return False
    
    def _has_basic_js_errors(self, code: str) -> bool:
        """Wykryj podstawowe błędy składni JS."""
        # Sprawdź balans nawiasów
        brackets = {'(': ')', '[': ']', '{': '}'}
        stack = []
        
        for char in code:
            if char in brackets.keys():
                stack.append(char)
            elif char in brackets.values():
                if not stack or brackets[stack.pop()] != char:
                    return True
        
        return len(stack) > 0
