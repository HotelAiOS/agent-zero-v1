import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class MarkdownGenerator:
    """Generator dokumentacji Markdown z AST"""
    
    def generate_file_docs(self, parsed_data: Dict[str, Any]) -> str:
        """Generuj dokumentację Markdown dla pliku"""
        md = []
        
        # Nagłówek
        filepath = parsed_data.get("filepath", "Unknown")
        md.append(f"# {Path(filepath).stem}\n")
        
        # Module docstring
        if parsed_data.get("module_docstring"):
            md.append(f"{parsed_data['module_docstring']}\n")
        
        # Imports
        if parsed_data.get("imports"):
            md.append("## Imports\n")
            for imp in parsed_data["imports"]:
                if imp["type"] == "import":
                    md.append(f"- `{', '.join(imp['names'])}`")
                else:
                    md.append(f"- `from {imp['module']} import {', '.join(imp['names'])}`")
            md.append("")
        
        # Classes
        if parsed_data.get("classes"):
            md.append("## Classes\n")
            for cls in parsed_data["classes"]:
                md.append(self._generate_class_docs(cls))
        
        # Functions
        if parsed_data.get("functions"):
            md.append("## Functions\n")
            for func in parsed_data["functions"]:
                md.append(self._generate_function_docs(func))
        
        return "\n".join(md)
    
    def _generate_class_docs(self, cls: Dict[str, Any]) -> str:
        """Generuj dokumentację klasy"""
        md = []
        
        # Nagłówek klasy
        bases = f"({', '.join(cls['bases'])})" if cls['bases'] else ""
        md.append(f"### `{cls['name']}{bases}`\n")
        
        # Docstring
        if cls.get("docstring"):
            md.append(f"{cls['docstring']}\n")
        
        # Metody
        if cls.get("methods"):
            md.append("**Methods:**\n")
            for method in cls["methods"]:
                args = ", ".join(method["args"])
                returns = f" -> {method['returns']}" if method['returns'] else ""
                md.append(f"- `{method['name']}({args}){returns}`")
                if method.get("docstring"):
                    # Pierwsze linię docstringu
                    first_line = method["docstring"].split('\n')[0]
                    md.append(f"  - {first_line}")
            md.append("")
        
        return "\n".join(md)
    
    def _generate_function_docs(self, func: Dict[str, Any]) -> str:
        """Generuj dokumentację funkcji"""
        md = []
        
        # Signature
        args = ", ".join(func["args"])
        returns = f" -> {func['returns']}" if func['returns'] else ""
        md.append(f"### `{func['name']}({args}){returns}`\n")
        
        # Decorators
        if func.get("decorators"):
            md.append(f"**Decorators:** `{'`, `'.join(func['decorators'])}`\n")
        
        # Docstring
        if func.get("docstring"):
            md.append(f"{func['docstring']}\n")
        
        return "\n".join(md)
    
    def generate_module_index(self, parsed_files: Dict[str, Dict]) -> str:
        """Generuj index wszystkich modułów"""
        md = ["# Module Index\n"]
        
        for filepath, data in sorted(parsed_files.items()):
            module_name = Path(filepath).stem
            md.append(f"- [{module_name}]({module_name}.md)")
            if data.get("module_docstring"):
                first_line = data["module_docstring"].split('\n')[0]
                md.append(f"  - {first_line}")
        
        return "\n".join(md)
