import ast
import logging
from typing import Dict, List, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class ASTParser:
    """Parser do analizy kodu Python i wydobywania dokumentacji"""
    
    def __init__(self):
        self.parsed_files = {}
    
    def parse_file(self, filepath: Path) -> Dict[str, Any]:
        """Parsuj plik Python i wydobądź strukturę"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source, filename=str(filepath))
            
            result = {
                "filepath": str(filepath),
                "classes": [],
                "functions": [],
                "imports": [],
                "module_docstring": ast.get_docstring(tree)
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    result["classes"].append(self._parse_class(node))
                elif isinstance(node, ast.FunctionDef):
                    if not self._is_in_class(node, tree):
                        result["functions"].append(self._parse_function(node))
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    result["imports"].append(self._parse_import(node))
            
            self.parsed_files[str(filepath)] = result
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return {}
    
    def _parse_class(self, node: ast.ClassDef) -> Dict[str, Any]:
        """Parsuj klasę"""
        return {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "bases": [self._get_name(base) for base in node.bases],
            "methods": [
                self._parse_function(item) 
                for item in node.body 
                if isinstance(item, ast.FunctionDef)
            ],
            "line": node.lineno
        }
    
    def _parse_function(self, node: ast.FunctionDef) -> Dict[str, Any]:
        """Parsuj funkcję"""
        return {
            "name": node.name,
            "docstring": ast.get_docstring(node),
            "args": [arg.arg for arg in node.args.args],
            "returns": self._get_name(node.returns) if node.returns else None,
            "decorators": [self._get_name(dec) for dec in node.decorator_list],
            "line": node.lineno
        }
    
    def _parse_import(self, node) -> Dict[str, Any]:
        """Parsuj import"""
        if isinstance(node, ast.Import):
            return {
                "type": "import",
                "names": [alias.name for alias in node.names]
            }
        else:  # ImportFrom
            return {
                "type": "from_import",
                "module": node.module,
                "names": [alias.name for alias in node.names]
            }
    
    def _get_name(self, node) -> str:
        """Wydobądź nazwę z node"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_name(node.value)}.{node.attr}"
        return str(node)
    
    def _is_in_class(self, node, tree) -> bool:
        """Sprawdź czy funkcja jest wewnątrz klasy"""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False
