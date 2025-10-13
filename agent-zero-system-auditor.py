#!/usr/bin/env python3
"""
Agent Zero V1 - System Audit & Inventory
==========================================
Kompletny audit systemu przed dalszą rozbudową:
- Analiza wszystkich plików Python
- Wykrywanie dead code i duplikatów  
- Dependency analysis
- Test coverage analysis
- Architecture pattern detection
- Database schema verification
- Docker health check
"""

import os
import sys
import json
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class AgentZeroSystemAuditor:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.report = {
            'audit_timestamp': datetime.now().isoformat(),
            'system_overview': {},
            'code_quality': {},
            'architecture_analysis': {},
            'infrastructure_status': {},
            'recommendations': [],
            'critical_issues': []
        }
    
    def run_complete_audit(self) -> Dict[str, Any]:
        """Uruchamia pełny audit systemu"""
        print("🔍 Starting Agent Zero V1 System Audit...")
        
        # 1. System Overview
        self._analyze_system_structure()
        
        # 2. Code Quality Analysis  
        self._analyze_code_quality()
        
        # 3. Architecture Pattern Detection
        self._analyze_architecture_patterns()
        
        # 4. Infrastructure Status
        self._check_infrastructure_status()
        
        # 5. Generate Recommendations
        self._generate_recommendations()
        
        # 6. Save Report
        self._save_audit_report()
        
        return self.report
    
    def _analyze_system_structure(self):
        """Analizuje strukturę systemu"""
        print("📁 Analyzing system structure...")
        
        python_files = list(self.root_path.rglob("*.py"))
        config_files = list(self.root_path.rglob("*.yaml")) + list(self.root_path.rglob("*.yml")) + list(self.root_path.rglob("*.json"))
        docker_files = list(self.root_path.rglob("Dockerfile*")) + list(self.root_path.rglob("docker-compose*"))
        
        self.report['system_overview'] = {
            'total_python_files': len(python_files),
            'total_config_files': len(config_files),
            'total_docker_files': len(docker_files),
            'directory_structure': self._analyze_directory_structure(),
            'file_size_analysis': self._analyze_file_sizes(python_files)
        }
        
    def _analyze_directory_structure(self) -> Dict[str, int]:
        """Analizuje strukturę katalogów"""
        dirs = {}
        for item in self.root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                py_count = len(list(item.rglob("*.py")))
                dirs[item.name] = py_count
        return dirs
    
    def _analyze_file_sizes(self, python_files: List[Path]) -> Dict[str, Any]:
        """Analizuje rozmiary plików - identyfikuje potencjalne monolity"""
        sizes = []
        large_files = []
        
        for file in python_files:
            try:
                size = file.stat().st_size
                sizes.append(size)
                if size > 50000:  # Pliki >50KB mogą być problemem
                    large_files.append({
                        'file': str(file.relative_to(self.root_path)),
                        'size_kb': round(size / 1024, 2)
                    })
            except:
                continue
                
        return {
            'average_file_size_kb': round(sum(sizes) / len(sizes) / 1024, 2) if sizes else 0,
            'large_files_count': len(large_files),
            'large_files': large_files
        }
    
    def _analyze_code_quality(self):
        """Analizuje jakość kodu"""
        print("🧪 Analyzing code quality...")
        
        # Import analysis - wykrywa nieużywane importy
        unused_imports = self._find_unused_imports()
        
        # Function complexity analysis
        complex_functions = self._find_complex_functions()
        
        # Duplicate code detection
        duplicates = self._find_duplicate_code()
        
        # Test coverage analysis
        test_coverage = self._analyze_test_coverage()
        
        self.report['code_quality'] = {
            'unused_imports': unused_imports,
            'complex_functions': complex_functions,
            'duplicate_code': duplicates,
            'test_coverage': test_coverage,
            'code_smells': self._detect_code_smells()
        }
        
        # Dodaj critical issues jeśli znalezione
        if len(unused_imports) > 50:
            self.report['critical_issues'].append("Too many unused imports detected")
        if len(duplicates) > 20:
            self.report['critical_issues'].append("Significant code duplication found")
    
    def _find_unused_imports(self) -> List[str]:
        """Szuka nieużywanych importów"""
        unused = []
        try:
            # Używamy unimport jeśli dostępny
            result = subprocess.run(['unimport', '--check', str(self.root_path)], 
                                  capture_output=True, text=True)
            if result.stdout:
                unused = result.stdout.splitlines()[:50]  # Limit to first 50
        except FileNotFoundError:
            unused = ["unimport tool not available - install with: pip install unimport"]
        
        return unused
    
    def _find_complex_functions(self) -> List[Dict[str, Any]]:
        """Znajduje zbyt złożone funkcje"""
        complex_funcs = []
        
        for py_file in self.root_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Szuka funkcji z dużą liczbą linii
                lines = content.split('\n')
                in_function = False
                func_start = 0
                func_name = ""
                
                for i, line in enumerate(lines):
                    if re.match(r'^\s*def\s+(\w+)', line):
                        if in_function and i - func_start > 100:  # Funkcje >100 linii
                            complex_funcs.append({
                                'file': str(py_file.relative_to(self.root_path)),
                                'function': func_name,
                                'lines': i - func_start,
                                'start_line': func_start
                            })
                        
                        in_function = True
                        func_start = i
                        func_name = re.match(r'^\s*def\s+(\w+)', line).group(1)
                        
            except Exception:
                continue
                
        return complex_funcs[:20]  # Top 20 most complex
    
    def _find_duplicate_code(self) -> List[str]:
        """Szuka duplikatów kodu"""
        duplicates = []
        
        # Prosta analiza - szuka plików o podobnych nazwach
        py_files = list(self.root_path.rglob("*.py"))
        names = [f.name for f in py_files]
        
        seen = set()
        for name in names:
            base_name = name.replace('-', '_').replace('_fixed', '').replace('_v2', '').replace('_production', '')
            if base_name in seen:
                duplicates.append(f"Possible duplicate: {name}")
            seen.add(base_name)
            
        return duplicates
    
    def _analyze_test_coverage(self) -> Dict[str, Any]:
        """Analizuje pokrycie testami"""
        py_files = list(self.root_path.rglob("*.py"))
        test_files = [f for f in py_files if 'test' in f.name.lower()]
        
        # Szuka plików bez testów
        files_without_tests = []
        for py_file in py_files:
            if 'test' not in py_file.name.lower():
                # Szuka odpowiadającego pliku testowego
                possible_test_names = [
                    py_file.parent / f"test_{py_file.name}",
                    py_file.parent / f"{py_file.stem}_test.py",
                    self.root_path / "tests" / f"test_{py_file.name}"
                ]
                
                has_test = any(test_path.exists() for test_path in possible_test_names)
                if not has_test:
                    files_without_tests.append(str(py_file.relative_to(self.root_path)))
        
        return {
            'total_files': len(py_files),
            'test_files': len(test_files),
            'files_without_tests': len(files_without_tests),
            'estimated_coverage_percent': round((1 - len(files_without_tests) / len(py_files)) * 100, 2) if py_files else 0,
            'missing_tests_sample': files_without_tests[:10]
        }
    
    def _detect_code_smells(self) -> List[str]:
        """Wykrywa code smells"""
        smells = []
        
        # Szuka długich plików
        for py_file in self.root_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                    if lines > 1000:
                        smells.append(f"Very long file ({lines} lines): {py_file.relative_to(self.root_path)}")
            except:
                continue
        
        # Szuka plików z dużą liczbą importów
        for py_file in self.root_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    import_count = len(re.findall(r'^\s*(?:import|from)\s+', content, re.MULTILINE))
                    if import_count > 30:
                        smells.append(f"Too many imports ({import_count}): {py_file.relative_to(self.root_path)}")
            except:
                continue
                
        return smells[:20]  # Limit to first 20
    
    def _analyze_architecture_patterns(self):
        """Analizuje wzorce architektoniczne"""
        print("🏗️ Analyzing architecture patterns...")
        
        # Szuka wzorców FastAPI, Flask, Django
        api_frameworks = self._detect_api_frameworks()
        
        # Szuka wzorców baz danych
        database_patterns = self._detect_database_patterns()
        
        # Szuka wzorców Docker/Kubernetes
        containerization = self._detect_containerization_patterns()
        
        # Szuka wzorców AI/ML
        ai_patterns = self._detect_ai_patterns()
        
        self.report['architecture_analysis'] = {
            'api_frameworks': api_frameworks,
            'database_patterns': database_patterns,
            'containerization': containerization,
            'ai_ml_patterns': ai_patterns,
            'service_architecture': self._analyze_service_architecture()
        }
    
    def _detect_api_frameworks(self) -> Dict[str, List[str]]:
        """Wykrywa używane frameworki API"""
        frameworks = {'fastapi': [], 'flask': [], 'django': [], 'other': []}
        
        for py_file in self.root_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    if 'from fastapi' in content or 'import fastapi' in content:
                        frameworks['fastapi'].append(str(py_file.relative_to(self.root_path)))
                    elif 'from flask' in content or 'import flask' in content:
                        frameworks['flask'].append(str(py_file.relative_to(self.root_path)))
                    elif 'django' in content:
                        frameworks['django'].append(str(py_file.relative_to(self.root_path)))
            except:
                continue
                
        return frameworks
    
    def _detect_database_patterns(self) -> Dict[str, List[str]]:
        """Wykrywa wzorce baz danych"""
        db_patterns = {'neo4j': [], 'postgresql': [], 'sqlite': [], 'redis': [], 'mongodb': []}
        
        for py_file in self.root_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    if 'neo4j' in content or 'cypher' in content:
                        db_patterns['neo4j'].append(str(py_file.relative_to(self.root_path)))
                    if 'postgresql' in content or 'psycopg' in content:
                        db_patterns['postgresql'].append(str(py_file.relative_to(self.root_path)))
                    if 'sqlite' in content:
                        db_patterns['sqlite'].append(str(py_file.relative_to(self.root_path)))
                    if 'redis' in content:
                        db_patterns['redis'].append(str(py_file.relative_to(self.root_path)))
                    if 'mongodb' in content or 'pymongo' in content:
                        db_patterns['mongodb'].append(str(py_file.relative_to(self.root_path)))
            except:
                continue
                
        return db_patterns
    
    def _detect_containerization_patterns(self) -> Dict[str, int]:
        """Wykrywa wzorce konteneryzacji"""
        docker_files = len(list(self.root_path.rglob("Dockerfile*")))
        compose_files = len(list(self.root_path.rglob("docker-compose*")))
        k8s_files = len(list(self.root_path.rglob("*.yaml"))) + len(list(self.root_path.rglob("*.yml")))
        
        return {
            'dockerfile_count': docker_files,
            'docker_compose_count': compose_files,
            'kubernetes_manifests': k8s_files
        }
    
    def _detect_ai_patterns(self) -> Dict[str, List[str]]:
        """Wykrywa wzorce AI/ML"""
        ai_patterns = {'ollama': [], 'openai': [], 'transformers': [], 'langchain': [], 'ml_frameworks': []}
        
        for py_file in self.root_path.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    
                    if 'ollama' in content:
                        ai_patterns['ollama'].append(str(py_file.relative_to(self.root_path)))
                    if 'openai' in content:
                        ai_patterns['openai'].append(str(py_file.relative_to(self.root_path)))
                    if 'transformers' in content or 'huggingface' in content:
                        ai_patterns['transformers'].append(str(py_file.relative_to(self.root_path)))
                    if 'langchain' in content:
                        ai_patterns['langchain'].append(str(py_file.relative_to(self.root_path)))
                    if any(framework in content for framework in ['tensorflow', 'pytorch', 'scikit-learn', 'pandas']):
                        ai_patterns['ml_frameworks'].append(str(py_file.relative_to(self.root_path)))
            except:
                continue
                
        return ai_patterns
    
    def _analyze_service_architecture(self) -> Dict[str, Any]:
        """Analizuje architekturę serwisów"""
        # Szuka plików server/service
        server_files = []
        for py_file in self.root_path.rglob("*.py"):
            if any(keyword in py_file.name.lower() for keyword in ['server', 'service', 'api', 'gateway']):
                server_files.append(str(py_file.relative_to(self.root_path)))
        
        # Analiza portów w plikach konfiguracyjnych
        ports_used = self._find_ports_in_configs()
        
        return {
            'server_files_count': len(server_files),
            'server_files': server_files[:10],
            'ports_identified': ports_used,
            'microservices_pattern': len(server_files) > 3  # Heurystyka
        }
    
    def _find_ports_in_configs(self) -> List[int]:
        """Znajduje używane porty w konfiguracji"""
        ports = []
        
        # Szuka w plikach Docker Compose
        for compose_file in self.root_path.rglob("docker-compose*"):
            try:
                with open(compose_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Regex do znajdowania portów w formacie "8080:80"
                    port_matches = re.findall(r'(\d{4,5}):\d{4,5}', content)
                    ports.extend([int(p) for p in port_matches])
            except:
                continue
        
        return sorted(list(set(ports)))
    
    def _check_infrastructure_status(self):
        """Sprawdza status infrastruktury"""
        print("🔧 Checking infrastructure status...")
        
        # Docker status
        docker_status = self._check_docker_status()
        
        # Requirements analysis  
        requirements_analysis = self._analyze_requirements()
        
        # Config files analysis
        config_analysis = self._analyze_config_files()
        
        self.report['infrastructure_status'] = {
            'docker': docker_status,
            'requirements': requirements_analysis,
            'configuration': config_analysis
        }
    
    def _check_docker_status(self) -> Dict[str, Any]:
        """Sprawdza status Docker"""
        try:
            # Sprawdz czy Docker jest dostępny
            result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
            docker_available = result.returncode == 0
            
            if docker_available:
                # Sprawdź uruchomione kontenery
                containers = subprocess.run(['docker', 'ps'], capture_output=True, text=True)
                running_containers = len(containers.stdout.splitlines()) - 1 if containers.returncode == 0 else 0
                
                return {
                    'available': True,
                    'version': result.stdout.strip(),
                    'running_containers': running_containers
                }
            else:
                return {'available': False, 'error': result.stderr}
                
        except FileNotFoundError:
            return {'available': False, 'error': 'Docker not installed'}
    
    def _analyze_requirements(self) -> Dict[str, Any]:
        """Analizuje pliki requirements"""
        req_files = list(self.root_path.rglob("requirements*.txt")) + list(self.root_path.rglob("pyproject.toml"))
        
        dependencies = []
        conflicts = []
        
        for req_file in req_files:
            try:
                with open(req_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if req_file.suffix == '.txt':
                        deps = [line.strip() for line in content.splitlines() 
                               if line.strip() and not line.startswith('#')]
                        dependencies.extend(deps)
            except:
                continue
        
        # Szuka potencjalnych konfliktów wersji
        package_versions = {}
        for dep in dependencies:
            if '==' in dep:
                package, version = dep.split('==', 1)
                if package in package_versions and package_versions[package] != version:
                    conflicts.append(f"{package}: {package_versions[package]} vs {version}")
                package_versions[package] = version
        
        return {
            'requirements_files': [str(f.relative_to(self.root_path)) for f in req_files],
            'total_dependencies': len(set(dependencies)),
            'version_conflicts': conflicts,
            'sample_dependencies': dependencies[:10]
        }
    
    def _analyze_config_files(self) -> Dict[str, Any]:
        """Analizuje pliki konfiguracyjne"""
        config_files = (list(self.root_path.rglob("*.yaml")) + 
                       list(self.root_path.rglob("*.yml")) + 
                       list(self.root_path.rglob("*.json")) +
                       list(self.root_path.rglob(".env*")))
        
        # Szuka wrażliwych danych w konfiguracji
        sensitive_patterns = ['password', 'secret', 'key', 'token', 'api_key']
        sensitive_files = []
        
        for config_file in config_files:
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    if any(pattern in content for pattern in sensitive_patterns):
                        sensitive_files.append(str(config_file.relative_to(self.root_path)))
            except:
                continue
        
        return {
            'config_files_count': len(config_files),
            'config_files': [str(f.relative_to(self.root_path)) for f in config_files],
            'sensitive_config_files': sensitive_files
        }
    
    def _generate_recommendations(self):
        """Generuje rekomendacje na podstawie analizy"""
        print("💡 Generating recommendations...")
        
        recommendations = []
        
        # Code quality recommendations
        if len(self.report['code_quality']['unused_imports']) > 20:
            recommendations.append("HIGH: Remove unused imports to improve code quality")
        
        if len(self.report['code_quality']['complex_functions']) > 5:
            recommendations.append("MEDIUM: Refactor complex functions (>100 lines)")
        
        if self.report['code_quality']['test_coverage']['estimated_coverage_percent'] < 50:
            recommendations.append("HIGH: Improve test coverage (currently <50%)")
        
        # Architecture recommendations
        api_frameworks = self.report['architecture_analysis']['api_frameworks']
        if len(api_frameworks['fastapi']) > 0 and len(api_frameworks['flask']) > 0:
            recommendations.append("MEDIUM: Consider standardizing on single API framework")
        
        # Infrastructure recommendations  
        if not self.report['infrastructure_status']['docker']['available']:
            recommendations.append("CRITICAL: Install Docker for containerization")
        
        if len(self.report['infrastructure_status']['requirements']['version_conflicts']) > 0:
            recommendations.append("HIGH: Resolve dependency version conflicts")
        
        # Security recommendations
        sensitive_configs = self.report['infrastructure_status']['configuration']['sensitive_config_files']
        if len(sensitive_configs) > 0:
            recommendations.append("HIGH: Move sensitive data to environment variables")
        
        self.report['recommendations'] = recommendations
    
    def _save_audit_report(self):
        """Zapisuje raport audytu"""
        # Utwórz katalog stabilization jeśli nie istnieje
        stabilization_dir = self.root_path / "stabilization"
        stabilization_dir.mkdir(exist_ok=True)
        
        # Zapisz pełny raport JSON
        report_file = stabilization_dir / "system_audit_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False)
        
        # Zapisz readable summary
        self._save_readable_summary()
        
        print(f"✅ Audit complete! Reports saved to:")
        print(f"   - {report_file}")
        print(f"   - {stabilization_dir / 'audit_summary.md'}")
    
    def _save_readable_summary(self):
        """Zapisuje czytelne podsumowanie"""
        stabilization_dir = self.root_path / "stabilization"
        summary_file = stabilization_dir / "audit_summary.md"
        
        summary = f"""# Agent Zero V1 - System Audit Report

**Audit Date:** {self.report['audit_timestamp']}

## 📊 System Overview
- **Python Files:** {self.report['system_overview']['total_python_files']}
- **Config Files:** {self.report['system_overview']['total_config_files']}
- **Docker Files:** {self.report['system_overview']['total_docker_files']}
- **Large Files (>50KB):** {self.report['system_overview']['file_size_analysis']['large_files_count']}

## 🧪 Code Quality Analysis
- **Unused Imports:** {len(self.report['code_quality']['unused_imports'])}
- **Complex Functions:** {len(self.report['code_quality']['complex_functions'])}
- **Duplicate Code Issues:** {len(self.report['code_quality']['duplicate_code'])}
- **Test Coverage:** {self.report['code_quality']['test_coverage']['estimated_coverage_percent']}%

## 🏗️ Architecture Analysis
- **API Frameworks:** FastAPI: {len(self.report['architecture_analysis']['api_frameworks']['fastapi'])}, Flask: {len(self.report['architecture_analysis']['api_frameworks']['flask'])}
- **Database Patterns:** Neo4j: {len(self.report['architecture_analysis']['database_patterns']['neo4j'])}, SQLite: {len(self.report['architecture_analysis']['database_patterns']['sqlite'])}
- **Docker Files:** {self.report['architecture_analysis']['containerization']['dockerfile_count']}
- **Microservices Pattern:** {'Yes' if self.report['architecture_analysis']['service_architecture']['microservices_pattern'] else 'No'}

## 🚨 Critical Issues
{chr(10).join(f'- {issue}' for issue in self.report['critical_issues']) if self.report['critical_issues'] else '- None detected'}

## 💡 Top Recommendations
{chr(10).join(f'- {rec}' for rec in self.report['recommendations'][:5])}

## 📁 Directory Structure
{chr(10).join(f'- {name}: {count} Python files' for name, count in self.report['system_overview']['directory_structure'].items())}

---
*Generated by Agent Zero V1 System Auditor*
"""
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary)

def main():
    """Main function"""
    if len(sys.argv) > 1:
        root_path = sys.argv[1]
    else:
        root_path = os.getcwd()
    
    print(f"🎯 Starting audit of: {root_path}")
    
    auditor = AgentZeroSystemAuditor(root_path)
    report = auditor.run_complete_audit()
    
    print(f"\n🎉 Audit completed successfully!")
    print(f"📈 Found {report['system_overview']['total_python_files']} Python files")
    print(f"⚠️  {len(report['critical_issues'])} critical issues identified")
    print(f"💡 {len(report['recommendations'])} recommendations generated")
    
    return report

if __name__ == "__main__":
    main()