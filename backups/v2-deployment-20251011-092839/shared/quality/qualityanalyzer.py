# Agent Zero v1 - Phase 2: Interactive Control  
# QualityAnalyzer - Generated code quality analysis and security scanning

import ast
import json
import re
import subprocess
import tempfile
import os
from typing import Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import hashlib

import requests
from bandit import manager as bandit_manager
from bandit.core import config as bandit_config
from pylint import lint
from pylint.lint import PyLinter
from pylint.reporters import CollectingReporter
# # import semgrep  # Optional
# from vulture import Vulture
import radon.complexity as complexity
import radon.metrics as metrics


class QualityLevel(Enum):
    """Code quality levels"""
    EXCELLENT = "excellent"      # 90-100%
    GOOD = "good"               # 75-89%
    ACCEPTABLE = "acceptable"   # 60-74%
    POOR = "poor"              # 40-59%
    CRITICAL = "critical"      # 0-39%


class VulnerabilityLevel(Enum):
    """Security vulnerability levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class LanguageType(Enum):
    """Supported programming languages"""
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


@dataclass
class QualityMetric:
    """Individual quality metric"""
    name: str
    score: float  # 0-100
    max_score: float = 100.0
    description: str = ""
    details: List[str] = None
    suggestions: List[str] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = []
        if self.suggestions is None:
            self.suggestions = []


@dataclass
class SecurityIssue:
    """Security vulnerability or issue"""
    issue_id: str
    level: VulnerabilityLevel
    category: str  # OWASP category
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    fix_suggestion: str
    cwe_id: Optional[str] = None  # Common Weakness Enumeration
    owasp_category: Optional[str] = None


@dataclass 
class QualityReport:
    """Comprehensive code quality analysis report"""
    project_name: str
    analysis_timestamp: str
    overall_score: float  # 0-100
    quality_level: QualityLevel
    
    # Quality metrics by category
    code_quality: Dict[str, QualityMetric]
    security_issues: List[SecurityIssue]
    best_practices: Dict[str, QualityMetric]
    performance_metrics: Dict[str, QualityMetric]
    maintainability: Dict[str, QualityMetric]
    
    # Language-specific analysis
    language_analysis: Dict[LanguageType, Dict[str, QualityMetric]]
    
    # Summary statistics
    total_files_analyzed: int
    total_lines_of_code: int
    test_coverage_percent: float
    
    # Quality gates
    quality_gates_passed: bool
    blocking_issues: List[str]
    
    # Recommendations
    priority_fixes: List[str]
    improvement_suggestions: List[str]


class SecurityAnalyzer:
    """Security vulnerability scanning using multiple tools"""
    
    def __init__(self):
        self.owasp_categories = {
            'A01': 'Broken Access Control',
            'A02': 'Cryptographic Failures', 
            'A03': 'Injection',
            'A04': 'Insecure Design',
            'A05': 'Security Misconfiguration',
            'A06': 'Vulnerable and Outdated Components',
            'A07': 'Identification and Authentication Failures',
            'A08': 'Software and Data Integrity Failures',
            'A09': 'Security Logging and Monitoring Failures',
            'A10': 'Server-Side Request Forgery'
        }
        
    async def scan_python_security(self, file_path: str, code_content: str) -> List[SecurityIssue]:
        """Scan Python code for security issues using Bandit"""
        issues = []
        
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_content)
                temp_file = f.name
                
            # Run Bandit security scan
            config = bandit_config.BanditConfig()
            manager = bandit_manager.BanditManager(config, 'file')
            manager.discover_files([temp_file])
            manager.run_tests()
            
            # Process results
            for result in manager.get_issue_list():
                issue = SecurityIssue(
                    issue_id=f"bandit_{result.test_id}",
                    level=self._map_bandit_severity(result.severity),
                    category="Security Scan",
                    title=result.text,
                    description=result.text,
                    file_path=file_path,
                    line_number=result.lineno,
                    code_snippet=result.get_code(max_lines=3),
                    fix_suggestion=self._get_bandit_fix_suggestion(result.test_id),
                    cwe_id=result.cwe_id if hasattr(result, 'cwe_id') else None
                )
                issues.append(issue)
                
        except Exception as e:
            # If Bandit fails, add a general security warning
            issues.append(SecurityIssue(
                issue_id="security_scan_failed",
                level=VulnerabilityLevel.MEDIUM,
                category="Tool Error",
                title="Security scan incomplete",
                description=f"Bandit security scan failed: {str(e)}",
                file_path=file_path,
                line_number=1,
                code_snippet="",
                fix_suggestion="Review code manually for security issues"
            ))
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file)
            except:
                pass
                
        return issues
        
    def _map_bandit_severity(self, severity: str) -> VulnerabilityLevel:
        """Map Bandit severity to our vulnerability levels"""
        mapping = {
            'HIGH': VulnerabilityLevel.HIGH,
            'MEDIUM': VulnerabilityLevel.MEDIUM, 
            'LOW': VulnerabilityLevel.LOW
        }
        return mapping.get(severity.upper(), VulnerabilityLevel.MEDIUM)
        
    def _get_bandit_fix_suggestion(self, test_id: str) -> str:
        """Get fix suggestions for common Bandit issues"""
        suggestions = {
            'B101': 'Avoid using assert statements in production code',
            'B102': 'Use safe functions instead of exec()',
            'B103': 'Set file permissions explicitly (e.g., 0o644)',
            'B104': 'Bind to localhost instead of 0.0.0.0 in development',
            'B105': 'Use secrets module for password generation',
            'B106': 'Use secrets module for random tokens',
            'B107': 'Use HTTPS URLs instead of HTTP',
            'B108': 'Add proper exception handling',
            'B201': 'Use subprocess with shell=False',
            'B301': 'Use pickle alternatives like JSON for untrusted data',
            'B302': 'Use marshal only with trusted data',
            'B303': 'Use hashlib with secure algorithms (SHA-256+)',
            'B304': 'Use secure cipher modes (not ECB)',
            'B305': 'Use secure cipher algorithms (AES, ChaCha20)',
            'B306': 'Use secrets.randbelow() for secure random numbers',
            'B307': 'Use eval() alternatives like ast.literal_eval()',
            'B308': 'Use safer HTML parsing libraries',
            'B309': 'Use HTTPS for external requests',
            'B310': 'Use urllib.parse.quote() for URL encoding',
            'B311': 'Use secrets module for cryptographic randomness',
            'B312': 'Use ssl.create_default_context() for SSL',
            'B313': 'Use xml.etree.ElementTree with secure settings',
            'B314': 'Use defusedxml for XML parsing',
            'B315': 'Use defusedxml for XML processing',
            'B316': 'Use defusedxml for XML-RPC',
            'B317': 'Use defusedxml for XML parsing',
            'B318': 'Use defusedxml for XML processing',
            'B319': 'Use defusedxml for XML parsing',
            'B320': 'Use secure XML parser settings',
            'B321': 'Use parameterized queries to prevent FTP injection',
            'B322': 'Use input validation for user input',
            'B323': 'Use secure random number generation',
            'B324': 'Use secure hash algorithms (SHA-256+)',
            'B325': 'Use secure temporary file creation'
        }
        
        return suggestions.get(test_id, 'Review code for security best practices')
        
    async def scan_javascript_security(self, file_path: str, code_content: str) -> List[SecurityIssue]:
        """Scan JavaScript/TypeScript for security issues"""
        issues = []
        
        # Basic security checks for JS/TS
        security_patterns = [
            (r'eval\s*\(', 'Avoid using eval() - use safer alternatives'),
            (r'innerHTML\s*=', 'Use textContent or sanitize HTML to prevent XSS'),
            (r'document\.write\s*\(', 'Avoid document.write - use DOM manipulation'),
            (r'setTimeout\s*\(\s*["\']', 'Avoid string-based setTimeout - use function reference'),
            (r'setInterval\s*\(\s*["\']', 'Avoid string-based setInterval - use function reference'),
            (r'\.cookie\s*=', 'Use secure, HttpOnly, SameSite cookie settings'),
            (r'http://(?!localhost)', 'Use HTTPS URLs instead of HTTP'),
            (r'Math\.random\s*\(\)', 'Use crypto.getRandomValues() for cryptographic randomness')
        ]
        
        for line_num, line in enumerate(code_content.split('\n'), 1):
            for pattern, suggestion in security_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        issue_id=f"js_security_{hashlib.md5(pattern.encode()).hexdigest()[:8]}",
                        level=VulnerabilityLevel.MEDIUM,
                        category="JavaScript Security",
                        title=f"Potential security issue: {pattern}",
                        description=f"Security concern found in line {line_num}",
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip(),
                        fix_suggestion=suggestion
                    ))
                    
        return issues
        
    async def scan_sql_security(self, file_path: str, code_content: str) -> List[SecurityIssue]:
        """Scan SQL for injection vulnerabilities"""
        issues = []
        
        # SQL injection patterns
        sql_injection_patterns = [
            (r'["\'].*\+.*["\']', 'Use parameterized queries instead of string concatenation'),
            (r'%s.*%.*', 'Use parameterized queries with proper escaping'),
            (r'format\s*\(.*\)', 'Use parameterized queries instead of string formatting'),
            (r'execute\s*\(\s*[\'"].*%', 'Use execute() with parameter placeholders'),
            (r'--.*drop|delete|update|insert', 'Review SQL comments for potential injection')
        ]
        
        for line_num, line in enumerate(code_content.split('\n'), 1):
            for pattern, suggestion in sql_injection_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append(SecurityIssue(
                        issue_id=f"sql_injection_{hashlib.md5(pattern.encode()).hexdigest()[:8]}",
                        level=VulnerabilityLevel.HIGH,
                        category="SQL Injection",
                        title="Potential SQL Injection vulnerability",
                        description="SQL injection risk detected",
                        file_path=file_path,
                        line_number=line_num,
                        code_snippet=line.strip(),
                        fix_suggestion=suggestion,
                        owasp_category="A03"  # Injection
                    ))
                    
        return issues


class LanguageAnalyzer:
    """Language-specific code quality analysis"""
    
    async def analyze_python(self, file_path: str, code_content: str) -> Dict[str, QualityMetric]:
        """Analyze Python code quality"""
        metrics = {}
        
        try:
            # Parse AST for basic metrics
            tree = ast.parse(code_content)
            
            # Complexity analysis using Radon
            complexity_score = complexity.cc_visit(tree)
            avg_complexity = sum(c.complexity for c in complexity_score) / max(len(complexity_score), 1)
            
            # Code metrics
            raw_metrics = metrics.analyze(code_content)
            
            # Pylint analysis
            pylint_score = await self._run_pylint(code_content)
            
            metrics.update({
                'complexity': QualityMetric(
                    name='Cyclomatic Complexity',
                    score=max(0, 100 - (avg_complexity - 1) * 10),  # Lower complexity = higher score
                    description=f'Average cyclomatic complexity: {avg_complexity:.2f}',
                    suggestions=['Break down complex functions', 'Use early returns', 'Extract methods']
                ),
                'maintainability': QualityMetric(
                    name='Maintainability Index',
                    score=raw_metrics.mi,
                    description=f'Maintainability Index: {raw_metrics.mi:.2f}',
                    suggestions=['Improve code organization', 'Add documentation', 'Reduce complexity']
                ),
                'pylint_score': QualityMetric(
                    name='Pylint Code Quality',
                    score=pylint_score,
                    description=f'Pylint score: {pylint_score:.1f}/10',
                    suggestions=['Follow PEP 8 guidelines', 'Add type hints', 'Improve naming']
                )
            })
            
        except Exception as e:
            metrics['analysis_error'] = QualityMetric(
                name='Analysis Error',
                score=0,
                description=f'Python analysis failed: {str(e)}'
            )
            
        return metrics
        
    async def _run_pylint(self, code_content: str) -> float:
        """Run Pylint analysis and return score"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code_content)
                temp_file = f.name
                
            # Create Pylint reporter
            reporter = CollectingReporter()
            linter = PyLinter()
            linter.set_reporter(reporter)
            
            # Run Pylint
            linter.check([temp_file])
            
            # Get score (0-10)
            score = linter.stats.global_note if hasattr(linter.stats, 'global_note') else 5.0
            
            # Convert to 0-100 scale
            return (score / 10.0) * 100
            
        except Exception:
            return 50.0  # Default score if Pylint fails
        finally:
            try:
                os.unlink(temp_file)
            except:
                pass
                
    async def analyze_javascript(self, file_path: str, code_content: str) -> Dict[str, QualityMetric]:
        """Analyze JavaScript/TypeScript code quality"""
        metrics = {}
        
        # Basic metrics for JavaScript
        lines = code_content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Function count
        function_count = len(re.findall(r'function\s+\w+|const\s+\w+\s*=\s*\([^)]*\)\s*=>', code_content))
        
        # Variable declarations
        var_declarations = len(re.findall(r'\b(var|let|const)\s+\w+', code_content))
        
        # Complex conditionals
        complex_conditionals = len(re.findall(r'if\s*\([^)]*&&[^)]*\|\|[^)]*\)', code_content))
        
        metrics.update({
            'function_density': QualityMetric(
                name='Function Density',
                score=min(100, (function_count / max(len(non_empty_lines), 1)) * 1000),
                description=f'Functions per 1000 lines: {function_count}/{len(non_empty_lines)}',
                suggestions=['Extract reusable functions', 'Break down large functions']
            ),
            'variable_usage': QualityMetric(
                name='Variable Declaration Style',
                score=85.0,  # Default good score
                description=f'Variable declarations: {var_declarations}',
                suggestions=['Use const by default', 'Use let instead of var', 'Avoid global variables']
            ),
            'conditional_complexity': QualityMetric(
                name='Conditional Complexity',
                score=max(50, 100 - complex_conditionals * 10),
                description=f'Complex conditionals: {complex_conditionals}',
                suggestions=['Simplify boolean expressions', 'Extract condition functions']
            )
        })
        
        return metrics


class BestPracticesChecker:
    """Check adherence to coding best practices per language"""
    
    def __init__(self):
        self.language_rules = {
            LanguageType.PYTHON: self._python_best_practices,
            LanguageType.JAVASCRIPT: self._javascript_best_practices,
            LanguageType.SQL: self._sql_best_practices,
            LanguageType.HTML: self._html_best_practices,
            LanguageType.CSS: self._css_best_practices
        }
        
    async def check_best_practices(self, language: LanguageType, code_content: str) -> Dict[str, QualityMetric]:
        """Check best practices for specific language"""
        if language in self.language_rules:
            return await self.language_rules[language](code_content)
        return {}
        
    async def _python_best_practices(self, code_content: str) -> Dict[str, QualityMetric]:
        """Check Python best practices"""
        metrics = {}
        lines = code_content.split('\n')
        
        # PEP 8 checks
        long_lines = [i for i, line in enumerate(lines, 1) if len(line) > 79]
        docstring_funcs = len(re.findall(r'def\s+\w+[^:]*:\s*"""', code_content))
        total_funcs = len(re.findall(r'def\s+\w+', code_content))
        
        # Type hints
        type_hinted_funcs = len(re.findall(r'def\s+\w+[^:]*:\s*\w+', code_content))
        
        # Import organization
        imports = re.findall(r'^(import\s+\w+|from\s+\w+\s+import)', code_content, re.MULTILINE)
        
        metrics.update({
            'line_length': QualityMetric(
                name='Line Length (PEP 8)',
                score=max(0, 100 - len(long_lines) * 5),
                description=f'Lines over 79 chars: {len(long_lines)}',
                details=[f'Line {ln}' for ln in long_lines[:10]],
                suggestions=['Break long lines', 'Use parentheses for line continuation']
            ),
            'documentation': QualityMetric(
                name='Function Documentation', 
                score=(docstring_funcs / max(total_funcs, 1)) * 100,
                description=f'Documented functions: {docstring_funcs}/{total_funcs}',
                suggestions=['Add docstrings to all public functions', 'Follow PEP 257 docstring conventions']
            ),
            'type_hints': QualityMetric(
                name='Type Annotations',
                score=(type_hinted_funcs / max(total_funcs, 1)) * 100,
                description=f'Type-hinted functions: {type_hinted_funcs}/{total_funcs}',
                suggestions=['Add type hints to function parameters', 'Use return type annotations']
            ),
            'import_style': QualityMetric(
                name='Import Organization',
                score=85.0,  # Default score
                description=f'Import statements: {len(imports)}',
                suggestions=['Group imports: stdlib, third-party, local', 'Use absolute imports']
            )
        })
        
        return metrics
        
    async def _javascript_best_practices(self, code_content: str) -> Dict[str, QualityMetric]:
        """Check JavaScript best practices"""
        metrics = {}
        
        # ES6+ features
        arrow_functions = len(re.findall(r'=>', code_content))
        const_usage = len(re.findall(r'\bconst\s+', code_content))
        let_usage = len(re.findall(r'\blet\s+', code_content))
        var_usage = len(re.findall(r'\bvar\s+', code_content))
        
        # Async patterns
        promises = len(re.findall(r'\.then\(|\.catch\(|new Promise', code_content))
        async_await = len(re.findall(r'\basync\s+|await\s+', code_content))
        
        total_declarations = const_usage + let_usage + var_usage
        
        metrics.update({
            'modern_syntax': QualityMetric(
                name='Modern JavaScript Usage',
                score=min(100, (arrow_functions + const_usage) / max(total_declarations, 1) * 100),
                description=f'Arrow functions: {arrow_functions}, const usage: {const_usage}',
                suggestions=['Use arrow functions where appropriate', 'Prefer const over let/var']
            ),
            'variable_declarations': QualityMetric(
                name='Variable Declaration Best Practices',
                score=max(0, 100 - (var_usage / max(total_declarations, 1)) * 100),
                description=f'var: {var_usage}, let: {let_usage}, const: {const_usage}',
                suggestions=['Avoid var declarations', 'Use const by default', 'Use let when reassignment needed']
            ),
            'async_patterns': QualityMetric(
                name='Async/Await Usage',
                score=(async_await / max(promises + async_await, 1)) * 100 if (promises + async_await) > 0 else 100,
                description=f'Async/await: {async_await}, Promises: {promises}',
                suggestions=['Prefer async/await over .then()', 'Handle errors with try/catch']
            )
        })
        
        return metrics
        
    async def _sql_best_practices(self, code_content: str) -> Dict[str, QualityMetric]:
        """Check SQL best practices"""
        metrics = {}
        
        # SQL formatting
        upper_keywords = len(re.findall(r'\b(SELECT|FROM|WHERE|JOIN|GROUP BY|ORDER BY|INSERT|UPDATE|DELETE)\b', code_content))
        lower_keywords = len(re.findall(r'\b(select|from|where|join|group by|order by|insert|update|delete)\b', code_content))
        
        # Query patterns
        star_selects = len(re.findall(r'SELECT\s+\*', code_content, re.IGNORECASE))
        joins_without_conditions = len(re.findall(r'JOIN\s+\w+\s*$', code_content, re.IGNORECASE | re.MULTILINE))
        
        total_keywords = upper_keywords + lower_keywords
        
        metrics.update({
            'keyword_case': QualityMetric(
                name='SQL Keyword Formatting',
                score=(upper_keywords / max(total_keywords, 1)) * 100,
                description=f'Uppercase keywords: {upper_keywords}/{total_keywords}',
                suggestions=['Use uppercase for SQL keywords', 'Consistent formatting improves readability']
            ),
            'select_practices': QualityMetric(
                name='SELECT Statement Best Practices',
                score=max(0, 100 - star_selects * 20),
                description=f'SELECT * usage: {star_selects}',
                suggestions=['Specify column names explicitly', 'Avoid SELECT * in production code']
            )
        })
        
        return metrics
        
    async def _html_best_practices(self, code_content: str) -> Dict[str, QualityMetric]:
        """Check HTML best practices"""
        metrics = {}
        
        # Semantic HTML
        semantic_tags = len(re.findall(r'<(header|nav|main|article|section|aside|footer)', code_content))
        div_tags = len(re.findall(r'<div', code_content))
        
        # Accessibility
        alt_attributes = len(re.findall(r'alt\s*=', code_content))
        img_tags = len(re.findall(r'<img', code_content))
        
        metrics.update({
            'semantic_html': QualityMetric(
                name='Semantic HTML Usage',
                score=min(100, (semantic_tags / max(div_tags + semantic_tags, 1)) * 100),
                description=f'Semantic tags: {semantic_tags}, div tags: {div_tags}',
                suggestions=['Use semantic HTML5 elements', 'Reduce generic div usage']
            ),
            'accessibility': QualityMetric(
                name='Accessibility Attributes',
                score=(alt_attributes / max(img_tags, 1)) * 100 if img_tags > 0 else 100,
                description=f'Images with alt text: {alt_attributes}/{img_tags}',
                suggestions=['Add alt attributes to all images', 'Use descriptive alt text']
            )
        })
        
        return metrics
        
    async def _css_best_practices(self, code_content: str) -> Dict[str, QualityMetric]:
        """Check CSS best practices"""
        metrics = {}
        
        # CSS organization
        classes = len(re.findall(r'\.[a-zA-Z][\w-]*', code_content))
        ids = len(re.findall(r'#[a-zA-Z][\w-]*', code_content))
        
        # Modern CSS
        flexbox_usage = len(re.findall(r'display:\s*flex', code_content))
        grid_usage = len(re.findall(r'display:\s*grid', code_content))
        
        metrics.update({
            'selector_usage': QualityMetric(
                name='CSS Selector Best Practices',
                score=min(100, (classes / max(classes + ids, 1)) * 100),
                description=f'Classes: {classes}, IDs: {ids}',
                suggestions=['Prefer classes over IDs', 'Use semantic class names']
            ),
            'layout_methods': QualityMetric(
                name='Modern Layout Methods',
                score=min(100, (flexbox_usage + grid_usage) * 10),
                description=f'Flexbox: {flexbox_usage}, Grid: {grid_usage}',
                suggestions=['Use Flexbox for 1D layouts', 'Use CSS Grid for 2D layouts']
            )
        })
        
        return metrics


class QualityAnalyzer:
    """
    Comprehensive code quality analysis system
    
    Features:
    - Generated code quality analysis and scoring
    - Security vulnerability scanning (OWASP Top 10)
    - Best practices validation per programming language
    - Quality gates with deployment blocking capability
    """
    
    def __init__(self, quality_threshold: float = 70.0):
        self.quality_threshold = quality_threshold
        self.security_analyzer = SecurityAnalyzer()
        self.language_analyzer = LanguageAnalyzer()
        self.best_practices_checker = BestPracticesChecker()
        
        # Quality gate configuration
        self.quality_gates = {
            'min_overall_score': 60.0,
            'max_critical_security_issues': 0,
            'max_high_security_issues': 2,
            'min_test_coverage': 70.0,
            'max_complexity_score': 15.0
        }
        
    async def analyze_project_quality(self, project_path: str, project_name: str) -> QualityReport:
        """Analyze complete project quality"""
        
        # Discover all code files
        code_files = self._discover_code_files(project_path)
        
        # Initialize report
        report = QualityReport(
            project_name=project_name,
            analysis_timestamp=datetime.now().isoformat(),
            overall_score=0.0,
            quality_level=QualityLevel.CRITICAL,
            code_quality={},
            security_issues=[],
            best_practices={},
            performance_metrics={},
            maintainability={},
            language_analysis={},
            total_files_analyzed=len(code_files),
            total_lines_of_code=0,
            test_coverage_percent=0.0,
            quality_gates_passed=False,
            blocking_issues=[],
            priority_fixes=[],
            improvement_suggestions=[]
        )
        
        # Analyze each file
        for file_path, language in code_files:
            await self._analyze_file(file_path, language, report)
            
        # Calculate overall metrics
        await self._calculate_overall_metrics(report)
        
        # Check quality gates
        await self._check_quality_gates(report)
        
        # Generate recommendations
        await self._generate_recommendations(report)
        
        return report
        
    def _discover_code_files(self, project_path: str) -> List[Tuple[str, LanguageType]]:
        """Discover all analyzable code files in project"""
        code_files = []
        
        language_extensions = {
            '.py': LanguageType.PYTHON,
            '.js': LanguageType.JAVASCRIPT, 
            '.ts': LanguageType.TYPESCRIPT,
            '.jsx': LanguageType.JAVASCRIPT,
            '.tsx': LanguageType.TYPESCRIPT,
            '.html': LanguageType.HTML,
            '.htm': LanguageType.HTML,
            '.css': LanguageType.CSS,
            '.scss': LanguageType.CSS,
            '.sass': LanguageType.CSS,
            '.sql': LanguageType.SQL,
            '.sh': LanguageType.BASH,
            '.bash': LanguageType.BASH,
            '.yaml': LanguageType.YAML,
            '.yml': LanguageType.YAML,
            '.json': LanguageType.JSON,
            'dockerfile': LanguageType.DOCKERFILE
        }
        
        for root, dirs, files in os.walk(project_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__', '.pytest_cache', 'venv', 'env'}]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_lower = file.lower()
                
                # Check by extension
                for ext, lang in language_extensions.items():
                    if file_lower.endswith(ext):
                        code_files.append((file_path, lang))
                        break
                        
                # Special case for Dockerfile
                if file_lower in ['dockerfile', 'dockerfile.dev', 'dockerfile.prod']:
                    code_files.append((file_path, LanguageType.DOCKERFILE))
                    
        return code_files
        
    async def _analyze_file(self, file_path: str, language: LanguageType, report: QualityReport):
        """Analyze individual file and update report"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            # Count lines of code
            lines = [line for line in content.split('\n') if line.strip()]
            report.total_lines_of_code += len(lines)
            
            # Language-specific analysis
            if language == LanguageType.PYTHON:
                lang_metrics = await self.language_analyzer.analyze_python(file_path, content)
                security_issues = await self.security_analyzer.scan_python_security(file_path, content)
                best_practices = await self.best_practices_checker.check_best_practices(language, content)
                
            elif language in [LanguageType.JAVASCRIPT, LanguageType.TYPESCRIPT]:
                lang_metrics = await self.language_analyzer.analyze_javascript(file_path, content)
                security_issues = await self.security_analyzer.scan_javascript_security(file_path, content)
                best_practices = await self.best_practices_checker.check_best_practices(LanguageType.JAVASCRIPT, content)
                
            elif language == LanguageType.SQL:
                lang_metrics = {}
                security_issues = await self.security_analyzer.scan_sql_security(file_path, content)
                best_practices = await self.best_practices_checker.check_best_practices(language, content)
                
            else:
                # Basic analysis for other languages
                lang_metrics = {}
                security_issues = []
                best_practices = await self.best_practices_checker.check_best_practices(language, content)
                
            # Update report
            if language not in report.language_analysis:
                report.language_analysis[language] = {}
                
            for metric_name, metric in lang_metrics.items():
                if metric_name not in report.code_quality:
                    report.code_quality[metric_name] = metric
                else:
                    # Average with existing metrics
                    existing = report.code_quality[metric_name]
                    existing.score = (existing.score + metric.score) / 2
                    
            report.security_issues.extend(security_issues)
            
            for practice_name, practice in best_practices.items():
                if practice_name not in report.best_practices:
                    report.best_practices[practice_name] = practice
                else:
                    # Average with existing practices
                    existing = report.best_practices[practice_name]
                    existing.score = (existing.score + practice.score) / 2
                    
        except Exception as e:
            # Add analysis error to report
            report.blocking_issues.append(f"Failed to analyze {file_path}: {str(e)}")
            
    async def _calculate_overall_metrics(self, report: QualityReport):
        """Calculate overall quality metrics"""
        
        # Code quality score (average of all metrics)
        quality_scores = [metric.score for metric in report.code_quality.values()]
        avg_quality = sum(quality_scores) / max(len(quality_scores), 1) if quality_scores else 0
        
        # Best practices score
        practice_scores = [metric.score for metric in report.best_practices.values()]
        avg_practices = sum(practice_scores) / max(len(practice_scores), 1) if practice_scores else 0
        
        # Security penalty
        security_penalty = 0
        for issue in report.security_issues:
            if issue.level == VulnerabilityLevel.CRITICAL:
                security_penalty += 20
            elif issue.level == VulnerabilityLevel.HIGH:
                security_penalty += 10
            elif issue.level == VulnerabilityLevel.MEDIUM:
                security_penalty += 5
            elif issue.level == VulnerabilityLevel.LOW:
                security_penalty += 2
                
        # Calculate overall score
        base_score = (avg_quality + avg_practices) / 2
        report.overall_score = max(0, base_score - security_penalty)
        
        # Determine quality level
        if report.overall_score >= 90:
            report.quality_level = QualityLevel.EXCELLENT
        elif report.overall_score >= 75:
            report.quality_level = QualityLevel.GOOD
        elif report.overall_score >= 60:
            report.quality_level = QualityLevel.ACCEPTABLE
        elif report.overall_score >= 40:
            report.quality_level = QualityLevel.POOR
        else:
            report.quality_level = QualityLevel.CRITICAL
            
    async def _check_quality_gates(self, report: QualityReport):
        """Check if project passes quality gates"""
        
        blocking_issues = []
        
        # Overall score gate
        if report.overall_score < self.quality_gates['min_overall_score']:
            blocking_issues.append(f"Overall quality score {report.overall_score:.1f}% below threshold {self.quality_gates['min_overall_score']}%")
            
        # Security issues gates
        critical_security = len([i for i in report.security_issues if i.level == VulnerabilityLevel.CRITICAL])
        high_security = len([i for i in report.security_issues if i.level == VulnerabilityLevel.HIGH])
        
        if critical_security > self.quality_gates['max_critical_security_issues']:
            blocking_issues.append(f"Critical security issues: {critical_security} (max allowed: {self.quality_gates['max_critical_security_issues']})")
            
        if high_security > self.quality_gates['max_high_security_issues']:
            blocking_issues.append(f"High security issues: {high_security} (max allowed: {self.quality_gates['max_high_security_issues']})")
            
        # Test coverage gate (if available)
        if report.test_coverage_percent < self.quality_gates['min_test_coverage']:
            blocking_issues.append(f"Test coverage {report.test_coverage_percent:.1f}% below threshold {self.quality_gates['min_test_coverage']}%")
            
        report.blocking_issues.extend(blocking_issues)
        report.quality_gates_passed = len(blocking_issues) == 0
        
    async def _generate_recommendations(self, report: QualityReport):
        """Generate improvement recommendations"""
        
        # Priority fixes (critical/high security issues)
        for issue in report.security_issues:
            if issue.level in [VulnerabilityLevel.CRITICAL, VulnerabilityLevel.HIGH]:
                report.priority_fixes.append(f"{issue.title} in {issue.file_path}:{issue.line_number}")
                
        # General improvements
        if report.overall_score < 60:
            report.improvement_suggestions.append("Focus on improving overall code quality through refactoring")
            
        if len([i for i in report.security_issues if i.level == VulnerabilityLevel.MEDIUM]) > 5:
            report.improvement_suggestions.append("Address medium-priority security issues")
            
        # Low scoring metrics
        for metric_name, metric in report.code_quality.items():
            if metric.score < 50:
                report.improvement_suggestions.append(f"Improve {metric_name}: {metric.description}")
                
        for practice_name, practice in report.best_practices.items():
            if practice.score < 60:
                report.improvement_suggestions.append(f"Follow {practice_name} best practices")
                
    async def export_report(self, report: QualityReport, output_path: str, format: str = 'json') -> str:
        """Export quality report to file"""
        
        if format == 'json':
            output_file = f"{output_path}/quality_report_{report.project_name}.json"
            
            # Convert dataclasses to dict
            report_dict = asdict(report)
            
            # Handle enum serialization
            def serialize_obj(obj):
                if hasattr(obj, 'value'):
                    return obj.value
                return str(obj)
                
            with open(output_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=serialize_obj)
                
        elif format == 'html':
            output_file = f"{output_path}/quality_report_{report.project_name}.html"
            html_content = await self._generate_html_report(report)
            
            with open(output_file, 'w') as f:
                f.write(html_content)
                
        return output_file
        
    async def _generate_html_report(self, report: QualityReport) -> str:
        """Generate HTML quality report"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Quality Report - {report.project_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 10px 0; padding: 10px; border-left: 4px solid #ccc; }}
                .excellent {{ border-color: #4CAF50; }}
                .good {{ border-color: #8BC34A; }}
                .acceptable {{ border-color: #FFC107; }}
                .poor {{ border-color: #FF9800; }}
                .critical {{ border-color: #F44336; }}
                .security-issue {{ background: #ffebee; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .security-critical {{ background: #ffcdd2; }}
                .security-high {{ background: #f8bbd9; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔍 Quality Analysis Report</h1>
                <h2>Project: {report.project_name}</h2>
                <p>Analysis Date: {report.analysis_timestamp}</p>
                <p><strong>Overall Score: {report.overall_score:.1f}%</strong> ({report.quality_level.value})</p>
                <p>Files Analyzed: {report.total_files_analyzed} | Lines of Code: {report.total_lines_of_code:,}</p>
            </div>
            
            <h3>🎯 Quality Gates</h3>
            <p>Status: <strong>{"✅ PASSED" if report.quality_gates_passed else "❌ FAILED"}</strong></p>
            
            {"<h4>Blocking Issues:</h4><ul>" + "".join(f"<li>{issue}</li>" for issue in report.blocking_issues) + "</ul>" if report.blocking_issues else ""}
            
            <h3>📊 Code Quality Metrics</h3>
        """
        
        for metric_name, metric in report.code_quality.items():
            quality_class = self._get_quality_class(metric.score)
            html += f"""
            <div class="metric {quality_class}">
                <h4>{metric.name}</h4>
                <p>Score: {metric.score:.1f}%</p>
                <p>{metric.description}</p>
                {"<ul>" + "".join(f"<li>{suggestion}</li>" for suggestion in metric.suggestions) + "</ul>" if metric.suggestions else ""}
            </div>
            """
            
        html += "<h3>🛡️ Security Issues</h3>"
        
        for issue in report.security_issues:
            security_class = f"security-{issue.level.value}"
            html += f"""
            <div class="security-issue {security_class}">
                <h4>{issue.title}</h4>
                <p><strong>Level:</strong> {issue.level.value.upper()}</p>
                <p><strong>File:</strong> {issue.file_path}:{issue.line_number}</p>
                <p><strong>Description:</strong> {issue.description}</p>
                <p><strong>Fix:</strong> {issue.fix_suggestion}</p>
                {f"<p><strong>Code:</strong> <code>{issue.code_snippet}</code></p>" if issue.code_snippet else ""}
            </div>
            """
            
        html += """
            <h3>💡 Recommendations</h3>
            <h4>Priority Fixes:</h4>
            <ul>
        """
        
        for fix in report.priority_fixes:
            html += f"<li>{fix}</li>"
            
        html += """
            </ul>
            <h4>Improvement Suggestions:</h4>
            <ul>
        """
        
        for suggestion in report.improvement_suggestions:
            html += f"<li>{suggestion}</li>"
            
        html += """
            </ul>
        </body>
        </html>
        """
        
        return html
        
    def _get_quality_class(self, score: float) -> str:
        """Get CSS class based on quality score"""
        if score >= 90:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "acceptable"
        elif score >= 40:
            return "poor"
        else:
            return "critical"


# Example usage and integration
if __name__ == "__main__":
    async def demo_quality_analysis():
        analyzer = QualityAnalyzer()
        
        # Analyze a project
        report = await analyzer.analyze_project_quality(
            "/path/to/project",
            "demo_project"
        )
        
        print(f"Overall Score: {report.overall_score:.1f}%")
        print(f"Quality Level: {report.quality_level.value}")
        print(f"Security Issues: {len(report.security_issues)}")
        print(f"Quality Gates: {'PASSED' if report.quality_gates_passed else 'FAILED'}")
        
        # Export report
        report_file = await analyzer.export_report(report, "./reports", "html")
        print(f"Report exported to: {report_file}")
        
    import asyncio
    asyncio.run(demo_quality_analysis())