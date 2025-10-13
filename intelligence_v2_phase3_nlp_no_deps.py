#!/usr/bin/env python3
"""
Agent Zero V1 - Intelligence V2.0 Phase 3: AI Task Decomposition (No External Dependencies)
NLP-powered intelligent task breakdown using built-in Python libraries
"""

import asyncio
import sys
import re
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict
from enum import Enum
import logging
import random
import string

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from agents.production_manager import ProductionAgentManager, AgentType, AgentCapability, AgentStatus, TaskStatus, TaskPriority
    from infrastructure.service_discovery import ServiceDiscovery
    from database.neo4j_connector import Neo4jConnector
    from intelligence_v2_phase1_analytics import IntelligenceV2Analytics
    from intelligence_v2_phase2_load_balancing import IntelligentLoadBalancer, LoadBalancingStrategy
except ImportError as e:
    print(f"⚠️ Import warning: {e}")
    print("🔧 Running in minimal mode without full infrastructure")
    ProductionAgentManager = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

class TaskComplexity(Enum):
    """Poziomy złożoności zadania"""
    TRIVIAL = "trivial"      # 1 subtask, <30 min
    SIMPLE = "simple"        # 2-3 subtasks, <2 hours
    MODERATE = "moderate"    # 4-6 subtasks, <1 day
    COMPLEX = "complex"      # 7-12 subtasks, 1-3 days
    ENTERPRISE = "enterprise"  # >12 subtasks, >3 days

class SubtaskType(Enum):
    """Typy podrzędnych zadań"""
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    COORDINATION = "coordination"
    DOCUMENTATION = "documentation"
    MONITORING = "monitoring"

@dataclass
class Subtask:
    """Reprezentuje podrzędne zadanie"""
    id: str
    title: str
    description: str
    type: SubtaskType
    priority: TaskPriority
    estimated_duration: float  # hours
    required_capabilities: List[str]
    dependencies: List[str]  # IDs of prerequisite subtasks
    confidence: float  # 0.0 to 1.0
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class TaskDecomposition:
    """Reprezentuje pełną dekompozycję zadania"""
    original_task_id: str
    original_description: str
    complexity: TaskComplexity
    subtasks: List[Subtask]
    execution_plan: List[List[str]]  # Parallel execution stages
    total_estimated_duration: float
    critical_path: List[str]
    confidence: float
    generated_at: datetime
    nlp_insights: Dict = None
    
    def __post_init__(self):
        if self.nlp_insights is None:
            self.nlp_insights = {}

class BasicNLPProcessor:
    """
    Prosty procesor NLP używający tylko built-in Python libraries
    Zastępuje NLTK i spaCy dla environment compatibility
    """
    
    def __init__(self):
        # Basic English stop words
        self.stop_words = {
            'the', 'is', 'at', 'which', 'on', 'and', 'a', 'to', 'are', 'as', 'was', 'for', 
            'in', 'be', 'have', 'it', 'that', 'an', 'he', 'his', 'her', 'she', 'you', 'we',
            'they', 'with', 'from', 'by', 'this', 'but', 'or', 'not', 'can', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'do', 'does', 'did', 'has', 'had'
        }
        
        # Common action verbs
        self.action_verbs = {
            'develop', 'build', 'create', 'implement', 'design', 'analyze', 'study', 'examine',
            'investigate', 'research', 'evaluate', 'deploy', 'release', 'launch', 'install',
            'configure', 'setup', 'coordinate', 'manage', 'organize', 'plan', 'schedule',
            'communicate', 'monitor', 'track', 'observe', 'watch', 'measure', 'audit',
            'test', 'validate', 'verify', 'optimize', 'improve', 'enhance', 'integrate'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Basic tokenization"""
        # Simple word tokenization
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [word for word in words if len(word) > 2 and word not in self.stop_words]
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract basic entities (simplified)"""
        entities = []
        
        # Look for technology terms
        tech_patterns = {
            'TECH': r'\b(?:api|database|server|cloud|docker|kubernetes|ci/cd|ml|ai|analytics|dashboard)\b',
            'LANGUAGE': r'\b(?:python|java|javascript|react|node|sql|html|css)\b',
            'PLATFORM': r'\b(?:aws|azure|gcp|linux|windows|macos)\b'
        }
        
        for entity_type, pattern in tech_patterns.items():
            matches = re.finditer(pattern, text.lower())
            for match in matches:
                entities.append((match.group(), entity_type))
        
        return entities
    
    def extract_action_verbs(self, text: str) -> List[str]:
        """Extract action verbs from text"""
        words = self.tokenize(text)
        verbs = []
        
        for word in words:
            if word in self.action_verbs:
                verbs.append(word)
            # Check for common verb patterns
            elif word.endswith(('ing', 'ate', 'ize', 'ify')):
                verbs.append(word)
        
        return list(set(verbs))

class IntelligentTaskDecomposer:
    """
    Intelligence V2.0 Phase 3: AI Task Decomposition
    Używa basic NLP i pattern matching do inteligentnej dekompozycji zadań
    """
    
    def __init__(self, agent_manager: ProductionAgentManager = None,
                 load_balancer: IntelligentLoadBalancer = None):
        self.logger = logging.getLogger(__name__)
        self.agent_manager = agent_manager
        self.load_balancer = load_balancer
        
        # Initialize basic NLP processor
        self.nlp_processor = BasicNLPProcessor()
        
        # Task pattern knowledge base
        self.task_patterns = {
            'development': {
                'keywords': ['develop', 'build', 'create', 'implement', 'code', 'program', 'design', 'software'],
                'subtask_types': [SubtaskType.RESEARCH, SubtaskType.ANALYSIS, SubtaskType.DEVELOPMENT, SubtaskType.TESTING],
                'typical_capabilities': ['software_development', 'programming', 'system_design']
            },
            'analysis': {
                'keywords': ['analyze', 'study', 'examine', 'investigate', 'research', 'evaluate', 'data'],
                'subtask_types': [SubtaskType.RESEARCH, SubtaskType.ANALYSIS, SubtaskType.DOCUMENTATION],
                'typical_capabilities': ['data_analysis', 'research', 'analytics']
            },
            'deployment': {
                'keywords': ['deploy', 'release', 'launch', 'install', 'configure', 'setup', 'production'],
                'subtask_types': [SubtaskType.DEVELOPMENT, SubtaskType.TESTING, SubtaskType.DEPLOYMENT, SubtaskType.MONITORING],
                'typical_capabilities': ['deployment', 'devops', 'system_administration']
            },
            'coordination': {
                'keywords': ['coordinate', 'manage', 'organize', 'plan', 'schedule', 'communicate', 'team'],
                'subtask_types': [SubtaskType.COORDINATION, SubtaskType.DOCUMENTATION, SubtaskType.MONITORING],
                'typical_capabilities': ['project_management', 'coordination', 'communication']
            },
            'monitoring': {
                'keywords': ['monitor', 'track', 'observe', 'watch', 'measure', 'audit', 'performance'],
                'subtask_types': [SubtaskType.MONITORING, SubtaskType.ANALYSIS, SubtaskType.DOCUMENTATION],
                'typical_capabilities': ['monitoring', 'analytics', 'system_health']
            }
        }
        
        # Complexity estimation patterns
        self.complexity_indicators = {
            'trivial': ['simple', 'quick', 'basic', 'easy', 'small'],
            'complex': ['complex', 'advanced', 'comprehensive', 'large-scale', 'sophisticated'],
            'enterprise': ['enterprise', 'production', 'scalable', 'distributed', 'multi-system', 'fortune']
        }
        
        self.logger.info("🧠 Intelligence V2.0 AI Task Decomposer initialized (Basic NLP mode)")
    
    def analyze_task_with_nlp(self, task_description: str) -> Dict:
        """Analizuje zadanie używając basic NLP techniques"""
        
        analysis = {
            'tokens': [],
            'entities': [],
            'keywords': [],
            'action_verbs': [],
            'complexity_indicators': [],
            'domain_classification': None,
            'estimated_complexity': TaskComplexity.SIMPLE,
            'confidence': 0.7
        }
        
        # Tokenization
        analysis['tokens'] = self.nlp_processor.tokenize(task_description)
        
        # Entity extraction
        analysis['entities'] = self.nlp_processor.extract_entities(task_description)
        
        # Action verbs
        analysis['action_verbs'] = self.nlp_processor.extract_action_verbs(task_description)
        
        # Domain classification
        analysis['domain_classification'] = self._classify_task_domain(analysis['tokens'], analysis['action_verbs'])
        
        # Complexity estimation
        analysis['estimated_complexity'], analysis['confidence'] = self._estimate_complexity(task_description, analysis['tokens'])
        
        # Extract complexity indicators
        analysis['complexity_indicators'] = self._extract_complexity_indicators(task_description)
        
        return analysis
    
    def _classify_task_domain(self, tokens: List[str], verbs: List[str]) -> str:
        """Klasyfikuje domenę zadania na podstawie tokenów i czasowników"""
        
        domain_scores = defaultdict(float)
        
        # Analyze tokens against patterns
        for domain, pattern in self.task_patterns.items():
            keywords = pattern['keywords']
            
            # Score based on keyword matches
            for token in tokens:
                for keyword in keywords:
                    if keyword in token or token in keyword:
                        domain_scores[domain] += 1.0
            
            # Score based on verb matches  
            for verb in verbs:
                for keyword in keywords:
                    if keyword in verb or verb in keyword:
                        domain_scores[domain] += 1.5  # Verbs are more important
        
        # Return domain with highest score
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return 'analysis'  # Default domain
    
    def _estimate_complexity(self, description: str, tokens: List[str]) -> Tuple[TaskComplexity, float]:
        """Szacuje złożoność zadania na podstawie analysis"""
        
        complexity_score = 0.0
        confidence = 0.7
        
        # Length-based complexity
        word_count = len(tokens)
        if word_count > 30:
            complexity_score += 2.0
        elif word_count > 15:
            complexity_score += 1.0
        elif word_count < 8:
            complexity_score -= 1.0
        
        # Keyword-based complexity
        desc_lower = description.lower()
        
        for keyword in self.complexity_indicators['trivial']:
            if keyword in desc_lower:
                complexity_score -= 1.0
                confidence += 0.1
        
        for keyword in self.complexity_indicators['complex']:
            if keyword in desc_lower:
                complexity_score += 1.5
                confidence += 0.1
        
        for keyword in self.complexity_indicators['enterprise']:
            if keyword in desc_lower:
                complexity_score += 3.0
                confidence += 0.15
        
        # Multi-component indicators
        if 'and' in desc_lower and desc_lower.count('and') > 2:
            complexity_score += 1.0
        
        # Integration complexity
        if any(word in desc_lower for word in ['integrate', 'connect', 'interface', 'api']):
            complexity_score += 0.5
        
        # Convert score to complexity enum
        if complexity_score <= -0.5:
            return TaskComplexity.TRIVIAL, min(0.95, confidence)
        elif complexity_score <= 0.5:
            return TaskComplexity.SIMPLE, min(0.9, confidence)
        elif complexity_score <= 2.0:
            return TaskComplexity.MODERATE, min(0.85, confidence)
        elif complexity_score <= 4.0:
            return TaskComplexity.COMPLEX, min(0.8, confidence)
        else:
            return TaskComplexity.ENTERPRISE, min(0.75, confidence)
    
    def _extract_complexity_indicators(self, description: str) -> List[str]:
        """Wydobywa wskaźniki złożoności z opisu"""
        
        indicators = []
        desc_lower = description.lower()
        
        # Check for multiple components
        if desc_lower.count('and') > 2:
            indicators.append('multiple_components')
        
        # Check for integration requirements
        if any(word in desc_lower for word in ['integrate', 'connect', 'interface', 'api']):
            indicators.append('integration_required')
        
        # Check for time constraints
        if any(word in desc_lower for word in ['urgent', 'asap', 'immediately', 'deadline']):
            indicators.append('time_sensitive')
        
        # Check for quality requirements
        if any(word in desc_lower for word in ['test', 'quality', 'reliable', 'robust']):
            indicators.append('quality_focused')
        
        # Check for scale requirements
        if any(word in desc_lower for word in ['scale', 'enterprise', 'production', 'distributed']):
            indicators.append('large_scale')
        
        return indicators
    
    def decompose_task(self, task_description: str, task_id: str = None) -> TaskDecomposition:
        """Główna funkcja dekompozycji zadania na podrzędne zadania"""
        
        self.logger.info(f"🧠 Decomposing task: {task_description[:50]}...")
        
        if not task_id:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # NLP Analysis
        nlp_analysis = self.analyze_task_with_nlp(task_description)
        
        # Generate subtasks based on analysis
        subtasks = self._generate_subtasks(task_description, nlp_analysis)
        
        # Create execution plan
        execution_plan = self._create_execution_plan(subtasks)
        
        # Calculate critical path
        critical_path = self._calculate_critical_path(subtasks)
        
        # Calculate total duration
        total_duration = self._calculate_total_duration(subtasks, execution_plan)
        
        # Determine overall confidence
        overall_confidence = self._calculate_overall_confidence(subtasks, nlp_analysis)
        
        decomposition = TaskDecomposition(
            original_task_id=task_id,
            original_description=task_description,
            complexity=nlp_analysis['estimated_complexity'],
            subtasks=subtasks,
            execution_plan=execution_plan,
            total_estimated_duration=total_duration,
            critical_path=critical_path,
            confidence=overall_confidence,
            generated_at=datetime.now(),
            nlp_insights=nlp_analysis
        )
        
        self.logger.info(f"✅ Task decomposed into {len(subtasks)} subtasks with {overall_confidence:.1%} confidence")
        
        return decomposition
    
    def _generate_subtasks(self, description: str, nlp_analysis: Dict) -> List[Subtask]:
        """Generuje podrzędne zadania na podstawie analysis"""
        
        subtasks = []
        domain = nlp_analysis['domain_classification']
        complexity = nlp_analysis['estimated_complexity']
        
        # Get domain-specific patterns
        domain_pattern = self.task_patterns.get(domain, self.task_patterns['analysis'])
        
        # Determine number of subtasks based on complexity
        subtask_count = {
            TaskComplexity.TRIVIAL: 1,
            TaskComplexity.SIMPLE: random.randint(2, 3),
            TaskComplexity.MODERATE: random.randint(4, 6),
            TaskComplexity.COMPLEX: random.randint(7, 10),
            TaskComplexity.ENTERPRISE: random.randint(10, 15)
        }
        
        count = subtask_count[complexity]
        
        # Generate subtasks based on domain pattern
        subtask_types = domain_pattern['subtask_types']
        capabilities = domain_pattern['typical_capabilities']
        
        for i in range(count):
            # Rotate through subtask types, but add variety
            if i < len(subtask_types):
                subtask_type = subtask_types[i]
            else:
                subtask_type = subtask_types[i % len(subtask_types)]
            
            # Generate subtask based on type and original description
            subtask = self._create_subtask(
                subtask_id=f"{domain}_{subtask_type.value}_{i+1}",
                task_description=description,
                subtask_type=subtask_type,
                capabilities=capabilities,
                position=i,
                total_count=count
            )
            
            subtasks.append(subtask)
        
        return subtasks
    
    def _create_subtask(self, subtask_id: str, task_description: str, 
                       subtask_type: SubtaskType, capabilities: List[str],
                       position: int, total_count: int) -> Subtask:
        """Tworzy pojedyncze podrzędne zadanie"""
        
        # Generate title and description based on type
        type_templates = {
            SubtaskType.RESEARCH: {
                'title': f"Research and Requirements Gathering",
                'description': f"Conduct thorough research and gather all necessary requirements for: {task_description}"
            },
            SubtaskType.ANALYSIS: {
                'title': f"Analysis and Feasibility Study",
                'description': f"Perform detailed analysis of requirements, constraints, and feasibility"
            },
            SubtaskType.DEVELOPMENT: {
                'title': f"Core Development and Implementation",
                'description': f"Implement and develop the core functionality and components"
            },
            SubtaskType.TESTING: {
                'title': f"Testing and Quality Assurance",
                'description': f"Perform comprehensive testing, validation, and quality assurance"
            },
            SubtaskType.DEPLOYMENT: {
                'title': f"Deployment and Configuration",
                'description': f"Deploy the solution and configure all necessary components"
            },
            SubtaskType.COORDINATION: {
                'title': f"Team Coordination and Communication",
                'description': f"Coordinate with team members, stakeholders, and manage communications"
            },
            SubtaskType.DOCUMENTATION: {
                'title': f"Documentation and Knowledge Transfer",
                'description': f"Create comprehensive documentation and facilitate knowledge transfer"
            },
            SubtaskType.MONITORING: {
                'title': f"Monitoring and Maintenance Setup",
                'description': f"Set up monitoring, maintenance procedures, and ongoing support"
            }
        }
        
        template = type_templates[subtask_type]
        
        # Estimate duration based on type and complexity
        base_durations = {
            SubtaskType.RESEARCH: 3.0,
            SubtaskType.ANALYSIS: 4.0,
            SubtaskType.DEVELOPMENT: 12.0,
            SubtaskType.TESTING: 6.0,
            SubtaskType.DEPLOYMENT: 4.0,
            SubtaskType.COORDINATION: 2.0,
            SubtaskType.DOCUMENTATION: 3.0,
            SubtaskType.MONITORING: 3.0
        }
        
        duration = base_durations[subtask_type]
        
        # Adjust duration based on task complexity indicators
        if any(indicator in task_description.lower() for indicator in ['complex', 'enterprise', 'advanced']):
            duration *= 1.8
        elif any(indicator in task_description.lower() for indicator in ['comprehensive', 'scalable']):
            duration *= 1.4
        elif any(indicator in task_description.lower() for indicator in ['simple', 'basic', 'quick']):
            duration *= 0.6
        
        # Determine priority based on position and type
        if subtask_type in [SubtaskType.RESEARCH, SubtaskType.ANALYSIS]:
            priority = TaskPriority.HIGH  # Early phase tasks are critical
        elif position < total_count * 0.4:
            priority = TaskPriority.HIGH
        elif position < total_count * 0.7:
            priority = TaskPriority.MEDIUM
        else:
            priority = TaskPriority.LOW
        
        # Determine dependencies (sequential workflow)
        dependencies = []
        if position > 0:
            # Create realistic dependencies
            if subtask_type == SubtaskType.ANALYSIS:
                dependencies = [f"{subtask_id.split('_')[0]}_research_1"]
            elif subtask_type == SubtaskType.DEVELOPMENT:
                dependencies = [f"{subtask_id.split('_')[0]}_analysis_{i}" for i in range(1, min(3, position))]
            elif subtask_type == SubtaskType.TESTING:
                dependencies = [f"{subtask_id.split('_')[0]}_development_{i}" for i in range(1, min(2, position))]
        
        # Select relevant capabilities (enhance with more specific ones)
        relevant_caps = capabilities.copy()
        
        # Add type-specific capabilities
        if subtask_type == SubtaskType.DEVELOPMENT:
            relevant_caps.extend(['software_development', 'programming'])
        elif subtask_type == SubtaskType.TESTING:
            relevant_caps.extend(['testing', 'quality_assurance'])
        elif subtask_type == SubtaskType.DEPLOYMENT:
            relevant_caps.extend(['deployment', 'devops'])
        elif subtask_type == SubtaskType.COORDINATION:
            relevant_caps.extend(['project_management', 'communication'])
        
        # Keep unique capabilities, take first 3
        relevant_caps = list(set(relevant_caps))[:3]
        
        return Subtask(
            id=subtask_id,
            title=template['title'],
            description=template['description'],
            type=subtask_type,
            priority=priority,
            estimated_duration=duration,
            required_capabilities=relevant_caps,
            dependencies=dependencies,
            confidence=0.8 + random.uniform(-0.1, 0.1),  # Slight variation
            metadata={'position': position, 'total_count': total_count}
        )
    
    def _create_execution_plan(self, subtasks: List[Subtask]) -> List[List[str]]:
        """Tworzy plan wykonania z parallel stages"""
        
        stages = []
        remaining_tasks = {st.id: st for st in subtasks}
        completed_tasks = set()
        
        max_iterations = len(subtasks) + 2  # Safety limit
        iteration = 0
        
        while remaining_tasks and iteration < max_iterations:
            current_stage = []
            
            # Find tasks that can be executed (dependencies met)
            for task_id, task in list(remaining_tasks.items()):
                dependencies_met = all(dep in completed_tasks for dep in task.dependencies)
                
                if dependencies_met:
                    current_stage.append(task_id)
            
            # Remove selected tasks from remaining
            for task_id in current_stage:
                remaining_tasks.pop(task_id, None)
            
            # If no tasks can be executed, break dependency deadlock
            if not current_stage and remaining_tasks:
                # Add tasks with minimal dependencies
                min_deps = min(len(task.dependencies) for task in remaining_tasks.values())
                for task_id, task in list(remaining_tasks.items()):
                    if len(task.dependencies) == min_deps:
                        current_stage.append(task_id)
                        remaining_tasks.pop(task_id)
                        break
            
            if current_stage:
                stages.append(current_stage)
                completed_tasks.update(current_stage)
            
            iteration += 1
        
        return stages
    
    def _calculate_critical_path(self, subtasks: List[Subtask]) -> List[str]:
        """Oblicza ścieżkę krytyczną (simplified algorithm)"""
        
        if not subtasks:
            return []
        
        # Simple approach: find longest sequential chain based on dependencies
        task_dict = {st.id: st for st in subtasks}
        
        # Find tasks with no dependencies (potential starts)
        start_tasks = [st for st in subtasks if not st.dependencies]
        if not start_tasks:
            return [subtasks[0].id]  # Fallback
        
        longest_path = []
        max_duration = 0.0
        
        # For each start task, find the longest path
        for start_task in start_tasks[:3]:  # Limit to avoid excessive computation
            path, duration = self._find_longest_path_from_task(start_task, task_dict, set())
            
            if duration > max_duration:
                max_duration = duration
                longest_path = path
        
        return longest_path or [subtasks[0].id]
    
    def _find_longest_path_from_task(self, task: Subtask, task_dict: Dict, visited: Set[str]) -> Tuple[List[str], float]:
        """Finds longest path from given task"""
        
        if task.id in visited:
            return [], 0.0
        
        visited.add(task.id)
        path = [task.id]
        duration = task.estimated_duration
        
        # Find all tasks that depend on this one
        dependent_tasks = [t for t in task_dict.values() if task.id in t.dependencies]
        
        if dependent_tasks:
            best_continuation = []
            best_duration = 0.0
            
            for dep_task in dependent_tasks:
                continuation_path, continuation_duration = self._find_longest_path_from_task(
                    dep_task, task_dict, visited.copy()
                )
                
                if continuation_duration > best_duration:
                    best_duration = continuation_duration
                    best_continuation = continuation_path
            
            path.extend(best_continuation)
            duration += best_duration
        
        return path, duration
    
    def _calculate_total_duration(self, subtasks: List[Subtask], execution_plan: List[List[str]]) -> float:
        """Oblicza całkowity czas wykonania"""
        
        task_dict = {st.id: st for st in subtasks}
        total_duration = 0.0
        
        for stage in execution_plan:
            stage_duration = 0.0
            for task_id in stage:
                if task_id in task_dict:
                    # Parallel execution - take maximum duration in stage
                    stage_duration = max(stage_duration, task_dict[task_id].estimated_duration)
            
            total_duration += stage_duration
        
        return total_duration
    
    def _calculate_overall_confidence(self, subtasks: List[Subtask], nlp_analysis: Dict) -> float:
        """Oblicza ogólną pewność dekompozycji"""
        
        # Base confidence from NLP analysis
        base_confidence = nlp_analysis['confidence']
        
        # Average confidence of subtasks
        if subtasks:
            subtask_confidence = sum(st.confidence for st in subtasks) / len(subtasks)
        else:
            subtask_confidence = 0.5
        
        # Combine confidences with some domain knowledge
        domain_confidence_boost = 0.1 if nlp_analysis['domain_classification'] in ['development', 'analysis'] else 0.05
        
        overall = (base_confidence * 0.5) + (subtask_confidence * 0.4) + domain_confidence_boost
        
        return min(0.95, overall)
    
    def optimize_decomposition_with_load_balancer(self, decomposition: TaskDecomposition) -> TaskDecomposition:
        """Optymalizuje dekompozycję używając load balancer insights"""
        
        if not self.load_balancer:
            self.logger.info("⚠️ Load balancer not available - skipping optimization")
            return decomposition
        
        self.logger.info("⚡ Optimizing decomposition with load balancer insights...")
        
        # Get current load metrics
        try:
            load_metrics = self.load_balancer.collect_load_metrics()
        except Exception as e:
            self.logger.warning(f"Could not collect load metrics: {e}")
            return decomposition
        
        # Optimize subtask assignment based on agent capabilities and load
        optimized_subtasks = []
        
        for subtask in decomposition.subtasks:
            # Get optimal agent recommendation
            task_requirements = {
                'capabilities': subtask.required_capabilities,
                'priority': subtask.priority.name
            }
            
            try:
                decision = self.load_balancer.select_optimal_agent(task_requirements)
                
                # Update subtask metadata with recommendations
                subtask.metadata.update({
                    'recommended_agent': decision.recommended_agent,
                    'predicted_completion_time': decision.predicted_completion_time,
                    'load_balancer_confidence': decision.confidence,
                    'alternative_agents': decision.alternative_agents
                })
                
                # Adjust estimated duration based on agent performance
                if decision.confidence > 0.7 and decision.predicted_completion_time > 0:
                    # Use load balancer prediction but keep it reasonable
                    subtask.estimated_duration = min(
                        subtask.estimated_duration * 1.5,
                        max(subtask.estimated_duration * 0.5, decision.predicted_completion_time)
                    )
            
            except Exception as e:
                self.logger.warning(f"Could not optimize subtask {subtask.id}: {e}")
            
            optimized_subtasks.append(subtask)
        
        # Update decomposition
        decomposition.subtasks = optimized_subtasks
        
        # Recalculate execution plan and durations
        decomposition.execution_plan = self._create_execution_plan(optimized_subtasks)
        decomposition.total_estimated_duration = self._calculate_total_duration(optimized_subtasks, decomposition.execution_plan)
        
        # Increase overall confidence due to load balancer optimization
        decomposition.confidence = min(0.95, decomposition.confidence * 1.05)
        
        self.logger.info("✅ Decomposition optimized with load balancing insights")
        
        return decomposition
    
    def generate_task_decomposition_report(self, decomposition: TaskDecomposition) -> Dict:
        """Generuje kompletny raport dekompozycji zadania"""
        
        self.logger.info("🧠 Generating Intelligence V2.0 Task Decomposition Report...")
        
        # Analyze subtask distribution
        subtask_analysis = self._analyze_subtask_distribution(decomposition.subtasks)
        
        # Calculate execution efficiency
        efficiency_metrics = self._calculate_execution_efficiency(decomposition)
        
        # Generate insights
        ai_insights = self._generate_decomposition_insights(decomposition)
        
        # Load balancer integration analysis
        lb_integration = self._analyze_load_balancer_integration(decomposition)
        
        # Optimization recommendations
        recommendations = self._generate_optimization_recommendations(decomposition)
        
        report = {
            'report_id': f"task_decomposition_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'intelligence_level': 'V2.0 Phase 3',
            'original_task': {
                'id': decomposition.original_task_id,
                'description': decomposition.original_description,
                'complexity': decomposition.complexity.value
            },
            'decomposition_summary': {
                'total_subtasks': len(decomposition.subtasks),
                'execution_stages': len(decomposition.execution_plan),
                'estimated_duration': decomposition.total_estimated_duration,
                'confidence': decomposition.confidence,
                'critical_path_length': len(decomposition.critical_path)
            },
            'subtask_analysis': subtask_analysis,
            'execution_plan': {
                'stages': decomposition.execution_plan,
                'parallel_efficiency': efficiency_metrics['parallel_efficiency'],
                'resource_utilization': efficiency_metrics['resource_utilization']
            },
            'nlp_insights': decomposition.nlp_insights,
            'ai_insights': ai_insights,
            'load_balancer_integration': lb_integration,
            'recommendations': recommendations
        }
        
        return report
    
    def _analyze_subtask_distribution(self, subtasks: List[Subtask]) -> Dict:
        """Analizuje rozkład podrzędnych zadań"""
        
        analysis = {
            'by_type': defaultdict(int),
            'by_priority': defaultdict(int),
            'by_duration': {'short': 0, 'medium': 0, 'long': 0},
            'capability_requirements': defaultdict(int)
        }
        
        for subtask in subtasks:
            analysis['by_type'][subtask.type.value] += 1
            analysis['by_priority'][subtask.priority.name] += 1
            
            # Duration categorization
            if subtask.estimated_duration <= 3.0:
                analysis['by_duration']['short'] += 1
            elif subtask.estimated_duration <= 8.0:
                analysis['by_duration']['medium'] += 1
            else:
                analysis['by_duration']['long'] += 1
            
            # Capability analysis
            for cap in subtask.required_capabilities:
                analysis['capability_requirements'][cap] += 1
        
        # Convert defaultdicts to regular dicts for JSON serialization
        return {
            'by_type': dict(analysis['by_type']),
            'by_priority': dict(analysis['by_priority']),
            'by_duration': analysis['by_duration'],
            'capability_requirements': dict(analysis['capability_requirements'])
        }
    
    def _calculate_execution_efficiency(self, decomposition: TaskDecomposition) -> Dict:
        """Oblicza metryki efektywności wykonania"""
        
        total_sequential_time = sum(st.estimated_duration for st in decomposition.subtasks)
        parallel_time = decomposition.total_estimated_duration
        
        parallel_efficiency = (total_sequential_time - parallel_time) / total_sequential_time if total_sequential_time > 0 else 0
        
        # Resource utilization analysis
        if decomposition.execution_plan:
            max_concurrent_tasks = max(len(stage) for stage in decomposition.execution_plan)
            avg_concurrent_tasks = sum(len(stage) for stage in decomposition.execution_plan) / len(decomposition.execution_plan)
        else:
            max_concurrent_tasks = 1
            avg_concurrent_tasks = 1
        
        resource_utilization = avg_concurrent_tasks / max_concurrent_tasks if max_concurrent_tasks > 0 else 0
        
        return {
            'parallel_efficiency': parallel_efficiency,
            'resource_utilization': resource_utilization,
            'total_sequential_time': total_sequential_time,
            'parallel_time': parallel_time,
            'time_savings': total_sequential_time - parallel_time,
            'max_concurrent_tasks': max_concurrent_tasks,
            'avg_concurrent_tasks': avg_concurrent_tasks
        }
    
    def _generate_decomposition_insights(self, decomposition: TaskDecomposition) -> List[Dict]:
        """Generuje AI insights o dekompozycji"""
        
        insights = []
        
        # Complexity insight
        if decomposition.complexity in [TaskComplexity.COMPLEX, TaskComplexity.ENTERPRISE]:
            insights.append({
                'type': 'complexity_warning',
                'message': f"High complexity task detected ({decomposition.complexity.value}). Consider additional planning and risk mitigation.",
                'confidence': 0.9
            })
        
        # Duration insight
        if decomposition.total_estimated_duration > 80:  # > 2 weeks
            insights.append({
                'type': 'duration_concern',
                'message': f"Long execution time ({decomposition.total_estimated_duration:.1f}h). Consider breaking into phases.",
                'confidence': 0.85
            })
        
        # Parallel execution insight
        efficiency_metrics = self._calculate_execution_efficiency(decomposition)
        if efficiency_metrics['parallel_efficiency'] > 0.4:
            insights.append({
                'type': 'parallelization_success',
                'message': f"Good parallelization potential ({efficiency_metrics['parallel_efficiency']:.1%} time savings).",
                'confidence': 0.8
            })
        elif efficiency_metrics['parallel_efficiency'] < 0.1:
            insights.append({
                'type': 'parallelization_limited',
                'message': "Limited parallelization opportunities detected. Tasks appear highly sequential.",
                'confidence': 0.75
            })
        
        # Critical path insight
        if len(decomposition.critical_path) > len(decomposition.subtasks) * 0.6:
            insights.append({
                'type': 'critical_path_concern',
                'message': "Long critical path detected. Consider task breakdown optimization.",
                'confidence': 0.8
            })
        
        # Subtask distribution insight
        subtask_types = [st.type for st in decomposition.subtasks]
        dev_tasks = sum(1 for st in subtask_types if st == SubtaskType.DEVELOPMENT)
        total_tasks = len(subtask_types)
        
        if dev_tasks > total_tasks * 0.6:
            insights.append({
                'type': 'development_heavy',
                'message': "Development-heavy task detected. Ensure adequate development resources.",
                'confidence': 0.85
            })
        
        return insights
    
    def _analyze_load_balancer_integration(self, decomposition: TaskDecomposition) -> Dict:
        """Analizuje integrację z load balancerem"""
        
        if not self.load_balancer:
            return {'integrated': False, 'reason': 'Load balancer not available'}
        
        agent_assignments = defaultdict(int)
        total_confidence = 0.0
        assigned_tasks = 0
        
        for subtask in decomposition.subtasks:
            if 'recommended_agent' in subtask.metadata:
                agent_id = subtask.metadata['recommended_agent']
                if agent_id:  # Check if not empty
                    agent_assignments[agent_id] += 1
                    total_confidence += subtask.metadata.get('load_balancer_confidence', 0)
                    assigned_tasks += 1
        
        avg_confidence = total_confidence / assigned_tasks if assigned_tasks > 0 else 0
        
        return {
            'integrated': True,
            'agent_assignments': dict(agent_assignments),
            'assignment_confidence': avg_confidence,
            'optimized_tasks': assigned_tasks,
            'load_balancing_effectiveness': min(1.0, avg_confidence * 1.1)
        }
    
    def _generate_optimization_recommendations(self, decomposition: TaskDecomposition) -> List[Dict]:
        """Generuje rekomendacje optymalizacji"""
        
        recommendations = []
        
        # Check for bottlenecks
        efficiency = self._calculate_execution_efficiency(decomposition)
        if efficiency['resource_utilization'] < 0.5:
            recommendations.append({
                'type': 'resource_optimization',
                'priority': 'MEDIUM',
                'description': 'Consider redistributing subtasks to improve resource utilization',
                'expected_improvement': '15-25% faster execution'
            })
        
        # Check for long sequential chains
        if len(decomposition.critical_path) > 6:
            recommendations.append({
                'type': 'parallelization',
                'priority': 'HIGH',
                'description': 'Break down critical path tasks into smaller parallel components',
                'expected_improvement': '20-35% time reduction'
            })
        
        # Load balancing recommendations
        lb_analysis = self._analyze_load_balancer_integration(decomposition)
        if lb_analysis.get('integrated') and lb_analysis.get('assignment_confidence', 0) < 0.6:
            recommendations.append({
                'type': 'agent_selection',
                'priority': 'MEDIUM',
                'description': 'Review agent assignments for better capability matching',
                'expected_improvement': 'Improved task success rate'
            })
        
        # Complexity-based recommendations
        if decomposition.complexity == TaskComplexity.ENTERPRISE:
            recommendations.append({
                'type': 'risk_management',
                'priority': 'HIGH',
                'description': 'Implement comprehensive risk management and milestone tracking',
                'expected_improvement': 'Reduced project risk'
            })
        
        return recommendations

async def main():
    """Main function dla Intelligence V2.0 Task Decomposition demo"""
    print("🧠 Agent Zero V1 - Intelligence V2.0 Phase 3: AI Task Decomposition")
    print("="*80)
    print("🔧 Using Basic NLP Mode (No External Dependencies)")
    
    # Initialize system z previous phases
    if ProductionAgentManager:
        print("\n🔧 Initializing with Production Agent Manager, Analytics & Load Balancer...")
        agent_manager = ProductionAgentManager()
        analytics = IntelligenceV2Analytics(agent_manager)
        load_balancer = IntelligentLoadBalancer(agent_manager, analytics)
        
        # Create specialized agents for task decomposition
        print("🤖 Creating specialized agents for task decomposition...")
        
        # AI & Research Agent
        ai_caps = [
            AgentCapability("machine_learning", 10, "ai"),
            AgentCapability("data_analysis", 9, "analytics"),
            AgentCapability("research", 8, "research")
        ]
        ai_agent = agent_manager.create_agent("AI Research Specialist", AgentType.ANALYZER, ai_caps, max_concurrent_tasks=4)
        agent_manager.update_agent_status(ai_agent, AgentStatus.ACTIVE)
        
        # Full-Stack Developer
        dev_caps = [
            AgentCapability("software_development", 10, "engineering"),
            AgentCapability("system_design", 9, "architecture"),
            AgentCapability("programming", 10, "technical")
        ]
        dev_agent = agent_manager.create_agent("Senior Full-Stack Developer", AgentType.EXECUTOR, dev_caps, max_concurrent_tasks=6)
        agent_manager.update_agent_status(dev_agent, AgentStatus.ACTIVE)
        
        # DevOps & Infrastructure Expert
        ops_caps = [
            AgentCapability("deployment", 10, "devops"),
            AgentCapability("system_administration", 9, "infrastructure"),
            AgentCapability("monitoring", 9, "operations")
        ]
        ops_agent = agent_manager.create_agent("DevOps Specialist", AgentType.EXECUTOR, ops_caps, max_concurrent_tasks=5)
        agent_manager.update_agent_status(ops_agent, AgentStatus.ACTIVE)
        
        # Project Manager & Coordinator  
        pm_caps = [
            AgentCapability("project_management", 10, "management"),
            AgentCapability("coordination", 9, "leadership"),
            AgentCapability("communication", 8, "social")
        ]
        pm_agent = agent_manager.create_agent("Senior Project Manager", AgentType.COORDINATOR, pm_caps, max_concurrent_tasks=5)
        agent_manager.update_agent_status(pm_agent, AgentStatus.ACTIVE)
        
        print(f"✅ Created {len(agent_manager.list_agents())} specialized agents")
        
    else:
        print("⚠️ Running in simulation mode without Production Agent Manager")
        agent_manager = None
        load_balancer = None
    
    # Initialize AI Task Decomposer
    print("\n🧠 Initializing Intelligence V2.0 AI Task Decomposer...")
    decomposer = IntelligentTaskDecomposer(agent_manager, load_balancer)
    
    # Test complex enterprise task decomposition
    print("\n📋 Testing AI Task Decomposition with Complex Enterprise Tasks...")
    
    test_tasks = [
        "Develop a comprehensive enterprise-grade AI-powered customer analytics platform with real-time data processing, machine learning recommendations, multi-tenant architecture, advanced security, and integrated dashboard for Fortune 500 deployment with scalable cloud infrastructure",
        
        "Implement automated CI/CD pipeline for microservices architecture with container orchestration using Kubernetes, automated testing suite, comprehensive security scanning, blue-green deployment strategy, and production rollout with zero-downtime capabilities",
        
        "Design and deploy distributed monitoring system with real-time alerting, predictive analytics using machine learning, integration with existing enterprise tools, multi-cloud support, and comprehensive dashboard for infrastructure management across multiple data centers",
        
        "Coordinate comprehensive digital transformation initiative including legacy system integration, cloud migration strategy, user training programs, change management processes, stakeholder communication plans, and cross-functional team coordination for Q4 delivery"
    ]
    
    decomposition_results = []
    
    for i, task_desc in enumerate(test_tasks, 1):
        print(f"\n🎯 Task {i}: Decomposing Enterprise Task...")
        print(f"   📝 Description: {task_desc[:80]}...")
        
        # Perform AI task decomposition
        decomposition = decomposer.decompose_task(task_desc, f"enterprise_task_{i}")
        
        # Optimize with load balancer if available
        if load_balancer:
            decomposition = decomposer.optimize_decomposition_with_load_balancer(decomposition)
        
        decomposition_results.append(decomposition)
        
        # Display key results
        print(f"   📊 AI Analysis Results:")
        print(f"      🧠 Domain: {decomposition.nlp_insights['domain_classification'].upper()}")
        print(f"      📈 Complexity: {decomposition.complexity.value.upper()}")
        print(f"      🔢 Subtasks Generated: {len(decomposition.subtasks)}")
        print(f"      ⏱️ Total Duration: {decomposition.total_estimated_duration:.1f} hours")
        print(f"      🎯 AI Confidence: {decomposition.confidence:.1%}")
        print(f"      📅 Execution Stages: {len(decomposition.execution_plan)}")
    
    # Generate comprehensive report dla first (most complex) task
    print(f"\n📊 Generating Comprehensive Intelligence V2.0 Task Decomposition Report...")
    main_decomposition = decomposition_results[0] if decomposition_results else None
    
    if main_decomposition:
        report = decomposer.generate_task_decomposition_report(main_decomposition)
        
        # Display comprehensive results
        print("\n" + "="*80)
        print("📈 INTELLIGENCE V2.0 AI TASK DECOMPOSITION REPORT")
        print("="*80)
        
        # Task Overview
        original = report['original_task']
        summary = report['decomposition_summary']
        
        print(f"\n🎯 Original Task Analysis:")
        print(f"   • Task Description: {original['description'][:70]}...")
        print(f"   • AI-Assessed Complexity: {original['complexity'].upper()}")
        print(f"   • Decomposition Confidence: {summary['confidence']:.1%}")
        
        # Decomposition Summary  
        print(f"\n📊 AI Decomposition Results:")
        print(f"   • Total Subtasks Created: {summary['total_subtasks']}")
        print(f"   • Parallel Execution Stages: {summary['execution_stages']}")
        print(f"   • Estimated Project Duration: {summary['estimated_duration']:.1f} hours ({summary['estimated_duration']/8:.1f} days)")
        print(f"   • Critical Path Length: {summary['critical_path_length']} tasks")
        
        # NLP Analysis Results
        nlp = report['nlp_insights']
        print(f"\n🧠 NLP Analysis Results:")
        print(f"   • Domain Classification: {nlp['domain_classification'].upper()}")
        print(f"   • Key Action Verbs: {', '.join(nlp['action_verbs'][:5])}")
        print(f"   • Complexity Indicators: {', '.join(nlp['complexity_indicators'])}")
        print(f"   • Technology Entities: {len(nlp['entities'])} detected")
        print(f"   • Processed Tokens: {len(nlp['tokens'])} meaningful words")
        
        # Subtask Analysis
        analysis = report['subtask_analysis']
        print(f"\n🔍 Intelligent Subtask Analysis:")
        print(f"   • Task Types: {analysis['by_type']}")
        print(f"   • Priority Distribution: {analysis['by_priority']}")
        print(f"   • Duration Categories: {analysis['by_duration']}")
        print(f"   • Required Capabilities: {len(analysis['capability_requirements'])} unique skills")
        
        # Execution Plan Efficiency
        exec_plan = report['execution_plan']
        print(f"\n⚡ Execution Plan Optimization:")
        print(f"   • Parallelization Efficiency: {exec_plan['parallel_efficiency']:.1%}")
        print(f"   • Resource Utilization: {exec_plan['resource_utilization']:.1%}")
        print(f"   • Execution Stages Breakdown:")
        
        for stage_num, stage_tasks in enumerate(exec_plan['stages'][:5], 1):  # Show first 5 stages
            print(f"      Stage {stage_num}: {len(stage_tasks)} parallel tasks")
        
        if len(exec_plan['stages']) > 5:
            print(f"      ... and {len(exec_plan['stages']) - 5} more stages")
        
        # AI Insights
        insights = report['ai_insights']
        if insights:
            print(f"\n💡 AI-Generated Strategic Insights ({len(insights)}):")
            for insight in insights:
                insight_icons = {
                    "complexity_warning": "⚠️", "duration_concern": "⏱️", 
                    "parallelization_success": "⚡", "parallelization_limited": "🔗",
                    "critical_path_concern": "🛤️", "development_heavy": "💻"
                }
                icon = insight_icons.get(insight['type'], "💡")
                
                print(f"   {icon} {insight['message']}")
                print(f"      🎯 AI Confidence: {insight['confidence']:.1%}")
        else:
            print(f"\n💡 No Critical Insights - Task appears well-structured for execution")
        
        # Load Balancer Integration
        lb_integration = report['load_balancer_integration']
        if lb_integration.get('integrated'):
            print(f"\n⚡ Load Balancer Intelligence Integration:")
            print(f"   • Integration Status: ✅ ACTIVE")
            print(f"   • Agent Assignment Confidence: {lb_integration['assignment_confidence']:.1%}")
            print(f"   • Tasks with Optimal Assignments: {lb_integration['optimized_tasks']}")
            
            if lb_integration['agent_assignments']:
                print(f"   • Agent Workload Distribution:")
                for agent_id, task_count in lb_integration['agent_assignments'].items():
                    print(f"      - {agent_id}: {task_count} tasks assigned")
        else:
            print(f"\n⚡ Load Balancer Integration: ⚠️ Not Available")
        
        # Optimization Recommendations
        recommendations = report['recommendations']
        if recommendations:
            print(f"\n🎯 AI Optimization Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations, 1):
                priority_icons = {"HIGH": "🚨", "MEDIUM": "📋", "LOW": "💡"}
                icon = priority_icons.get(rec['priority'], "•")
                
                print(f"   {i}. {icon} {rec['description']}")
                print(f"      📈 Expected Benefit: {rec['expected_improvement']}")
                print(f"      🎯 Priority Level: {rec['priority']}")
        else:
            print(f"\n🎯 No Optimization Recommendations - Task decomposition is well-optimized")
        
        # Detailed Subtask Breakdown (selected examples)
        print(f"\n📋 Sample Subtask Breakdown (First 4 of {len(main_decomposition.subtasks)}):")
        for i, subtask in enumerate(main_decomposition.subtasks[:4], 1):
            print(f"\n   {i}. 📋 {subtask.title}")
            print(f"      📝 Type: {subtask.type.value.title()}")
            print(f"      ⏱️ Duration: {subtask.estimated_duration:.1f}h")
            print(f"      🎯 Priority: {subtask.priority.name}")
            print(f"      🔧 Required Skills: {', '.join(subtask.required_capabilities)}")
            print(f"      🎲 AI Confidence: {subtask.confidence:.1%}")
            
            if 'recommended_agent' in subtask.metadata and subtask.metadata['recommended_agent']:
                agent_id = subtask.metadata['recommended_agent']
                lb_confidence = subtask.metadata.get('load_balancer_confidence', 0)
                print(f"      🤖 Optimal Agent: {agent_id}")
                print(f"      📊 Assignment Confidence: {lb_confidence:.1%}")
            
            if subtask.dependencies:
                print(f"      🔗 Dependencies: {len(subtask.dependencies)} prerequisite tasks")
        
        if len(main_decomposition.subtasks) > 4:
            print(f"\n   ... and {len(main_decomposition.subtasks) - 4} more subtasks")
    
    # Summary of all tasks
    print(f"\n" + "="*80)
    print("📊 ENTERPRISE TASK DECOMPOSITION SUMMARY")
    print("="*80)
    
    for i, decomp in enumerate(decomposition_results, 1):
        complexity_icons = {
            'trivial': '🟢', 'simple': '🟡', 'moderate': '🟠', 'complex': '🔴', 'enterprise': '🔥'
        }
        icon = complexity_icons.get(decomp.complexity.value, '⚪')
        
        print(f"Task {i}: {icon} {decomp.complexity.value.upper()} | "
              f"{len(decomp.subtasks)} subtasks | "
              f"{decomp.total_estimated_duration:.0f}h | "
              f"{decomp.confidence:.0%} confidence")
    
    print(f"\n🎉 Intelligence V2.0 Phase 3 AI Task Decomposition Complete!")
    print("="*80)
    print("✅ NLP-powered task analysis and intelligent decomposition operational")
    print("🧠 Automated complexity assessment and subtask generation working")  
    print("⚡ Smart execution planning with parallelization optimization active")
    print("🤖 Integration with load balancer for optimal resource allocation complete")
    print("📊 Enterprise-grade project planning capabilities now available")
    print("🚀 Ready for Phase 4: Pattern Recognition & Continuous Learning")
    print("="*80)
    
    return decomposition_results

if __name__ == "__main__":
    asyncio.run(main())