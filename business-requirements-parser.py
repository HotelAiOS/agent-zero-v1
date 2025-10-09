"""
Business Requirements Parser for Agent Zero V1
Natural language to technical specs converter

Core implementation for Priority 2 tasks
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import re
import json

class IntentType(Enum):
    """Types of business intents"""
    CREATE = "create"
    UPDATE = "update" 
    ANALYZE = "analyze"
    PROCESS = "process"
    GENERATE = "generate"
    SEARCH = "search"
    DELETE = "delete"
    OPTIMIZE = "optimize"

class ComplexityLevel(Enum):
    """Task complexity levels"""
    SIMPLE = "simple"      # Single agent, < 5min
    MODERATE = "moderate"  # Multiple agents, < 30min 
    COMPLEX = "complex"    # Multi-step, > 30min
    ENTERPRISE = "enterprise"  # Long-running, multiple systems

@dataclass
class BusinessIntent:
    """Extracted business intent from natural language"""
    primary_action: IntentType
    target_entities: List[str]  # What to act upon
    constraints: Dict[str, Any]  # Limitations, requirements
    context: Dict[str, Any]      # Additional context
    complexity: ComplexityLevel
    confidence: float           # 0-1 confidence in parsing
    
@dataclass  
class TechnicalSpec:
    """Technical specification for execution"""
    agent_sequence: List[str]   # Which agents to use
    dependencies: List[str]     # Task dependencies
    estimated_cost: float       # USD cost estimate
    estimated_time: int         # Minutes estimate
    required_tools: List[str]   # Tools/APIs needed
    success_criteria: List[str] # How to measure success
    rollback_plan: Optional[str] = None

class BusinessRequirementsParser:
    """
    Natural language to technical specs converter
    
    Philosophy: Bridge business language and technical execution
    """
    
    def __init__(self):
        # Action keywords mapping
        self.action_keywords = {
            IntentType.CREATE: ['create', 'build', 'make', 'generate', 'add', 'new'],
            IntentType.UPDATE: ['update', 'modify', 'change', 'edit', 'fix', 'improve'],
            IntentType.ANALYZE: ['analyze', 'examine', 'review', 'check', 'audit', 'evaluate'],
            IntentType.PROCESS: ['process', 'handle', 'manage', 'execute', 'run'],
            IntentType.GENERATE: ['generate', 'produce', 'output', 'render', 'compile'],
            IntentType.SEARCH: ['find', 'search', 'locate', 'discover', 'lookup'],
            IntentType.DELETE: ['delete', 'remove', 'clean', 'purge', 'clear'],
            IntentType.OPTIMIZE: ['optimize', 'improve', 'enhance', 'speed up', 'reduce']
        }
        
        # Entity patterns
        self.entity_patterns = {
            'file': r'\b(?:file|document|pdf|csv|json|xml)\b',
            'database': r'\b(?:database|db|table|sql|query)\b', 
            'api': r'\b(?:api|endpoint|service|microservice)\b',
            'code': r'\b(?:code|function|class|module|script)\b',
            'report': r'\b(?:report|analysis|summary|dashboard)\b',
            'email': r'\b(?:email|message|notification)\b',
            'user': r'\b(?:user|customer|client|person)\b',
            'system': r'\b(?:system|platform|application|app)\b'
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            ComplexityLevel.SIMPLE: ['simple', 'quick', 'basic', 'one', 'single'],
            ComplexityLevel.MODERATE: ['multiple', 'several', 'few', 'batch'],
            ComplexityLevel.COMPLEX: ['complex', 'advanced', 'detailed', 'comprehensive'],
            ComplexityLevel.ENTERPRISE: ['enterprise', 'large-scale', 'production', 'critical']
        }
    
    def parse_intent(self, business_request: str) -> BusinessIntent:
        """
        Extract business intent from natural language
        
        Examples:
        - "Create a user registration API" → CREATE intent, API entity
        - "Analyze sales data from last quarter" → ANALYZE intent, data entity
        """
        
        request_lower = business_request.lower()
        
        # 1. Identify action
        primary_action = self._identify_action(request_lower)
        
        # 2. Extract entities
        target_entities = self._extract_entities(request_lower)
        
        # 3. Find constraints
        constraints = self._extract_constraints(business_request)
        
        # 4. Determine complexity
        complexity = self._assess_complexity(request_lower, target_entities)
        
        # 5. Extract context
        context = {
            'original_request': business_request,
            'word_count': len(business_request.split()),
            'technical_terms': self._count_technical_terms(request_lower)
        }
        
        # 6. Calculate confidence
        confidence = self._calculate_confidence(primary_action, target_entities, complexity)
        
        return BusinessIntent(
            primary_action=primary_action,
            target_entities=target_entities,
            constraints=constraints,
            context=context,
            complexity=complexity,
            confidence=confidence
        )
    
    def _identify_action(self, request: str) -> IntentType:
        """Identify primary action from text"""
        action_scores = {}
        
        for intent_type, keywords in self.action_keywords.items():
            score = sum(1 for keyword in keywords if keyword in request)
            if score > 0:
                action_scores[intent_type] = score
        
        if action_scores:
            return max(action_scores.items(), key=lambda x: x[1])[0]
        
        # Default fallback
        return IntentType.PROCESS
    
    def _extract_entities(self, request: str) -> List[str]:
        """Extract target entities from text"""
        entities = []
        
        for entity_type, pattern in self.entity_patterns.items():
            if re.search(pattern, request):
                entities.append(entity_type)
        
        # Add specific entities mentioned
        entity_words = ['user', 'data', 'report', 'system', 'database', 'api', 'file']
        for word in entity_words:
            if word in request and word not in entities:
                entities.append(word)
        
        return entities
    
    def _extract_constraints(self, request: str) -> Dict[str, Any]:
        """Extract constraints and requirements"""
        constraints = {}
        
        # Time constraints
        time_patterns = {
            'urgent': r'\b(?:urgent|asap|immediately|now)\b',
            'deadline': r'\b(?:by|until|before|deadline)\s+(\w+)',
            'duration': r'\b(?:in|within)\s+(\d+)\s*(minutes?|hours?|days?)\b'
        }
        
        for constraint_type, pattern in time_patterns.items():
            match = re.search(pattern, request.lower())
            if match:
                constraints[constraint_type] = match.group(1) if match.groups() else True
        
        # Quality constraints  
        quality_words = ['high quality', 'production ready', 'enterprise grade', 'secure']
        for quality in quality_words:
            if quality in request.lower():
                constraints['quality_requirement'] = quality
        
        # Budget constraints
        budget_pattern = r'\\$([0-9,]+)'
        budget_match = re.search(budget_pattern, request)
        if budget_match:
            constraints['budget_limit'] = float(budget_match.group(1).replace(',', ''))
        
        return constraints
    
    def _assess_complexity(self, request: str, entities: List[str]) -> ComplexityLevel:
        """Assess task complexity based on indicators"""
        
        # Count complexity indicators
        complexity_scores = {}
        for level, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in request)
            complexity_scores[level] = score
        
        # Factor in entity count
        entity_count = len(entities)
        if entity_count >= 4:
            complexity_scores[ComplexityLevel.COMPLEX] += 2
        elif entity_count >= 2:
            complexity_scores[ComplexityLevel.MODERATE] += 1
        
        # Factor in word count
        word_count = len(request.split())
        if word_count > 50:
            complexity_scores[ComplexityLevel.COMPLEX] += 1
        elif word_count > 20:
            complexity_scores[ComplexityLevel.MODERATE] += 1
        
        # Return highest scoring complexity
        if complexity_scores:
            return max(complexity_scores.items(), key=lambda x: x[1])[0]
        
        return ComplexityLevel.SIMPLE
    
    def _count_technical_terms(self, request: str) -> int:
        """Count technical terms to assess complexity"""
        technical_terms = [
            'api', 'database', 'sql', 'json', 'xml', 'http', 'rest', 'microservice',
            'authentication', 'authorization', 'encryption', 'algorithm', 'framework',
            'deployment', 'docker', 'kubernetes', 'cloud', 'aws', 'azure'
        ]
        
        return sum(1 for term in technical_terms if term in request)
    
    def _calculate_confidence(self, action: IntentType, entities: List[str], 
                            complexity: ComplexityLevel) -> float:
        """Calculate confidence in intent parsing"""
        confidence = 0.5  # Base confidence
        
        # Higher confidence if clear action identified
        if action != IntentType.PROCESS:  # Not default fallback
            confidence += 0.2
        
        # Higher confidence if entities identified
        if entities:
            confidence += min(len(entities) * 0.1, 0.2)
        
        # Adjust based on complexity (simpler = more confident)
        complexity_adjustments = {
            ComplexityLevel.SIMPLE: 0.1,
            ComplexityLevel.MODERATE: 0.0,
            ComplexityLevel.COMPLEX: -0.1,
            ComplexityLevel.ENTERPRISE: -0.2
        }
        confidence += complexity_adjustments.get(complexity, 0.0)
        
        return min(max(confidence, 0.0), 1.0)  # Clamp to 0-1
    
    def generate_technical_spec(self, intent: BusinessIntent) -> TechnicalSpec:
        """
        Convert business intent to technical specification
        
        Maps business intent to technical implementation plan
        """
        
        # Agent selection based on intent and entities
        agent_sequence = self._select_agents(intent)
        
        # Estimate cost and time
        estimated_cost, estimated_time = self._estimate_resources(intent)
        
        # Determine required tools
        required_tools = self._determine_tools(intent)
        
        # Define success criteria
        success_criteria = self._define_success_criteria(intent)
        
        # Identify dependencies
        dependencies = self._identify_dependencies(intent)
        
        return TechnicalSpec(
            agent_sequence=agent_sequence,
            dependencies=dependencies,
            estimated_cost=estimated_cost,
            estimated_time=estimated_time,
            required_tools=required_tools,
            success_criteria=success_criteria,
            rollback_plan=self._create_rollback_plan(intent)
        )
    
    def _select_agents(self, intent: BusinessIntent) -> List[str]:
        """Select appropriate agents for the task"""
        agents = []
        
        # Agent mapping based on intent type
        intent_agent_map = {
            IntentType.CREATE: ['code_generator', 'architect'],
            IntentType.ANALYZE: ['data_analyst', 'researcher'],
            IntentType.GENERATE: ['content_creator', 'code_generator'],
            IntentType.PROCESS: ['data_processor', 'orchestrator'],
            IntentType.SEARCH: ['researcher', 'data_retriever'],
            IntentType.UPDATE: ['code_modifier', 'data_updater'],
            IntentType.DELETE: ['cleanup_agent', 'data_manager'],
            IntentType.OPTIMIZE: ['performance_optimizer', 'code_refactor']
        }
        
        base_agents = intent_agent_map.get(intent.primary_action, ['orchestrator'])
        agents.extend(base_agents)
        
        # Add entity-specific agents
        if 'database' in intent.target_entities:
            agents.append('database_specialist')
        if 'api' in intent.target_entities:
            agents.append('api_specialist')
        if 'file' in intent.target_entities:
            agents.append('file_processor')
        
        # Add complexity-based agents
        if intent.complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE]:
            agents.insert(0, 'project_manager')  # Add at beginning
            agents.append('quality_assurance')   # Add at end
        
        return list(dict.fromkeys(agents))  # Remove duplicates, preserve order
    
    def _estimate_resources(self, intent: BusinessIntent) -> tuple[float, int]:
        """Estimate cost and time for the task"""
        
        # Base estimates by complexity
        base_estimates = {
            ComplexityLevel.SIMPLE: (0.01, 5),      # $0.01, 5 minutes
            ComplexityLevel.MODERATE: (0.05, 20),   # $0.05, 20 minutes  
            ComplexityLevel.COMPLEX: (0.20, 60),    # $0.20, 60 minutes
            ComplexityLevel.ENTERPRISE: (1.00, 240) # $1.00, 240 minutes
        }
        
        base_cost, base_time = base_estimates[intent.complexity]
        
        # Adjust based on entities count
        entity_multiplier = 1.0 + (len(intent.target_entities) * 0.2)
        
        # Adjust based on constraints
        constraint_multiplier = 1.0
        if 'urgent' in intent.constraints:
            constraint_multiplier *= 0.8  # Faster but potentially higher cost
        if 'quality_requirement' in intent.constraints:
            constraint_multiplier *= 1.5  # Higher quality = more time/cost
        
        final_cost = base_cost * entity_multiplier * constraint_multiplier
        final_time = int(base_time * entity_multiplier * constraint_multiplier)
        
        return final_cost, final_time
    
    def _determine_tools(self, intent: BusinessIntent) -> List[str]:
        """Determine required tools for the task"""
        tools = ['orchestrator']  # Always needed
        
        # Intent-based tools
        intent_tools = {
            IntentType.CREATE: ['code_editor', 'file_creator'],
            IntentType.ANALYZE: ['data_analyzer', 'chart_generator'], 
            IntentType.GENERATE: ['template_engine', 'content_generator'],
            IntentType.SEARCH: ['search_engine', 'web_scraper'],
            IntentType.PROCESS: ['data_processor', 'workflow_engine']
        }
        
        tools.extend(intent_tools.get(intent.primary_action, []))
        
        # Entity-based tools
        if 'database' in intent.target_entities:
            tools.extend(['sql_executor', 'database_connector'])
        if 'api' in intent.target_entities:
            tools.extend(['http_client', 'api_tester'])
        if 'file' in intent.target_entities:
            tools.extend(['file_reader', 'file_writer'])
        
        return list(set(tools))  # Remove duplicates
    
    def _define_success_criteria(self, intent: BusinessIntent) -> List[str]:
        """Define how to measure task success"""
        criteria = []
        
        # Intent-based criteria
        if intent.primary_action == IntentType.CREATE:
            criteria.append("Artifact successfully created and accessible")
        elif intent.primary_action == IntentType.ANALYZE:
            criteria.append("Analysis completed with actionable insights")
        elif intent.primary_action == IntentType.GENERATE:
            criteria.append("Content generated meets specified requirements")
        
        # Add quality criteria if specified
        if 'quality_requirement' in intent.constraints:
            criteria.append(f"Output meets {intent.constraints['quality_requirement']} standards")
        
        # Add performance criteria
        criteria.append("Task completed within estimated time and budget")
        criteria.append("No errors or exceptions during execution")
        
        return criteria
    
    def _identify_dependencies(self, intent: BusinessIntent) -> List[str]:
        """Identify task dependencies"""
        dependencies = []
        
        # Entity-based dependencies
        if 'database' in intent.target_entities:
            dependencies.append("Database connection available")
        if 'api' in intent.target_entities:
            dependencies.append("API endpoints accessible")
        if 'file' in intent.target_entities:
            dependencies.append("File system access granted")
        
        # Complexity-based dependencies
        if intent.complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE]:
            dependencies.append("Sufficient computational resources")
            dependencies.append("Backup systems operational")
        
        return dependencies
    
    def _create_rollback_plan(self, intent: BusinessIntent) -> Optional[str]:
        """Create rollback plan for the task"""
        
        if intent.complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ENTERPRISE]:
            return "Create checkpoint before execution, monitor progress, rollback to checkpoint if failure rate exceeds 20%"
        elif intent.primary_action in [IntentType.DELETE, IntentType.UPDATE]:
            return "Create backup before modification, restore from backup if needed"
        
        return None

# Example usage and testing
if __name__ == "__main__":
    parser = BusinessRequirementsParser()
    
    # Test cases
    test_requests = [
        "Create a user authentication API with JWT tokens",
        "Analyze sales data from the last quarter and generate a report", 
        "Build a simple todo list application",
        "Optimize database queries for better performance"
    ]
    
    for request in test_requests:
        print(f"\\nRequest: {request}")
        intent = parser.parse_intent(request)
        spec = parser.generate_technical_spec(intent)
        
        print(f"Intent: {intent.primary_action.value}")
        print(f"Entities: {intent.target_entities}")
        print(f"Complexity: {intent.complexity.value}")
        print(f"Confidence: {intent.confidence:.2f}")
        print(f"Agents: {spec.agent_sequence}")
        print(f"Est. Cost: ${spec.estimated_cost:.3f}")
        print(f"Est. Time: {spec.estimated_time} minutes")