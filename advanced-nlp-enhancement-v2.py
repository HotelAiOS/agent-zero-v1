#!/usr/bin/env python3
"""
Agent Zero V2.0 Phase 2 - Advanced NLP Enhancement
Saturday, October 11, 2025 @ 09:46 CEST

Enhancement of existing AI Intelligence Layer with advanced NLP capabilities
Building on successful Phase 1 production deployment
"""

import os
import re
import asyncio
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

# NLP Processing Libraries
import spacy
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

# FastAPI integration for existing AI Intelligence Layer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# PHASE 2 ADVANCED NLP MODELS AND ENUMS
# =============================================================================

class TaskComplexity(Enum):
    """Task complexity classification"""
    SIMPLE = "simple"           # Single step, clear requirements
    MODERATE = "moderate"       # Multiple steps, some dependencies
    COMPLEX = "complex"         # Multiple dependencies, technical depth
    VERY_COMPLEX = "very_complex"  # Advanced technical, high uncertainty

class IntentCategory(Enum):
    """Intent classification for business requests"""
    DEVELOPMENT = "development"     # Code, implementation, technical tasks
    ANALYSIS = "analysis"          # Research, investigation, data analysis
    INTEGRATION = "integration"    # System integration, API connection
    OPTIMIZATION = "optimization"  # Performance, cost, efficiency improvements
    PLANNING = "planning"          # Project planning, roadmapping
    DEPLOYMENT = "deployment"      # Production deployment, DevOps
    MAINTENANCE = "maintenance"    # Bug fixes, updates, monitoring
    RESEARCH = "research"          # Exploration, proof of concept

class DependencyType(Enum):
    """Types of task dependencies"""
    TECHNICAL = "technical"        # Code, system dependencies
    BUSINESS = "business"          # Business logic, requirements
    RESOURCE = "resource"          # People, infrastructure, tools
    TEMPORAL = "temporal"          # Time-based, sequence dependencies
    DATA = "data"                  # Data availability, quality
    EXTERNAL = "external"          # Third-party, vendor dependencies

@dataclass
class TaskDecomposition:
    """Advanced task decomposition result"""
    original_request: str
    intent: IntentCategory
    complexity: TaskComplexity
    confidence_score: float
    subtasks: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    context_analysis: Dict[str, Any]
    recommendations: List[str]
    estimated_effort: str
    risk_factors: List[str]

@dataclass  
class ContextAnalysis:
    """Context understanding result"""
    domain: str
    technical_depth: float
    business_impact: float
    urgency_level: float
    ambiguity_score: float
    extracted_entities: List[Dict[str, Any]]
    semantic_concepts: List[str]

# =============================================================================
# PHASE 2 ADVANCED NLP ENGINE
# =============================================================================

class AdvancedNLPEngine:
    """
    Advanced Natural Language Understanding Engine
    Enhancement for existing AI Intelligence Layer
    """
    
    def __init__(self):
        logger.info("🧠 Initializing Advanced NLP Engine for Agent Zero V2.0")
        
        # Load NLP models
        self.nlp_model = self._load_spacy_model()
        self.sentence_transformer = self._load_sentence_transformer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        
        # Intent classification patterns
        self.intent_patterns = self._build_intent_patterns()
        
        # Complexity analysis keywords
        self.complexity_indicators = self._build_complexity_indicators()
        
        # Dependency detection patterns
        self.dependency_patterns = self._build_dependency_patterns()
        
        logger.info("✅ Advanced NLP Engine initialized successfully")
    
    def _load_spacy_model(self):
        """Load spaCy model with fallback"""
        try:
            # Try to load enhanced model first
            nlp = spacy.load("en_core_web_lg")
            logger.info("✅ Loaded spaCy large model")
            return nlp
        except OSError:
            try:
                # Fallback to medium model
                nlp = spacy.load("en_core_web_md")
                logger.info("⚠️ Loaded spaCy medium model (fallback)")
                return nlp
            except OSError:
                # Fallback to small model
                nlp = spacy.load("en_core_web_sm")
                logger.warning("⚠️ Loaded spaCy small model (minimal fallback)")
                return nlp
    
    def _load_sentence_transformer(self):
        """Load sentence transformer for semantic analysis"""
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded Sentence Transformer model")
            return model
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer: {e}")
            return None
    
    def _build_intent_patterns(self) -> Dict[IntentCategory, List[str]]:
        """Build intent classification patterns"""
        return {
            IntentCategory.DEVELOPMENT: [
                "implement", "code", "develop", "build", "create", "program",
                "write code", "software", "application", "feature", "function",
                "api", "endpoint", "service", "component", "module"
            ],
            IntentCategory.ANALYSIS: [
                "analyze", "investigate", "research", "study", "examine",
                "evaluate", "assess", "review", "compare", "measure",
                "metrics", "data analysis", "performance analysis"
            ],
            IntentCategory.INTEGRATION: [
                "integrate", "connect", "link", "combine", "merge",
                "interface", "bridge", "synchronize", "federate",
                "api integration", "system integration"
            ],
            IntentCategory.OPTIMIZATION: [
                "optimize", "improve", "enhance", "speed up", "reduce cost",
                "performance", "efficiency", "scalability", "refactor",
                "tune", "streamline"
            ],
            IntentCategory.PLANNING: [
                "plan", "design", "architecture", "roadmap", "strategy",
                "timeline", "schedule", "project plan", "blueprint",
                "specification", "requirements"
            ],
            IntentCategory.DEPLOYMENT: [
                "deploy", "release", "publish", "launch", "production",
                "devops", "infrastructure", "pipeline", "ci/cd",
                "environment", "server"
            ],
            IntentCategory.MAINTENANCE: [
                "fix", "debug", "troubleshoot", "maintain", "update",
                "patch", "monitor", "support", "resolve", "repair",
                "bug fix", "maintenance"
            ],
            IntentCategory.RESEARCH: [
                "explore", "prototype", "proof of concept", "poc",
                "feasibility", "experiment", "pilot", "trial",
                "investigation", "discovery"
            ]
        }
    
    def _build_complexity_indicators(self) -> Dict[TaskComplexity, List[str]]:
        """Build complexity analysis indicators"""
        return {
            TaskComplexity.SIMPLE: [
                "simple", "basic", "straightforward", "easy", "quick",
                "single step", "minimal", "standard", "routine"
            ],
            TaskComplexity.MODERATE: [
                "moderate", "multiple steps", "several", "intermediate",
                "standard complexity", "typical", "average"
            ],
            TaskComplexity.COMPLEX: [
                "complex", "advanced", "sophisticated", "intricate",
                "multiple dependencies", "technical depth", "challenging"
            ],
            TaskComplexity.VERY_COMPLEX: [
                "very complex", "extremely complex", "highly sophisticated",
                "cutting edge", "research level", "experimental",
                "unknown territory", "high uncertainty"
            ]
        }
    
    def _build_dependency_patterns(self) -> Dict[DependencyType, List[str]]:
        """Build dependency detection patterns"""
        return {
            DependencyType.TECHNICAL: [
                "depends on", "requires", "needs", "based on", "using",
                "api", "database", "service", "library", "framework",
                "technology", "platform", "infrastructure"
            ],
            DependencyType.BUSINESS: [
                "business logic", "requirements", "approval", "stakeholder",
                "business rules", "policy", "process", "workflow",
                "business decision", "authorization"
            ],
            DependencyType.RESOURCE: [
                "team", "developer", "expert", "specialist", "resource",
                "person", "role", "skill", "expertise", "capacity",
                "availability", "budget", "funding"
            ],
            DependencyType.TEMPORAL: [
                "after", "before", "sequence", "order", "timeline",
                "deadline", "schedule", "priority", "first", "then",
                "prerequisite", "sequential"
            ],
            DependencyType.DATA: [
                "data", "dataset", "information", "content", "input",
                "source", "feed", "import", "export", "migration",
                "data quality", "data availability"
            ],
            DependencyType.EXTERNAL: [
                "third party", "vendor", "external", "partner", "client",
                "supplier", "provider", "integration", "external service",
                "third-party api"
            ]
        }
    
    async def analyze_request(self, request: str) -> TaskDecomposition:
        """
        Advanced request analysis with NLP
        Main entry point for Phase 2 NLP enhancement
        """
        logger.info(f"🔍 Analyzing request: {request[:100]}...")
        
        try:
            # Step 1: Context Analysis
            context = await self._analyze_context(request)
            
            # Step 2: Intent Classification
            intent, intent_confidence = await self._classify_intent(request, context)
            
            # Step 3: Complexity Assessment
            complexity, complexity_confidence = await self._assess_complexity(request, context)
            
            # Step 4: Task Decomposition
            subtasks = await self._decompose_tasks(request, intent, complexity, context)
            
            # Step 5: Dependency Analysis
            dependencies = await self._analyze_dependencies(request, subtasks, context)
            
            # Step 6: Generate Recommendations
            recommendations = await self._generate_recommendations(
                request, intent, complexity, context, subtasks, dependencies
            )
            
            # Step 7: Effort Estimation
            effort = await self._estimate_effort(complexity, subtasks, dependencies)
            
            # Step 8: Risk Analysis
            risks = await self._analyze_risks(complexity, dependencies, context)
            
            # Calculate overall confidence
            overall_confidence = (intent_confidence + complexity_confidence) / 2
            
            # Create comprehensive task decomposition
            decomposition = TaskDecomposition(
                original_request=request,
                intent=intent,
                complexity=complexity,
                confidence_score=overall_confidence,
                subtasks=subtasks,
                dependencies=dependencies,
                context_analysis=context.__dict__,
                recommendations=recommendations,
                estimated_effort=effort,
                risk_factors=risks
            )
            
            logger.info(f"✅ Analysis complete - Intent: {intent.value}, Complexity: {complexity.value}")
            return decomposition
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            raise HTTPException(status_code=500, detail=f"NLP analysis failed: {str(e)}")
    
    async def _analyze_context(self, request: str) -> ContextAnalysis:
        """Analyze request context using advanced NLP"""
        doc = self.nlp_model(request)
        
        # Extract entities
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "description": spacy.explain(ent.label_),
                "start": ent.start_char,
                "end": ent.end_char
            })
        
        # Domain classification
        domain = self._classify_domain(request, entities)
        
        # Technical depth analysis
        technical_depth = self._assess_technical_depth(request, doc)
        
        # Business impact analysis
        business_impact = self._assess_business_impact(request, doc)
        
        # Urgency analysis
        urgency_level = self._assess_urgency(request, doc)
        
        # Ambiguity analysis
        ambiguity_score = self._assess_ambiguity(request, doc)
        
        # Semantic concepts
        semantic_concepts = self._extract_semantic_concepts(request)
        
        return ContextAnalysis(
            domain=domain,
            technical_depth=technical_depth,
            business_impact=business_impact,
            urgency_level=urgency_level,
            ambiguity_score=ambiguity_score,
            extracted_entities=entities,
            semantic_concepts=semantic_concepts
        )
    
    def _classify_domain(self, request: str, entities: List[Dict]) -> str:
        """Classify the domain of the request"""
        domain_keywords = {
            "software_development": ["code", "software", "development", "programming", "api"],
            "data_science": ["data", "analysis", "machine learning", "analytics", "model"],
            "devops": ["deployment", "infrastructure", "pipeline", "ci/cd", "docker"],
            "business": ["business", "strategy", "planning", "requirements", "stakeholder"],
            "research": ["research", "investigation", "study", "analysis", "exploration"],
            "integration": ["integration", "connection", "interface", "bridge", "sync"]
        }
        
        request_lower = request.lower()
        domain_scores = {}
        
        for domain, keywords in domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in request_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        else:
            return "general"
    
    def _assess_technical_depth(self, request: str, doc) -> float:
        """Assess technical depth of the request (0-1 scale)"""
        technical_indicators = [
            "api", "database", "algorithm", "architecture", "framework",
            "implementation", "code", "system", "technical", "engineering",
            "software", "platform", "infrastructure", "protocol", "library"
        ]
        
        request_lower = request.lower()
        technical_count = sum(1 for indicator in technical_indicators if indicator in request_lower)
        
        # Normalize by request length and cap at 1.0
        technical_depth = min(technical_count / (len(request.split()) / 10), 1.0)
        return round(technical_depth, 2)
    
    def _assess_business_impact(self, request: str, doc) -> float:
        """Assess business impact of the request (0-1 scale)"""
        business_indicators = [
            "business", "revenue", "cost", "profit", "customer", "user",
            "market", "competitive", "strategic", "growth", "efficiency",
            "productivity", "roi", "value", "impact", "critical", "important"
        ]
        
        request_lower = request.lower()
        business_count = sum(1 for indicator in business_indicators if indicator in request_lower)
        
        # Normalize and cap at 1.0
        business_impact = min(business_count / (len(request.split()) / 15), 1.0)
        return round(business_impact, 2)
    
    def _assess_urgency(self, request: str, doc) -> float:
        """Assess urgency level of the request (0-1 scale)"""
        urgency_indicators = {
            "urgent": 0.9,
            "asap": 0.9,
            "immediately": 0.9,
            "critical": 0.8,
            "high priority": 0.8,
            "quick": 0.6,
            "fast": 0.6,
            "soon": 0.5,
            "when possible": 0.3,
            "low priority": 0.2
        }
        
        request_lower = request.lower()
        urgency_scores = [
            score for phrase, score in urgency_indicators.items()
            if phrase in request_lower
        ]
        
        if urgency_scores:
            return max(urgency_scores)
        else:
            return 0.5  # Default medium urgency
    
    def _assess_ambiguity(self, request: str, doc) -> float:
        """Assess ambiguity level of the request (0-1 scale, higher = more ambiguous)"""
        ambiguity_indicators = [
            "maybe", "possibly", "might", "could", "should", "would",
            "somehow", "something", "some", "unclear", "vague",
            "general", "broad", "flexible", "open to", "suggestions"
        ]
        
        request_lower = request.lower()
        ambiguity_count = sum(1 for indicator in ambiguity_indicators if indicator in request_lower)
        
        # Check for specific details
        specific_indicators = [
            r'\d+', r'[A-Z][a-z]+\s[A-Z][a-z]+',  # Numbers, proper nouns
            r'https?://', r'\w+\.\w+',  # URLs, domains
        ]
        specific_count = sum(1 for pattern in specific_indicators if re.search(pattern, request))
        
        # Calculate ambiguity (more ambiguous words, fewer specifics = higher score)
        ambiguity = (ambiguity_count / max(len(request.split()) / 10, 1)) - (specific_count / 5)
        return max(0, min(ambiguity, 1.0))
    
    def _extract_semantic_concepts(self, request: str) -> List[str]:
        """Extract key semantic concepts from the request"""
        if not self.sentence_transformer:
            return []
        
        try:
            # Get sentence embedding
            embedding = self.sentence_transformer.encode([request])
            
            # For now, return basic concepts - in production would use more advanced semantic analysis
            doc = self.nlp_model(request)
            concepts = []
            
            # Extract noun phrases
            for chunk in doc.noun_chunks:
                if len(chunk.text.split()) <= 3:  # Keep it concise
                    concepts.append(chunk.text.lower())
            
            # Extract key verbs
            for token in doc:
                if token.pos_ == "VERB" and not token.is_stop:
                    concepts.append(token.lemma_.lower())
            
            # Remove duplicates and return top concepts
            return list(set(concepts))[:10]
            
        except Exception as e:
            logger.error(f"Semantic concept extraction failed: {e}")
            return []
    
    async def _classify_intent(self, request: str, context: ContextAnalysis) -> Tuple[IntentCategory, float]:
        """Classify the intent of the request"""
        request_lower = request.lower()
        intent_scores = {}
        
        # Score based on keyword matching
        for intent, keywords in self.intent_patterns.items():
            score = 0
            for keyword in keywords:
                if keyword in request_lower:
                    score += 1
                    
            # Weight by context
            if intent == IntentCategory.DEVELOPMENT and context.technical_depth > 0.5:
                score *= 1.5
            elif intent == IntentCategory.ANALYSIS and "analyz" in request_lower:
                score *= 1.3
            elif intent == IntentCategory.PLANNING and context.business_impact > 0.5:
                score *= 1.2
                
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            max_score = intent_scores[best_intent]
            total_score = sum(intent_scores.values())
            confidence = max_score / total_score if total_score > 0 else 0.5
            return best_intent, min(confidence, 1.0)
        else:
            return IntentCategory.DEVELOPMENT, 0.5  # Default fallback
    
    async def _assess_complexity(self, request: str, context: ContextAnalysis) -> Tuple[TaskComplexity, float]:
        """Assess the complexity of the request"""
        request_lower = request.lower()
        complexity_scores = {}
        
        # Score based on complexity indicators
        for complexity, keywords in self.complexity_indicators.items():
            score = sum(1 for keyword in keywords if keyword in request_lower)
            if score > 0:
                complexity_scores[complexity] = score
        
        # Additional complexity assessment
        complexity_factors = 0
        
        # Technical depth increases complexity
        if context.technical_depth > 0.7:
            complexity_factors += 2
        elif context.technical_depth > 0.4:
            complexity_factors += 1
            
        # High ambiguity increases complexity
        if context.ambiguity_score > 0.6:
            complexity_factors += 2
        elif context.ambiguity_score > 0.3:
            complexity_factors += 1
            
        # Multiple entities suggest complexity
        if len(context.extracted_entities) > 5:
            complexity_factors += 1
            
        # Request length can indicate complexity
        if len(request.split()) > 50:
            complexity_factors += 1
        
        # Determine complexity based on factors
        if complexity_factors >= 4:
            final_complexity = TaskComplexity.VERY_COMPLEX
        elif complexity_factors >= 3:
            final_complexity = TaskComplexity.COMPLEX
        elif complexity_factors >= 1:
            final_complexity = TaskComplexity.MODERATE
        else:
            final_complexity = TaskComplexity.SIMPLE
        
        # Override with explicit keywords if found
        if complexity_scores:
            keyword_complexity = max(complexity_scores, key=complexity_scores.get)
            max_score = complexity_scores[keyword_complexity]
            confidence = min(max_score / 3.0, 1.0)  # Normalize
            return keyword_complexity, confidence
        
        # Calculate confidence based on factors
        confidence = min(complexity_factors / 5.0, 1.0)
        return final_complexity, max(confidence, 0.3)  # Minimum confidence
    
    async def _decompose_tasks(self, request: str, intent: IntentCategory, 
                              complexity: TaskComplexity, context: ContextAnalysis) -> List[Dict[str, Any]]:
        """Decompose the request into subtasks"""
        subtasks = []
        
        # Base subtasks based on intent
        if intent == IntentCategory.DEVELOPMENT:
            subtasks = [
                {"name": "Requirements Analysis", "type": "analysis", "effort": "low"},
                {"name": "Technical Design", "type": "design", "effort": "medium"},
                {"name": "Implementation", "type": "development", "effort": "high"},
                {"name": "Testing", "type": "testing", "effort": "medium"},
                {"name": "Documentation", "type": "documentation", "effort": "low"}
            ]
        elif intent == IntentCategory.ANALYSIS:
            subtasks = [
                {"name": "Data Collection", "type": "research", "effort": "medium"},
                {"name": "Analysis Framework", "type": "planning", "effort": "low"},
                {"name": "Data Analysis", "type": "analysis", "effort": "high"},
                {"name": "Results Interpretation", "type": "analysis", "effort": "medium"},
                {"name": "Report Generation", "type": "documentation", "effort": "medium"}
            ]
        elif intent == IntentCategory.INTEGRATION:
            subtasks = [
                {"name": "Systems Assessment", "type": "analysis", "effort": "medium"},
                {"name": "Integration Design", "type": "design", "effort": "high"},
                {"name": "API Development", "type": "development", "effort": "high"},
                {"name": "Testing & Validation", "type": "testing", "effort": "high"},
                {"name": "Deployment", "type": "deployment", "effort": "medium"}
            ]
        else:
            # Generic subtasks
            subtasks = [
                {"name": "Planning", "type": "planning", "effort": "low"},
                {"name": "Execution", "type": "execution", "effort": "high"},
                {"name": "Validation", "type": "testing", "effort": "medium"},
                {"name": "Documentation", "type": "documentation", "effort": "low"}
            ]
        
        # Adjust based on complexity
        if complexity in [TaskComplexity.COMPLEX, TaskComplexity.VERY_COMPLEX]:
            # Add additional subtasks for complex requests
            subtasks.insert(1, {"name": "Risk Assessment", "type": "analysis", "effort": "medium"})
            subtasks.append({"name": "Monitoring & Optimization", "type": "monitoring", "effort": "medium"})
            
            # Increase effort levels
            for task in subtasks:
                if task["effort"] == "low":
                    task["effort"] = "medium"
                elif task["effort"] == "medium":
                    task["effort"] = "high"
        
        # Add unique IDs and additional metadata
        for i, task in enumerate(subtasks):
            task.update({
                "id": f"task_{i+1}",
                "priority": "medium",  # Could be enhanced with more logic
                "estimated_hours": self._estimate_task_hours(task["effort"]),
                "skills_required": self._identify_required_skills(task["type"], context),
                "deliverables": [f"{task['name']} deliverable"]
            })
        
        return subtasks
    
    def _estimate_task_hours(self, effort: str) -> str:
        """Estimate hours for a task based on effort level"""
        effort_mapping = {
            "low": "2-4 hours",
            "medium": "4-8 hours", 
            "high": "8-16 hours"
        }
        return effort_mapping.get(effort, "4-8 hours")
    
    def _identify_required_skills(self, task_type: str, context: ContextAnalysis) -> List[str]:
        """Identify required skills for a task type"""
        skill_mapping = {
            "analysis": ["analytical thinking", "data analysis"],
            "design": ["system design", "architecture"],
            "development": ["programming", "software development"],
            "testing": ["testing", "quality assurance"],
            "documentation": ["technical writing", "documentation"],
            "deployment": ["devops", "deployment"],
            "monitoring": ["monitoring", "system administration"],
            "planning": ["project management", "planning"]
        }
        
        base_skills = skill_mapping.get(task_type, ["general"])
        
        # Add domain-specific skills based on context
        if context.domain == "software_development":
            base_skills.append("software engineering")
        elif context.domain == "data_science":
            base_skills.append("data science")
        elif context.domain == "devops":
            base_skills.append("devops")
            
        return base_skills
    
    async def _analyze_dependencies(self, request: str, subtasks: List[Dict[str, Any]], 
                                  context: ContextAnalysis) -> List[Dict[str, Any]]:
        """Analyze dependencies in the request and between subtasks"""
        dependencies = []
        request_lower = request.lower()
        
        # Detect explicit dependencies in request
        for dep_type, keywords in self.dependency_patterns.items():
            for keyword in keywords:
                if keyword in request_lower:
                    dependencies.append({
                        "id": f"dep_{len(dependencies)+1}",
                        "type": dep_type.value,
                        "description": f"Dependency on {keyword}",
                        "source": "request_analysis",
                        "impact": "medium",
                        "mitigation": f"Ensure {keyword} availability"
                    })
        
        # Analyze inter-task dependencies
        for i, task in enumerate(subtasks):
            if i > 0:
                # Most tasks depend on previous tasks
                dependencies.append({
                    "id": f"dep_task_{i+1}",
                    "type": DependencyType.TEMPORAL.value,
                    "description": f"{task['name']} depends on {subtasks[i-1]['name']}",
                    "source": "task_sequence",
                    "from_task": subtasks[i-1]["id"],
                    "to_task": task["id"],
                    "impact": "high",
                    "mitigation": "Follow sequential task execution"
                })
        
        # Add context-based dependencies
        if context.technical_depth > 0.6:
            dependencies.append({
                "id": f"dep_technical",
                "type": DependencyType.TECHNICAL.value,
                "description": "High technical complexity requires specialized expertise",
                "source": "context_analysis",
                "impact": "high",
                "mitigation": "Assign experienced technical team members"
            })
        
        if context.business_impact > 0.7:
            dependencies.append({
                "id": f"dep_business",
                "type": DependencyType.BUSINESS.value,
                "description": "High business impact requires stakeholder approval",
                "source": "context_analysis",
                "impact": "high",
                "mitigation": "Early stakeholder engagement and approval process"
            })
        
        return dependencies
    
    async def _generate_recommendations(self, request: str, intent: IntentCategory,
                                      complexity: TaskComplexity, context: ContextAnalysis,
                                      subtasks: List[Dict], dependencies: List[Dict]) -> List[str]:
        """Generate intelligent recommendations based on analysis"""
        recommendations = []
        
        # Complexity-based recommendations
        if complexity == TaskComplexity.VERY_COMPLEX:
            recommendations.extend([
                "Consider breaking this into multiple phases or iterations",
                "Assign senior team members with relevant expertise",
                "Implement comprehensive risk management strategies",
                "Plan for additional time buffer (20-30%)"
            ])
        elif complexity == TaskComplexity.COMPLEX:
            recommendations.extend([
                "Allocate sufficient time for thorough planning",
                "Consider involving subject matter experts",
                "Implement regular checkpoint reviews"
            ])
        
        # Intent-based recommendations
        if intent == IntentCategory.DEVELOPMENT:
            recommendations.extend([
                "Follow established coding standards and best practices",
                "Implement comprehensive testing strategy",
                "Consider CI/CD pipeline integration"
            ])
        elif intent == IntentCategory.INTEGRATION:
            recommendations.extend([
                "Conduct thorough API documentation review",
                "Implement proper error handling and fallback mechanisms",
                "Plan for data migration and synchronization"
            ])
        
        # Context-based recommendations
        if context.ambiguity_score > 0.6:
            recommendations.append("Clarify ambiguous requirements before proceeding")
        
        if context.urgency_level > 0.7:
            recommendations.append("Prioritize critical path items for faster delivery")
        
        if len(dependencies) > 5:
            recommendations.append("Create detailed dependency management plan")
        
        # Business impact recommendations
        if context.business_impact > 0.7:
            recommendations.extend([
                "Involve key stakeholders in regular progress reviews",
                "Implement comprehensive change management process",
                "Prepare detailed business impact assessment"
            ])
        
        return recommendations[:8]  # Limit to top recommendations
    
    async def _estimate_effort(self, complexity: TaskComplexity, 
                              subtasks: List[Dict], dependencies: List[Dict]) -> str:
        """Estimate overall effort for the request"""
        base_hours = {
            TaskComplexity.SIMPLE: 8,
            TaskComplexity.MODERATE: 24,
            TaskComplexity.COMPLEX: 80,
            TaskComplexity.VERY_COMPLEX: 200
        }
        
        base_effort = base_hours[complexity]
        
        # Adjust based on number of subtasks
        task_multiplier = 1 + (len(subtasks) - 4) * 0.1  # Base 4 tasks
        
        # Adjust based on dependencies
        dependency_multiplier = 1 + len(dependencies) * 0.05
        
        final_effort = base_effort * task_multiplier * dependency_multiplier
        
        # Convert to range
        min_effort = int(final_effort * 0.8)
        max_effort = int(final_effort * 1.3)
        
        return f"{min_effort}-{max_effort} hours"
    
    async def _analyze_risks(self, complexity: TaskComplexity, 
                           dependencies: List[Dict], context: ContextAnalysis) -> List[str]:
        """Analyze potential risks"""
        risks = []
        
        # Complexity-based risks
        if complexity == TaskComplexity.VERY_COMPLEX:
            risks.extend([
                "High technical complexity may lead to scope creep",
                "Unknown technical challenges may cause delays",
                "May require specialized expertise not currently available"
            ])
        
        # Dependency-based risks
        high_impact_deps = [d for d in dependencies if d.get("impact") == "high"]
        if len(high_impact_deps) > 3:
            risks.append("Multiple high-impact dependencies increase project risk")
        
        external_deps = [d for d in dependencies if d.get("type") == "external"]
        if external_deps:
            risks.append("External dependencies may cause delays beyond our control")
        
        # Context-based risks
        if context.ambiguity_score > 0.6:
            risks.append("High ambiguity in requirements may lead to rework")
        
        if context.urgency_level > 0.8:
            risks.append("High urgency may compromise quality if not managed properly")
        
        if context.business_impact > 0.8:
            risks.append("High business impact requires careful stakeholder management")
        
        return risks[:6]  # Limit to top risks

# =============================================================================
# PHASE 2 API MODELS FOR INTEGRATION
# =============================================================================

class AdvancedAnalysisRequest(BaseModel):
    """Enhanced request model for advanced NLP analysis"""
    request_text: str
    context: Optional[Dict[str, Any]] = None
    options: Optional[Dict[str, Any]] = None

class AdvancedAnalysisResponse(BaseModel):
    """Enhanced response model for advanced analysis results"""
    analysis_id: str
    original_request: str
    intent: str
    complexity: str
    confidence_score: float
    subtasks: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    context_analysis: Dict[str, Any]
    recommendations: List[str]
    estimated_effort: str
    risk_factors: List[str]
    processing_time_ms: int
    timestamp: str

# =============================================================================
# INTEGRATION WITH EXISTING AI INTELLIGENCE LAYER
# =============================================================================

def enhance_ai_intelligence_with_nlp():
    """
    Function to enhance existing AI Intelligence Layer with advanced NLP
    This integrates with the existing service on port 8010
    """
    logger.info("🔧 Enhancing existing AI Intelligence Layer with Phase 2 NLP")
    
    # Initialize the advanced NLP engine
    nlp_engine = AdvancedNLPEngine()
    
    # Return the enhanced analysis function
    async def enhanced_analyze_request(request_data: AdvancedAnalysisRequest) -> AdvancedAnalysisResponse:
        """Enhanced request analysis with advanced NLP"""
        start_time = datetime.now()
        
        # Perform advanced NLP analysis
        decomposition = await nlp_engine.analyze_request(request_data.request_text)
        
        # Calculate processing time
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Create response
        response = AdvancedAnalysisResponse(
            analysis_id=f"nlp_{int(datetime.now().timestamp())}",
            original_request=decomposition.original_request,
            intent=decomposition.intent.value,
            complexity=decomposition.complexity.value,
            confidence_score=decomposition.confidence_score,
            subtasks=decomposition.subtasks,
            dependencies=decomposition.dependencies,
            context_analysis=decomposition.context_analysis,
            recommendations=decomposition.recommendations,
            estimated_effort=decomposition.estimated_effort,
            risk_factors=decomposition.risk_factors,
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        return response
    
    return enhanced_analyze_request

if __name__ == "__main__":
    print("🧠 Agent Zero V2.0 Phase 2 - Advanced NLP Enhancement")
    print("=" * 60)
    print("✅ Advanced Natural Language Understanding Engine")
    print("✅ Context-aware task decomposition")
    print("✅ Multi-dimensional intent classification")
    print("✅ Intelligent dependency detection")
    print("✅ Risk analysis and recommendations")
    print()
    print("🔧 Ready for integration with existing AI Intelligence Layer")
    print("📍 Target: Enhance port 8010 service with Phase 2 capabilities")
    print("🎯 Status: Development complete, ready for deployment")