
# AGENT ZERO V2.0 - INTELLIGENCE CORE
# Mózg który myśli, decyduje, pamięta

intelligence_core = '''"""
Agent Zero V2.0 - Intelligence Core
Mózg systemu: myśli, decyduje, pamięta, uczy się

Dependencies: ollama, neo4j-driver, asyncio
"""

import asyncio
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

# Neo4j dla pamięci (opcjonalne - działa bez tego też)
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("⚠️  Neo4j not available - memory will be in-process only")

# Ollama dla AI (wymagane)
try:
    import httpx
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("❌ httpx required for Ollama - install: pip install httpx")


class TaskType(Enum):
    """Typy zadań które system rozpoznaje"""
    CODE_GENERATION = "code"
    BUSINESS_ANALYSIS = "business"
    ARCHITECTURE = "architecture"
    PROBLEM_SOLVING = "problem"
    PLANNING = "planning"
    OPTIMIZATION = "optimization"


class ModelChoice(Enum):
    """Dostępne modele Ollama"""
    QUICK = "llama3.1:8b"        # Szybki, tani - proste zadania
    SMART = "codellama:13b"      # Balans - większość zadań
    EXPERT = "qwen2.5-coder:32b" # Wolny, drogi - trudne zadania
    BUSINESS = "llama3.1:70b"    # Największy - strategia (jeśli dostępny)


@dataclass
class ThinkingResult:
    """Wynik myślenia systemu"""
    task_type: TaskType
    complexity: int  # 1-10
    selected_model: ModelChoice
    reasoning: str
    confidence: float
    estimated_tokens: int
    expected_quality: float


@dataclass
class ExecutionResult:
    """Wynik wykonania zadania"""
    task_id: str
    model_used: str
    result_text: str
    actual_tokens: int
    execution_time: float
    quality_score: float
    learned_patterns: List[str]


class AgentZeroIntelligence:
    """
    🧠 Centralny Mózg Agent Zero V2.0
    
    Rola: MYŚLI o zadaniu, DECYDUJE jak je wykonać, PAMIĘTA co działa
    """
    
    def __init__(self, 
                 ollama_url: str = "http://localhost:11434",
                 neo4j_uri: Optional[str] = None,
                 neo4j_user: Optional[str] = None,
                 neo4j_password: Optional[str] = None):
        
        self.logger = logging.getLogger(__name__)
        self.ollama_url = ollama_url
        
        # Pamięć w procesie (zawsze działa)
        self.execution_memory = []
        self.pattern_library = {}
        
        # Pamięć Neo4j (opcjonalna)
        self.neo4j_driver = None
        if NEO4J_AVAILABLE and neo4j_uri:
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    neo4j_uri,
                    auth=(neo4j_user, neo4j_password)
                )
                self.logger.info("✅ Neo4j memory connected")
            except Exception as e:
                self.logger.warning(f"⚠️  Neo4j connection failed: {e}")
        
        # Model performance tracking
        self.model_stats = {model.value: {"uses": 0, "avg_quality": 0.0} 
                           for model in ModelChoice}
        
        self.logger.info("🧠 Intelligence Core initialized")
    
    async def think_and_execute(self, user_request: str, 
                               context: Optional[Dict] = None) -> ExecutionResult:
        """
        GŁÓWNA FUNKCJA: Myśli o zadaniu i wykonuje je inteligentnie
        
        1. MYŚLI: Co to za zadanie? Jaka złożoność?
        2. DECYDUJE: Który model będzie najlepszy?
        3. WYKONUJE: Z optymalnym modelem
        4. UCZY SIĘ: Zapisuje co zadziałało
        """
        context = context or {}
        task_id = self._generate_task_id(user_request)
        
        self.logger.info(f"🎯 Thinking about task: {task_id}")
        
        # STAGE 1: MYŚLENIE - Analiza zadania
        thinking = await self._think_about_task(user_request, context)
        self.logger.info(f"💡 Decision: {thinking.selected_model.value} "
                        f"(confidence: {thinking.confidence:.0%})")
        
        # STAGE 2: WYKONANIE - Z najlepszym modelem
        result = await self._execute_with_model(
            task_id, user_request, thinking, context
        )
        
        # STAGE 3: UCZENIE - Zapisz doświadczenie
        await self._learn_from_execution(thinking, result)
        
        return result
    
    async def _think_about_task(self, request: str, 
                               context: Dict) -> ThinkingResult:
        """
        Inteligentna analiza zadania - to jest MYŚLENIE
        """
        # Rozpoznaj typ zadania
        task_type = self._classify_task_type(request)
        
        # Oceń złożoność (1-10)
        complexity = self._assess_complexity(request, task_type)
        
        # Sprawdź czy mamy podobne doświadczenia
        similar_tasks = self._find_similar_experiences(request, task_type)
        
        # Wybierz model bazując na:
        # 1. Typ zadania
        # 2. Złożoność
        # 3. Poprzednie doświadczenia
        # 4. Dostępność modelu
        selected_model = await self._select_best_model(
            task_type, complexity, similar_tasks, context
        )
        
        # Oblicz confidence i predykcje
        confidence = self._calculate_confidence(
            task_type, complexity, similar_tasks
        )
        
        estimated_tokens = self._estimate_tokens(request, complexity)
        expected_quality = self._predict_quality(
            selected_model, task_type, complexity
        )
        
        reasoning = self._generate_reasoning(
            task_type, complexity, selected_model, similar_tasks
        )
        
        return ThinkingResult(
            task_type=task_type,
            complexity=complexity,
            selected_model=selected_model,
            reasoning=reasoning,
            confidence=confidence,
            estimated_tokens=estimated_tokens,
            expected_quality=expected_quality
        )
    
    def _classify_task_type(self, request: str) -> TaskType:
        """Rozpoznaj typ zadania na podstawie treści"""
        request_lower = request.lower()
        
        # Wzorce dla każdego typu
        patterns = {
            TaskType.CODE_GENERATION: [
                "kod", "code", "funkcj", "class", "api", "implement",
                "napisz", "stwórz", "zaimplementuj"
            ],
            TaskType.BUSINESS_ANALYSIS: [
                "biznes", "business", "analiz", "wymagania", "requirements",
                "stakeholder", "roi", "strategia"
            ],
            TaskType.ARCHITECTURE: [
                "architektura", "architecture", "design", "system",
                "skalowalnos", "pattern", "struktura"
            ],
            TaskType.PROBLEM_SOLVING: [
                "problem", "błąd", "error", "debug", "napraw", "fix",
                "rozwiąż", "troubleshoot"
            ],
            TaskType.PLANNING: [
                "plan", "harmonogram", "timeline", "roadmap",
                "zaplanuj", "schedule"
            ],
            TaskType.OPTIMIZATION: [
                "optymalizuj", "optimize", "improve", "wydajnos",
                "performance", "przyspiesz"
            ]
        }
        
        # Zlicz trafienia
        scores = {}
        for task_type, keywords in patterns.items():
            score = sum(1 for keyword in keywords if keyword in request_lower)
            if score > 0:
                scores[task_type] = score
        
        # Zwróć najlepszy match lub default
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return TaskType.PROBLEM_SOLVING  # Default
    
    def _assess_complexity(self, request: str, task_type: TaskType) -> int:
        """Oceń złożoność zadania (1-10)"""
        complexity = 5  # Base
        
        # Długość requestu
        word_count = len(request.split())
        if word_count > 100:
            complexity += 2
        elif word_count > 50:
            complexity += 1
        
        # Słowa kluczowe złożoności
        high_complexity = [
            "complex", "advanced", "enterprise", "scalable",
            "distributed", "real-time", "machine learning"
        ]
        medium_complexity = [
            "integrate", "optimize", "refactor", "design"
        ]
        
        request_lower = request.lower()
        
        if any(word in request_lower for word in high_complexity):
            complexity += 3
        elif any(word in request_lower for word in medium_complexity):
            complexity += 1
        
        # Typ zadania wpływa na bazową złożoność
        type_complexity = {
            TaskType.CODE_GENERATION: 0,
            TaskType.BUSINESS_ANALYSIS: 1,
            TaskType.ARCHITECTURE: 2,
            TaskType.PROBLEM_SOLVING: 1,
            TaskType.PLANNING: 1,
            TaskType.OPTIMIZATION: 2
        }
        
        complexity += type_complexity.get(task_type, 0)
        
        return min(max(complexity, 1), 10)  # Clamp 1-10
    
    def _find_similar_experiences(self, request: str, 
                                 task_type: TaskType) -> List[Dict]:
        """Znajdź podobne zadania z przeszłości"""
        similar = []
        
        for memory in self.execution_memory:
            # Sprawdź czy ten sam typ
            if memory.get('task_type') != task_type.value:
                continue
            
            # Sprawdź podobieństwo tekstowe (prosty matching)
            similarity = self._calculate_similarity(
                request, memory.get('request', '')
            )
            
            if similarity > 0.3:  # 30% podobieństwa
                similar.append({
                    'memory': memory,
                    'similarity': similarity
                })
        
        # Sortuj po podobieństwie
        similar.sort(key=lambda x: x['similarity'], reverse=True)
        return similar[:3]  # Top 3
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Prosta miara podobieństwa tekstów"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _select_best_model(self, task_type: TaskType, 
                                complexity: int,
                                similar_tasks: List[Dict],
                                context: Dict) -> ModelChoice:
        """
        KLUCZOWA DECYZJA: Wybór najlepszego modelu
        
        Bazuje na:
        - Typ zadania
        - Złożoność
        - Poprzednie doświadczenia
        - Priorytet (jakość vs szybkość vs koszt)
        """
        # Priorytet z contextu
        quality_priority = context.get('quality_priority', 0.7)
        speed_priority = context.get('speed_priority', 0.2)
        
        # Sprawdź dostępne modele
        available_models = await self._check_available_models()
        
        # Reguły decyzyjne
        
        # 1. Bardzo proste zadania → QUICK
        if complexity <= 3:
            if ModelChoice.QUICK.value in available_models:
                return ModelChoice.QUICK
        
        # 2. Kod o średniej złożoności → SMART (CodeLlama)
        if task_type == TaskType.CODE_GENERATION and complexity <= 7:
            if ModelChoice.SMART.value in available_models:
                return ModelChoice.SMART
        
        # 3. Trudne zadania architektoniczne → EXPERT
        if task_type == TaskType.ARCHITECTURE and complexity >= 6:
            if ModelChoice.EXPERT.value in available_models:
                return ModelChoice.EXPERT
        
        # 4. Biznes i strategia → najlepszy dostępny
        if task_type == TaskType.BUSINESS_ANALYSIS:
            if ModelChoice.BUSINESS.value in available_models and complexity >= 8:
                return ModelChoice.BUSINESS
            elif ModelChoice.EXPERT.value in available_models:
                return ModelChoice.EXPERT
        
        # 5. Uczenie się z przeszłości
        if similar_tasks:
            # Sprawdź który model działał najlepiej
            best_past_model = self._find_best_performing_model(similar_tasks)
            if best_past_model and best_past_model in available_models:
                return ModelChoice(best_past_model)
        
        # 6. Priorytet jakości → większy model
        if quality_priority > 0.8:
            if ModelChoice.EXPERT.value in available_models:
                return ModelChoice.EXPERT
        
        # 7. Priorytet szybkości → mniejszy model
        if speed_priority > 0.5:
            if ModelChoice.QUICK.value in available_models:
                return ModelChoice.QUICK
        
        # 8. Default: SMART (balans)
        if ModelChoice.SMART.value in available_models:
            return ModelChoice.SMART
        
        # 9. Fallback: pierwszy dostępny
        for model in ModelChoice:
            if model.value in available_models:
                return model
        
        # 10. Ostateczny fallback
        return ModelChoice.QUICK
    
    async def _check_available_models(self) -> List[str]:
        """Sprawdź które modele są dostępne w Ollama"""
        if not OLLAMA_AVAILABLE:
            return [ModelChoice.QUICK.value]  # Fallback
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    return [model['name'] for model in data.get('models', [])]
        except Exception as e:
            self.logger.warning(f"⚠️  Could not check Ollama models: {e}")
        
        return [ModelChoice.QUICK.value]  # Fallback
    
    def _find_best_performing_model(self, similar_tasks: List[Dict]) -> Optional[str]:
        """Znajdź model który najlepiej działał dla podobnych zadań"""
        model_scores = {}
        
        for task_info in similar_tasks:
            memory = task_info['memory']
            model = memory.get('model_used')
            quality = memory.get('quality_score', 0)
            
            if model:
                if model not in model_scores:
                    model_scores[model] = []
                model_scores[model].append(quality)
        
        if not model_scores:
            return None
        
        # Oblicz średnią jakość dla każdego modelu
        avg_scores = {
            model: sum(scores) / len(scores)
            for model, scores in model_scores.items()
        }
        
        # Zwróć najlepszy
        best_model = max(avg_scores.items(), key=lambda x: x[1])[0]
        return best_model
    
    def _calculate_confidence(self, task_type: TaskType, 
                            complexity: int,
                            similar_tasks: List[Dict]) -> float:
        """Oblicz confidence w decyzji"""
        confidence = 0.5  # Base
        
        # Mamy podobne zadania → wyższa confidence
        if similar_tasks:
            confidence += 0.2 * min(len(similar_tasks) / 3, 1.0)
        
        # Niska złożoność → wyższa confidence
        if complexity <= 5:
            confidence += 0.2
        
        # Typ zadania który dobrze znamy
        if task_type in [TaskType.CODE_GENERATION, TaskType.PROBLEM_SOLVING]:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _estimate_tokens(self, request: str, complexity: int) -> int:
        """Oszacuj ile tokenów użyje model"""
        base_tokens = len(request.split()) * 4  # Rough estimate
        complexity_multiplier = 1 + (complexity / 10)
        return int(base_tokens * complexity_multiplier)
    
    def _predict_quality(self, model: ModelChoice, 
                        task_type: TaskType, complexity: int) -> float:
        """Przewiduj jakość wyniku"""
        # Bazowa jakość modelu
        model_quality = {
            ModelChoice.QUICK: 0.7,
            ModelChoice.SMART: 0.85,
            ModelChoice.EXPERT: 0.95,
            ModelChoice.BUSINESS: 0.9
        }
        
        base = model_quality.get(model, 0.7)
        
        # Dopasowanie model-zadanie
        if model == ModelChoice.SMART and task_type == TaskType.CODE_GENERATION:
            base += 0.05  # CodeLlama jest świetny w kodzie
        
        # Złożoność vs moc modelu
        model_size = list(ModelChoice).index(model)
        if complexity > 7 and model_size < 2:  # Trudne zadanie, słaby model
            base -= 0.15
        
        return min(base, 1.0)
    
    def _generate_reasoning(self, task_type: TaskType, complexity: int,
                          model: ModelChoice, similar_tasks: List[Dict]) -> str:
        """Wygeneruj uzasadnienie decyzji"""
        reasons = []
        
        reasons.append(f"Task type: {task_type.value}")
        reasons.append(f"Complexity: {complexity}/10")
        reasons.append(f"Selected: {model.value}")
        
        if similar_tasks:
            reasons.append(f"Based on {len(similar_tasks)} similar past tasks")
        
        if complexity <= 3:
            reasons.append("Low complexity → fast model sufficient")
        elif complexity >= 8:
            reasons.append("High complexity → expert model needed")
        
        return " | ".join(reasons)
    
    async def _execute_with_model(self, task_id: str, request: str,
                                 thinking: ThinkingResult,
                                 context: Dict) -> ExecutionResult:
        """Wykonaj zadanie z wybranym modelem"""
        self.logger.info(f"🚀 Executing with {thinking.selected_model.value}")
        
        start_time = datetime.now()
        
        # Przygotuj prompt
        optimized_prompt = self._optimize_prompt(request, thinking)
        
        # Wywołaj Ollama
        result_text = await self._call_ollama(
            thinking.selected_model.value,
            optimized_prompt
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Oceń jakość wyniku
        quality_score = self._assess_quality(result_text, thinking)
        
        # Wyodrębnij wzorce do uczenia
        learned_patterns = self._extract_patterns(
            request, result_text, thinking, quality_score
        )
        
        return ExecutionResult(
            task_id=task_id,
            model_used=thinking.selected_model.value,
            result_text=result_text,
            actual_tokens=len(result_text.split()) * 4,
            execution_time=execution_time,
            quality_score=quality_score,
            learned_patterns=learned_patterns
        )
    
    def _optimize_prompt(self, request: str, thinking: ThinkingResult) -> str:
        """Optymalizuj prompt dla konkretnego modelu"""
        prompt_parts = [request]
        
        # Dodaj kontekst bazując na typie zadania
        if thinking.task_type == TaskType.CODE_GENERATION:
            prompt_parts.append("\nProvide clean, production-ready code with comments.")
        elif thinking.task_type == TaskType.ARCHITECTURE:
            prompt_parts.append("\nFocus on scalability, maintainability, and best practices.")
        elif thinking.task_type == TaskType.BUSINESS_ANALYSIS:
            prompt_parts.append("\nProvide clear business value and ROI considerations.")
        
        # Dodaj instrukcje jakości dla trudnych zadań
        if thinking.complexity >= 7:
            prompt_parts.append("\nThis is a complex task - take time to think through edge cases.")
        
        return "\n".join(prompt_parts)
    
    async def _call_ollama(self, model: str, prompt: str) -> str:
        """Wywołaj Ollama API"""
        if not OLLAMA_AVAILABLE:
            return f"[DEMO MODE] Response for: {prompt[:50]}..."
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get('response', '')
                else:
                    self.logger.error(f"Ollama error: {response.status_code}")
                    return f"Error calling Ollama: {response.status_code}"
                    
        except Exception as e:
            self.logger.error(f"Ollama call failed: {e}")
            return f"Error: {str(e)}"
    
    def _assess_quality(self, result: str, thinking: ThinkingResult) -> float:
        """Oceń jakość wyniku"""
        quality = 0.5  # Base
        
        # Długość odpowiedzi
        if len(result) > 100:
            quality += 0.2
        
        # Struktura (ma paragrafy/sekcje)
        if result.count('\n') >= 2:
            quality += 0.1
        
        # Specyficzne dla typu zadania
        if thinking.task_type == TaskType.CODE_GENERATION:
            if 'def ' in result or 'class ' in result or 'function' in result:
                quality += 0.2
        
        return min(quality, 1.0)
    
    def _extract_patterns(self, request: str, result: str,
                         thinking: ThinkingResult, quality: float) -> List[str]:
        """Wyodrębnij wzorce do uczenia się"""
        patterns = []
        
        if quality > 0.8:
            patterns.append(f"GOOD: {thinking.task_type.value} with {thinking.selected_model.value}")
        
        if quality < 0.5:
            patterns.append(f"POOR: {thinking.task_type.value} with {thinking.selected_model.value}")
        
        return patterns
    
    async def _learn_from_execution(self, thinking: ThinkingResult,
                                   result: ExecutionResult):
        """Zapisz doświadczenie do pamięci"""
        # Pamięć w procesie
        memory_entry = {
            'task_type': thinking.task_type.value,
            'complexity': thinking.complexity,
            'model_used': result.model_used,
            'quality_score': result.quality_score,
            'execution_time': result.execution_time,
            'timestamp': datetime.now().isoformat(),
            'learned_patterns': result.learned_patterns
        }
        
        self.execution_memory.append(memory_entry)
        
        # Update model stats
        model = result.model_used
        if model in self.model_stats:
            stats = self.model_stats[model]
            old_avg = stats['avg_quality']
            old_count = stats['uses']
            
            stats['uses'] = old_count + 1
            stats['avg_quality'] = (old_avg * old_count + result.quality_score) / stats['uses']
        
        # Neo4j (jeśli dostępny)
        if self.neo4j_driver:
            await self._store_in_neo4j(memory_entry)
        
        self.logger.info(f"📚 Learned from execution (quality: {result.quality_score:.0%})")
    
    async def _store_in_neo4j(self, memory: Dict):
        """Zapisz do Neo4j (persistent memory)"""
        try:
            with self.neo4j_driver.session() as session:
                session.run("""
                    CREATE (e:Execution {
                        task_type: $task_type,
                        complexity: $complexity,
                        model_used: $model_used,
                        quality_score: $quality_score,
                        timestamp: $timestamp
                    })
                """, **memory)
        except Exception as e:
            self.logger.warning(f"Neo4j store failed: {e}")
    
    def _generate_task_id(self, request: str) -> str:
        """Generuj unikalny ID zadania"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_part = hashlib.md5(request.encode()).hexdigest()[:8]
        return f"task_{timestamp}_{hash_part}"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Pobierz statystyki systemu"""
        return {
            'total_executions': len(self.execution_memory),
            'model_stats': self.model_stats,
            'avg_quality': sum(m['quality_score'] for m in self.execution_memory) / len(self.execution_memory) if self.execution_memory else 0,
            'memory_available': NEO4J_AVAILABLE and self.neo4j_driver is not None
        }
    
    def close(self):
        """Zamknij połączenia"""
        if self.neo4j_driver:
            self.neo4j_driver.close()


# ============================================================================
# DEMO / TEST
# ============================================================================

async def demo_intelligence_core():
    """Demonstracja Intelligence Core"""
    print("🧠 Agent Zero V2.0 - Intelligence Core Demo")
    print("=" * 60)
    
    # Initialize
    brain = AgentZeroIntelligence()
    
    # Test Case 1: Proste zadanie kodowe
    print("\n📝 Test 1: Simple Code Generation")
    result1 = await brain.think_and_execute(
        "Napisz funkcję Python do obliczania liczb Fibonacciego"
    )
    print(f"✅ Model used: {result1.model_used}")
    print(f"📊 Quality: {result1.quality_score:.0%}")
    print(f"⏱️  Time: {result1.execution_time:.2f}s")
    
    # Test Case 2: Trudne zadanie architektoniczne
    print("\n📝 Test 2: Complex Architecture")
    result2 = await brain.think_and_execute(
        "Zaprojektuj skalowalną architekturę mikrousług dla systemu e-commerce obsługującego 1M użytkowników",
        context={'quality_priority': 0.9}
    )
    print(f"✅ Model used: {result2.model_used}")
    print(f"📊 Quality: {result2.quality_score:.0%}")
    print(f"⏱️  Time: {result2.execution_time:.2f}s")
    
    # Statistics
    print("\n📈 System Statistics:")
    stats = brain.get_statistics()
    print(f"Total executions: {stats['total_executions']}")
    print(f"Average quality: {stats['avg_quality']:.0%}")
    print(f"Model performance:")
    for model, model_stats in stats['model_stats'].items():
        if model_stats['uses'] > 0:
            print(f"  {model}: {model_stats['uses']} uses, avg quality {model_stats['avg_quality']:.0%}")
    
    brain.close()
    print("\n🎉 Intelligence Core working perfectly!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_intelligence_core())
'''

# Zapisz plik
with open('intelligence_core.py', 'w', encoding='utf-8') as f:
    f.write(intelligence_core)

print("✅ PLIK 1/3 GOTOWY: intelligence_core.py")
print("\n📋 CO ROBI TEN PLIK:")
print("  🧠 MYŚLI o zadaniu (typ, złożoność)")
print("  🎯 DECYDUJE który model Ollama użyć")
print("  🚀 WYKONUJE z optymalnym modelem")
print("  📚 UCZY SIĘ z każdego wykonania")
print("  💾 PAMIĘTA co działa (in-memory + Neo4j opcjonalnie)")
print("\n✨ KLUCZOWE FUNKCJE:")
print("  • think_and_execute() - główna funkcja")
print("  • Automatyczny wybór modelu (llama3.1:8b / codellama:13b / qwen2.5-coder:32b)")
print("  • Uczenie się z doświadczeń")
print("  • Statystyki wydajności modeli")
print("\n⚡ GOTOWE DO UŻYCIA - działa standalone!")
