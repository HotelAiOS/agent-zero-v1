"""
Knowledge Graph - Pamięć i uczenie się systemu Agent Zero

System zapisuje każdą akcję, decyzję i wygenerowany kod w Neo4j.
Agenci mogą pytać graf o podobne zadania i uczyć się z przeszłości.

Zgodne z architekturą z dokumentacji - fundament inteligencji systemu.
"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from neo4j import GraphDatabase
import json

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Graf wiedzy oparty o Neo4j.
    
    Zapisuje:
    - Projekty i ich strukturę
    - Wygenerowany kod przez agentów
    - Decyzje architektoniczne
    - Wzorce i learning
    
    Przykład użycia:
        >>> kg = KnowledgeGraph()
        >>> await kg.connect()
        >>> await kg.record_code_generation(
        ...     agent_id="backend_001",
        ...     task="user registration",
        ...     code="...",
        ...     model="deepseek-coder"
        ... )
    """
    
    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        username: str = "neo4j",
        password: str = "agent-pass"
    ):
        """
        Inicjalizacja Knowledge Graph.
        
        Args:
            uri: Neo4j URI
            username: Nazwa użytkownika
            password: Hasło
        """
        self.uri = uri
        self.username = username
        self.password = password
        self.driver = None
        
    async def connect(self):
        """Połącz z Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            
            # Sprawdź połączenie
            with self.driver.session() as session:
                session.run("RETURN 1")
            
            logger.info("✅ Knowledge Graph połączony z Neo4j")
            
            # Stwórz indeksy dla performance
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"❌ Błąd połączenia z Neo4j: {e}")
            raise
    
    async def _create_indexes(self):
        """Stwórz indeksy dla szybkiego wyszukiwania"""
        with self.driver.session() as session:
            # Indeks na ID projektów
            session.run(
                "CREATE INDEX project_id_index IF NOT EXISTS "
                "FOR (p:Project) ON (p.project_id)"
            )
            
            # Indeks na ID agentów
            session.run(
                "CREATE INDEX agent_id_index IF NOT EXISTS "
                "FOR (a:Agent) ON (a.agent_id)"
            )
            
            # Indeks na typy zadań
            session.run(
                "CREATE INDEX task_type_index IF NOT EXISTS "
                "FOR (t:Task) ON (t.task_type)"
            )
            
        logger.info("✅ Indeksy Neo4j utworzone")
    
    async def record_code_generation(
        self,
        agent_id: str,
        task_description: str,
        generated_code: str,
        model_used: str,
        processing_time: float,
        success: bool,
        context: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None
    ) -> str:
        """
        Zapisz wygenerowany kod w grafie wiedzy.
        
        Args:
            agent_id: ID agenta który generował
            task_description: Opis zadania
            generated_code: Wygenerowany kod
            model_used: Model AI użyty (np. "deepseek-coder:33b")
            processing_time: Czas generacji (sekundy)
            success: Czy sukces
            context: Dodatkowy kontekst
            project_id: ID projektu (opcjonalnie)
            
        Returns:
            task_id: Unikalny ID zadania w grafie
            
        Przykład:
            >>> task_id = await kg.record_code_generation(
            ...     agent_id="backend_001",
            ...     task_description="Create user registration endpoint",
            ...     generated_code="from fastapi import...",
            ...     model_used="deepseek-coder:33b",
            ...     processing_time=1211.95,
            ...     success=True
            ... )
        """
        task_id = f"task_{datetime.now().timestamp()}"
        
        with self.driver.session() as session:
            # Cypher query - tworzy węzły i relacje
            query = """
            // Znajdź lub stwórz Agenta
            MERGE (agent:Agent {agent_id: $agent_id})
            ON CREATE SET agent.created_at = datetime()
            
            // Znajdź lub stwórz Model
            MERGE (model:AIModel {name: $model_used})
            ON CREATE SET model.created_at = datetime()
            
            // Stwórz Task
            CREATE (task:Task {
                task_id: $task_id,
                description: $task_description,
                task_type: $task_type,
                success: $success,
                processing_time: $processing_time,
                created_at: datetime(),
                timestamp: $timestamp
            })
            
            // Stwórz Code Node
            CREATE (code:GeneratedCode {
                code_id: $task_id + '_code',
                content: $generated_code,
                language: $language,
                lines: $code_lines,
                created_at: datetime()
            })
            
            // Relacje
            CREATE (agent)-[:GENERATED]->(task)
            CREATE (task)-[:USED_MODEL]->(model)
            CREATE (task)-[:PRODUCED]->(code)
            
            // Opcjonalnie - projekt
            WITH task, agent
            OPTIONAL MATCH (project:Project {project_id: $project_id})
            FOREACH (p IN CASE WHEN project IS NOT NULL THEN [project] ELSE [] END |
                CREATE (p)-[:HAS_TASK]->(task)
            )
            
            RETURN task.task_id as task_id
            """
            
            # Wykryj język programowania
            language = self._detect_language(generated_code)
            
            # Parametry
            params = {
                "agent_id": agent_id,
                "task_id": task_id,
                "task_description": task_description,
                "task_type": self._extract_task_type(task_description),
                "model_used": model_used,
                "generated_code": generated_code,
                "language": language,
                "code_lines": len(generated_code.split('\n')),
                "processing_time": processing_time,
                "success": success,
                "timestamp": datetime.now().isoformat(),
                "project_id": project_id
            }
            
            # Wykonaj
            result = session.run(query, params)
            created_task_id = result.single()["task_id"]
            
            logger.info(f"✅ Kod zapisany w Knowledge Graph: {created_task_id}")
            
            return created_task_id
    
    async def find_similar_tasks(
        self,
        task_description: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Znajdź podobne zadania z przeszłości.
        
        Args:
            task_description: Opis zadania
            limit: Max wyników
            
        Returns:
            Lista podobnych zadań z kodem i kontekstem
            
        Przykład:
            >>> similar = await kg.find_similar_tasks("user registration")
            >>> for task in similar:
            ...     print(f"Podobne: {task['description']}")
            ...     print(f"Kod: {task['code'][:100]}...")
        """
        task_type = self._extract_task_type(task_description)
        
        with self.driver.session() as session:
            query = """
            MATCH (task:Task {task_type: $task_type, success: true})
            MATCH (task)-[:PRODUCED]->(code:GeneratedCode)
            MATCH (agent:Agent)-[:GENERATED]->(task)
            MATCH (task)-[:USED_MODEL]->(model:AIModel)
            
            RETURN 
                task.task_id as task_id,
                task.description as description,
                task.processing_time as processing_time,
                code.content as code,
                code.language as language,
                agent.agent_id as agent_id,
                model.name as model_used,
                task.created_at as created_at
            
            ORDER BY task.created_at DESC
            LIMIT $limit
            """
            
            result = session.run(query, {
                "task_type": task_type,
                "limit": limit
            })
            
            similar_tasks = []
            for record in result:
                similar_tasks.append({
                    "task_id": record["task_id"],
                    "description": record["description"],
                    "code": record["code"],
                    "language": record["language"],
                    "agent_id": record["agent_id"],
                    "model_used": record["model_used"],
                    "processing_time": record["processing_time"],
                    "created_at": str(record["created_at"])
                })
            
            logger.info(f"🔍 Znaleziono {len(similar_tasks)} podobnych zadań")
            
            return similar_tasks
    
    async def get_agent_stats(self, agent_id: str) -> Dict[str, Any]:
        """
        Pobierz statystyki agenta.
        
        Returns:
            Dict ze statystykami: total_tasks, success_rate, avg_time, etc.
        """
        with self.driver.session() as session:
            query = """
            MATCH (agent:Agent {agent_id: $agent_id})-[:GENERATED]->(task:Task)
            OPTIONAL MATCH (task)-[:PRODUCED]->(code:GeneratedCode)
            
            RETURN 
                count(task) as total_tasks,
                sum(CASE WHEN task.success THEN 1 ELSE 0 END) as successful_tasks,
                avg(task.processing_time) as avg_processing_time,
                sum(code.lines) as total_code_lines
            """
            
            result = session.run(query, {"agent_id": agent_id})
            record = result.single()
            
            if not record:
                return {"agent_id": agent_id, "total_tasks": 0}
            
            total = record["total_tasks"]
            successful = record["successful_tasks"]
            
            return {
                "agent_id": agent_id,
                "total_tasks": total,
                "successful_tasks": successful,
                "success_rate": (successful / total * 100) if total > 0 else 0,
                "avg_processing_time": record["avg_processing_time"],
                "total_code_lines": record["total_code_lines"]
            }
    
    def _extract_task_type(self, description: str) -> str:
        """Wyciągnij typ zadania z opisu (prosty classifier)"""
        desc_lower = description.lower()
        
        if "registration" in desc_lower or "register" in desc_lower:
            return "user_registration"
        elif "authentication" in desc_lower or "login" in desc_lower:
            return "authentication"
        elif "api" in desc_lower or "endpoint" in desc_lower:
            return "api_endpoint"
        elif "database" in desc_lower or "schema" in desc_lower:
            return "database"
        elif "frontend" in desc_lower or "ui" in desc_lower:
            return "frontend"
        else:
            return "general"
    
    def _detect_language(self, code: str) -> str:
        """Wykryj język programowania z kodu"""
        code_lower = code.lower()
        
        if "from fastapi" in code_lower or "import fastapi" in code_lower:
            return "python_fastapi"
        elif "def " in code_lower or "class " in code_lower:
            return "python"
        elif "const " in code_lower or "function" in code_lower:
            return "javascript"
        elif "interface " in code_lower or ": React.FC" in code:
            return "typescript_react"
        else:
            return "unknown"
    
    async def close(self):
        """Zamknij połączenie"""
        if self.driver:
            self.driver.close()
            logger.info("🔌 Knowledge Graph disconnected")


# Singleton instance
knowledge_graph = KnowledgeGraph()
