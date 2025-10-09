#!/usr/bin/env bash
set -e

REPO_DIR="agent-zero-v1"

# 1. Sklonuj repo, jeśli nie ma
if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/HotelAiOS/agent-zero-v1.git "$REPO_DIR"
fi

cd "$REPO_DIR"

# 2. Utwórz i aktywuj virtualenv
python3 -m venv .venv
# aktywacja w tym skrypcie
source .venv/bin/activate

# 3. Zaktualizuj pip i zainstaluj driver Neo4j
pip install --upgrade pip
pip install neo4j-driver

# 4. Wygeneruj intelligence_core.py
mkdir -p app/ai
cat > app/ai/intelligence_core.py << 'EOF'
import os
import asyncio
import logging
from neo4j import GraphDatabase

class PatternMiningEngine:
    def optimize_model_choice(self, model: str, task_type: str) -> str:
        return model

class AgentZeroIntelligence:
    """
    Centralny AI Brain:
    – rozumie typ zadania
    – wybiera model Ollama
    – wykonuje prompt
    – zapisuje wynik w Neo4j
    """

    def __init__(self, neo4j_uri=None, neo4j_user=None, neo4j_password=None):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password)
        )
        self.models = {
            "quick":  "llama3.1:8b",
            "smart":  "codellama:13b",
            "expert": "qwen2.5-coder:32b"
        }
        self.pattern_engine = PatternMiningEngine()
        self.last_model = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    async def think_and_execute(self, user_request: str) -> str:
        task_type = self._analyze_request(user_request)
        model = self.pattern_engine.optimize_model_choice(
            self.models[task_type], task_type
        )
        self.last_model = model
        self.logger.info(f"[Brain] Selected model: {model}")

        result = await self._execute_with_ollama(model, user_request)
        self._learn_from_result(task_type, model, result)
        return result

    def _analyze_request(self, text: str) -> str:
        length = len(text)
        if length < 50:
            return "quick"
        if length < 200:
            return "smart"
        return "expert"

    async def _execute_with_ollama(self, model: str, prompt: str) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                "ollama", "run", model, "--prompt", prompt,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            out, err = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(err.decode().strip())
            return out.decode().strip()
        except FileNotFoundError:
            self.logger.warning("Ollama CLI nie znaleziono – zwracam stub")
            return f"[stub:{model}] {prompt}"

    def _learn_from_result(self, task_type: str, model: str, result: str):
        try:
            with self.driver.session() as session:
                session.write_transaction(
                    self._create_record, task_type, model, result
                )
            self.logger.info("[Brain] Zapisano wynik w Neo4j")
        except Exception as e:
            self.logger.error(f"[Brain] Błąd zapisu w Neo4j: {e}")

    @staticmethod
    def _create_record(tx, task_type, model, result):
        tx.run(
            """
            MERGE (t:TaskType {name: $task_type})
            MERGE (m:Model {name: $model})
            CREATE (e:Execution {result: $result, timestamp: datetime()})
            MERGE (e)-[:OF_TASK]->(t)
            MERGE (e)-[:USED_MODEL]->(m)
            """,
            task_type=task_type, model=model, result=result
        )

if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "Napisz funkcję do obliczania fibonacci w Pythonie."
    brain = AgentZeroIntelligence()
    result = asyncio.run(brain.think_and_execute(text))
    print("\n=== RESULT ===\n", result)
EOF

# 5. Szybki test
echo "=== URUCHAMIAM TEST ==="
python app/ai/intelligence_core.py "Testuję mózg Agent Zero"

# 6. Informacja końcowa
echo "✓ Gotowe. Aby pracować dalej, w nowym terminalu włącz virtualenv:"
echo "  cd $REPO_DIR && source .venv/bin/activate"
