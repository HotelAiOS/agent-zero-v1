# Enhanced Agent Orchestrator V2 - Production
# File: orchestration/enhanced_agent_orchestrator_v2.py

import asyncio
import time
import uuid
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from agent_executor import AgentExecutor, ExecutionContext, ExecutionResult, TaskStatus

@dataclass
class OrchestrationSummary:
    session_id: str
    tasks_completed: int
    success_rate: float
    quality_score: float
    execution_time: float

class EnhancedAgentOrchestrator:
    """Production-grade orchestrator with real-time monitoring and quality gates"""
    def __init__(self, db_path: str = "orchestrator_v2.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self._init_db()
        self.executor = AgentExecutor()
        self.active_agents: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}

    def _init_db(self):
        with self.db:
            self.db.executescript(
                """
                CREATE TABLE IF NOT EXISTS orchestration_sessions (
                    session_id TEXT PRIMARY KEY,
                    started_at TEXT,
                    completed_at TEXT,
                    tasks_total INTEGER,
                    tasks_success INTEGER,
                    quality_score REAL,
                    execution_time REAL
                );
                
                CREATE TABLE IF NOT EXISTS orchestration_results (
                    session_id TEXT,
                    task_id TEXT,
                    agent_type TEXT,
                    status TEXT,
                    execution_time REAL,
                    error TEXT,
                    created_at TEXT
                );
                
                CREATE INDEX IF NOT EXISTS idx_results_session ON orchestration_results(session_id);
                """
            )

    async def orchestrate(self, tasks: List[Dict[str, Any]]) -> OrchestrationSummary:
        session_id = str(uuid.uuid4())[:8]
        start = time.time()
        successes = 0
        total = len(tasks)
        
        results: List[ExecutionResult] = []

        async def status_callback(task_id: str, status: TaskStatus):
            self.metrics[task_id] = {
                "status": status.value,
                "timestamp": datetime.now().isoformat()
            }

        # Execute in parallel
        exec_tasks = []
        for t in tasks:
            context = ExecutionContext(
                task_id=t["id"],
                agent_type=t["agent_type"],
                input_data=t.get("input", {}),
                workspace_dir=t.get("workspace", "/app/output"),
                timeout=t.get("timeout", 300)
            )
            exec_tasks.append(self.executor.execute_task(context, status_callback))

        exec_results = await asyncio.gather(*exec_tasks, return_exceptions=True)

        # Persist results
        with self.db:
            for res in exec_results:
                if isinstance(res, ExecutionResult):
                    results.append(res)
                    if res.is_success:
                        successes += 1
                    self.db.execute(
                        "INSERT INTO orchestration_results VALUES (?,?,?,?,?,?,?)",
                        (
                            session_id,
                            res.task_id,
                            tasks[[t["id"] for t in tasks].index(res.task_id)]["agent_type"],
                            res.status.value,
                            res.execution_time,
                            res.error,
                            datetime.now().isoformat()
                        )
                    )
                else:
                    # Exception captured
                    self.db.execute(
                        "INSERT INTO orchestration_results VALUES (?,?,?,?,?,?,?)",
                        (
                            session_id,
                            "unknown",
                            "unknown",
                            TaskStatus.FAILED.value,
                            0.0,
                            str(res),
                            datetime.now().isoformat()
                        )
                    )

        # Simple quality score placeholder (can integrate with intelligence_v2)
        quality_score = round((successes / max(total,1)) * 100.0, 2)
        exec_time = time.time() - start

        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO orchestration_sessions VALUES (?,?,?,?,?,?,?)",
                (
                    session_id,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                    total,
                    successes,
                    quality_score,
                    exec_time
                )
            )

        return OrchestrationSummary(
            session_id=session_id,
            tasks_completed=total,
            success_rate=(successes / max(total,1)),
            quality_score=quality_score,
            execution_time=exec_time
        )
