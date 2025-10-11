#!/usr/bin/env python3
"""
📊 Agent Zero V1 - Monitoring Dashboard
=======================================
Real-time monitoring dla systemu zintegrowanego
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import sqlite3
from dataclasses import asdict

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

# Import głównego systemu
from integrated_system import ProductionIntegratedSystem, IntegratedEnhancedTaskDecomposer

app = FastAPI(title="Agent Zero V1 - Monitoring Dashboard")
templates = Jinja2Templates(directory="templates")

# Global instances
monitoring_system = ProductionIntegratedSystem()
task_decomposer = IntegratedEnhancedTaskDecomposer()

@app.on_event("startup")
async def startup():
    await monitoring_system.initialize()

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main monitoring dashboard"""

    # Pobierz ostatnie zadania
    recent_tasks = await task_decomposer.get_stored_tasks(20)

    # Statystyki
    stats = {
        "total_tasks": len(recent_tasks),
        "avg_confidence": sum(t.ai_reasoning.confidence_score for t in recent_tasks if t.ai_reasoning) / len(recent_tasks) if recent_tasks else 0,
        "high_confidence_tasks": len([t for t in recent_tasks if t.ai_reasoning and t.ai_reasoning.confidence_score > 90]),
        "total_hours": sum(t.estimated_hours for t in recent_tasks),
        "by_priority": {}
    }

    # Grupuj według priorytetu
    for task in recent_tasks:
        priority = task.priority.value
        if priority not in stats["by_priority"]:
            stats["by_priority"][priority] = 0
        stats["by_priority"][priority] += 1

    # System health
    health = {
        "neo4j": monitoring_system.neo4j_driver is not None,
        "redis": monitoring_system.redis_client is not None,
        "rabbitmq": monitoring_system.rabbitmq_connection is not None,
        "ai_engine": True
    }

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Agent Zero V1 - Monitoring Dashboard</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 20px; }}
            .stat-card {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .stat-value {{ font-size: 2em; font-weight: bold; color: #667eea; }}
            .stat-label {{ color: #666; margin-top: 5px; }}
            .health-status {{ display: flex; gap: 10px; flex-wrap: wrap; }}
            .health-item {{ padding: 10px 15px; border-radius: 5px; color: white; font-weight: bold; }}
            .healthy {{ background: #4CAF50; }}
            .unhealthy {{ background: #f44336; }}
            .tasks-section {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .task-item {{ border-left: 4px solid #667eea; padding: 15px; margin: 10px 0; background: #f9f9f9; border-radius: 5px; }}
            .task-title {{ font-weight: bold; color: #333; }}
            .task-meta {{ color: #666; font-size: 0.9em; margin-top: 5px; }}
            .confidence {{ color: #4CAF50; font-weight: bold; }}
            .priority-critical {{ border-left-color: #f44336; }}
            .priority-high {{ border-left-color: #ff9800; }}
            .priority-medium {{ border-left-color: #2196F3; }}
            .priority-low {{ border-left-color: #4CAF50; }}
        </style>
        <script>
            // Auto-refresh co 30 sekund
            setTimeout(() => location.reload(), 30000);
        </script>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Agent Zero V1 - Monitoring Dashboard</h1>
            <p>Production Integrated System with AI Intelligence Layer</p>
            <p><strong>Last Update:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </div>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{stats["total_tasks"]}</div>
                <div class="stat-label">Total Tasks</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["avg_confidence"]:.1f}%</div>
                <div class="stat-label">Average AI Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["high_confidence_tasks"]}</div>
                <div class="stat-label">High Confidence Tasks (>90%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats["total_hours"]:.1f}h</div>
                <div class="stat-label">Total Estimated Hours</div>
            </div>
        </div>

        <div class="tasks-section">
            <h2>System Health</h2>
            <div class="health-status">
                <div class="health-item {'healthy' if health['neo4j'] else 'unhealthy'}">
                    Neo4j: {'✅ Connected' if health['neo4j'] else '❌ Disconnected'}
                </div>
                <div class="health-item {'healthy' if health['redis'] else 'unhealthy'}">
                    Redis: {'✅ Connected' if health['redis'] else '❌ Disconnected'}
                </div>
                <div class="health-item {'healthy' if health['rabbitmq'] else 'unhealthy'}">
                    RabbitMQ: {'✅ Connected' if health['rabbitmq'] else '❌ Disconnected'}
                </div>
                <div class="health-item healthy">
                    AI Engine: ✅ Active
                </div>
            </div>
        </div>

        <div class="tasks-section">
            <h2>Recent Enhanced Tasks</h2>
    """

    for task in recent_tasks[:10]:
        priority_class = f"priority-{task.priority.value.lower()}"
        confidence = task.ai_reasoning.confidence_score if task.ai_reasoning else 0

        html_content += f"""
            <div class="task-item {priority_class}">
                <div class="task-title">{task.title}</div>
                <div class="task-meta">
                    Type: {task.task_type.value} | 
                    Priority: {task.priority.value} | 
                    Hours: {task.estimated_hours:.1f}h |
                    <span class="confidence">AI Confidence: {confidence:.1f}%</span>
                </div>
                <div style="color: #666; margin-top: 5px;">{task.description[:100]}...</div>
            </div>
        """

    html_content += """
        </div>
    </body>
    </html>
    """

    return html_content

@app.get("/api/stats")
async def get_stats():
    """API endpoint dla statystyk"""
    tasks = await task_decomposer.get_stored_tasks(100)

    return {
        "total_tasks": len(tasks),
        "avg_confidence": sum(t.ai_reasoning.confidence_score for t in tasks if t.ai_reasoning) / len(tasks) if tasks else 0,
        "by_type": {},
        "by_priority": {},
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    print("🖥️ Starting Agent Zero V1 Monitoring Dashboard...")
    print("📊 Dashboard będzie dostępny na: http://localhost:8080")

    uvicorn.run(
        "monitoring-dashboard:app",
        host="0.0.0.0", 
        port=8080,
        reload=True,
        log_level="info"
    )
