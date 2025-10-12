#!/usr/bin/env python3
"""Agent Zero V2.0 - Simplified Working Version"""

import sys, os, sqlite3, json
from datetime import datetime
from typing import Dict, Any

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    print("Install: pip install fastapi uvicorn")
    sys.exit(1)

app = FastAPI(title="Agent Zero V2.0", version="2.0.0")

@app.get("/")
def root():
    return {"platform": "Agent Zero V2.0", "status": "operational", "version": "2.0.0"}

@app.get("/health")
def health():
    try:
        conn = sqlite3.connect("agent_zero.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchone()[0]
        conn.close()
        return {"status": "healthy", "tables": tables, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/agents")
def list_agents():
    try:
        conn = sqlite3.connect("agent_zero.db")
        cursor = conn.cursor()
        cursor.execute("SELECT agent_id, name, skills, status FROM agents")
        agents = [{"id": r[0], "name": r[1], "skills": r[2], "status": r[3]} for r in cursor.fetchall()]
        conn.close()
        return {"agents": agents, "count": len(agents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/projects")
def list_projects():
    try:
        conn = sqlite3.connect("agent_zero.db")
        cursor = conn.cursor()
        cursor.execute("SELECT project_id, name, description, status FROM projects")
        projects = [{"id": r[0], "name": r[1], "desc": r[2], "status": r[3]} for r in cursor.fetchall()]
        conn.close()
        return {"projects": projects, "count": len(projects)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/team/recommend")
def recommend_team(payload: dict):
    project = payload.get("project_name", "New Project")
    skills = payload.get("skills", [])
    
    try:
        conn = sqlite3.connect("agent_zero.db")
        cursor = conn.cursor()
        cursor.execute("SELECT agent_id, name, skills FROM agents WHERE status='available'")
        agents = cursor.fetchall()
        conn.close()
        
        recommendations = []
        for agent_id, name, skills_json in agents:
            agent_skills = json.loads(skills_json) if skills_json else {}
            score = sum(agent_skills.get(s, 0) for s in skills) / max(len(skills), 1)
            recommendations.append({"agent": agent_id, "name": name, "score": round(score, 2)})
        
        recommendations.sort(key=lambda x: x["score"], reverse=True)
        return {"project": project, "recommendations": recommendations[:3]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("🚀 Agent Zero V2.0 Starting...")
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
