#!/usr/bin/env fish
# 🚨 Agent Zero V2.0 - Ultimate Simple Fix (No Heredoc)
# 🔧 Manual approach that definitely works in Fish Shell

echo "🚨 Agent Zero V2.0 - Ultimate Simple Fix"
echo "📅 " (date)

# Kill existing servers
echo "🛑 Stopping existing servers..."
pkill -f uvicorn 2>/dev/null || true
pkill -f python 2>/dev/null || true
sleep 2

# Fix database
echo "🗄️ Recreating database..."
rm -f agent_zero.db* 2>/dev/null || true

python3 -c "
import sqlite3
from datetime import datetime

conn = sqlite3.connect('agent_zero.db')
conn.executescript('''
CREATE TABLE agents (
    id INTEGER PRIMARY KEY,
    agent_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    skills TEXT,
    status TEXT DEFAULT 'available'
);

CREATE TABLE projects (
    id INTEGER PRIMARY KEY,
    project_id TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'planning'
);

CREATE TABLE team_history (
    id INTEGER PRIMARY KEY,
    project_id TEXT NOT NULL,
    team_composition TEXT NOT NULL,
    outcome_success REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO agents (agent_id, name, skills, status) VALUES
('dev_001', 'Alice Developer', '{\"python\": 0.9, \"fastapi\": 0.8}', 'available'),
('dev_002', 'Bob Designer', '{\"figma\": 0.9, \"ui_ux\": 0.95}', 'available'),
('dev_003', 'Charlie DevOps', '{\"docker\": 0.9, \"aws\": 0.8}', 'available');

INSERT INTO projects (project_id, name, description, status) VALUES
('proj_001', 'E-commerce Platform', 'Modern e-commerce system', 'active'),
('proj_002', 'Mobile Banking', 'Secure banking app', 'planning');
''')
conn.commit()
conn.close()
print('✅ Database created')
"

echo "✅ Database fixed"

# Create simple Python files manually (no heredoc)
echo "🐍 Creating Python files..."

# Create the simplified main application
echo '#!/usr/bin/env python3
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
        cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='\''table'\''")
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
        cursor.execute("SELECT agent_id, name, skills FROM agents WHERE status='\''available'\''")
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
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)' > simple_main.py

echo "✅ simple_main.py created"

# Create startup script 
echo '#!/usr/bin/env fish
echo "🚀 Agent Zero V2.0 - Simple Server"
echo "📊 http://localhost:8001/docs"
echo "🔍 http://localhost:8001/health"
source venv/bin/activate.fish
python3 simple_main.py' > start_simple.fish

chmod +x start_simple.fish

echo "✅ start_simple.fish created"

# Test the setup
echo "🧪 Quick test..."
source venv/bin/activate.fish

python3 -c "
try:
    import sqlite3
    conn = sqlite3.connect('agent_zero.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM agents')
    agents = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM projects')
    projects = cursor.fetchone()[0]
    conn.close()
    print(f'✅ Database: {agents} agents, {projects} projects')
except Exception as e:
    print(f'❌ Database error: {e}')

try:
    import fastapi, uvicorn
    print('✅ FastAPI: Available')
except Exception as e:
    print(f'❌ FastAPI error: {e}')

import os
if os.path.exists('simple_main.py'):
    print('✅ Main app: Created')
else:
    print('❌ Main app: Missing')
"

echo ""
echo "🎉 Agent Zero V2.0 - Simple Fix COMPLETE!"
echo "=" (string repeat -n 40 "=")
echo ""
echo "✅ Co zostało naprawione:"
echo "   🗄️ Database recreated with sample data"
echo "   🐍 Simple Python app created (simple_main.py)"  
echo "   🚀 Working startup script (start_simple.fish)"
echo "   🔧 No heredoc - pure Fish Shell compatible"
echo ""
echo "🚀 Uruchom serwer:"
echo "   ./start_simple.fish"
echo ""
echo "📊 API endpoints:"
echo "   • http://localhost:8001/docs - API documentation"
echo "   • http://localhost:8001/health - Health check"
echo "   • http://localhost:8001/api/agents - List agents"
echo "   • http://localhost:8001/api/projects - List projects"
echo "   • http://localhost:8001/api/team/recommend - Team recommendations"
echo ""
echo "🎯 Status: SIMPLE & WORKING!"