"""
Test AI-Powered Agents z Knowledge Graph + Workspace Integration - KOMPLETNY

Ten test pokazuje pełny workflow:
1. Agent generuje kod z AI Brain (TWÓJ system)
2. Kod automatycznie zapisuje się do Neo4j Knowledge Graph
3. Kod zapisuje się do fizycznego workspace'u na dysku
4. System może znaleźć podobne zadania z przeszłości
5. Agent uczy się z historii i tworzy rzeczywiste projekty
"""
import asyncio
import logging
import sys
import os
import time
from pathlib import Path

# Import TWOJEGO systemu
from intelligent_agent import IntelligentAgent

# Import Knowledge Graph
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'knowledge'))
from knowledge_graph import knowledge_graph

# Import Workspace Manager
sys.path.append(os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

# WORKSPACE MANAGER - wbudowany w test
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

@dataclass
class WorkspaceConfig:
    """Konfiguracja workspace'u"""
    workspace_id: str
    name: str
    project_type: str
    base_path: Path
    agents: List[str]
    tech_stack: List[str]
    created_at: datetime
    status: str

@dataclass  
class CodeFile:
    """Reprezentacja pliku kodu"""
    file_id: str
    workspace_id: str
    agent_id: str
    task_id: str
    filename: str
    filepath: Path
    language: str
    content: str
    created_at: datetime

class SimpleWorkspaceManager:
    """Uproszczony Workspace Manager dla testu"""
    
    def __init__(self, base_dir: str = "./test_workspaces"):
        self.base_path = Path(base_dir)
        self.base_path.mkdir(exist_ok=True)
        self.workspaces: Dict[str, WorkspaceConfig] = {}
        self.code_files: Dict[str, CodeFile] = {}
        
    async def create_workspace(self, name: str, project_type: str, tech_stack: List[str]) -> str:
        """Utwórz workspace"""
        workspace_id = f"ws-{uuid.uuid4().hex[:8]}"
        workspace_path = self.base_path / workspace_id
        workspace_path.mkdir(exist_ok=True)
        
        # Struktura katalogów
        for dir_name in ["src", "tests", "docs", "config"]:
            (workspace_path / dir_name).mkdir(exist_ok=True)
        
        config = WorkspaceConfig(
            workspace_id=workspace_id,
            name=name,
            project_type=project_type,
            base_path=workspace_path,
            agents=[],
            tech_stack=tech_stack,
            created_at=datetime.now(),
            status="active"
        )
        
        self.workspaces[workspace_id] = config
        
        # Zapisz config
        config_file = workspace_path / "workspace_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "workspace_id": workspace_id,
                "name": name,
                "project_type": project_type,
                "base_path": str(workspace_path),
                "tech_stack": tech_stack,
                "created_at": config.created_at.isoformat(),
                "status": "active"
            }, f, indent=2)
        
        print(f"   ✅ Created workspace: {name} ({workspace_id})")
        return workspace_id
    
    async def save_generated_code(self, workspace_id: str, agent_id: str, 
                                 task_id: str, filename: str, content: str,
                                 language: str = "python") -> str:
        """Zapisz kod do workspace'u"""
        
        workspace = self.workspaces[workspace_id]
        
        # Determine file path
        if language == "python":
            if "test" in filename.lower():
                filepath = workspace.base_path / "tests" / filename
            else:
                filepath = workspace.base_path / "src" / filename
        else:
            filepath = workspace.base_path / "src" / filename
        
        # Create directory
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create record
        file_id = f"file-{uuid.uuid4().hex[:8]}"
        code_file = CodeFile(
            file_id=file_id,
            workspace_id=workspace_id,
            agent_id=agent_id,
            task_id=task_id,
            filename=filename,
            filepath=filepath,
            language=language,
            content=content,
            created_at=datetime.now()
        )
        
        self.code_files[file_id] = code_file
        
        print(f"   📁 Saved to workspace: {filename} ({len(content)} chars)")
        print(f"      Path: {filepath}")
        
        return file_id
    
    async def create_fastapi_structure(self, workspace_id: str):
        """Utwórz strukturę FastAPI"""
        workspace = self.workspaces[workspace_id]
        
        files_to_create = {
            "src/main.py": '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agent Zero Generated API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Agent Zero Generated API", "status": "online"}

@app.get("/health")
async def health():
    return {"status": "healthy", "generated_by": "agent_zero"}
''',
            "requirements.txt": '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
redis==5.0.1
''',
            "README.md": f'''# {workspace.name}

Generated by Agent Zero AI system.

## Tech Stack
{chr(10).join(f"- {tech}" for tech in workspace.tech_stack)}

## Usage
