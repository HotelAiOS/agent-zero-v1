"""
Workspace Manager - Agent Zero v1.0
Zarządza workspace'ami projektowymi, zapisuje kod do plików, integruje z Knowledge Graph
"""
import os
import json
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

# Import TWOJEGO existing code
from knowledge.knowledge_graph import KnowledgeGraph
from communication.intelligent_agent import IntelligentAgent
from communication.messagebus import MessageBus

@dataclass
class WorkspaceConfig:
    """Konfiguracja workspace'u"""
    workspace_id: str
    name: str
    project_type: str  # "saas", "ecommerce", "api", "microservices"
    base_path: Path
    agents: List[str]
    tech_stack: List[str]
    created_at: datetime
    status: str  # "active", "archived", "deleted"

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
    dependencies: List[str]
    
class WorkspaceManager:
    """Manager workspace'ów - integruje z TWOIM systemem"""
    
    def __init__(self, 
                 base_workspaces_dir: str = "./workspaces",
                 knowledge_graph: Optional[KnowledgeGraph] = None):
        
        self.base_path = Path(base_workspaces_dir)
        self.base_path.mkdir(exist_ok=True)
        
        # Integration z TWOIM Knowledge Graph
        self.knowledge_graph = knowledge_graph or KnowledgeGraph()
        
        # Workspace state
        self.active_workspaces: Dict[str, WorkspaceConfig] = {}
        self.code_files: Dict[str, CodeFile] = {}
        
        # Load existing workspaces
        self._load_existing_workspaces()
        
    def _load_existing_workspaces(self):
        """Załaduj istniejące workspace'y"""
        for workspace_dir in self.base_path.iterdir():
            if workspace_dir.is_dir():
                config_file = workspace_dir / "workspace_config.json"
                if config_file.exists():
                    with open(config_file) as f:
                        config_data = json.load(f)
                    
                    workspace = WorkspaceConfig(
                        workspace_id=config_data["workspace_id"],
                        name=config_data["name"],
                        project_type=config_data["project_type"],
                        base_path=Path(config_data["base_path"]),
                        agents=config_data["agents"],
                        tech_stack=config_data["tech_stack"],
                        created_at=datetime.fromisoformat(config_data["created_at"]),
                        status=config_data["status"]
                    )
                    
                    self.active_workspaces[workspace.workspace_id] = workspace
                    print(f"📁 Loaded workspace: {workspace.name} ({workspace.workspace_id})")
    
    async def create_workspace(self, 
                              name: str, 
                              project_type: str,
                              tech_stack: List[str],
                              template: Optional[str] = None) -> str:
        """Utwórz nowy workspace"""
        
        workspace_id = f"ws-{uuid.uuid4().hex[:8]}"
        workspace_path = self.base_path / workspace_id
        workspace_path.mkdir(exist_ok=True)
        
        # Struktura katalogów
        directories = [
            "src", "tests", "docs", "config", 
            "docker", "k8s", "scripts", "data"
        ]
        
        for dir_name in directories:
            (workspace_path / dir_name).mkdir(exist_ok=True)
        
        # Konfiguracja workspace'u
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
        
        # Zapisz config
        config_file = workspace_path / "workspace_config.json"
        with open(config_file, 'w') as f:
            json.dump({
                "workspace_id": workspace_id,
                "name": name,
                "project_type": project_type,
                "base_path": str(workspace_path),
                "agents": [],
                "tech_stack": tech_stack,
                "created_at": config.created_at.isoformat(),
                "status": "active"
            }, f, indent=2)
        
        self.active_workspaces[workspace_id] = config
        
        # Zapisz do Knowledge Graph
        await self._register_workspace_in_kg(config)
        
        print(f"✅ Created workspace: {name} ({workspace_id})")
        print(f"📁 Path: {workspace_path}")
        
        return workspace_id
    
    async def _register_workspace_in_kg(self, config: WorkspaceConfig):
        """Zarejestruj workspace w Knowledge Graph"""
        if not self.knowledge_graph:
            return
            
        try:
            # Create workspace node
            query = """
            CREATE (w:Workspace {
                workspace_id: $workspace_id,
                name: $name,
                project_type: $project_type,
                tech_stack: $tech_stack,
                created_at: $created_at,
                status: $status
            })
            """
            
            await self.knowledge_graph.execute_query(query, {
                "workspace_id": config.workspace_id,
                "name": config.name,
                "project_type": config.project_type,
                "tech_stack": config.tech_stack,
                "created_at": config.created_at.isoformat(),
                "status": config.status
            })
            
            print(f"📊 Workspace {config.workspace_id} registered in Knowledge Graph")
            
        except Exception as e:
            print(f"⚠️ Failed to register workspace in KG: {e}")
    
    async def save_generated_code(self, 
                                 workspace_id: str,
                                 agent_id: str,
                                 task_id: str,
                                 filename: str,
                                 content: str,
                                 language: str = "python") -> str:
        """Zapisz wygenerowany kod do workspace'u"""
        
        if workspace_id not in self.active_workspaces:
            raise ValueError(f"Workspace {workspace_id} not found")
        
        workspace = self.active_workspaces[workspace_id]
        
        # Determine file path based on language and type
        if language == "python":
            if filename.endswith("_test.py") or "test" in filename:
                filepath = workspace.base_path / "tests" / filename
            else:
                filepath = workspace.base_path / "src" / filename
        elif language == "dockerfile":
            filepath = workspace.base_path / "docker" / filename
        elif language == "yaml" and "k8s" in content.lower():
            filepath = workspace.base_path / "k8s" / filename
        else:
            filepath = workspace.base_path / "src" / filename
        
        # Create directory if not exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Create CodeFile record
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
            created_at=datetime.now(),
            dependencies=self._extract_dependencies(content, language)
        )
        
        self.code_files[file_id] = code_file
        
        # Register in Knowledge Graph
        await self._register_code_in_kg(code_file)
        
        print(f"💾 Saved code: {filename} ({len(content)} chars)")
        print(f"📁 Path: {filepath}")
        
        return file_id
    
    async def _register_code_in_kg(self, code_file: CodeFile):
        """Zarejestruj kod w Knowledge Graph"""
        if not self.knowledge_graph:
            return
            
        try:
            # Create code file node and relationships
            query = """
            MATCH (w:Workspace {workspace_id: $workspace_id})
            MATCH (a:Agent {agent_id: $agent_id})
            CREATE (f:CodeFile {
                file_id: $file_id,
                filename: $filename,
                language: $language,
                content_length: $content_length,
                created_at: $created_at,
                dependencies: $dependencies
            })
            CREATE (w)-[:CONTAINS]->(f)
            CREATE (a)-[:GENERATED]->(f)
            """
            
            await self.knowledge_graph.execute_query(query, {
                "workspace_id": code_file.workspace_id,
                "agent_id": code_file.agent_id,
                "file_id": code_file.file_id,
                "filename": code_file.filename,
                "language": code_file.language,
                "content_length": len(code_file.content),
                "created_at": code_file.created_at.isoformat(),
                "dependencies": code_file.dependencies
            })
            
        except Exception as e:
            print(f"⚠️ Failed to register code in KG: {e}")
    
    def _extract_dependencies(self, content: str, language: str) -> List[str]:
        """Wydobądź zależności z kodu"""
        dependencies = []
        
        if language == "python":
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('import '):
                    dep = line.replace('import ', '').split()[0]
                    dependencies.append(dep)
                elif line.startswith('from '):
                    dep = line.split()[1]
                    dependencies.append(dep)
        
        return list(set(dependencies))
    
    async def get_workspace_files(self, workspace_id: str) -> List[Dict]:
        """Pobierz listę plików w workspace"""
        files = []
        for file_id, code_file in self.code_files.items():
            if code_file.workspace_id == workspace_id:
                files.append({
                    "file_id": file_id,
                    "filename": code_file.filename,
                    "language": code_file.language,
                    "agent_id": code_file.agent_id,
                    "created_at": code_file.created_at.isoformat(),
                    "size": len(code_file.content),
                    "path": str(code_file.filepath)
                })
        
        return files
    
    async def create_project_structure(self, workspace_id: str, project_type: str):
        """Utwórz strukturę projektu na podstawie typu"""
        
        if workspace_id not in self.active_workspaces:
            raise ValueError(f"Workspace {workspace_id} not found")
            
        workspace = self.active_workspaces[workspace_id]
        
        if project_type == "fastapi":
            await self._create_fastapi_structure(workspace)
        elif project_type == "microservices":
            await self._create_microservices_structure(workspace)
        elif project_type == "saas":
            await self._create_saas_structure(workspace)
    
    async def _create_fastapi_structure(self, workspace: WorkspaceConfig):
        """Utwórz strukturę FastAPI"""
        
        # Main app structure
        files_to_create = {
            "src/main.py": '''from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Agent Zero API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "Agent Zero API"}

@app.get("/health")
async def health():
    return {"status": "healthy"}
''',
            "src/models/__init__.py": "",
            "src/routers/__init__.py": "",
            "src/dependencies.py": '''from fastapi import Depends
# Add common dependencies here
''',
            "requirements.txt": '''fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
''',
            "Dockerfile": '''FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
''',
            "docker-compose.yml": '''version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
    volumes:
      - ./src:/app/src
'''
        }
        
        for filepath, content in files_to_create.items():
            full_path = workspace.base_path / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(full_path, 'w') as f:
                f.write(content)
            
            print(f"📄 Created: {filepath}")

# Integration z TWOIM systemem
class AgentWorkspaceIntegration:
    """Integracja Workspace Manager z TWOIMI agentami"""
    
    def __init__(self, workspace_manager: WorkspaceManager):
        self.workspace_manager = workspace_manager
        self.message_bus = None
        
    async def initialize(self):
        """Inicjalizuj integrację"""
        # Connect to YOUR MessageBus
        self.message_bus = MessageBus()
        await self.message_bus.connect()
        
        # Subscribe to code generation events
        await self.message_bus.subscribe(
            "agent.*.code_generated",
            self._handle_code_generated
        )
        
        print("✅ Workspace integration initialized")
    
    async def _handle_code_generated(self, message_data: Dict):
        """Handle code generated by YOUR agents"""
        try:
            agent_id = message_data.get("agent_id")
            task_id = message_data.get("task_id")
            code = message_data.get("code")
            filename = message_data.get("filename", f"generated_{task_id}.py")
            language = message_data.get("language", "python")
            workspace_id = message_data.get("workspace_id")
            
            if workspace_id and code:
                file_id = await self.workspace_manager.save_generated_code(
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                    task_id=task_id,
                    filename=filename,
                    content=code,
                    language=language
                )
                
                print(f"💾 Auto-saved generated code: {filename} ({file_id})")
                
        except Exception as e:
            print(f"❌ Error handling code generation: {e}")

# CLI Extension for YOUR system
class WorkspaceCLI:
    """CLI rozszerzenie dla workspace management"""
    
    def __init__(self):
        self.workspace_manager = WorkspaceManager()
        self.integration = AgentWorkspaceIntegration(self.workspace_manager)
        
    async def run_workspace_commands(self):
        """CLI dla workspace management"""
        
        await self.integration.initialize()
        
        print("📁 WORKSPACE MANAGER")
        print("=" * 40)
        
        while True:
            print("\n🎮 WORKSPACE COMMANDS:")
            print("1️⃣  Utwórz workspace")
            print("2️⃣  Lista workspace'ów")
            print("3️⃣  Pokaż pliki workspace'u")
            print("4️⃣  Utwórz strukturę projektu")
            print("0️⃣  Powrót")
            
            choice = input("\n👉 Wybierz: ").strip()
            
            if choice == "0":
                break
            elif choice == "1":
                await self._create_workspace_interactive()
            elif choice == "2":
                self._list_workspaces()
            elif choice == "3":
                await self._show_workspace_files()
            elif choice == "4":
                await self._create_project_structure()
    
    async def _create_workspace_interactive(self):
        """Interaktywne tworzenie workspace'u"""
        print("\n📁 TWORZENIE WORKSPACE'U")
        
        name = input("Nazwa workspace'u: ").strip()
        if not name:
            print("❌ Nazwa nie może być pusta")
            return
        
        print("Dostępne typy projektów:")
        project_types = ["saas", "api", "microservices", "ecommerce", "custom"]
        for i, ptype in enumerate(project_types, 1):
            print(f"{i}. {ptype}")
        
        try:
            choice = int(input("Wybierz typ (numer): ")) - 1
            if 0 <= choice < len(project_types):
                project_type = project_types[choice]
            else:
                project_type = "custom"
        except ValueError:
            project_type = "custom"
        
        tech_stack = input("Tech stack (oddzielone przecinkami): ").strip()
        tech_list = [t.strip() for t in tech_stack.split(",") if t.strip()] if tech_stack else []
        
        workspace_id = await self.workspace_manager.create_workspace(
            name=name,
            project_type=project_type,
            tech_stack=tech_list
        )
        
        print(f"✅ Workspace utworzony: {workspace_id}")
    
    def _list_workspaces(self):
        """Lista workspace'ów"""
        print("\n📁 WORKSPACE'Y:")
        
        if not self.workspace_manager.active_workspaces:
            print("❌ Brak workspace'ów")
            return
        
        for ws_id, workspace in self.workspace_manager.active_workspaces.items():
            print(f"🏢 {workspace.name} ({ws_id})")
            print(f"   Typ: {workspace.project_type}")
            print(f"   Tech: {', '.join(workspace.tech_stack)}")
            print(f"   Ścieżka: {workspace.base_path}")
            print(f"   Status: {workspace.status}")
            print(f"   Utworzony: {workspace.created_at.strftime('%Y-%m-%d %H:%M')}")

# Test function
async def test_workspace_integration():
    """Test integracji workspace managera"""
    
    # Initialize workspace manager z TWOIM Knowledge Graph
    kg = KnowledgeGraph()
    await kg.connect()
    
    workspace_manager = WorkspaceManager(knowledge_graph=kg)
    
    # Create test workspace
    workspace_id = await workspace_manager.create_workspace(
        name="Test SaaS Platform",
        project_type="saas",
        tech_stack=["fastapi", "postgresql", "redis", "kubernetes"]
    )
    
    # Create project structure
    await workspace_manager.create_project_structure(workspace_id, "fastapi")
    
    # Simulate code generation (like YOUR agents would do)
    sample_code = '''
from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

app = FastAPI()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class UserLogin(BaseModel):
    username: str
    password: str

@app.post("/login")
async def login(user: UserLogin):
    # Authentication logic here
    return {"access_token": "example_token", "token_type": "bearer"}
'''
    
    file_id = await workspace_manager.save_generated_code(
        workspace_id=workspace_id,
        agent_id="backend_ai_001",
        task_id="auth_endpoint",
        filename="auth_endpoints.py",
        content=sample_code,
        language="python"
    )
    
    print(f"✅ Test completed - file saved: {file_id}")
    
    # Show workspace files
    files = await workspace_manager.get_workspace_files(workspace_id)
    print("\n📁 Workspace files:")
    for file_info in files:
        print(f"   📄 {file_info['filename']} ({file_info['size']} chars)")

if __name__ == "__main__":
    # Test the integration
    asyncio.run(test_workspace_integration())
