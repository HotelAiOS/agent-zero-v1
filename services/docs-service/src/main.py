from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
import os
from pathlib import Path
from typing import Optional

from .models.schemas import (
    ParseRequest, ParseResponse, GenerateDocsRequest, 
    GenerateDocsResponse, WatchStatus
)
from .parser.ast_parser import ASTParser
from .generator.markdown_gen import MarkdownGenerator
from .watcher.file_watcher import FileWatcher

# Logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Agent Zero Docs Service",
    version="1.0.0",
    description="Automatyczna generacja dokumentacji z kodu Python"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Components
parser = ASTParser()
generator = MarkdownGenerator()
watcher: Optional[FileWatcher] = None

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Agent Zero Docs Service v1.0.0"}

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "watcher_active": watcher is not None and watcher.observer.is_alive()
    }

@app.post("/parse", response_model=ParseResponse)
async def parse_file(request: ParseRequest):
    """Parsuj pojedynczy plik Python"""
    try:
        filepath = Path(request.filepath)
        
        if not filepath.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        if not filepath.suffix == '.py':
            raise HTTPException(status_code=400, detail="Not a Python file")
        
        # Parsuj plik
        parsed_data = parser.parse_file(filepath)
        
        # Generuj markdown
        markdown = generator.generate_file_docs(parsed_data)
        
        return ParseResponse(
            filepath=str(filepath),
            success=True,
            markdown=markdown
        )
    
    except Exception as e:
        logger.error(f"Parse failed: {e}")
        return ParseResponse(
            filepath=request.filepath,
            success=False,
            error=str(e)
        )

@app.post("/generate", response_model=GenerateDocsResponse)
async def generate_docs(request: GenerateDocsRequest):
    """Generuj dokumentację dla całego katalogu"""
    try:
        source_dir = Path(request.directory)
        output_dir = Path(request.output_dir)
        
        if not source_dir.exists():
            raise HTTPException(status_code=404, detail="Directory not found")
        
        # Utwórz katalog wyjściowy
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Znajdź wszystkie pliki Python
        py_files = list(source_dir.rglob("*.py"))
        files_processed = 0
        files_generated = 0
        errors = []
        
        for py_file in py_files:
            try:
                # Parsuj
                parsed_data = parser.parse_file(py_file)
                
                # Generuj markdown
                markdown = generator.generate_file_docs(parsed_data)
                
                # Zapisz
                relative_path = py_file.relative_to(source_dir)
                output_file = output_dir / relative_path.with_suffix('.md')
                output_file.parent.mkdir(parents=True, exist_ok=True)
                
                output_file.write_text(markdown, encoding='utf-8')
                
                files_processed += 1
                files_generated += 1
                
            except Exception as e:
                logger.error(f"Failed to process {py_file}: {e}")
                errors.append(f"{py_file}: {str(e)}")
                files_processed += 1
        
        # Generuj index
        index_md = generator.generate_module_index(parser.parsed_files)
        (output_dir / "INDEX.md").write_text(index_md, encoding='utf-8')
        
        return GenerateDocsResponse(
            files_processed=files_processed,
            files_generated=files_generated,
            output_directory=str(output_dir),
            errors=errors
        )
    
    except Exception as e:
        logger.error(f"Generate docs failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/watch/start")
async def start_watching(directories: list[str], background_tasks: BackgroundTasks):
    """Rozpocznij obserwowanie katalogów"""
    global watcher
    
    if watcher and watcher.observer.is_alive():
        return {"message": "Watcher already running"}
    
    def on_file_change(filepath: Path):
        """Callback przy zmianie pliku"""
        try:
            parsed_data = parser.parse_file(filepath)
            markdown = generator.generate_file_docs(parsed_data)
            
            output_file = Path("docs/api") / filepath.with_suffix('.md').name
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(markdown, encoding='utf-8')
            
            logger.info(f"Updated docs for: {filepath}")
        except Exception as e:
            logger.error(f"Failed to update docs for {filepath}: {e}")
    
    watch_dirs = [Path(d) for d in directories]
    watcher = FileWatcher(watch_dirs, on_file_change)
    
    background_tasks.add_task(watcher.start)
    
    return {"message": "Watcher started", "directories": directories}

@app.post("/watch/stop")
async def stop_watching():
    """Zatrzymaj watchera"""
    global watcher
    
    if watcher and watcher.observer.is_alive():
        watcher.stop()
        return {"message": "Watcher stopped"}
    
    return {"message": "Watcher not running"}

@app.get("/watch/status", response_model=WatchStatus)
async def watcher_status():
    """Status watchera"""
    if watcher and watcher.observer.is_alive():
        return WatchStatus(
            watching=True,
            directories=[str(d) for d in watcher.watch_dirs],
            files_watched=len(parser.parsed_files)
        )
    
    return WatchStatus(
        watching=False,
        directories=[],
        files_watched=0
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
