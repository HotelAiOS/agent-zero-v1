from pydantic import BaseModel, Field
from typing import List, Optional
from pathlib import Path

class ParseRequest(BaseModel):
    """Żądanie parsowania pliku"""
    filepath: str = Field(..., description="Ścieżka do pliku Python")

class ParseResponse(BaseModel):
    """Odpowiedź parsowania"""
    filepath: str
    success: bool
    markdown: Optional[str] = None
    error: Optional[str] = None

class GenerateDocsRequest(BaseModel):
    """Żądanie generacji dokumentacji"""
    directory: str = Field(..., description="Katalog do przetworzenia")
    output_dir: str = Field(default="docs/api", description="Katalog wyjściowy")

class GenerateDocsResponse(BaseModel):
    """Odpowiedź generacji"""
    files_processed: int
    files_generated: int
    output_directory: str
    errors: List[str] = Field(default_factory=list)

class WatchStatus(BaseModel):
    """Status watchera"""
    watching: bool
    directories: List[str]
    files_watched: int
