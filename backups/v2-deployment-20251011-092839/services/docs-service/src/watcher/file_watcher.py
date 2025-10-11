import logging
import time
from pathlib import Path
from typing import Callable, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

logger = logging.getLogger(__name__)

class PythonFileHandler(FileSystemEventHandler):
    """Handler dla zmian w plikach Python"""
    
    def __init__(self, callback: Callable[[Path], None]):
        self.callback = callback
        self.debounce_time = 1.0
        self.pending_files: Set[Path] = set()
        self.last_event_time = {}
    
    def on_modified(self, event: FileSystemEvent):
        """Plik został zmodyfikowany"""
        if event.is_directory or not event.src_path.endswith('.py'):
            return
        
        filepath = Path(event.src_path)
        current_time = time.time()
        
        # Debounce - ignoruj szybkie wielokrotne zmiany
        if filepath in self.last_event_time:
            if current_time - self.last_event_time[filepath] < self.debounce_time:
                return
        
        self.last_event_time[filepath] = current_time
        logger.info(f"File modified: {filepath}")
        self.callback(filepath)
    
    def on_created(self, event: FileSystemEvent):
        """Nowy plik został utworzony"""
        if event.is_directory or not event.src_path.endswith('.py'):
            return
        
        filepath = Path(event.src_path)
        logger.info(f"File created: {filepath}")
        self.callback(filepath)

class FileWatcher:
    """Watcher dla zmian w plikach projektu"""
    
    def __init__(self, watch_dirs: list[Path], callback: Callable[[Path], None]):
        self.watch_dirs = watch_dirs
        self.callback = callback
        self.observer = Observer()
        self.handler = PythonFileHandler(callback)
    
    def start(self):
        """Uruchom watchera"""
        for watch_dir in self.watch_dirs:
            if watch_dir.exists():
                self.observer.schedule(
                    self.handler,
                    str(watch_dir),
                    recursive=True
                )
                logger.info(f"Watching directory: {watch_dir}")
            else:
                logger.warning(f"Directory does not exist: {watch_dir}")
        
        self.observer.start()
        logger.info("FileWatcher started")
    
    def stop(self):
        """Zatrzymaj watchera"""
        self.observer.stop()
        self.observer.join()
        logger.info("FileWatcher stopped")
