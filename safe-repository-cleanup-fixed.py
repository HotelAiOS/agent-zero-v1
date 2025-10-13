#!/usr/bin/env python3
"""
Safe Repository Cleanup for Agent Zero V1 - FIXED VERSION
PRZENOSI (nie usuwa) podejrzane foldery do _quarantine
Tworzy manifest do łatwego przywracania
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

class SafeRepositoryCleanup:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.quarantine_dir = self.base_path / f"_quarantine_{self.timestamp}"
        self.manifest = {
            "cleanup_date": datetime.now().isoformat(),
            "moved_directories": {},
            "moved_files": {},
            "statistics": {}
        }
    
    def create_quarantine_directory(self):
        """Tworzy folder quarantine z timestamp"""
        self.quarantine_dir.mkdir(exist_ok=True)
        print(f"📁 Created quarantine directory: {self.quarantine_dir}")
    
    def move_to_quarantine(self, source_path, reason="cleanup"):
        """Bezpiecznie przenosi folder/plik do quarantine"""
        source = Path(source_path)
        if not source.exists():
            print(f"⚠️  {source} doesn't exist, skipping...")
            return False
            
        # Destination w quarantine
        dest = self.quarantine_dir / source.name
        
        # Jeśli destination już istnieje, dodaj suffix
        counter = 1
        original_dest = dest
        while dest.exists():
            dest = Path(f"{original_dest}_{counter}")
            counter += 1
        
        try:
            # Przenieś folder/plik
            shutil.move(str(source), str(dest))
            
            # Zapisz w manifest
            size_mb = self.get_directory_size(dest) / (1024 * 1024)
            self.manifest["moved_directories"][str(source)] = {
                "moved_to": str(dest),
                "reason": reason,
                "size_mb": round(size_mb, 2),
                "file_count": self.count_files(dest)
            }
            
            print(f"✅ Moved {source} → {dest} ({size_mb:.1f}MB, reason: {reason})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to move {source}: {e}")
            return False
    
    def get_directory_size(self, path):
        """Oblicza rozmiar folderu w bytach"""
        total = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                try:
                    total += os.path.getsize(os.path.join(root, file))
                except (OSError, FileNotFoundError):
                    pass
        return total
    
    def count_files(self, path):
        """Liczy pliki w folderze"""
        if path.is_file():
            return 1
        count = 0
        for root, dirs, files in os.walk(path):
            count += len(files)
        return count
    
    def cleanup_repository(self):
        """Główna funkcja cleanup - przenosi podejrzane foldery"""
        
        print("🧹 Starting Safe Repository Cleanup...")
        self.create_quarantine_directory()
        
        # FOLDERY DO PRZENIESIENIA (wysokie prawdopodobieństwo że niepotrzebne)
        high_priority_moves = [
            ("backups", "backup_folder_23k_files"),
            ("backups_20251011_235951", "dated_backup_folder"), 
            ("backups_20251012_103431", "dated_backup_folder"),
            ("backup_kaizen_mock_20251012_180933", "dated_backup_folder"),
            ("_cleanup_review", "previous_cleanup_attempt"),
            ("venv", "virtual_environment"),
            ("venv_v2", "virtual_environment"), 
            ("venv_nlu", "virtual_environment"),
            ("agent-zero-env", "virtual_environment"),
        ]
        
        # POTENCJALNIE LEGACY/DUPLICATE FOLDERS
        medium_priority_moves = [
            ("phase2-service", "development_phase_folder"),
            ("phase3-development", "development_phase_folder"),
            ("phase3-service", "development_phase_folder"), 
            ("phase3-priority2", "development_phase_folder"),
            ("phase3-priority3", "development_phase_folder"),
            ("phase4-analysis", "development_phase_folder"),
            ("phase4-ollama", "development_phase_folder"),
            ("intelligence_v2", "versioned_duplicate"),
            ("agent-zero-v1-integrated", "integration_test_folder"),
            ("agent-zero-v1", "duplicate_folder_name"),
        ]
        
        # CACHE I TEMPORARY FILES
        cache_moves = [
            ("__pycache__", "python_cache"),
            (".pytest_cache", "test_cache"), 
            ("logs", "log_files"),
            ("audit_reports", "temporary_reports"),
        ]
        
        # Wykonaj przeniesienia
        moved_count = 0
        total_size_freed = 0
        
        for folder, reason in high_priority_moves + medium_priority_moves + cache_moves:
            if self.move_to_quarantine(folder, reason):
                moved_count += 1
                if folder in self.manifest["moved_directories"]:
                    total_size_freed += self.manifest["moved_directories"][folder]["size_mb"]
        
        # PATTERN-BASED CLEANUP (pliki po extensions)
        self.cleanup_by_patterns()
        
        # Zapisz manifest
        self.save_manifest()
        
        # Podsumowanie
        self.print_cleanup_summary(moved_count, total_size_freed)
        
        return self.manifest
    
    def cleanup_by_patterns(self):
        """Czyści pliki według wzorców (*.pyc, *.log, etc.)"""
        
        patterns_to_move = [
            ("*.pyc", "compiled_python"),
            ("*.pyo", "compiled_python"),
            ("*.pyd", "compiled_python"), 
            ("core.*", "core_dumps"),
            (".DS_Store", "macos_files"),
            ("Thumbs.db", "windows_files"),
        ]
        
        for pattern, reason in patterns_to_move:
            files = list(self.base_path.rglob(pattern))
            if files:
                pattern_dir = self.quarantine_dir / f"pattern_{pattern.replace('*', 'wildcard').replace('.', '_')}"
                pattern_dir.mkdir(exist_ok=True)
                
                for file in files:
                    try:
                        dest_file = pattern_dir / file.name
                        # Jeśli plik o tej nazwie już istnieje, dodaj numer
                        counter = 1
                        original_dest = dest_file
                        while dest_file.exists():
                            stem = original_dest.stem
                            suffix = original_dest.suffix
                            dest_file = pattern_dir / f"{stem}_{counter}{suffix}"
                            counter += 1
                        
                        shutil.move(str(file), str(dest_file))
                        
                    except Exception as e:
                        print(f"⚠️  Could not move {file}: {e}")
    
    def save_manifest(self):
        """Zapisuje manifest przeniesień"""
        manifest_file = self.quarantine_dir / "quarantine_manifest.json"
        
        # Dodaj statystyki
        self.manifest["statistics"] = {
            "total_moved_directories": len(self.manifest["moved_directories"]),
            "total_size_freed_mb": sum(item["size_mb"] for item in self.manifest["moved_directories"].values()),
            "total_files_moved": sum(item["file_count"] for item in self.manifest["moved_directories"].values())
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(self.manifest, f, indent=2)
            
        # Również markdown summary
        self.create_markdown_summary()
        
        print(f"📋 Manifest saved: {manifest_file}")
    
    def create_markdown_summary(self):
        """Tworzy czytelne podsumowanie w markdown"""
        summary_file = self.quarantine_dir / "CLEANUP_SUMMARY.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# Safe Cleanup Summary - {self.timestamp}\n\n")
            f.write(f"**Cleanup Date:** {self.manifest['cleanup_date']}\n\n")
            
            f.write("## 📊 Statistics\n")
            stats = self.manifest["statistics"]
            f.write(f"- **Directories Moved:** {stats['total_moved_directories']}\n")
            f.write(f"- **Total Size Freed:** {stats['total_size_freed_mb']:.1f} MB\n")
            f.write(f"- **Files Moved:** {stats['total_files_moved']}\n\n")
            
            f.write("## 📁 Moved Directories\n\n")
            for original, info in self.manifest["moved_directories"].items():
                f.write(f"### `{original}`\n")
                f.write(f"- **Moved to:** `{info['moved_to']}`\n")
                f.write(f"- **Reason:** {info['reason']}\n") 
                f.write(f"- **Size:** {info['size_mb']} MB\n")
                f.write(f"- **Files:** {info['file_count']}\n\n")
            
            f.write("## 🔄 How to Restore\n\n")
            f.write("If you need to restore any directory:\n\n")
            f.write("```bash\n")
            f.write("# Restore specific directory\n")
            f.write(f"mv {self.quarantine_dir}/DIRECTORY_NAME ./\n\n")
            f.write("# Or restore everything (NOT recommended)\n") 
            f.write(f"cp -r {self.quarantine_dir}/* ./\n")
            f.write("```\n")
        
        print(f"📝 Summary created: {summary_file}")
    
    def print_cleanup_summary(self, moved_count, total_size_freed):
        """Wyświetla podsumowanie cleanup"""
        print("\n" + "="*60)
        print("🎉 SAFE CLEANUP COMPLETED!")
        print("="*60)
        print(f"📦 Directories moved to quarantine: {moved_count}")
        print(f"💾 Disk space freed: {total_size_freed:.1f} MB")
        print(f"📁 Quarantine location: {self.quarantine_dir}")
        print(f"📋 Manifest: {self.quarantine_dir}/quarantine_manifest.json")
        print(f"📝 Summary: {self.quarantine_dir}/CLEANUP_SUMMARY.md")
        print("\n🔄 To restore any directory:")
        print(f"   mv {self.quarantine_dir}/DIRECTORY_NAME ./")
        print("\n🗑️  When confident, you can delete quarantine:")
        print(f"   rm -rf {self.quarantine_dir}")
        print("="*60)

def main():
    """Uruchamia safe cleanup"""
    cleanup = SafeRepositoryCleanup()
    
    print("🛡️  SAFE REPOSITORY CLEANUP")
    print("="*40)
    print("This script will MOVE (not delete) suspicious directories to quarantine.")
    print("You can review and restore anything important later.")
    print("")
    
    response = input("Continue with safe cleanup? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("❌ Cleanup cancelled.")
        return
    
    try:
        manifest = cleanup.cleanup_repository()
        print("\n✅ Safe cleanup completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Cleanup failed: {e}")
        return

if __name__ == "__main__":
    main()