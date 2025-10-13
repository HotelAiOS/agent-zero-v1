#!/usr/bin/env python3
"""
Final Repository Cleanup Round 2 for Agent Zero V1
Addresses remaining large folders and suspicious files
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

class FinalRepositoryCleanup:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.quarantine_dir = self.base_path / f"_quarantine_round2_{self.timestamp}"
        
    def create_quarantine_directory(self):
        """Tworzy folder quarantine round 2"""
        self.quarantine_dir.mkdir(exist_ok=True)
        print(f"📁 Created round 2 quarantine: {self.quarantine_dir}")
    
    def move_to_quarantine(self, source_path, reason="cleanup"):
        """Bezpiecznie przenosi folder/plik do quarantine round 2"""
        source = Path(source_path)
        if not source.exists():
            print(f"⚠️  {source} doesn't exist, skipping...")
            return False
            
        dest = self.quarantine_dir / source.name
        
        # Jeśli destination już istnieje, dodaj suffix
        counter = 1
        original_dest = dest
        while dest.exists():
            dest = Path(f"{original_dest}_{counter}")
            counter += 1
        
        try:
            shutil.move(str(source), str(dest))
            size_mb = self.get_directory_size(dest) / (1024 * 1024)
            print(f"✅ Moved {source} → {dest} ({size_mb:.1f}MB, reason: {reason})")
            return True
            
        except Exception as e:
            print(f"❌ Failed to move {source}: {e}")
            return False
    
    def get_directory_size(self, path):
        """Oblicza rozmiar folderu w bytach"""
        if not path.exists():
            return 0
            
        if path.is_file():
            return path.stat().st_size
            
        total = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                try:
                    total += os.path.getsize(os.path.join(root, file))
                except (OSError, FileNotFoundError):
                    pass
        return total
    
    def final_cleanup(self):
        """Final aggressive cleanup based on verification results"""
        
        print("🧹 Starting FINAL Repository Cleanup Round 2...")
        self.create_quarantine_directory()
        
        moved_count = 0
        total_size_freed = 0
        
        # TARGET: Large folders identified in verification
        large_folders_to_move = [
            ("data", "large_data_folder_517M"),
            ("shared", "shared_folder_286M"), 
            ("cli", "cli_folder_26M"),
            ("stabilization", "stabilization_folder_4_5M"),
            ("models", "models_folder_1_7M"),
        ]
        
        # BACKUP FILES in root - all those .backup files
        backup_patterns = [
            ("*.backup", "backup_files"),
            ("*.backup-*", "backup_files_with_timestamps"),
            ("*.backup.*", "backup_files_with_extensions"),
        ]
        
        # WEIRD FILES - those =X.X.X files
        weird_files = [
            ("=0.104.0", "pip_install_artifacts"),
            ("=0.13.0", "pip_install_artifacts"),
            ("=0.24.0", "pip_install_artifacts"),
            ("=1.24.0", "pip_install_artifacts"),
            ("=1.3.0", "pip_install_artifacts"),
            ("=2.0.0", "pip_install_artifacts"),
            ("=23.0.0", "pip_install_artifacts"),
            ("=5.0.0", "pip_install_artifacts"),
        ]
        
        # HIDDEN FOLDERS that might be large
        hidden_folders = [
            (".venv", "hidden_virtual_env"),
            (".security", "security_cache"),
        ]
        
        print("\n🎯 Moving large folders...")
        for folder, reason in large_folders_to_move:
            if self.move_to_quarantine(folder, reason):
                moved_count += 1
                
        print("\n🗑️ Cleaning backup files...")
        for pattern, reason in backup_patterns:
            files = list(self.base_path.glob(pattern))
            for file in files:
                if self.move_to_quarantine(file, reason):
                    moved_count += 1
        
        print("\n🔧 Cleaning weird pip artifacts...")
        for file, reason in weird_files:
            if self.move_to_quarantine(file, reason):
                moved_count += 1
                
        print("\n👁️ Checking hidden folders...")
        for folder, reason in hidden_folders:
            folder_path = Path(folder)
            if folder_path.exists():
                size_mb = self.get_directory_size(folder_path) / (1024 * 1024)
                if size_mb > 5:  # Only move if > 5MB
                    if self.move_to_quarantine(folder, reason):
                        moved_count += 1
        
        # Calculate total size freed
        if self.quarantine_dir.exists():
            total_size_freed = self.get_directory_size(self.quarantine_dir) / (1024 * 1024)
        
        self.print_final_summary(moved_count, total_size_freed)
        
        return moved_count, total_size_freed
    
    def print_final_summary(self, moved_count, total_size_freed):
        """Wyświetla podsumowanie final cleanup"""
        print("\n" + "="*60)
        print("🎉 FINAL CLEANUP ROUND 2 COMPLETED!")
        print("="*60)
        print(f"📦 Items moved to quarantine: {moved_count}")
        print(f"💾 Additional space freed: {total_size_freed:.1f} MB")
        print(f"📁 Round 2 quarantine: {self.quarantine_dir}")
        print("\n🎯 Repository should now be much cleaner!")
        print(f"\n🔄 To restore anything:")
        print(f"   mv {self.quarantine_dir}/ITEM_NAME ./")
        print("\n🗑️  When confident, delete both quarantines:")
        print("   rm -rf _quarantine_*")
        print("="*60)

def main():
    """Uruchamia final cleanup"""
    cleanup = FinalRepositoryCleanup()
    
    print("🛡️  FINAL REPOSITORY CLEANUP - ROUND 2")
    print("="*50)
    print("This will move remaining large folders and suspicious files.")
    print("Targets: data/, shared/, cli/, backup files, pip artifacts")
    print("")
    
    response = input("Continue with final cleanup? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("❌ Final cleanup cancelled.")
        return
    
    try:
        moved_count, size_freed = cleanup.final_cleanup()
        
        print(f"\n✅ Final cleanup completed!")
        print(f"   Moved {moved_count} items, freed {size_freed:.1f}MB")
        print(f"\n🔍 Run verification again:")
        print("   python3 pre-github-verification.py")
        
    except Exception as e:
        print(f"\n❌ Final cleanup failed: {e}")
        return

if __name__ == "__main__":
    main()