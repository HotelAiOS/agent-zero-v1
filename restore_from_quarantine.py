#!/usr/bin/env python3
"""
Narzędzie do przywracania z quarantine
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

def list_quarantine_folders():
    """Lista dostępnych folderów quarantine"""
    quarantine_folders = list(Path(".").glob("_quarantine_*"))
    if not quarantine_folders:
        print("No quarantine folders found.")
        return []
    
    print("Available quarantine folders:")
    for i, folder in enumerate(quarantine_folders, 1):
        manifest_file = folder / "quarantine_manifest.json"
        if manifest_file.exists():
            with open(manifest_file) as f:
                manifest = json.load(f)
            print(f"{i}. {folder.name} ({manifest['cleanup_date']})")
            print(f"   - {len(manifest['moved_directories'])} directories")
            print(f"   - {manifest['statistics']['total_size_freed_mb']:.1f} MB")
        else:
            print(f"{i}. {folder.name} (no manifest)")
    
    return quarantine_folders

def restore_selective(quarantine_path):
    """Selektywne przywracanie z quarantine"""
    manifest_file = quarantine_path / "quarantine_manifest.json"
    
    if not manifest_file.exists():
        print("No manifest found. Cannot perform selective restore.")
        return
    
    with open(manifest_file) as f:
        manifest = json.load(f)
    
    print(f"\nDirectories in quarantine ({quarantine_path}):")
    items = list(manifest["moved_directories"].items())
    
    for i, (original, info) in enumerate(items, 1):
        print(f"{i:2d}. {original}")  
        print(f"     → {info['reason']} ({info['size_mb']} MB, {info['file_count']} files)")
    
    print("\nEnter numbers to restore (comma-separated), or 'all' for everything:")
    selection = input("Restore: ").strip()
    
    if selection.lower() == 'all':
        indices = list(range(len(items)))
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',') if x.strip()]
        except ValueError:
            print("Invalid selection.")
            return
    
    for idx in indices:
        if 0 <= idx < len(items):
            original, info = items[idx]
            moved_path = Path(info["moved_to"])
            
            if moved_path.exists():
                try:
                    shutil.move(str(moved_path), original)
                    print(f"✅ Restored: {original}")
                except Exception as e:
                    print(f"❌ Failed to restore {original}: {e}")
            else:
                print(f"⚠️  {moved_path} not found in quarantine")

if __name__ == "__main__":
    quarantine_folders = list_quarantine_folders()
    
    if not quarantine_folders:
        exit(1)
    
    if len(quarantine_folders) == 1:
        selected = quarantine_folders[0]
    else:
        try:
            choice = int(input("\nSelect quarantine folder: ")) - 1
            selected = quarantine_folders[choice]
        except (ValueError, IndexError):
            print("Invalid selection.")
            exit(1)
    
    restore_selective(selected)
