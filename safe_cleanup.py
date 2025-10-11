#!/usr/bin/env python3
"""
GitHub-based Safe Cleanup Script for Agent Zero V1
Przenosi potencjalne śmieci do folderu review zamiast usuwania
"""

import os
import shutil
import subprocess
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

class SafeCleanupManager:
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.cleanup_dir = self.project_root / "_cleanup_review"
        self.log_file = self.cleanup_dir / "cleanup_log.txt"
        
    def analyze_and_move_junk(self):
        """Główna funkcja - analiza i przeniesienie śmieci"""
        
        # Stwórz folder review
        self.cleanup_dir.mkdir(exist_ok=True)
        
        # Identyfikuj pliki do przeniesienia
        files_to_move = self.identify_junk_files()
        
        # Generuj raport
        self.generate_cleanup_report(files_to_move)
        
        # Przenieś pliki z potwierdzeniem
        self.move_files_with_confirmation(files_to_move)
        
        return files_to_move
    
    def identify_junk_files(self) -> Dict[str, List[Path]]:
        """Identyfikuje potencjalne śmieci na podstawie wzorców"""
        
        junk_categories = {
            'backup_files': [],
            'duplicate_versions': [],
            'old_configs': [],
            'temp_files': [],
            'database_files': []
        }
        
        # 1. Backup files - 100% pewności
        for pattern in ['*.bak', '*.backup', '*_backup*', '*.yml.bak_*']:
            junk_categories['backup_files'].extend(self.project_root.glob(pattern))
        
        # 2. Database files w repo - nie powinny być w Git
        db_files = list(self.project_root.glob('*.db')) + list(self.project_root.glob('*.sqlite*'))
        junk_categories['database_files'] = db_files
        
        # 3. Duplicate versions - analizuj podobne nazwy
        junk_categories['duplicate_versions'] = self.find_version_duplicates()
        
        # 4. Old configs - wielokrotne wersje tego samego
        junk_categories['old_configs'] = self.find_config_duplicates()
        
        # 5. Temp i test files
        temp_patterns = ['*_temp.py', '*_test_*.py', '*_backup.py', '*_old.py', '*_fixed.py']
        for pattern in temp_patterns:
            junk_categories['temp_files'].extend(self.project_root.glob(pattern))
        
        return junk_categories
    
    def find_version_duplicates(self) -> List[Path]:
        """Znajduje duplikaty bazując na nazwach i timestamp"""
        
        duplicates = []
        
        # Grupuj pliki po base name (bez -fixed, -v2, etc)
        file_groups = {}
        
        for py_file in self.project_root.glob('*.py'):
            # Wyciągnij base name
            base_name = self.extract_base_name(py_file.stem)
            
            if base_name in file_groups:
                file_groups[base_name].append(py_file)
            else:
                file_groups[base_name] = [py_file]
        
        # Dla każdej grupy, zachowaj najnowszy/najlepszy
        for base_name, files in file_groups.items():
            if len(files) > 1:
                # Sortuj po dacie modyfikacji (najnowszy pierwszy)
                files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
                
                # Pierwszy to najnowszy - zachowaj
                # Reszta to potencjalne duplikaty
                potential_dupes = files[1:]
                
                # Dodatkowa weryfikacja - nie ruszaj jeśli różnice w zawartości
                keeper = files[0]
                for dupe in potential_dupes:
                    if self.are_significantly_different(keeper, dupe):
                        continue  # Zachowaj jeśli znaczące różnice
                    duplicates.append(dupe)
        
        return duplicates
    
    def extract_base_name(self, filename: str) -> str:
        """Wyciąga bazową nazwę bez suffixów typu -fixed, -v2"""
        
        # Usuń typowe suffixes
        suffixes_to_remove = [
            '-fixed', '-complete', '-standalone', '-production', 
            '-enhanced', '-v2', '-v3', '-final', '-new', '-old',
            '_fixed', '_complete', '_standalone', '_production'
        ]
        
        base = filename.lower()
        for suffix in suffixes_to_remove:
            base = base.replace(suffix, '')
        
        return base
    
    def find_config_duplicates(self) -> List[Path]:
        """Znajduje duplikaty konfiguracji"""
        
        config_files = []
        
        # Docker compose duplikates
        docker_files = list(self.project_root.glob('docker-compose*.yml'))
        if len(docker_files) > 1:
            # Zachowaj docker-compose.yml lub najnowszy
            docker_files.sort(key=lambda f: (
                0 if f.name == 'docker-compose.yml' else 1,  # Priorytet dla standardowej nazwy
                -f.stat().st_mtime  # Potem najnowszy
            ))
            config_files.extend(docker_files[1:])  # Reszta to duplikaty
        
        # Config.py duplikates  
        config_py_files = list(self.project_root.glob('config*.py'))
        if len(config_py_files) > 1:
            config_py_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
            config_files.extend(config_py_files[1:])
        
        return config_files
    
    def are_significantly_different(self, file1: Path, file2: Path) -> bool:
        """Sprawdza czy pliki różnią się znacząco"""
        
        try:
            with open(file1, 'r', encoding='utf-8', errors='ignore') as f1:
                content1 = f1.read()
            with open(file2, 'r', encoding='utf-8', errors='ignore') as f2:
                content2 = f2.read()
            
            # Usuń whitespace i komentarze dla porównania
            clean1 = self.clean_content_for_comparison(content1)
            clean2 = self.clean_content_for_comparison(content2)
            
            # Jeśli różnica > 20% to znaczące różnice
            if len(clean1) == 0 and len(clean2) == 0:
                return False
            
            similarity = self.calculate_similarity(clean1, clean2)
            return similarity < 0.8  # Mniej niż 80% podobieństwa = znaczące różnice
            
        except Exception:
            return True  # W razie błędu, zachowaj bezpiecznie
    
    def clean_content_for_comparison(self, content: str) -> str:
        """Czyści zawartość do porównania"""
        import re
        
        # Usuń komentarze
        content = re.sub(r'#.*', '', content)
        # Usuń puste linie
        content = re.sub(r'\n\s*\n', '\n', content)
        # Normalizuj whitespace
        content = re.sub(r'\s+', ' ', content)
        
        return content.strip()
    
    def calculate_similarity(self, text1: str, text2: str) -> float:
        """Oblicza podobieństwo między tekstami"""
        
        if len(text1) == 0 and len(text2) == 0:
            return 1.0
        if len(text1) == 0 or len(text2) == 0:
            return 0.0
        
        # Prosta miara podobieństwa bazowana na common substring
        common_length = 0
        min_length = min(len(text1), len(text2))
        
        for i in range(min_length):
            if text1[i] == text2[i]:
                common_length += 1
            else:
                break
        
        return common_length / max(len(text1), len(text2))
    
    def generate_cleanup_report(self, files_to_move: Dict[str, List[Path]]):
        """Generuje szczegółowy raport cleanup"""
        
        report = []
        report.append(f"# Agent Zero V1 - Cleanup Report")
        report.append(f"Generated: {datetime.now()}")
        report.append(f"=" * 50)
        
        total_files = sum(len(files) for files in files_to_move.values())
        report.append(f"\nTotal files to review: {total_files}")
        
        for category, files in files_to_move.items():
            if files:
                report.append(f"\n## {category.upper()} ({len(files)} files):")
                for file_path in files:
                    size_kb = file_path.stat().st_size / 1024
                    modified = datetime.fromtimestamp(file_path.stat().st_mtime)
                    report.append(f"  - {file_path.name} ({size_kb:.1f}KB, {modified.strftime('%Y-%m-%d %H:%M')})")
        
        # Zapisz raport
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        # Wyświetl na konsoli
        print('\n'.join(report))
    
    def move_files_with_confirmation(self, files_to_move: Dict[str, List[Path]]):
        """Przenosi pliki z opcją potwierdzenia"""
        
        total_files = sum(len(files) for files in files_to_move.values())
        
        if total_files == 0:
            print("✅ Nie znaleziono plików do przeniesienia.")
            return
        
        print(f"\n🔍 Znaleziono {total_files} plików do przeglądu.")
        print(f"📁 Zostaną przeniesione do: {self.cleanup_dir}")
        
        choice = input("\nWybierz akcję:\n1. Przenieś wszystkie\n2. Przegląd po kategorii\n3. Anuluj\nWybór (1/2/3): ")
        
        if choice == '1':
            self.move_all_files(files_to_move)
        elif choice == '2':
            self.move_with_category_review(files_to_move)
        else:
            print("❌ Anulowano.")
    
    def move_all_files(self, files_to_move: Dict[str, List[Path]]):
        """Przenosi wszystkie pliki"""
        
        moved_count = 0
        
        for category, files in files_to_move.items():
            category_dir = self.cleanup_dir / category
            category_dir.mkdir(exist_ok=True)
            
            for file_path in files:
                try:
                    destination = category_dir / file_path.name
                    shutil.move(str(file_path), str(destination))
                    moved_count += 1
                    print(f"📦 Przeniesiono: {file_path.name}")
                except Exception as e:
                    print(f"❌ Błąd przenoszenia {file_path.name}: {e}")
        
        print(f"\n✅ Przeniesiono {moved_count} plików do {self.cleanup_dir}")
        print(f"📝 Szczegóły w: {self.log_file}")
    
    def move_with_category_review(self, files_to_move: Dict[str, List[Path]]):
        """Przenosi z przeglądem po kategoriach"""
        
        for category, files in files_to_move.items():
            if not files:
                continue
                
            print(f"\n📂 Kategoria: {category} ({len(files)} plików)")
            for file_path in files:
                print(f"  - {file_path.name}")
            
            choice = input(f"Przenieść pliki z kategorii '{category}'? (y/n/s=skip): ")
            
            if choice.lower() == 'y':
                category_dir = self.cleanup_dir / category
                category_dir.mkdir(exist_ok=True)
                
                for file_path in files:
                    try:
                        destination = category_dir / file_path.name
                        shutil.move(str(file_path), str(destination))
                        print(f"📦 Przeniesiono: {file_path.name}")
                    except Exception as e:
                        print(f"❌ Błąd: {e}")
            elif choice.lower() == 's':
                continue
            else:
                print("⏭️ Pominięto kategorię")


# Funkcja uruchomieniowa
def main():
    print("🧹 Agent Zero V1 - Safe Cleanup Tool")
    print("=" * 40)
    
    cleaner = SafeCleanupManager()
    files_found = cleaner.analyze_and_move_junk()
    
    print("\n🎯 Cleanup zakończony!")
    print(f"📁 Pliki przeniesione do: {cleaner.cleanup_dir}")
    print("💡 Możesz je przejrzeć i usunąć ręcznie lub przywrócić jeśli potrzebne.")

if __name__ == "__main__":
    main()
