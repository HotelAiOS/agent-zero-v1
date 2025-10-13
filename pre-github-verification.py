#!/usr/bin/env python3
"""
Pre-GitHub Verification Script for Agent Zero V1
Weryfikuje czy zostały tylko ważne pliki przed push na GitHub
"""

import os
import json
from pathlib import Path
from datetime import datetime
import subprocess

class RepositoryVerification:
    def __init__(self, base_path="."):
        self.base_path = Path(base_path)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.verification_report = {
            "verification_date": datetime.now().isoformat(),
            "essential_files": {},
            "suspicious_files": {},
            "missing_critical": {},
            "recommendations": [],
            "git_status": {},
            "size_analysis": {}
        }
    
    def get_git_status(self):
        """Sprawdza status git repository"""
        try:
            # Git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=self.base_path)
            status_lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            # Git branch
            branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                         capture_output=True, text=True, cwd=self.base_path)
            current_branch = branch_result.stdout.strip()
            
            # Untracked files count
            untracked = [line for line in status_lines if line.startswith('??')]
            modified = [line for line in status_lines if line.startswith(' M')]
            added = [line for line in status_lines if line.startswith('A ')]
            
            self.verification_report["git_status"] = {
                "current_branch": current_branch,
                "total_changes": len(status_lines),
                "untracked_files": len(untracked),
                "modified_files": len(modified),
                "added_files": len(added),
                "status_detail": status_lines[:20]  # First 20 changes
            }
            
            return True
        except Exception as e:
            print(f"⚠️ Git status check failed: {e}")
            return False
    
    def verify_essential_structure(self):
        """Weryfikuje czy są wszystkie kluczowe komponenty Agent Zero V1"""
        
        # CORE AGENT ZERO V1 COMPONENTS
        essential_paths = {
            # Main application structure
            "src/": "Core source directory",
            "src/core/": "Core system components", 
            "src/agents/": "Agent system",
            "src/api/": "API layer",
            "src/intelligence/": "Intelligence layer V2.0",
            "src/database/": "Database integration",
            "src/websocket/": "WebSocket service",
            
            # Configuration
            "config/": "Configuration files",
            "docker-compose.yml": "Docker orchestration",
            "requirements.txt": "Python dependencies",
            ".env.example": "Environment template",
            
            # Documentation
            "README.md": "Main documentation",
            "docs/": "Documentation directory",
            
            # Tests
            "tests/": "Test suite",
            
            # Scripts
            "scripts/": "Utility scripts",
            "run.py": "Main entry point",
            "cli.py": "Command line interface"
        }
        
        found_essential = {}
        missing_critical = {}
        
        for path, description in essential_paths.items():
            full_path = self.base_path / path
            if full_path.exists():
                if full_path.is_dir():
                    file_count = len(list(full_path.rglob("*")))
                    size_mb = self.get_directory_size(full_path) / (1024 * 1024)
                else:
                    file_count = 1
                    size_mb = full_path.stat().st_size / (1024 * 1024)
                
                found_essential[path] = {
                    "description": description,
                    "type": "directory" if full_path.is_dir() else "file",
                    "size_mb": round(size_mb, 2),
                    "file_count": file_count
                }
            else:
                missing_critical[path] = description
        
        self.verification_report["essential_files"] = found_essential
        self.verification_report["missing_critical"] = missing_critical
        
        return len(missing_critical) == 0
    
    def scan_for_suspicious_files(self):
        """Skanuje w poszukiwaniu podejrzanych plików które mogły zostać"""
        
        suspicious_patterns = [
            # Backup/temp files that might remain
            ("*backup*", "Backup files"),
            ("*temp*", "Temporary files"), 
            ("*.tmp", "Temporary files"),
            ("*cache*", "Cache files"),
            ("*log*", "Log files"),
            
            # Virtual environments
            ("venv*/", "Virtual environment"),
            ("*env*/", "Environment folders"),
            ("node_modules/", "Node modules"),
            
            # IDE/Editor files
            (".vscode/", "VS Code settings"),
            (".idea/", "IntelliJ IDEA"),
            ("*.swp", "Vim swap files"),
            ("*.swo", "Vim swap files"),
            
            # OS files
            (".DS_Store", "macOS files"),
            ("Thumbs.db", "Windows files"),
            ("desktop.ini", "Windows files"),
            
            # Development artifacts
            ("*.pyc", "Python compiled"),
            ("__pycache__/", "Python cache"),
            (".pytest_cache/", "Pytest cache"),
            ("*.egg-info/", "Python package info")
        ]
        
        suspicious_found = {}
        
        for pattern, description in suspicious_patterns:
            matches = list(self.base_path.glob(pattern)) + list(self.base_path.rglob(pattern))
            
            if matches:
                for match in matches:
                    rel_path = match.relative_to(self.base_path)
                    size_mb = self.get_directory_size(match) / (1024 * 1024) if match.is_dir() else match.stat().st_size / (1024 * 1024)
                    
                    suspicious_found[str(rel_path)] = {
                        "pattern": pattern,
                        "description": description,
                        "size_mb": round(size_mb, 2),
                        "type": "directory" if match.is_dir() else "file"
                    }
        
        self.verification_report["suspicious_files"] = suspicious_found
        return len(suspicious_found) == 0
    
    def analyze_repository_size(self):
        """Analizuje rozmiary folderów"""
        
        directory_sizes = {}
        
        for item in self.base_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                size_mb = self.get_directory_size(item) / (1024 * 1024)
                file_count = len(list(item.rglob("*")))
                
                directory_sizes[item.name] = {
                    "size_mb": round(size_mb, 2),
                    "file_count": file_count
                }
        
        # Sort by size
        sorted_dirs = sorted(directory_sizes.items(), key=lambda x: x[1]["size_mb"], reverse=True)
        
        self.verification_report["size_analysis"] = {
            "total_directories": len(directory_sizes),
            "largest_directories": sorted_dirs[:10],
            "total_size_mb": round(sum(d["size_mb"] for d in directory_sizes.values()), 2)
        }
        
        return sorted_dirs
    
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
    
    def generate_recommendations(self):
        """Generuje rekomendacje na podstawie weryfikacji"""
        
        recommendations = []
        
        # Critical missing files
        if self.verification_report["missing_critical"]:
            recommendations.append({
                "severity": "HIGH",
                "category": "Missing Files",
                "message": f"Missing {len(self.verification_report['missing_critical'])} critical files/directories",
                "action": "Review and restore essential components before GitHub push"
            })
        
        # Suspicious files found
        if self.verification_report["suspicious_files"]:
            large_suspicious = [f for f, info in self.verification_report["suspicious_files"].items() 
                              if info["size_mb"] > 10]
            if large_suspicious:
                recommendations.append({
                    "severity": "MEDIUM", 
                    "category": "Cleanup Needed",
                    "message": f"Found {len(large_suspicious)} large suspicious files (>10MB)",
                    "action": "Clean up before pushing to GitHub"
                })
        
        # Repository size
        total_size = self.verification_report["size_analysis"]["total_size_mb"]
        if total_size > 500:
            recommendations.append({
                "severity": "MEDIUM",
                "category": "Repository Size",
                "message": f"Repository is large ({total_size:.1f}MB)",
                "action": "Consider cleaning up large directories"
            })
        
        # Git status
        git_status = self.verification_report["git_status"]
        if git_status.get("untracked_files", 0) > 50:
            recommendations.append({
                "severity": "LOW",
                "category": "Git Status", 
                "message": f"Many untracked files ({git_status['untracked_files']})",
                "action": "Review .gitignore or stage important files"
            })
        
        # All good case
        if not recommendations:
            recommendations.append({
                "severity": "INFO",
                "category": "Status",
                "message": "Repository looks clean and ready for GitHub",
                "action": "Proceed with git push"
            })
        
        self.verification_report["recommendations"] = recommendations
        return recommendations
    
    def run_full_verification(self):
        """Uruchamia pełną weryfikację"""
        print("🔍 Starting Repository Verification...")
        print("="*50)
        
        # Git status
        print("📊 Checking Git status...")
        self.get_git_status()
        
        # Essential structure
        print("🏗️  Verifying essential structure...")
        structure_ok = self.verify_essential_structure()
        
        # Suspicious files
        print("🕵️  Scanning for suspicious files...")
        clean_scan = self.scan_for_suspicious_files()
        
        # Size analysis
        print("📏 Analyzing repository size...")
        self.analyze_repository_size()
        
        # Generate recommendations
        print("💡 Generating recommendations...")
        recommendations = self.generate_recommendations()
        
        # Save report
        self.save_verification_report()
        
        # Print summary
        self.print_verification_summary()
        
        return self.verification_report
    
    def save_verification_report(self):
        """Zapisuje raport weryfikacji"""
        report_file = self.base_path / f"verification_report_{self.timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(self.verification_report, f, indent=2)
        
        # Markdown summary
        self.create_verification_markdown()
        
        print(f"📋 Verification report saved: {report_file}")
    
    def create_verification_markdown(self):
        """Tworzy markdown summary weryfikacji"""
        summary_file = self.base_path / f"VERIFICATION_SUMMARY_{self.timestamp}.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# Repository Verification Report - {self.timestamp}\n\n")
            f.write(f"**Verification Date:** {self.verification_report['verification_date']}\n\n")
            
            # Git Status
            git_status = self.verification_report["git_status"]
            f.write("## 📊 Git Status\n")
            f.write(f"- **Branch:** {git_status.get('current_branch', 'unknown')}\n")
            f.write(f"- **Total Changes:** {git_status.get('total_changes', 0)}\n")
            f.write(f"- **Untracked Files:** {git_status.get('untracked_files', 0)}\n")
            f.write(f"- **Modified Files:** {git_status.get('modified_files', 0)}\n\n")
            
            # Essential Files
            f.write("## ✅ Essential Components Found\n")
            for path, info in self.verification_report["essential_files"].items():
                f.write(f"- **{path}** - {info['description']} ({info['size_mb']}MB)\n")
            f.write("\n")
            
            # Missing Critical
            if self.verification_report["missing_critical"]:
                f.write("## ❌ Missing Critical Components\n")
                for path, desc in self.verification_report["missing_critical"].items():
                    f.write(f"- **{path}** - {desc}\n")
                f.write("\n")
            
            # Suspicious Files
            if self.verification_report["suspicious_files"]:
                f.write("## ⚠️ Suspicious Files Found\n")
                for path, info in self.verification_report["suspicious_files"].items():
                    f.write(f"- **{path}** - {info['description']} ({info['size_mb']}MB)\n")
                f.write("\n")
            
            # Size Analysis
            f.write("## 📏 Size Analysis\n")
            size_info = self.verification_report["size_analysis"]
            f.write(f"- **Total Size:** {size_info['total_size_mb']}MB\n")
            f.write(f"- **Total Directories:** {size_info['total_directories']}\n\n")
            
            f.write("### Largest Directories:\n")
            for dir_name, info in size_info["largest_directories"][:5]:
                f.write(f"- **{dir_name}:** {info['size_mb']}MB ({info['file_count']} files)\n")
            f.write("\n")
            
            # Recommendations
            f.write("## 💡 Recommendations\n")
            for rec in self.verification_report["recommendations"]:
                icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️", "INFO": "✅"}[rec["severity"]]
                f.write(f"### {icon} {rec['category']} ({rec['severity']})\n")
                f.write(f"**Issue:** {rec['message']}\n\n")
                f.write(f"**Action:** {rec['action']}\n\n")
        
        print(f"📝 Verification summary: {summary_file}")
    
    def print_verification_summary(self):
        """Wyświetla podsumowanie weryfikacji"""
        print("\n" + "="*60)
        print("🔍 REPOSITORY VERIFICATION COMPLETED")
        print("="*60)
        
        # Git status
        git_status = self.verification_report["git_status"]
        print(f"📊 Git: {git_status.get('current_branch', 'unknown')} branch, {git_status.get('total_changes', 0)} changes")
        
        # Essential components
        essential_count = len(self.verification_report["essential_files"])
        missing_count = len(self.verification_report["missing_critical"])
        print(f"✅ Essential: {essential_count} found, {missing_count} missing")
        
        # Suspicious files
        suspicious_count = len(self.verification_report["suspicious_files"])
        if suspicious_count > 0:
            suspicious_size = sum(info["size_mb"] for info in self.verification_report["suspicious_files"].values())
            print(f"⚠️  Suspicious: {suspicious_count} files ({suspicious_size:.1f}MB)")
        else:
            print("✅ No suspicious files found")
        
        # Total size
        total_size = self.verification_report["size_analysis"]["total_size_mb"]
        print(f"📏 Repository Size: {total_size:.1f}MB")
        
        # Recommendations
        print(f"\n💡 Recommendations: {len(self.verification_report['recommendations'])}")
        for rec in self.verification_report["recommendations"]:
            icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️", "INFO": "✅"}[rec["severity"]]
            print(f"   {icon} {rec['message']}")
        
        print("="*60)

def main():
    """Uruchamia weryfikację repozytorium"""
    verifier = RepositoryVerification()
    
    print("🛡️  REPOSITORY VERIFICATION FOR GITHUB")
    print("="*45)
    print("This will verify if repository is ready for GitHub push.")
    print("")
    
    try:
        report = verifier.run_full_verification()
        
        # Quick decision helper
        high_issues = [r for r in report["recommendations"] if r["severity"] == "HIGH"]
        medium_issues = [r for r in report["recommendations"] if r["severity"] == "MEDIUM"]
        
        if high_issues:
            print(f"\n🚨 {len(high_issues)} HIGH severity issues found!")
            print("   RECOMMENDATION: Fix issues before GitHub push")
        elif medium_issues:
            print(f"\n⚠️ {len(medium_issues)} MEDIUM severity issues found")
            print("   RECOMMENDATION: Consider cleanup, but can proceed")
        else:
            print("\n✅ Repository looks ready for GitHub!")
            print("   RECOMMENDATION: Safe to proceed with git push")
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return

if __name__ == "__main__":
    main()