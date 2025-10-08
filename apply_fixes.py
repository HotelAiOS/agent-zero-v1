#!/usr/bin/env python3
"""
Agent Zero V1 - Automated Critical Fixes Application
Applies all critical fixes to the project

Usage:
    python apply_fixes.py --project-root /path/to/agent-zero-v1
    python apply_fixes.py --project-root /path/to/agent-zero-v1 --skip-docker
"""

import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


class AgentZeroFixer:
    """Automated fix application for Agent Zero V1"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / f"backups_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.fixes_applied = []

        # Mapping: source file -> target file in project
        self.file_mappings = {
            "neo4j_client.py": "shared/knowledge/neo4j_client.py",
            "agent_executor.py": "src/core/agent_executor.py",
            "task_decomposer.py": "shared/orchestration/task_decomposer.py",
            "docker-compose.yml": "docker-compose.yml"
        }

    def backup_files(self, files: list) -> bool:
        """Create backup of files before modification"""
        print(f"\n📦 Creating backup in {self.backup_dir}")
        self.backup_dir.mkdir(exist_ok=True, parents=True)

        for file_path in files:
            if file_path.exists():
                backup_path = self.backup_dir / file_path.name
                shutil.copy2(file_path, backup_path)
                print(f"  ✅ Backed up: {file_path.name}")
            else:
                print(f"  ⚠️  File not found (will be created): {file_path.name}")
        return True

    def apply_fix(self, source_file: str, target_relative: str, fix_name: str) -> bool:
        """Apply a single fix"""
        print(f"\n🔧 Applying {fix_name}...")

        source = Path(source_file)
        target = self.project_root / target_relative

        # Check source exists
        if not source.exists():
            print(f"  ❌ Source file not found: {source_file}")
            print(f"     Make sure {source_file} is in the same directory as this script")
            return False

        # Create target directory if needed
        target.parent.mkdir(parents=True, exist_ok=True)

        # Backup if target exists
        if target.exists():
            self.backup_files([target])

        # Copy fixed file
        shutil.copy2(source, target)
        print(f"  ✅ Applied: {target_relative}")
        self.fixes_applied.append(fix_name)
        return True

    def apply_all_fixes(self) -> bool:
        """Apply all fixes"""
        print("=" * 80)
        print("AGENT ZERO V1 - AUTOMATED CRITICAL FIXES")
        print("=" * 80)
        print(f"Project root: {self.project_root}")
        print(f"Backup directory: {self.backup_dir}")

        success = True

        # Apply each fix
        success &= self.apply_fix(
            "neo4j_client.py",
            self.file_mappings["neo4j_client.py"],
            "Neo4j Client Fix (A0-5)"
        )

        success &= self.apply_fix(
            "agent_executor.py",
            self.file_mappings["agent_executor.py"],
            "AgentExecutor Fix (A0-6)"
        )

        success &= self.apply_fix(
            "task_decomposer.py",
            self.file_mappings["task_decomposer.py"],
            "TaskDecomposer Fix (TECH-001)"
        )

        success &= self.apply_fix(
            "docker-compose.yml",
            self.file_mappings["docker-compose.yml"],
            "Docker Compose Configuration"
        )

        return success

    def verify_fixes(self) -> bool:
        """Verify that fixes were applied correctly"""
        print("\n✓ Verifying fixes...")

        checks = [
            (self.project_root / path, name)
            for name, path in [
                ("Neo4j Client", self.file_mappings["neo4j_client.py"]),
                ("AgentExecutor", self.file_mappings["agent_executor.py"]),
                ("TaskDecomposer", self.file_mappings["task_decomposer.py"]),
                ("Docker Compose", self.file_mappings["docker-compose.yml"])
            ]
        ]

        all_ok = True
        for file_path, name in checks:
            if file_path.exists():
                print(f"  ✅ {name}: OK")
            else:
                print(f"  ❌ {name}: MISSING")
                all_ok = False

        return all_ok

    def restart_services(self) -> bool:
        """Restart Docker services"""
        print("\n🔄 Restarting Docker services...")

        try:
            os.chdir(self.project_root)

            # Stop services
            print("  Stopping services...")
            subprocess.run(["docker-compose", "down"], check=True)
            print("  ✅ Services stopped")

            # Start services
            print("  Starting services...")
            subprocess.run(["docker-compose", "up", "-d"], check=True)
            print("  ✅ Services started")

            # Wait for health checks
            print("  Waiting for services to be healthy (30s)...")
            import time
            time.sleep(30)

            return True
        except subprocess.CalledProcessError as e:
            print(f"  ❌ Docker restart failed: {e}")
            return False
        except Exception as e:
            print(f"  ❌ Error: {e}")
            return False

    def run(self, skip_docker: bool = False) -> bool:
        """Run all fixes"""

        # Verify project root
        if not self.project_root.exists():
            print(f"❌ Project root does not exist: {self.project_root}")
            return False

        # Apply fixes
        success = self.apply_all_fixes()

        if not success:
            print("\n❌ Some fixes failed to apply")
            print("   Check that all fix files are in the same directory as this script:")
            print("   - neo4j_client.py")
            print("   - agent_executor.py")
            print("   - task_decomposer.py")
            print("   - docker-compose.yml")
            return False

        # Verify
        if not self.verify_fixes():
            print("\n❌ Verification failed")
            return False

        # Restart services
        if not skip_docker:
            self.restart_services()
        else:
            print("\n⚠️  Skipping Docker restart (--skip-docker)")
            print("   Run manually: cd", self.project_root, "&& docker-compose up -d")

        # Summary
        print("\n" + "=" * 80)
        print("✅ ALL FIXES APPLIED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nFixes applied:")
        for fix in self.fixes_applied:
            print(f"  ✅ {fix}")
        print(f"\nBackup location: {self.backup_dir}")
        print("\nNext steps:")
        print("1. Test Neo4j connection:")
        print("   docker exec -it agent-zero-neo4j cypher-shell -u neo4j -p agent-pass")
        print("\n2. Check service health:")
        print("   docker-compose ps")
        print("\n3. View logs:")
        print("   docker-compose logs -f neo4j")
        print("\n4. Run integration tests:")
        print("   pytest tests/test_full_integration.py -v")

        return True


def main():
    parser = argparse.ArgumentParser(
        description="Apply Agent Zero V1 critical fixes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Apply all fixes and restart Docker services
  python apply_fixes.py --project-root /home/user/agent-zero-v1

  # Apply fixes but skip Docker restart
  python apply_fixes.py --project-root /home/user/agent-zero-v1 --skip-docker
        """
    )
    parser.add_argument(
        "--project-root",
        type=str,
        required=True,
        help="Path to agent-zero-v1 project root directory"
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker service restart (apply fixes only)"
    )

    args = parser.parse_args()

    # Verify current directory contains fix files
    required_files = ["neo4j_client.py", "agent_executor.py", "task_decomposer.py", "docker-compose.yml"]
    missing = [f for f in required_files if not Path(f).exists()]

    if missing:
        print("❌ Error: Missing fix files in current directory:")
        for f in missing:
            print(f"   - {f}")
        print("\nMake sure you run this script from the directory containing all fix files.")
        sys.exit(1)

    # Run fixer
    fixer = AgentZeroFixer(args.project_root)
    success = fixer.run(skip_docker=args.skip_docker)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
