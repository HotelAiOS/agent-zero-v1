#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
System Audit for Agent Zero V1
- Skanuje repozytorium i generuje raport audytowy.
- Zgodny z planem: Audit i Stabilizacja → inventory, duplikaty, dead code, brak testów, konflikty config, zależności, health.

Wyniki:
- audit_reports/audit_summary.json (zbiorczy JSON)
- audit_reports/audit_summary.md   (czytelne podsumowanie)
- audit_reports/dupe_map.json      (mapa duplikatów)
- audit_reports/unused_report.json (podejrzenie martwego kodu - heurystyka)
- audit_reports/config_audit.json  (spójność konfiguracji)
- audit_reports/dep_audit.json     (zależności, wersje, konflikty)
- audit_reports/test_coverage.json (pokrycie testami na poziomie plików - heurystyka)

Wymagania: Python 3.11+
Opcjonalnie: git, pip, venv, flake8/isort/black (jeśli chcesz rozszerzyć audyt jakości)
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

# Konfiguracja
REPO_ROOT = Path(__file__).resolve().parent
REPORT_DIR = REPO_ROOT / "audit_reports"
INCLUDE_EXT = {".py", ".yml", ".yaml", ".toml", ".json", ".md", ".sh", ".fish"}
PY_EXT = {".py"}
IGNORE_DIRS = {
    ".git",
    ".github",
    "__pycache__",
    "venv",
    ".venv",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".security",
    "exported-assets",
    "exported-assets (1)",
    "backups",
    "backups_",
    "checkpoints",
}
# katalogi z artefaktami binarnymi/danymi testowymi, które mogą fałszować duplikaty
BINARY_EXT = {".db", ".pkl", ".zip", ".png", ".jpeg", ".jpg", ".gif", ".pdf", ".html", ".log", ".csv", ".txt"}
CONFIG_FILES = [
    "pyproject.toml",
    "requirements.txt",
    "requirements-production.txt",
    "requirements-v2.txt",
    "requirements-intelligence-v2.txt",
    "config.yaml",
    ".env",
    ".env.example",
    "docker-compose.yml",
]

TEST_DIR_HINTS = {"tests", "test", "integrationtests.py"}
TEST_FILE_PATTERNS = [
    re.compile(r"^test_.*\.py$"),
    re.compile(r".*_test\.py$"),
]

# Heurystyka "dead code": nazwy symboli niezreferowanych nigdzie indziej
# Bez zależności zewn., robimy conservative check (AST import map + prosta referencja)
# Dla bardziej dokładnych wyników można spiąć z "vulture" poza tym skryptem.


def is_ignored(path: Path) -> bool:
    parts = set(path.parts)
    if any(ignored in parts for ignored in IGNORE_DIRS):
        return True
    return False


def walk_files(root: Path) -> List[Path]:
    files = []
    for p in root.rglob("*"):
        if p.is_file():
            if is_ignored(p):
                continue
            if p.suffix.lower() in INCLUDE_EXT or (p.suffix.lower() in BINARY_EXT):
                files.append(p)
    return files


def hash_file(path: Path, block_size: int = 65536) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(block_size), b""):
                h.update(chunk)
    except Exception:
        return ""
    return h.hexdigest()


def detect_duplicates(files: List[Path]) -> Dict[str, List[str]]:
    # Duplikaty wykrywamy po hash-u pliku. Filtrujemy binaria, aby nie rozdmuchać raportu.
    dup_map: Dict[str, List[str]] = defaultdict(list)
    for f in files:
        if f.suffix.lower() in BINARY_EXT:
            continue
        h = hash_file(f)
        if h:
            dup_map[h].append(str(f.relative_to(REPO_ROOT)))
    # Zostaw tylko te grupy, które mają >1
    return {k: v for k, v in dup_map.items() if len(v) > 1}


def list_python_files(files: List[Path]) -> List[Path]:
    return [f for f in files if f.suffix.lower() in PY_EXT]


def parse_imports(py_file: Path) -> Tuple[Set[str], Set[str]]:
    """Zwraca (importowane_moduły, zdefiniowane_nazwy) dla heurystyki dead code"""
    imported: Set[str] = set()
    defined: Set[str] = set()
    try:
        src = py_file.read_text(encoding="utf-8", errors="ignore")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
            elif isinstance(node, ast.FunctionDef):
                defined.add(node.name)
            elif isinstance(node, ast.ClassDef):
                defined.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        defined.add(target.id)
    except Exception:
        pass
    return imported, defined


def build_reference_index(py_files: List[Path]) -> Dict[str, Set[str]]:
    """
    Buduje prosty indeks referencji: nazwa_symbolu -> z jakich plików referencja
    Szukamy substringów nazw zdefiniowanych; heurystyka, ale szybka i lokalna.
    """
    index: Dict[str, Set[str]] = defaultdict(set)
    contents: Dict[Path, str] = {}
    # Wczytaj treść 1x
    for p in py_files:
        try:
            contents[p] = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            contents[p] = ""
    # Zbierz definicje
    definitions: Dict[Path, Set[str]] = {}
    for p in py_files:
        _, defined = parse_imports(p)
        definitions[p] = defined
    # Szukaj referencji
    for p in py_files:
        content = contents[p]
        for other, defs in definitions.items():
            for name in defs:
                # Ograniczamy "fałszywe trafienia" – tylko całe słowa
                # Używamy regexu z \b
                try:
                    if re.search(rf"\b{name}\b", content):
                        index[name].add(str(p.relative_to(REPO_ROOT)))
                except re.error:
                    # Nazwa może mieć nietypowe znaki - fallback
                    if name in content:
                        index[name].add(str(p.relative_to(REPO_ROOT)))
    return index


def detect_dead_code(py_files: List[Path]) -> Dict[str, Dict[str, List[str]]]:
    """
    Heurystyka: jeżeli symbol zdefiniowany nie jest referencjonowany poza plikiem,
    uznajemy jako 'potentially_unused'. To nie jest w 100% precyzyjne, ale daje sygnały do przeglądu.
    """
    definitions: Dict[str, Set[str]] = {}
    for p in py_files:
        _, defined = parse_imports(p)
        definitions[str(p.relative_to(REPO_ROOT))] = defined

    ref_index = build_reference_index(py_files)
    unused: Dict[str, List[str]] = defaultdict(list)
    for file_rel, defs in definitions.items():
        for name in defs:
            refs = ref_index.get(name, set())
            # jeśli nazwa występuje tylko w tym samym pliku -> kandydat do unused
            if refs == {file_rel} or len(refs) == 0:
                unused[file_rel].append(name)

    return {"potentially_unused": dict(unused)}


def detect_missing_tests(py_files: List[Path]) -> Dict[str, List[str]]:
    """
    Mapuje pliki produkcyjne do plików testowych heurystycznie:
    - Szuka w katalogach tests lub plików spełniających wzorce testowe
    - Plik py bez parującego testu dodaje do 'missing_tests'
    """
    test_files: List[Path] = []
    prod_files: List[Path] = []
    for p in py_files:
        rel = str(p.relative_to(REPO_ROOT))
        dirname = p.parent.name.lower()
        if any(h in rel for h in TEST_DIR_HINTS) or any(regex.match(p.name) for regex in TEST_FILE_PATTERNS):
            test_files.append(p)
        else:
            prod_files.append(p)

    test_names = {p.name for p in test_files}
    missing: List[str] = []
    for pf in prod_files:
        # heurystyka parująca: test_<nazwa> lub <nazwa>_test
        candidates = {f"test_{pf.name}", re.sub(r"\.py$", "_test.py", pf.name)}
        if not (candidates & test_names):
            missing.append(str(pf.relative_to(REPO_ROOT)))

    return {"missing_tests": missing, "tests_found": sorted(str(p.relative_to(REPO_ROOT)) for p in test_files)}


def audit_config() -> Dict[str, Dict]:
    """
    Sprawdza istnienie i podstawową spójność kluczowych plików konfiguracyjnych.
    """
    result: Dict[str, Dict] = {}
    for fname in CONFIG_FILES:
        p = REPO_ROOT / fname
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        result[fname] = {
            "exists": exists,
            "size": size,
            "path": str(p.relative_to(REPO_ROOT)) if exists else None,
        }
    # Dodatkowa walidacja docker-compose.yml (jeżeli istnieje)
    dc_path = REPO_ROOT / "docker-compose.yml"
    if dc_path.exists():
        try:
            content = dc_path.read_text(encoding="utf-8", errors="ignore")
            # Proste checki portów i usług
            services = re.findall(r"^\s{2,}([a-zA-Z0-9\-_]+):\s*$", content, flags=re.M)
            ports = re.findall(r"ports:\s*\n(?:\s*-\s*[\"']?(\d+):(\d+)[\"']?\s*\n)+", content, flags=re.M)
            result["docker_compose_analysis"] = {
                "service_count_guess": len(set(services)),
                "has_ports_mappings": bool(ports),
            }
        except Exception as e:
            result["docker_compose_analysis"] = {"error": str(e)}
    return result


def audit_dependencies() -> Dict[str, Dict]:
    """
    Parsuje pliki requirements* i pyproject.toml, wykrywa potencjalne konflikty wersji.
    Nie instaluje pakietów. Analiza heurystyczna.
    """
    req_files = [
        "requirements.txt",
        "requirements-production.txt",
        "requirements-v2.txt",
        "requirements-intelligence-v2.txt",
    ]
    req_map: Dict[str, Dict[str, str]] = {}
    version_re = re.compile(r"^\s*([a-zA-Z0-9_.\-]+)\s*(?:==|>=|<=|~=|>|<)\s*([a-zA-Z0-9_.\-]+)\s*$")

    for rf in req_files:
        p = REPO_ROOT / rf
        if not p.exists():
            continue
        packages: Dict[str, str] = {}
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # rozbij na 'pakiet' i resztę
            m = version_re.match(line)
            if m:
                packages[m.group(1).lower()] = line
            else:
                # brak blokady wersji – odnotuj surową linię
                pkg = line.split()[0].strip().lower()
                packages[pkg] = line
        req_map[rf] = packages

    # Wykryj konflikty: ta sama nazwa pakietu, różne deklaracje między plikami
    conflicts: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    for rf, pkgs in req_map.items():
        for name, spec in pkgs.items():
            conflicts[name][spec].append(rf)

    real_conflicts = {}
    for name, by_spec in conflicts.items():
        if len(by_spec) > 1:
            real_conflicts[name] = by_spec

    pyproject = REPO_ROOT / "pyproject.toml"
    pyproject_present = pyproject.exists()
    pyproject_size = pyproject.stat().st_size if pyproject_present else 0

    return {
        "requirements": req_map,
        "version_conflicts": real_conflicts,
        "pyproject": {"exists": pyproject_present, "size": pyproject_size},
    }


def git_status() -> Dict[str, str | List[str]]:
    def run(cmd: List[str]) -> Tuple[int, str]:
        try:
            res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
            return res.returncode, res.stdout.strip() or res.stderr.strip()
        except Exception as e:
            return 1, str(e)

    status = {}
    rc, out = run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status["branch"] = out if rc == 0 else "unknown"

    rc, out = run(["git", "status", "--porcelain"])
    changed = out.splitlines() if rc == 0 and out else []
    status["working_tree_changes"] = changed

    rc, out = run(["git", "log", "-1", "--pretty=%H %ci %s"])
    status["latest_commit"] = out if rc == 0 else "unknown"

    return status


def health_quick_checks() -> Dict[str, Dict]:
    """
    Szybkie sanity-check:
    - port conflicts (skrypt i plik lsof jeżeli obecny)
    - obecność głównych usług (pliki serwerów)
    """
    checks = {}

    # Sprawdź obecność kluczowych serwisów
    key_services = [
        "agent_zero_server.py",
        "collaboration_server.py",
        "analytics_server.py",
        "predictive_server.py",
        "team_formation_server.py",
        "quantum_intelligence_server.py",
    ]
    present = {svc: (REPO_ROOT / svc).exists() for svc in key_services}
    checks["key_services_present"] = present

    # LSOF helper (jeżeli repo zawiera skrypt lsof/port helper)
    lsof_path = REPO_ROOT / "lsof"
    checks["lsof_helper_exists"] = lsof_path.exists()

    return checks


def generate_markdown_summary(summary: Dict) -> str:
    lines = []
    lines.append("# System Audit Summary")
    lines.append("")
    lines.append(f"- Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Repo root: {REPO_ROOT.name}")
    lines.append("")
    git = summary.get("git", {})
    lines.append("## Git")
    lines.append(f"- Branch: {git.get('branch')}")
    lines.append(f"- Latest commit: {git.get('latest_commit')}")
    wc = git.get("working_tree_changes") or []
    lines.append(f"- Working tree changes: {len(wc)}")
    if wc:
        lines.append("")
        lines.append("```plain")
        lines.extend(wc[:50])
        if len(wc) > 50:
            lines.append("... (truncated)")
        lines.append("```")

    lines.append("")
    lines.append("## Inventory")
    inv = summary.get("inventory", {})
    lines.append(f"- Files scanned: {inv.get('files_count', 0)}")
    lines.append(f"- Python files: {inv.get('py_files_count', 0)}")
    lines.append(f"- Binary/data files: {inv.get('bin_files_count', 0)}")

    lines.append("")
    lines.append("## Duplicates")
    dup = summary.get("duplicates", {})
    lines.append(f"- Duplicate groups: {dup.get('groups_count', 0)}")
    if dup.get("examples"):
        lines.append("  - Example group:")
        for e in dup["examples"][:5]:
            lines.append(f"    - {e}")

    lines.append("")
    lines.append("## Dead code (heuristic)")
    dead = summary.get("dead_code", {})
    pc = dead.get("potentially_unused_count", 0)
    lines.append(f"- Potentially unused definitions: {pc}")
    if dead.get("examples"):
        lines.append("  - Example file:")
        lines.append("    ```plain")
        lines.extend(dead["examples"][:10])
        lines.append("    ```")

    lines.append("")
    lines.append("## Tests")
    tests = summary.get("tests", {})
    lines.append(f"- Tests found: {len(tests.get('tests_found', []))}")
    lines.append(f"- Files missing tests (heuristic): {len(tests.get('missing_tests', []))}")

    lines.append("")
    lines.append("## Config")
    cfg = summary.get("config", {})
    lines.append(f"- Checked config files: {len(cfg.get('files', {}))}")
    dca = cfg.get("docker_compose_analysis", {})
    if dca:
        lines.append(f"- docker-compose services guess: {dca.get('service_count_guess')}")
        lines.append(f"- docker-compose has ports mappings: {dca.get('has_ports_mappings')}")

    lines.append("")
    lines.append("## Dependencies")
    dep = summary.get("dependencies", {})
    conflicts = dep.get("version_conflicts", {})
    lines.append(f"- Requirements sets: {len(dep.get('requirements', {}))}")
    lines.append(f"- Version conflicts: {len(conflicts)}")

    lines.append("")
    lines.append("## Health quick checks")
    hq = summary.get("health", {})
    ksp = hq.get("key_services_present", {})
    present = [k for k, v in ksp.items() if v]
    missing = [k for k, v in ksp.items() if not v]
    lines.append(f"- Key services present: {len(present)}")
    if missing:
        lines.append(f"- Missing services: {', '.join(missing)}")
    lines.append(f"- lsof helper exists: {hq.get('lsof_helper_exists')}")

    return "\n".join(lines)


def main() -> int:
    REPORT_DIR.mkdir(exist_ok=True)

    all_files = walk_files(REPO_ROOT)
    py_files = list_python_files(all_files)
    bin_files = [f for f in all_files if f.suffix.lower() in BINARY_EXT]

    # Duplikaty
    dupe_map = detect_duplicates(all_files)
    dupe_groups = len(dupe_map)
    dupe_examples: List[str] = []
    for g in dupe_map.values():
        dupe_examples.extend(g[:3])
        if len(dupe_examples) > 10:
            break

    # Dead code (heurystyka)
    dead = detect_dead_code(py_files)
    potentially_unused = dead.get("potentially_unused", {})
    dead_count = sum(len(v) for v in potentially_unused.values())
    dead_examples: List[str] = []
    for f, names in potentially_unused.items():
        if names:
            dead_examples.append(f"{f}: {', '.join(names[:5])}")
        if len(dead_examples) > 10:
            break

    # Missing tests
    tests_info = detect_missing_tests(py_files)

    # Config
    cfg = audit_config()
    cfg_files_state = {k: v for k, v in cfg.items() if k not in {"docker_compose_analysis"}}

    # Dependencies
    deps = audit_dependencies()

    # Git
    g = git_status()

    # Health quick
    health = health_quick_checks()

    # Zapisz szczegółowe raporty
    (REPORT_DIR / "dupe_map.json").write_text(json.dumps(dupe_map, indent=2), encoding="utf-8")
    (REPORT_DIR / "unused_report.json").write_text(json.dumps(dead, indent=2), encoding="utf-8")
    (REPORT_DIR / "config_audit.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (REPORT_DIR / "dep_audit.json").write_text(json.dumps(deps, indent=2), encoding="utf-8")
    (REPORT_DIR / "test_coverage.json").write_text(json.dumps(tests_info, indent=2), encoding="utf-8")

    summary = {
        "git": g,
        "inventory": {
            "files_count": len(all_files),
            "py_files_count": len(py_files),
            "bin_files_count": len(bin_files),
        },
        "duplicates": {
            "groups_count": dupe_groups,
            "examples": dupe_examples,
        },
        "dead_code": {
            "potentially_unused_count": dead_count,
            "examples": dead_examples,
        },
        "tests": tests_info,
        "config": {
            "files": cfg_files_state,
            "docker_compose_analysis": cfg.get("docker_compose_analysis"),
        },
        "dependencies": deps,
        "health": health,
        "hints": {
            "next_steps": [
                "Uruchom vulture dla precyzyjnego dead code (opcjonalnie).",
                "Ujednolić wersje w requirements* aby usunąć konflikty.",
                "Zmapować brakujące testy do krytycznych plików (AgentExecutor, WebSocket, Neo4j).",
                "Zweryfikować docker-compose.yml i zdrowie usług (healthchecks).",
                "Docelowo włączyć ten audyt do CI (pre-merge gate).",
            ]
        },
    }

    (REPORT_DIR / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (REPORT_DIR / "audit_summary.md").write_text(generate_markdown_summary(summary), encoding="utf-8")

    print(f"[OK] Audit completed. Reports in: {REPORT_DIR}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)