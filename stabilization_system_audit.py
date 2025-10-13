```python
# File: stabilization/system_audit.py
import os
import subprocess

def audit_codebase(root_dir):
    """
    Tworzy raport inwentaryzacji kodu:
    - Lista plików
    - Status: exists, duplicate, dead code
    - Brak testów
    - Konflikty dependencies
    """
    report = {
        'files': [],
        'duplicates': [],
        'dead_code': [],
        'missing_tests': [],
        'dependency_issues': []
    }

    # Lista wszystkich plików źródłowych
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            path = os.path.join(subdir, file)
            if path.endswith('.py'):
                report['files'].append(path)

    # Wykrywanie dead code przez vulture
    try:
        output = subprocess.check_output(['vulture', root_dir, '--min-confidence', '80'])
        report['dead_code'] = output.decode().splitlines()
    except Exception:
        report['dependency_issues'].append('Vulture scan failed')

    # Wykrywanie duplikatów przez flake8
    try:
        dups = subprocess.check_output(['flake8', '--select', 'DUO', root_dir])
        report['duplicates'] = dups.decode().splitlines()
    except Exception:
        report['dependency_issues'].append('Flake8 DUO plugin not installed')

    # Analiza brakujących testów
    for f in report['files']:
        test_path = f.replace('.py', '_test.py')
        if not os.path.exists(test_path):
            report['missing_tests'].append(f)

    # Dependency issues
    try:
        audit = subprocess.check_output(['pip-audit', '--ignore', 'vulnerable'])
        report['dependency_issues'] += audit.decode().splitlines()
    except Exception:
        report['dependency_issues'].append('pip-audit not available')

    # Zapisz raport do pliku
    with open('stabilization/system_inventory_report.json', 'w') as f:
        import json
        json.dump(report, f, indent=2)

    print("System audit complete. Report saved to stabilization/system_inventory_report.json")

if __name__ == '__main__':
    audit_codebase(os.getcwd())
```