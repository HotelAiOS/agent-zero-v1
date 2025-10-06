import re

# Fix qualityanalyzer.py
with open("shared/quality/qualityanalyzer.py", "r") as f:
    content = f.read()

# Usuń problematyczne importy
content = content.replace("import semgrep", "# import semgrep")
content = content.replace("from vulture import Vulture", "# from vulture import Vulture")

with open("shared/quality/qualityanalyzer.py", "w") as f:
    f.write(content)

# Fix interactive_control_system.py imports
with open("shared/monitoring/interactive_control_system.py", "r") as f:
    content = f.read()

content = content.replace("from live_monitor import", "from shared.monitoring.livemonitor import")
content = content.replace("from quality_analyzer import", "from shared.quality.qualityanalyzer import")
content = content.replace("from performance_optimizer import", "from shared.performance.optimizer import")

with open("shared/monitoring/interactive_control_system.py", "w") as f:
    f.write(content)

print("✅ Naprawiono wszystkie importy!")
