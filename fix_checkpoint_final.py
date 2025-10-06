# Wczytaj plik
with open("shared/monitoring/livemonitor.py", "r") as f:
    content = f.read()

# Znajdź i zamień problematyczną linię
content = content.replace(
    "checkpoint_file = f\"{self.checkpoint_dir}/{checkpoint.project_name}_{checkpoint.checkpoint_id}.json\"",
    """# Handle both dict and ProjectCheckpoint
        if isinstance(checkpoint, dict):
            project_name = checkpoint.get("project_name", "unknown")
            checkpoint_id = checkpoint.get("checkpoint_id", "checkpoint")
            checkpoint_data = checkpoint
        else:
            project_name = checkpoint.project_name
            checkpoint_id = checkpoint.checkpoint_id
            checkpoint_data = {
                "project_name": checkpoint.project_name,
                "checkpoint_id": checkpoint.checkpoint_id,
                "timestamp": checkpoint.timestamp.isoformat(),
                "completed_tasks": checkpoint.completed_tasks,
                "current_phase": checkpoint.current_phase,
                "agent_states": checkpoint.agent_states,
                "execution_context": checkpoint.execution_context
            }
        
        checkpoint_file = f"{self.checkpoint_dir}/{project_name}_{checkpoint_id}.json\""""
)

# Usuń stary blok checkpoint_data (będzie duplikat)
import re
content = re.sub(
    r"checkpoint_data = \{\s+\"project_name\": checkpoint\.project_name,.*?\}",
    "",
    content,
    flags=re.DOTALL
)

with open("shared/monitoring/livemonitor.py", "w") as f:
    f.write(content)

print("✅ Naprawiono CheckpointManager!")
