with open("shared/monitoring/livemonitor.py", "r") as f:
    lines = f.readlines()

# Znajdź linię z błędem i zamień
new_lines = []
for i, line in enumerate(lines):
    if "checkpoint_file = f\"{self.checkpoint_dir}/{checkpoint.project_name" in line:
        # Dodaj obsługę dict przed tą linią
        indent = " " * 8
        new_lines.extend([
            indent + "# Handle both dict and ProjectCheckpoint\n",
            indent + "if isinstance(checkpoint, dict):\n",
            indent + "    project_name = checkpoint.get(\"project_name\", \"unknown\")\n",
            indent + "    checkpoint_id = checkpoint.get(\"checkpoint_id\", \"checkpoint\")\n",
            indent + "    checkpoint_data = checkpoint\n",
            indent + "else:\n",
            indent + "    project_name = checkpoint.project_name\n",
            indent + "    checkpoint_id = checkpoint.checkpoint_id\n",
            indent + "    checkpoint_data = {\n",
            indent + "        \"project_name\": checkpoint.project_name,\n",
            indent + "        \"checkpoint_id\": checkpoint.checkpoint_id,\n",
            indent + "        \"timestamp\": checkpoint.timestamp.isoformat(),\n",
            indent + "        \"completed_tasks\": checkpoint.completed_tasks,\n",
            indent + "        \"current_phase\": checkpoint.current_phase,\n",
            indent + "        \"agent_states\": checkpoint.agent_states,\n",
            indent + "        \"execution_context\": checkpoint.execution_context\n",
            indent + "    }\n",
            indent + "\n",
            indent + f"checkpoint_file = f\"{{self.checkpoint_dir}}/{{project_name}}_{{checkpoint_id}}.json\"\n"
        ])
    elif "checkpoint_data = {" in line and i > 0 and "checkpoint_file" not in lines[i-1]:
        # Pomiń stare linie checkpoint_data
        continue
    else:
        new_lines.append(line)

with open("shared/monitoring/livemonitor.py", "w") as f:
    f.writelines(new_lines)

print("✅ Naprawiono CheckpointManager (metoda 2)!")
