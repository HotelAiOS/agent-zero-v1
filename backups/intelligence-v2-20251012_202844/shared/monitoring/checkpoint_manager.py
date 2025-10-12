import json
import os
from datetime import datetime
from typing import Optional, List, Dict
from dataclasses import dataclass
import aiofiles


@dataclass
class ProjectCheckpoint:
    """Project execution checkpoint for resume capability"""
    project_name: str
    checkpoint_id: str
    timestamp: datetime
    completed_tasks: List[str]
    current_phase: str
    agent_states: Dict[str, Dict]
    execution_context: Dict


class CheckpointManager:
    """Manages project checkpoints for resume functionality"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        
    async def save_checkpoint(self, checkpoint) -> str:
        """Save project checkpoint to disk - accepts dict or ProjectCheckpoint"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # Handle both dict and ProjectCheckpoint object
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
        
        checkpoint_file = f"{self.checkpoint_dir}/{project_name}_{checkpoint_id}.json"
        
        async with aiofiles.open(checkpoint_file, "w") as f:
            await f.write(json.dumps(checkpoint_data, indent=2, default=str))
            
        return checkpoint_file
        
    async def load_checkpoint(self, project_name: str, checkpoint_id: str) -> Optional[ProjectCheckpoint]:
        """Load project checkpoint from disk"""
        checkpoint_file = f"{self.checkpoint_dir}/{project_name}_{checkpoint_id}.json"
        
        try:
            async with aiofiles.open(checkpoint_file, "r") as f:
                data = json.loads(await f.read())
                
            return ProjectCheckpoint(
                project_name=data["project_name"],
                checkpoint_id=data["checkpoint_id"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                completed_tasks=data["completed_tasks"],
                current_phase=data["current_phase"],
                agent_states=data["agent_states"],
                execution_context=data["execution_context"]
            )
        except FileNotFoundError:
            return None
            
    async def list_checkpoints(self, project_name: str) -> List[str]:
        """List all checkpoints for a project"""
        import glob
        
        if not os.path.exists(self.checkpoint_dir):
            return []
            
        pattern = f"{self.checkpoint_dir}/{project_name}_*.json"
        checkpoint_files = glob.glob(pattern)
        
        return [
            os.path.basename(f).replace(f"{project_name}_", "").replace(".json", "")
            for f in checkpoint_files
        ]
