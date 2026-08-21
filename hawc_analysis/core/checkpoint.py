"""
Checkpoint management for HAWC pipeline
Extracted from: main.py CheckpointManager class

Enables resume capability for interrupted runs.
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from collections import OrderedDict


class CheckpointManager:
    """Manage pipeline checkpoints and history for resume capability
    
    Tracks:
    - Execution history (all steps completed)
    - Current checkpoint (resume point)
    - Step data for inter-step communication
    """
    
    def __init__(self, checkpoint_dir: str):
        """Initialize checkpoint manager
        
        Parameters:
        -----------
        checkpoint_dir : str
            Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history_file = self.checkpoint_dir / 'pipeline_history.json'
        self.checkpoint_file = self.checkpoint_dir / 'current_checkpoint.json'
        
        # Initialize or load history and checkpoint
        self.history = self._load_history()
        self.current_checkpoint = self._load_checkpoint()
    
    def _load_history(self) -> Dict[str, Any]:
        """Load existing history or create new one
        
        Returns:
        --------
        dict
            Pipeline execution history
        """
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                return json.load(f)
        
        return {
            'created': datetime.now().isoformat(),
            'steps': OrderedDict(),
            'total_steps': 0,
            'current_step': None,
            'status': 'initialized'
        }
    
    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load existing checkpoint or create new one
        
        Returns:
        --------
        dict
            Current checkpoint state
        """
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        
        return {
            'current_step': None,
            'current_iteration': 0,
            'last_completed_step': None,
            'data': {}
        }
    
    def save_step(self, step_name: str, iteration: int, status: str, 
                  data: Dict[str, Any], metadata: Dict[str, Any] = None) -> None:
        """Save progress for a pipeline step
        
        Parameters:
        -----------
        step_name : str
            Name of the pipeline step (e.g., 'phase0', 'phase1')
        iteration : int
            Iteration number (for iterative steps, 0 for single)
        status : str
            Status of the step: 'running', 'completed', 'failed'
        data : dict
            Results data from the step
        metadata : dict, optional
            Additional metadata (e.g., timing, number of sources)
        """
        step_key = f"{step_name}_{iteration}" if iteration > 0 else step_name
        
        step_record = {
            'name': step_name,
            'iteration': iteration,
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'data': data,
            'metadata': metadata or {}
        }
        
        # Update history
        self.history['steps'][step_key] = step_record
        self.history['current_step'] = step_key
        self.history['status'] = status
        self._save_history()
        
        # Update checkpoint
        self.current_checkpoint['current_step'] = step_name
        self.current_checkpoint['current_iteration'] = iteration
        
        if status == 'completed':
            self.current_checkpoint['last_completed_step'] = step_key
        
        self.current_checkpoint['data'][step_key] = data
        self._save_checkpoint()
    
    def _save_history(self) -> None:
        """Save history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def _save_checkpoint(self) -> None:
        """Save checkpoint to file"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.current_checkpoint, f, indent=2, default=str)
            print(f"Checkpoint saved: {self.checkpoint_file}")
    
    def get_last_completed_step(self) -> Optional[str]:
        """Get the name of the last completed step
        
        Returns:
        --------
        str or None
            Last completed step name, or None if no steps completed
        """
        return self.current_checkpoint.get('last_completed_step')
    
    def get_step_data(self, step_name: str, iteration: int = 0) -> Optional[Dict]:
        """Retrieve saved data from a previous step
        
        Parameters:
        -----------
        step_name : str
            Name of the step
        iteration : int, optional
            Iteration number (default: 0)
        
        Returns:
        --------
        dict or None
            Step data if available, None otherwise
        """
        step_key = f"{step_name}_{iteration}" if iteration > 0 else step_name
        return self.current_checkpoint['data'].get(step_key)
    
    def has_checkpoint(self) -> bool:
        """Check if a checkpoint exists
        
        Returns:
        --------
        bool
            True if checkpoint file exists
        """
        return self.checkpoint_file.exists()
    
    def print_history(self) -> None:
        """Print the execution history to console
        
        Displays all completed steps with timestamps and status.
        """
        print("\n" + "="*80)
        print("PIPELINE EXECUTION HISTORY")
        print("="*80)
        
        if not self.history['steps']:
            print("No steps executed yet")
        else:
            for step_key, record in self.history['steps'].items():
                status = record['status'].upper()
                print(f"\n[{status}] {step_key}")
                print(f"  Timestamp: {record['timestamp']}")
                
                if record['metadata']:
                    for key, value in record['metadata'].items():
                        print(f"  {key}: {value}")
        
        print("\n" + "="*80 + "\n")
    
    def clear_history(self) -> None:
        """Clear checkpoint history (for new runs)
        
        WARNING: This removes all checkpoint data. Use with caution.
        """
        self.history_file.unlink(missing_ok=True)
        self.checkpoint_file.unlink(missing_ok=True)
        self.history = self._load_history()
        self.current_checkpoint = self._load_checkpoint()