"""
Seeding Base Module

Abstract base class and output dataclass for source detection seed generation.
Defines standardized interface for DRIPS and ALPS seeding methods.

Extracted from: pipeline_sourcedetector.py, pipeline_helpers.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
from astropy.wcs import WCS


@dataclass
class SeedingOutput:
    """Standardized output from seeding algorithms (DRIPS or ALPS)
    
    Attributes:
    -----------
    source_info_db : pd.DataFrame
        Source catalog with columns: 'ra', 'dec', 'Sigma Radius', 'TS', etc.
    
    baseline_model_path : Path
        Path to baseline threeML model file
    
    baseline_likelihood : float
        Log-likelihood of baseline model
    
    baseline_params : dict
        Baseline model parameters
    
    ts_values : dict
        TS (Test Statistic) for each source: {'Source_0': 25.3, ...}
    
    residual_map_path : Path
        Path to residual significance map after fitting baseline model
    
    checkpoint_data : dict
        Metadata: {'method': 'DRIPS', 'num_iterations': 5, 'num_sources': 3}
    
    num_sources : int
        Number of sources detected
    
    num_iterations : int
        Number of iterations performed
    
    method : str
        Seeding method used ('DRIPS' or 'ALPS')
    """
    
    source_info_db: pd.DataFrame
    baseline_model_path: Path
    baseline_likelihood: float
    baseline_params: Dict
    ts_values: Dict[str, float]
    residual_map_path: Path
    checkpoint_data: Dict
    num_sources: int
    num_iterations: int
    method: str = "Unknown"
    
    # Optional fields
    significance_map_path: Optional[Path] = None
    footprint: Optional[np.ndarray] = None
    wcs: Optional[WCS] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            'source_info_db': self.source_info_db.to_dict(orient='records'),
            'baseline_model_path': str(self.baseline_model_path),
            'baseline_likelihood': float(self.baseline_likelihood),
            'baseline_params': self.baseline_params,
            'ts_values': self.ts_values,
            'residual_map_path': str(self.residual_map_path),
            'checkpoint_data': self.checkpoint_data,
            'num_sources': int(self.num_sources),
            'num_iterations': int(self.num_iterations),
            'method': self.method,
        }
    
    def summary(self) -> str:
        """Return human-readable summary"""
        summary = f"""
Seeding Results ({self.method})
{'='*50}
Sources detected: {self.num_sources}
Iterations: {self.num_iterations}
Baseline log-likelihood: {self.baseline_likelihood:.2f}
Baseline model: {self.baseline_model_path}
Residual map: {self.residual_map_path}

Source Catalog:
{self.source_info_db.to_string()}

TS Values:
{pd.Series(self.ts_values).to_string()}
{'='*50}
        """
        return summary


class SeedingModule(ABC):
    """Abstract base class for source detection seeding algorithms
    
    Both DRIPS and ALPS seed models implement this interface.
    Standardizes input/output and checkpoint handling.
    
    Subclasses must implement:
    - run() : Main detection algorithm
    
    Attributes:
    -----------
    config : object
        Configuration manager (ConfigManager instance)
    
    logger : object
        Pipeline logger (PipelineLogger instance)
    
    directory_manager : object
        Directory structure manager (DirectoryManager instance)
    
    data_loader : object
        Data loading utilities (DataLoader class)
    
    plotting : object
        Plotting utilities (PlottingUtilities class)
    
    model_generator : object
        Model generation utilities (ModelGenerator class)
    """
    
    def __init__(
        self,
        config: object,
        logger: object,
        directory_manager: object,
        data_loader: object = None,
        plotting: object = None,
        model_generator: object = None,
    ):
        """Initialize seeding module
        
        Parameters:
        -----------
        config : object
            ConfigManager instance for configuration
        
        logger : object
            PipelineLogger instance
        
        directory_manager : object
            DirectoryManager instance
        
        data_loader : object, optional
            DataLoader class (default: imported from core)
        
        plotting : object, optional
            PlottingUtilities class (default: imported from core)
        
        model_generator : object, optional
            ModelGenerator class (default: imported from core)
        """
        self.config = config
        self.logger = logger
        self.directory_manager = directory_manager
        
        # Import from core if not provided
        if data_loader is None:
            from core.data_loading import DataLoader
            self.data_loader = DataLoader
        else:
            self.data_loader = data_loader
        
        if plotting is None:
            from core.plotting import PlottingUtilities
            self.plotting = PlottingUtilities
        else:
            self.plotting = plotting
        
        if model_generator is None:
            from core.model_generator import ModelGenerator
            self.model_generator = ModelGenerator
        else:
            self.model_generator = model_generator
        
        self.logger.info(f"Initialized {self.__class__.__name__}")
    
    @abstractmethod
    def run(self) -> SeedingOutput:
        """Run seeding algorithm
        
        Must be implemented by subclass.
        
        Returns:
        --------
        SeedingOutput
            Standardized output with detected sources and baseline model
        """
        pass
    
    def get_checkpoint_dir(self) -> Path:
        """Get checkpoint directory for this seeding method
        
        Returns:
        --------
        Path
            Checkpoint directory path
        """
        method_name = self.__class__.__name__.lower()
        checkpoint_dir = self.directory_manager.get_models_dir() / f".{method_name}_checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir
    
    def save_checkpoint(self, data: dict, checkpoint_name: str = "checkpoint.pkl") -> Path:
        """Save checkpoint data for resuming interrupted runs
        
        Parameters:
        -----------
        data : dict
            Data to checkpoint
        
        checkpoint_name : str, optional
            Checkpoint filename (default: "checkpoint.pkl")
        
        Returns:
        --------
        Path
            Path to saved checkpoint
        """
        import pickle
        
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)
            self.logger.info(f"Saved checkpoint: {checkpoint_path}")
            return checkpoint_path
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            return None
    
    def load_checkpoint(self, checkpoint_name: str = "checkpoint.pkl") -> Optional[dict]:
        """Load checkpoint data for resuming interrupted runs
        
        Parameters:
        -----------
        checkpoint_name : str, optional
            Checkpoint filename (default: "checkpoint.pkl")
        
        Returns:
        --------
        dict or None
            Loaded checkpoint data, or None if not found
        """
        import pickle
        
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        if not checkpoint_path.exists():
            self.logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            self.logger.info(f"Loaded checkpoint: {checkpoint_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def clear_checkpoint(self, checkpoint_name: str = "checkpoint.pkl") -> bool:
        """Clear checkpoint data
        
        Parameters:
        -----------
        checkpoint_name : str, optional
            Checkpoint filename (default: "checkpoint.pkl")
        
        Returns:
        --------
        bool
            True if successful
        """
        checkpoint_dir = self.get_checkpoint_dir()
        checkpoint_path = checkpoint_dir / checkpoint_name
        
        if checkpoint_path.exists():
            try:
                checkpoint_path.unlink()
                self.logger.info(f"Cleared checkpoint: {checkpoint_path}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to clear checkpoint: {e}")
                return False
        
        return True
    
    def log_seeding_start(self, method_name: str) -> None:
        """Log seeding start with method info
        
        Parameters:
        -----------
        method_name : str
            Name of seeding method (e.g., 'DRIPS', 'ALPS')
        """
        self.logger.info("="*60)
        self.logger.info(f"STARTING {method_name} SEEDING")
        self.logger.info("="*60)
    
    def log_seeding_complete(
        self,
        method_name: str,
        num_sources: int,
        num_iterations: int,
        baseline_likelihood: float
    ) -> None:
        """Log seeding completion with results
        
        Parameters:
        -----------
        method_name : str
            Name of seeding method
        
        num_sources : int
            Number of sources detected
        
        num_iterations : int
            Number of iterations performed
        
        baseline_likelihood : float
            Baseline model log-likelihood
        """
        self.logger.info("="*60)
        self.logger.info(f"COMPLETED {method_name} SEEDING")
        self.logger.info(f"Sources detected: {num_sources}")
        self.logger.info(f"Iterations: {num_iterations}")
        self.logger.info(f"Baseline log-likelihood: {baseline_likelihood:.2f}")
        self.logger.info("="*60)
    
    def validate_output(self, output: SeedingOutput) -> bool:
        """Validate SeedingOutput structure
        
        Parameters:
        -----------
        output : SeedingOutput
            Output to validate
        
        Returns:
        --------
        bool
            True if valid
        """
        required_fields = [
            'source_info_db',
            'baseline_model_path',
            'baseline_likelihood',
            'baseline_params',
            'ts_values',
            'residual_map_path',
            'checkpoint_data',
            'num_sources',
            'num_iterations',
        ]
        
        for field in required_fields:
            if not hasattr(output, field):
                self.logger.error(f"Missing required field: {field}")
                return False
        
        # Validate types
        if not isinstance(output.source_info_db, pd.DataFrame):
            self.logger.error("source_info_db must be pd.DataFrame")
            return False
        
        if not isinstance(output.baseline_model_path, Path):
            self.logger.error("baseline_model_path must be Path")
            return False
        
        if not isinstance(output.baseline_likelihood, (int, float)):
            self.logger.error("baseline_likelihood must be numeric")
            return False
        
        if not isinstance(output.ts_values, dict):
            self.logger.error("ts_values must be dict")
            return False
        
        self.logger.info("Output validation passed")
        return True
