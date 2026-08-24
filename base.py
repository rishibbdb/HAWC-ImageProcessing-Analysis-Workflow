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