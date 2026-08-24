"""
Configuration management for HAWC pipeline
Extracted from: main.py PipelineConfig class
"""

import yaml
from pathlib import Path
from typing import Any, Optional, Dict


class ConfigManager:
    """Load and manage configuration from YAML file
    
    Handles:
    - Loading YAML config files
    - Dot notation access (e.g., 'phase0.method')
    - Default values
    - Config validation
    - Backward compatibility with DRIPS format
    """
    
    def __init__(self, config_file: str):
        """Initialize config manager with YAML file
        
        Parameters:
        -----------
        config_file : str
            Path to configuration YAML file
        """
        self.config_file = Path(config_file)
        
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        if not isinstance(self.config, dict):
            raise ValueError("Config file must contain a YAML dictionary")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value using dot notation
        
        Parameters:
        -----------
        key : str
            Key path using dot notation: 'section.subsection.key'
            Example: 'phase0.method' or 'coordinates.ra'
        default : Any, optional
            Default value if key not found
        
        Returns:
        --------
        Any
            Configuration value or default
        
        Examples:
        ---------
        >>> config = ConfigManager('config.yaml')
        >>> method = config.get('phase0.method', 'image_seeds')
        >>> ra = config.get('coordinates.ra', 0.0)
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    def set(self, key: str, value: Any, save: bool = False) -> None:
        """Set config value using dot notation.

        Parameters
        ----------
        key : str
            Key path using dot notation.
        value : Any
            Value to assign.
        save : bool, optional
            If True, write the updated config back to the YAML file.
        """
        keys = key.split(".")
        d = self.config

        # Traverse/create intermediate dictionaries
        for k in keys[:-1]:
            if k not in d or not isinstance(d[k], dict):
                d[k] = {}
            d = d[k]

        d[keys[-1]] = value

        if save:
            with open(self.config_file, "w") as f:
                yaml.safe_dump(self.config, f, sort_keys=False)
    def __getitem__(self, key: str) -> Any:
        """Direct dictionary access
        
        Parameters:
        -----------
        key : str
            Top-level key (no dot notation)
        
        Returns:
        --------
        Any
            Configuration section
        
        Examples:
        ---------
        >>> config = ConfigManager('config.yaml')
        >>> coordinates = config['coordinates']
        """
        return self.config[key]
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists
        
        Parameters:
        -----------
        key : str
            Key path with dot notation
        
        Returns:
        --------
        bool
            True if key exists
        """
        return self.get(key, None) is not None
    
    def __repr__(self) -> str:
        """String representation of config"""
        return yaml.dump(self.config, default_flow_style=False)
    
    def validate(self) -> None:
        """Validate required config sections
        
        Raises:
        -------
        ValueError
            If required sections are missing
        """
        required_sections = ['metadata', 'coordinates', 'hawc', 'phase0', 'roi']
        missing = [s for s in required_sections if s not in self.config]
        
        if missing:
            raise ValueError(f"Missing required config sections: {missing}")
        
        # Validate coordinates (ra/dec OR l/b)
        coords = self.config.get('coordinates', {})
        has_ra_dec = 'ra' in coords and 'dec' in coords
        has_l_b = 'l' in coords and 'b' in coords
        
        if not (has_ra_dec or has_l_b):
            raise ValueError("Must provide (ra, dec) OR (l, b) in coordinates")
        
        # Validate phase0 method
        phase0 = self.config.get('phase0', {})
        method = phase0.get('method')
        
        if method not in ['image_seeds', 'alps_seeds']:
            raise ValueError(f"phase0.method must be 'image_seeds' or 'alps_seeds', got: {method}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return config as dictionary
        
        Returns:
        --------
        dict
            Full configuration dictionary
        """
        return self.config.copy()