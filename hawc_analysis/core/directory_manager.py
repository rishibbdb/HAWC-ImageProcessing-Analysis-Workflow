"""
Directory Manager for HAWC Pipeline

Manages the directory structure for fit results:
fit_name/
├── DataMapDir/
├── Main_Logs/
├── Models/
└── Results/
    ├── Step0-Allpoint-sources/
    ├── Step1-Testsource1extension/
    └── ...
"""

from pathlib import Path
from typing import Optional, List
from datetime import datetime


class DirectoryManager:
    """Manage pipeline directory structure for fit results
    
    Organizes output files according to HAWC pipeline conventions:
    - DataMap: Significance maps (if create_sig_map: True)
    - Logs: Pipeline log files
    - Models: Model files (.model) for each fitting step
    - FitResults: Results organized by fitting step
    """
    
    # Directory names (constants)
    DATAMAP_DIR = 'DataMapDir'
    LOGS_DIR = 'Main_Logs'
    MODELS_DIR = 'Models'
    FIT_RESULTS_DIR = 'Results'
    
    def __init__(self, base_dir: str, fit_name: str, logger: Optional[object] = None):
        """Initialize directory manager
        
        Parameters:
        -----------
        base_dir : str
            Base directory for output (e.g., '/path/to/output')
        fit_name : str
            Fit name from config (e.g., 'crab_fit')
        logger : object, optional
            Logger instance for messages
        """
        self.base_dir = Path(base_dir)
        self.fit_name = fit_name
        self.root_dir = self.base_dir / fit_name
        self.logger = logger
        
        # Create root directory
        self.root_dir.mkdir(parents=True, exist_ok=True)
        
        if self.logger:
            self.logger.info(f"Initialized DirectoryManager at {self.root_dir}")
    
    def create_structure(self) -> None:
        """Create complete directory structure
        
        Creates:
        - DataMap/
        - Logs/
        - Models/
        - FitResults/
        """
        for dirname in [self.DATAMAP_DIR, self.LOGS_DIR, self.MODELS_DIR, self.FIT_RESULTS_DIR]:
            dirpath = self.root_dir / dirname
            dirpath.mkdir(parents=True, exist_ok=True)
            
            if self.logger:
                self.logger.debug(f"Created directory: {dirpath}")
        
        if self.logger:
            self.logger.info(f"Directory structure created at {self.root_dir}")
    
    def get_root_dir(self) -> Path:
        """Get root directory path (fit_name/)
        
        Returns:
        --------
        Path
            Root directory path
        """
        return self.root_dir
    
    def get_datamap_dir(self) -> Path:
        """Get DataMap directory path
        
        Used for significance maps (if create_sig_map: True)
        
        Returns:
        --------
        Path
            DataMap/ directory path
        """
        dirpath = self.root_dir / self.DATAMAP_DIR
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath
    
    def get_logs_dir(self) -> Path:
        """Get Logs directory path
        
        Used for pipeline log files:
        - pipeline_TIMESTAMP.log
        - full_log_TIMESTAMP.log
        
        Returns:
        --------
        Path
            Logs/ directory path
        """
        dirpath = self.root_dir / self.LOGS_DIR
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath
    
    def get_models_dir(self) -> Path:
        """Get Models directory path
        
        Used for storing model files:
        - Step0-Allpoint-sources.model
        - Step1-Testsource1extension.model
        - Step1-Testsource1extension-Spectrum.model
        
        Returns:
        --------
        Path
            Models/ directory path
        """
        dirpath = self.root_dir / self.MODELS_DIR
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath
    
    def get_fit_results_dir(self) -> Path:
        """Get FitResults directory path
        
        Used for storing fit step results (parent directory)
        
        Returns:
        --------
        Path
            FitResults/ directory path
        """
        dirpath = self.root_dir / self.FIT_RESULTS_DIR
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath
    
    def get_step_results_dir(self, step_name: str) -> Path:
        """Get results directory for a specific fitting step
        
        Creates FitResults/{step_name}/ subdirectory
        
        Step naming conventions (user-defined based on fitting procedure):
        
        For DRIPS (Image Seeds):
        - Step0-Allpoint-sources (initial seed detection)
        - Step1-Testsource1extension (test source 1 for extension)
        - Step1-Testsource1extension-Spectrum (spectrum testing)
        - Step2-Testsource1extension-accepted (if source accepted)
        
        For ALPS (Iterative Hotspot):
        - Step0-1ps (1 point source added)
        - Step0-diffuse (if diffuse background)
        - Step0-2ps (2 point sources)
        - Step0-3ps (3 point sources)
        - etc.
        
        Parameters:
        -----------
        step_name : str
            Step name (user-defined based on fitting procedure)
        
        Returns:
        --------
        Path
            FitResults/{step_name}/ directory path
        
        Examples:
        ---------
        >>> # DRIPS naming
        >>> dirpath = dm.get_step_results_dir('Step0-Allpoint-sources')
        >>> # Returns: fit_name/FitResults/Step0-Allpoint-sources/
        
        >>> # ALPS naming
        >>> dirpath = dm.get_step_results_dir('Step0-1ps')
        >>> # Returns: fit_name/FitResults/Step0-1ps/
        
        >>> dirpath = dm.get_step_results_dir('Step0-diffuse')
        >>> # Returns: fit_name/FitResults/Step0-diffuse/
        """
        dirpath = self.root_dir / self.FIT_RESULTS_DIR / step_name
        dirpath.mkdir(parents=True, exist_ok=True)
        return dirpath
    
    def get_model_file_path(self, step_name: str) -> Path:
        """Get expected model file path for a step
        
        Parameters:
        -----------
        step_name : str
            Step name (e.g., 'Step0-Allpoint-sources')
        
        Returns:
        --------
        Path
            Models/{step_name}.model file path
        
        Examples:
        ---------
        >>> model_path = dm.get_model_file_path('Step0-Allpoint-sources')
        >>> # Returns: fit_name/Models/Step0-Allpoint-sources.model
        """
        return self.get_models_dir() / f"{step_name}.model"
    
    def get_fit_results_file_path(self, step_name: str, filename: str = 'fit_results.yaml') -> Path:
        """Get fit results file path for a step
        
        Parameters:
        -----------
        step_name : str
            Step name (e.g., 'Step0-Allpoint-sources')
        filename : str, optional
            Filename (default: 'fit_results.yaml')
        
        Returns:
        --------
        Path
            FitResults/{step_name}/{filename} file path
        
        Examples:
        ---------
        >>> results = dm.get_fit_results_file_path('Step0-Allpoint-sources')
        >>> # Returns: fit_name/FitResults/Step0-Allpoint-sources/fit_results.yaml
        
        >>> params = dm.get_fit_results_file_path('Step0-Allpoint-sources', 'parameters.yaml')
        >>> # Returns: fit_name/FitResults/Step0-Allpoint-sources/parameters.yaml
        """
        return self.get_step_results_dir(step_name) / filename
    
    def get_hdf5_file_path(self, step_name: str, map_type: str = 'model') -> Path:
        """Get HDF5 map file path for a step
        
        Parameters:
        -----------
        step_name : str
            Step name
        map_type : str, optional
            Map type: 'model' or 'residual' (default: 'model')
        
        Returns:
        --------
        Path
            FitResults/{step_name}/{map_type}_fit.hd5 file path
        
        Examples:
        ---------
        >>> hdf5_path = dm.get_hdf5_file_path('Step0-Allpoint-sources', 'model')
        >>> # Returns: fit_name/FitResults/Step0-Allpoint-sources/model_fit.hd5
        
        >>> residual_path = dm.get_hdf5_file_path('Step0-Allpoint-sources', 'residual')
        >>> # Returns: fit_name/FitResults/Step0-Allpoint-sources/residual_fit.hd5
        """
        if map_type not in ['model', 'residual']:
            raise ValueError(f"map_type must be 'model' or 'residual', got: {map_type}")
        
        return self.get_step_results_dir(step_name) / f"{map_type}_fit.hd5"
    
    def list_steps(self) -> List[str]:
        """List all completed fitting steps
        
        Returns:
        --------
        list of str
            Sorted list of step names (directories in FitResults/)
        
        Examples:
        ---------
        >>> steps = dm.list_steps()
        >>> print(steps)
        ['Step0-Allpoint-sources', 'Step1-Testsource1extension']
        """
        fit_results_dir = self.get_fit_results_dir()
        
        if not fit_results_dir.exists():
            return []
        
        steps = [d.name for d in fit_results_dir.iterdir() if d.is_dir()]
        return sorted(steps)
    
    def list_models(self) -> List[Path]:
        """List all model files
        
        Returns:
        --------
        list of Path
            List of .model files in Models/ directory
        
        Examples:
        ---------
        >>> models = dm.list_models()
        >>> for model_file in models:
        ...     print(model_file.name)
        Step0-Allpoint-sources.model
        Step1-Testsource1extension.model
        """
        models_dir = self.get_models_dir()
        
        if not models_dir.exists():
            return []
        
        return sorted(models_dir.glob('*.model'))
    
    def get_significance_map_path(self) -> Path:
        """Get expected significance map file path
        
        Returns:
        --------
        Path
            DataMap/sky_map.fits file path
        """
        return self.get_datamap_dir() / 'sky_map.fits'
    
    def print_structure(self) -> None:
        """Print directory structure tree to console
        
        Examples:
        ---------
        >>> dm.print_structure()
        fit_name/
        ├── DataMap/
        ├── Logs/
        ├── Models/
        └── FitResults/
            ├── Step0-Allpoint-sources/
            └── Step1-Testsource1extension/
        """
        print(f"\n{self.root_dir}/")
        
        # Print main directories
        for subdir in [self.DATAMAP_DIR, self.LOGS_DIR, self.MODELS_DIR, self.FIT_RESULTS_DIR]:
            dirpath = self.root_dir / subdir
            if dirpath.exists():
                if subdir == self.FIT_RESULTS_DIR:
                    print(f"├── {subdir}/")
                    # Print step subdirectories
                    steps = sorted(dirpath.iterdir())
                    for i, step_dir in enumerate(steps):
                        if step_dir.is_dir():
                            prefix = "│   └── " if i == len(steps) - 1 else "│   ├── "
                            print(f"{prefix}{step_dir.name}/")
                else:
                    print(f"├── {subdir}/")
        print()
    
    def get_summary(self) -> dict:
        """Get summary of current directory structure
        
        Returns:
        --------
        dict
            Summary with counts and status information
        
        Examples:
        ---------
        >>> summary = dm.get_summary()
        >>> print(f"Completed steps: {summary['num_steps']}")
        >>> print(f"Models created: {summary['num_models']}")
        """
        models = self.list_models()
        steps = self.list_steps()
        
        # Count log files
        logs_dir = self.get_logs_dir()
        log_files = list(logs_dir.glob('*.log')) if logs_dir.exists() else []
        
        # Check for significance map
        sig_map = self.get_significance_map_path()
        has_sig_map = sig_map.exists()
        
        return {
            'root_dir': str(self.root_dir),
            'fit_name': self.fit_name,
            'num_steps': len(steps),
            'completed_steps': steps,
            'num_models': len(models),
            'model_files': [m.name for m in models],
            'num_log_files': len(log_files),
            'has_significance_map': has_sig_map,
        }