"""
Logging system for HAWC pipeline
Extracted from: main.py PipelineLogger class

Dual-logging approach:
- Pipeline log: Only Pipeline messages
- Full log: All packages (threeML, astromodels, etc.)
"""

import logging
from pathlib import Path
from datetime import datetime


class PipelineLogger:
    """Centralized logging system with separate pipeline and full logs
    
    Creates two log files:
    1. pipeline_[timestamp].log - Only Pipeline messages
    2. full_log_[timestamp].log - All packages (root logger)
    
    Also outputs to console.
    """
    
    def __init__(self, log_dir: str, log_level: str = 'INFO'):
        """Initialize logging system
        
        Parameters:
        -----------
        log_dir : str
            Directory to save log files
        log_level : str, optional
            Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
            Default: 'INFO'
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for log files
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Validate log level
        log_level_upper = log_level.upper()
        if not hasattr(logging, log_level_upper):
            raise ValueError(f"Invalid log level: {log_level}")
        log_level_int = getattr(logging, log_level_upper)
        
        # ========== PIPELINE LOGGER (only Pipeline messages) ==========
        self.logger = logging.getLogger('Pipeline')
        self.logger.setLevel(log_level_int)
        self.logger.propagate = False  # Don't propagate to root logger
        
        # Clear any existing handlers
        self.logger.handlers = []
        
        # Pipeline log file handler
        pipeline_log_file = self.log_dir / f"pipeline_{timestamp}.log"
        pipeline_fh = logging.FileHandler(pipeline_log_file)
        pipeline_fh.setLevel(log_level_int)
        
        # Console handler for pipeline messages
        ch = logging.StreamHandler()
        ch.setLevel(log_level_int)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        pipeline_fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to pipeline logger
        self.logger.addHandler(pipeline_fh)
        self.logger.addHandler(ch)
        
        # ========== ROOT LOGGER (all packages) ==========
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level_int)
        
        # Clear any existing handlers from root logger
        root_logger.handlers = []
        
        # Full log file handler (captures all packages)
        full_log_file = self.log_dir / f"full_log_{timestamp}.log"
        full_fh = logging.FileHandler(full_log_file)
        full_fh.setLevel(log_level_int)
        full_fh.setFormatter(formatter)
        
        # Add full log handler to root logger
        root_logger.addHandler(full_fh)
        
        # Log initialization
        self.logger.info(f"Pipeline logger initialized")
        self.logger.info(f"Log level: {log_level_upper}")
        self.logger.info(f"Pipeline log: {pipeline_log_file}")
        self.logger.info(f"Full log: {full_log_file}")
    
    def info(self, msg: str) -> None:
        """Log info message
        
        Parameters:
        -----------
        msg : str
            Message to log
        """
        self.logger.info(msg)
    
    def debug(self, msg: str) -> None:
        """Log debug message
        
        Parameters:
        -----------
        msg : str
            Message to log
        """
        self.logger.debug(msg)
    
    def warning(self, msg: str) -> None:
        """Log warning message
        
        Parameters:
        -----------
        msg : str
            Message to log
        """
        self.logger.warning(msg)
    
    def error(self, msg: str) -> None:
        """Log error message
        
        Parameters:
        -----------
        msg : str
            Message to log
        """
        self.logger.error(msg)
    
    def critical(self, msg: str) -> None:
        """Log critical message
        
        Parameters:
        -----------
        msg : str
            Message to log
        """
        self.logger.critical(msg)
    
    def exception(self, msg: str) -> None:
        """Log exception with traceback
        
        Parameters:
        -----------
        msg : str
            Message to log
        """
        self.logger.exception(msg)