import os
import sys
import json
import yaml
import logging
import argparse
import traceback
from pathlib import Path
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from collections import OrderedDict
import shutil
from pipeline_sourcedetector import SourceSeedDetector
from pipeline_fitmodel import threeMLFit
 
class PipelineLogger:
    """Centralized logging system"""
    
    def __init__(self, log_dir: str, log_level: str = 'INFO'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('Pipeline')
        self.logger.setLevel(getattr(logging, log_level))
        
        # File handler
        log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(getattr(logging, log_level))
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
    
    def info(self, msg):
        self.logger.info(msg)
    
    def debug(self, msg):
        self.logger.debug(msg)
    
    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
    def critical(self, msg):
        self.logger.critical(msg)
 
 
class CheckpointManager:
    """Manage pipeline checkpoints and history for resume capability"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.checkpoint_dir / 'pipeline_history.json'
        self.checkpoint_file = self.checkpoint_dir / 'current_checkpoint.json'
        
        # Initialize or load history
        self.history = self._load_history()
        self.current_checkpoint = self._load_checkpoint()
    
    def _load_history(self) -> Dict[str, Any]:
        """Load existing history or create new one"""
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
        """Load existing checkpoint or create new one"""
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
                  data: Dict[str, Any], metadata: Dict[str, Any] = None):
        """
        Save progress for a step
        
        Parameters:
        -----------
        step_name : str
            Name of the pipeline step
        iteration : int
            Iteration number (for iterative steps)
        status : str
            Status of the step ('running', 'completed', 'failed')
        data : dict
            Data/results from the step
        metadata : dict, optional
            Additional metadata
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
    
    def _save_history(self):
        """Save history to file"""
        with open(self.history_file, 'w') as f:
            json.dump(self.history, f, indent=2, default=str)
    
    def _save_checkpoint(self):
        """Save checkpoint to file"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.current_checkpoint, f, indent=2, default=str)
    
    def get_last_completed_step(self) -> Optional[str]:
        """Get the last completed step"""
        return self.current_checkpoint.get('last_completed_step')
    
    def get_step_data(self, step_name: str, iteration: int = 0) -> Optional[Dict]:
        """Retrieve saved data from a previous step"""
        step_key = f"{step_name}_{iteration}" if iteration > 0 else step_name
        return self.current_checkpoint['data'].get(step_key)
    
    def print_history(self):
        """Print the execution history"""
        print("\n" + "="*80)
        print("PIPELINE EXECUTION HISTORY")
        print("="*80)
        for step_key, record in self.history['steps'].items():
            print(f"\n[{record['status'].upper()}] {step_key}")
            print(f"  Timestamp: {record['timestamp']}")
            if record['metadata']:
                print(f"  Metadata: {record['metadata']}")
        print("\n" + "="*80 + "\n")
 
 
class PipelineConfig:
    """Load and manage configuration from YAML file"""
    def __init__(self, config_file: str):
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self.config_file = config_file
    
    def get(self, key: str, default=None):
        """Get config value using dot notation: 'section.subsection.key'"""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default) if isinstance(value, dict) else default
        return value
    
    def __getitem__(self, key):
        return self.config[key]
    
    def __repr__(self):
        return yaml.dump(self.config, default_flow_style=False)
    
class SourceSearchPipeline:
    """Main pipeline orchestrator with logging and checkpoint management"""
    
    def __init__(self, config_file: str):
        """Initialize pipeline with configuration and logging"""
        self.config = PipelineConfig(config_file)
        
        # Get step name and iteration from config
        self.step_name = self.config.get('paths.step', 'DefaultStep')
        self.step_iteration = self.config.get('paths.step_iteration', 0)
        
        # Setup directories - hierarchical structure
        main_dir = Path(self.config.get('paths.main_dir'))
        main_dir.mkdir(parents=True, exist_ok=True)
        self.main_dir = main_dir
        
        # Create step-specific subdirectory with iteration number
        step_dir = main_dir / f'step{self.step_iteration}.{self.step_name}'
        step_dir.mkdir(parents=True, exist_ok=True)
        
        # Create logs and results subdirectories within step directory
        log_dir = step_dir / 'logs'
        checkpoint_dir = main_dir / 'checkpoints'
        output_dir = step_dir / 'results'
        
        log_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize logger and checkpoint manager
        self.logger = PipelineLogger(str(log_dir), self.config.get('logging.log_level', 'INFO'))
        self.checkpoint_mgr = CheckpointManager(str(checkpoint_dir))
        
        # Store paths
        self.step_dir = step_dir
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.output_dir = output_dir

    
    def run_fit(self):
        self.hal_fit = threeMLFit(config_path=str(self.config.config_file), model=self.model)
        self.hal_fit.run()
        self.logger.info(f"{self.step_name} completed successfully")

    def run_fit_witherrors(self):
        self.hal_fit = threeMLFit(config_path=str(self.config.config_file), model=self.model)
        self.hal_fit.run()
        self.logger.info(f"{self.step_name} completed successfully")
    
    def run_fit_withTS(self):
        self.hal_fit = threeMLFit(config_path=str(self.config.config_file), model=self.model)
        self.hal_fit.run()
        self.logger.info(f"{self.step_name} completed successfully")

    def run_fit_withTS_errors(self):
        self.hal_fit = threeMLFit(config_path=str(self.config.config_file), model=self.model)
        self.hal_fit.run()
        self.logger.info(f"{self.step_name} completed successfully")               
    
    def _run_seed_model(self, significance_map_filename: str = 'sky_map.fits'):
        """Run the seed model fitting step"""
        self.logger.info("Initializing SourceSeedDetector for seed model fitting...")
        
        self.checkpoint_mgr.save_step(
            step_name='SeedModelFit',
            iteration=0,
            status='running',
            data={
                'config_file': str(self.config.config_file),
                'output_dir': str(self.output_dir)
            },
            metadata={'step_type': 'seed_model_fitting'}
        )
        
        detector = SourceSeedDetector(str(self.config.config_file), step_path=str(self.step_dir))
        
        # Update detector to use the newly created significance map
        significance_map_path = self.output_dir / significance_map_filename
        if significance_map_path.exists():
            detector.initialmap = str(significance_map_path)
            self.logger.info(f"Updated detector to use significance map at {significance_map_path}")
        
        self.logger.info("Running source detection...")
        detector.run()
        
        self.checkpoint_mgr.save_step(
            step_name='SeedModelFit',
            iteration=0,
            status='completed',
            data={
                'detector_sources_ps': len(detector.ps_filtered_group),
                'detector_sources_ext': len(detector.ext_filtered_group),
                'filtered_df_rows': len(detector.filtered_df) if hasattr(detector, 'filtered_df') else 0,
                'output_yaml': str(self.output_dir / 'filtered_sources.yaml'),
                'output_model': str(self.output_dir / 'curModel.model')
            },
            metadata={
                'detector_class': 'SourceSeedDetector',
                'completion_time': datetime.now().isoformat()
            }
        )
        
        self.logger.info(f"Source detection completed:")
        self.logger.info(f"  - Point sources: {len(detector.ps_filtered_group)}")
        self.logger.info(f"  - Extended sources: {len(detector.ext_filtered_group)}")
        self.logger.info(f"  - Output directory: {self.output_dir}")

    
    def _run_source_detection(self):
        """Run the source detection step"""
        self.logger.info("Initializing SourceSeedDetector for source detection...")
        
        self.checkpoint_mgr.save_step(
            step_name='SourceDetection',
            iteration=0,
            status='running',
            data={'output_dir': str(self.output_dir)},
            metadata={'step_type': 'source_detection'}
        )
        
        try:            
            detector = SourceSeedDetector(str(self.config.config_file))
            detector.save_dir = str(self.output_dir)
            
            self.logger.info("Running source detection...")
            detector.run_source_detection()
            
            self.checkpoint_mgr.save_step(
                step_name='SourceDetection',
                iteration=0,
                status='completed',
                data={
                    'sources_detected': len(detector.filtered_df) if hasattr(detector, 'filtered_df') else 0,
                    'output_yaml': str(self.output_dir / 'filtered_sources.yaml')
                },
                metadata={'completion_time': datetime.now().isoformat()}
            )
            
            self.logger.info(f"Source detection completed. Output: {self.output_dir}")
            
        except Exception as e:
            self.checkpoint_mgr.save_step(
                step_name='SourceDetection',
                iteration=0,
                status='failed',
                data={'error': str(e)},
                metadata={'error_traceback': traceback.format_exc()}
            )
            raise
    
    def print_summary(self):
        """Print pipeline execution summary"""
        self.logger.info("\n" + "="*80)
        self.logger.info("PIPELINE EXECUTION SUMMARY")
        self.logger.info("="*80)
        self.logger.info(f"Step: {self.step_name}")
        self.logger.info(f"Status: {self.checkpoint_mgr.history.get('status', 'unknown')}")
        self.logger.info(f"Main output directory: {self.main_dir}")
        self.logger.info(f"Step output directory: {self.output_dir}")
        self.logger.info("="*80 + "\n")
        
        self.checkpoint_mgr.print_history()
 
    def make_maps(self, output_filename: str = "model_map.fits"):
        self.logger.info("Running HealpixSigFluxMap...")
        bins = self.config["fitting"]["bins"]
        det_res = self.config["paths"]["detector_response"]
        output_file = self.output_dir / output_filename

        data_dir = Path(self.config["paths"]["data_dir"])

        input_files = []
        for b in bins:
            matched_files = list(data_dir.glob(f"*{b}*"))
            if not matched_files:
                self.logger.warning(f"No files found for bin {b} in {data_dir}")
            input_files.extend(matched_files)

        # Convert Path objects to strings for subprocess
        input_files = [str(f) for f in input_files]

        # Coordinates
        ra = self.config["coordinates"]["ra"]
        dec = self.config["coordinates"]["dec"]
        roi_x = self.config["coordinates"]["roi_x"]
        roi_y = self.config["coordinates"]["roi_y"]

        # Build the command
        cmd = (
            ["pixi", "run", "aerie-apps-HealpixSigFluxMap"]
            + ["-i"] + input_files
            + ["-b"] + bins
            + ["-d", str(det_res)]
            + ["--index", "2.6"]
            + ["--pivot", "7"]
            + ["--window", str(ra), str(dec), str(roi_x), str(roi_y)]
            + ["--negFlux", "--negSignif"]
            + ["--extension", "0.5"]
            + ["-o", str(output_file)]
        )

        self.logger.info("Executing command:")
        self.logger.info(" ".join(cmd))

        subprocess.run(cmd, check=True)

        self.logger.info(f"Model map created at {output_file}")

    def check_residual(self):
        """Check residuals after fitting"""
        self.logger.info("Checking residuals...")

        self.logger.info("Residual check completed (placeholder)")

    def create_residualmaps(self):
        """Create residual maps after fitting"""
        self.logger.info("Creating residual maps...")

        self.logger.info("Residual map creation completed (placeholder)")

    def create_modelmaps(self):
        """Create model maps after fitting"""
        self.logger.info("Creating model maps...")

        self.logger.info("Model map creation completed (placeholder)")

    def run(self):
        """Execute the pipeline based on step configuration"""
        self.logger.info(f"Starting {self.step_name} step...")
        if self.step_name == 'SeedModelFit':
            # Define the significance map filename to use consistently
            significance_map_filename = 'sky_map.fits'
            
            print(f"Seed model directory: {self.step_dir}")
            self.model = self.output_dir / 'curModel.model'
            if os.path.exists(self.model):
                self.logger.info(f"Model file already exists at {self.model}, skipping seed model fitting")
            else:
                self.logger.info(f"Model file not found at {self.model}, running seed model fitting")
                
                # Check if significance map needs to be created
                significance_map_path = self.config.get('paths.significance_map')
                signif_map_path = self.output_dir / significance_map_filename
                
                # Check if either significance map exists
                if (significance_map_path and os.path.exists(significance_map_path)) or os.path.exists(signif_map_path):
                    self.logger.info(f"Significance map found, skipping map creation")
                else:
                    self.logger.info("Significance map not found, creating it now...")
                    self.make_maps(output_filename=significance_map_filename)
                
                self._run_seed_model(significance_map_filename)
            self.run_fit_withTS()
            self.create_modelmaps()
            self.create_residualmaps()
            self.check_residual()

def main():
    config_file = '/Users/rishi/Documents/Analysis/Sources/AstroImageDetection-fitmodel/config_crab.yaml'
    # pipeline = SourceSearchPipeline(config_file)
    # pipeline.run()
    pipeline = SourceSearchPipeline(config_file)
    pipeline.run()
    params = pipeline.hal_fit.params
    stats = pipeline.hal_fit.statistics

    # Print results
    print("\nFit Parameters:")
    for name, value in params.items():
        print(f"{name}: {value}")

    print("\nFit Statistics:")
    print(stats)



if __name__ == "__main__":
    main()