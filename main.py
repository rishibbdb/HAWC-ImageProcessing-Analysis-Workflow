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
from pipeline_helpers import *
from pipeline_hd5 import convert_hd5_to_fits
import healpy as hp

class PipelineLogger:
    """Centralized logging system with separate pipeline and full logs"""
    
    def __init__(self, log_dir: str, log_level: str = 'INFO'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create Pipeline logger (only Pipeline messages)
        self.logger = logging.getLogger('Pipeline')
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.propagate = False  # Don't propagate to root logger
        
        # File handler for pipeline.log
        pipeline_log_file = self.log_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        pipeline_fh = logging.FileHandler(pipeline_log_file)
        pipeline_fh.setLevel(getattr(logging, log_level))
        
        # Console handler for pipeline messages
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, log_level))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        pipeline_fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers to pipeline logger
        self.logger.addHandler(pipeline_fh)
        self.logger.addHandler(ch)
        
        # Create root logger (captures everything from all packages)
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level))
        
        # File handler for full_log.log (captures all packages)
        full_log_file = self.log_dir / f"full_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        full_fh = logging.FileHandler(full_log_file)
        full_fh.setLevel(getattr(logging, log_level))
        full_fh.setFormatter(formatter)
        
        # Add full log handler to root logger
        root_logger.addHandler(full_fh)
    
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
        
        #Main Dir
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
        self._setup_roi_template()

    def _setup_roi_template(self):
        """Setup ROI template: either use existing, create new, or use default"""
        roi_template_path = self.config.get('paths.roi_template')
        create_roi_flag = self.config.get('paths.create_roi_template', False)
        
        if roi_template_path and os.path.exists(roi_template_path):
            self.logger.info(f"Using existing ROI template: {roi_template_path}")
            self.roiTemplate = roi_template_path
        elif create_roi_flag:
            self.logger.info("Creating new ROI template...")
            self.NSIDE = 1024
            self.include_pixels_roitemplate = True
            self.create_roi_template()
            self.roiTemplate = self.main_dir / 'roiTemplate.fits'
            self.logger.info(f"ROI template created at: {self.roiTemplate}")
        else:
            self.logger.info("Using default circular ROI (no template)")
            self.roiTemplate = None

    def create_roi_template(self):
        if self.config.get('coordinates.l') is None:
            self.ra = float(self.config.get('coordinates.ra'))
            self.dec = float(self.config.get('coordinates.dec'))
            central_coord = SkyCoord(ra=self.ra*u.degree, dec=self.dec*u.degree, frame='icrs')
            self.l = central_coord.galactic.l.deg
            self.b = central_coord.galactic.b.deg
        else:
            self.l = float(self.config.get('coordinates.l'))
            self.b = float(self.config.get('coordinates.b'))
        self.x_length = self.config.get('coordinates.roi_x', 1.0)
        self.y_length = self.config.get('coordinates.roi_y', 1.0)
        l_max, l_min, b_max, b_min = self.l + self.x_length, self.l - self.x_length, self.b + self.y_length, self.b - self.y_length, 
        rectangle_region = [ (l_max, b_max), (l_min, b_max), (l_min, b_min), (l_max, b_min) ] 
        rectangle_region_coords=convert_coords(rectangle_region,'galactic')
        m=np.zeros(hp.nside2npix(self.NSIDE))
        common_vectors=coord_vectors(rectangle_region_coords)
        common_pix=hp.query_polygon(self.NSIDE,common_vectors,inclusive=self.include_pixels_roitemplate)
        save_ROI(common_pix, self.NSIDE, f"{self.main_dir}/roiTemplate.fits")
    
    def run_fit(self):
        self.hal_fit = threeMLFit(config_path=str(self.config.config_file), model=self.model, save_dir=self.step_dir, roiTemplate=self.roiTemplate, logger=self.logger)
        self.hal_fit.hal_fit()
        self.logger.info(f"{self.step_name} completed successfully")

    def run_fit_witherrors(self):
        self.hal_fit = threeMLFit(config_path=str(self.config.config_file), model=self.model, save_dir=self.step_dir, roiTemplate=self.roiTemplate, logger=self.logger)
        self.hal_fit.hal_fit_with_covariance()
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
        
        detector = SourceSeedDetector(str(self.config.config_file), step_path=str(self.step_dir), logger=self.logger)
        
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
 
    def make_maps(self, data_dir=None, input_pattern: str = None, output_filename: str = "skymap.fits"):
        """
        Create maps using HealpixSigFluxMap with flexible input pattern matching
        
        Parameters:
        -----------
        data_dir : Path or str, optional
            Directory containing input files. If None, uses config paths.data_dir
        input_pattern : str, optional
            Pattern to match input files. If None, matches 'bin*' files
            - 'bin' or None: matches bin files (sky map)
            - 'residual': matches residual_bin* files
            - 'model': matches model_bin* files
        output_filename : str
            Output filename for the map
        """
        self.logger.info(f"Running HealpixSigFluxMap with pattern '{input_pattern}'...")
        
        # Use provided data_dir or get from config
        if data_dir is None:
            data_dir = Path(self.config["paths"]["data_dir"])
        else:
            data_dir = Path(data_dir)
        
        bins = self.config["fitting"]["bins"]
        det_res = self.config["paths"]["detector_response"]
        output_file = self.output_dir / output_filename
        
        # Determine the file pattern to search for
        if input_pattern is None or input_pattern == 'bin':
            # Default: match bin files for sky map
            search_pattern = f"*bin*"
            map_type = "sky"
        elif input_pattern == 'residual':
            # Match residual_bin files
            search_pattern = f"*residual*bin*"
            map_type = "residual"
        elif input_pattern == 'model':
            # Match model_bin files
            search_pattern = f"*model*bin*"
            map_type = "model"
        else:
            # Custom pattern
            search_pattern = f"*{input_pattern}*"
            map_type = input_pattern
        
        # Collect input files
        input_files = []
        for b in bins:
            matched_files = list(data_dir.glob(f"{search_pattern}{b}*"))
            if not matched_files:
                self.logger.warning(f"No files found for bin {b} with pattern '{search_pattern}' in {data_dir}")
            input_files.extend(matched_files)
        
        if not input_files:
            self.logger.error(f"No input files found matching pattern '{search_pattern}' in {data_dir}")
            return None
        
        self.logger.info(f"Found {len(input_files)} input files for {map_type} map")
        
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
            + ["--window", str(ra), str(dec), str(roi_x + 6), str(roi_y + 6)]
            + ["--negFlux", "--negSignif"]
            + ["-o", str(output_file)]
        )
        
        self.logger.info("Executing HealpixSigFluxMap command:")
        self.logger.info(" ".join(cmd))
        
        try:
            subprocess.run(cmd, check=True)
            self.logger.info(f"{map_type.capitalize()} map created at {output_file}")
            return output_file
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create {map_type} map: {e}")
            raise

    def check_residual(self):
        """Check residuals after fitting"""
        self.logger.info("Checking residuals...")
        resmap_path = os.path.join(self.output_dir, 'residual.fits')
        if os.path.exists(resmap_path):
            self._run_seed_model(significance_map_filename = 'residual.fits')
        else:
            self.logger.info("Residual map not found")
        self.logger.info("Residual check completed (placeholder)")

    def create_residualmaps(self):
        """Create residual maps after fitting"""
        resmap_path = os.path.join(self.output_dir, 'residual.fits')
        if os.path.exists(resmap_path):
            self.logger.info("Residual map exists")
            return
        else:
            self.logger.info("Creating residual maps...")
            convert_hd5_to_fits(self.output_dir, 'residual_fit.hd5', 'residual')
            self.make_maps(self.output_dir, input_pattern='residual', output_filename='residual.fits')

        self.logger.info("Residual map creation completed (placeholder)")

    def create_modelmaps(self):
        """Create model maps after fitting"""
        self.logger.info("Creating model maps...")
        convert_hd5_to_fits(self.output_dir, 'model_fit.hd5', 'model')
        self.make_maps(self.output_dir, input_pattern='model', output_filename='model.fits')
        self.logger.info("Model map creation completed (placeholder)")

    def _save_fit_results(self, params=None, stats=None, ts_values=None):
        """Save fitting results to YAML file in step directory (not results subdirectory)"""
        results_file = self.step_dir / 'fit_results.yaml'
        
        results = {
            'fit_step': self.step_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add parameters if provided
        if params is not None:
            params_dict = {}
            if hasattr(params, 'to_dict'):
                params_dict = params.to_dict()
            elif isinstance(params, dict):
                params_dict = params
            else:
                params_dict = dict(params)
            results['parameters'] = params_dict
        
        # Add statistics if provided
        if stats is not None:
            stats_dict = {}
            if hasattr(stats, 'to_dict'):
                stats_dict = stats.to_dict()
            elif isinstance(stats, dict):
                stats_dict = stats
            else:
                stats_dict = dict(stats)
            results['statistics'] = stats_dict
        
        # Add TS values if provided
        if ts_values is not None:
            ts_dict = {}
            if isinstance(ts_values, dict):
                ts_dict = ts_values
            elif hasattr(ts_values, 'to_dict'):
                # Handle pandas Series
                ts_dict = ts_values.to_dict()
            else:
                try:
                    ts_dict = dict(ts_values)
                except (ValueError, TypeError):
                    # If conversion fails, convert to string representation
                    ts_dict = {'ts_values': str(ts_values)}
            results['ts_values'] = ts_dict
        
        # Save to YAML
        with open(results_file, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, sort_keys=False)
        
        self.logger.info(f"Fit results saved to: {results_file}")
 
    def _run_seed_model_on_residual(self, residual_map_path: str):
        """Run source detection on residual map to find excess hotspots"""
        self.logger.info(f"Initializing SourceSeedDetector for residual analysis...")
        
        # Create a sub-directory for residual analysis
        residual_output_dir = self.output_dir / 'residual_analysis'
        residual_output_dir.mkdir(parents=True, exist_ok=True)
        
        detector = SourceSeedDetector(str(self.config.config_file), step_path=str(self.step_dir))
        
        # Update detector to use residual map
        detector.initialmap = residual_map_path
        detector.out_dir = str(residual_output_dir)
        
        self.logger.info(f"Running source detection on residual map: {residual_map_path}")
        
        try:
            detector.run()
            self.logger.info(f"Residual analysis completed. Results saved to {residual_output_dir}")
            
            # Log summary of detected hotspots
            if hasattr(detector, 'filtered_df') and len(detector.filtered_df) > 0:
                self.logger.info(f"Found {len(detector.filtered_df)} excess hotspots in residual map")
            else:
                self.logger.info("No significant excess hotspots found in residual map")
        
        except Exception as e:
            self.logger.error(f"Error running source detection on residual map: {e}")
            raise

    def run(self):
        """Execute the pipeline based on step configuration"""
        self.logger.info(f"Starting {self.step_name} step...")
        if self.step_name == 'SeedModelFit':
            significance_map_filename = 'sky_map.fits'
            self.logger.info(f"Seed model directory: {self.step_dir}")
            self.model = self.output_dir / 'curModel.model'
            if os.path.exists(self.model):
                self.logger.info(f"Model file already exists at {self.model}, skipping seed model fitting")
            else:
                self.logger.info(f"Model file not found at {self.model}, running seed model fitting")
                

                significance_map_path = self.config.get('paths.significance_map')
                signif_map_path = self.output_dir / significance_map_filename
                if (significance_map_path and os.path.exists(significance_map_path)) or os.path.exists(signif_map_path):
                    self.logger.info(f"Significance map found, skipping map creation")
                else:
                    self.logger.info("Significance map not found, creating it now...")
                    data_dir = Path(self.config["paths"]["data_dir"])
                    self.make_maps(data_dir, output_filename=significance_map_filename)
                
                self._run_seed_model(significance_map_filename)
            # self.run_fit_witherrors()
            self.run_fit()
            # self.hal_fit.make_maps()
            params = self.hal_fit.params
            print(params.value)
            print('*'*80)
            print(params.unit)
            print('*'*80)
            print(params.positive_error)
            print('*'*80)
            print(params.negative_error)
            # stats = self.hal_fit.statistics
            # source, TS = self.hal_fit.get_TS()
            # self._save_fit_results(params=params, stats=stats, ts_values=TS)
            self.create_residualmaps()
            self.check_residual()
            self.create_modelmaps()
        # else:


def main():
    config_file = 'config_crab.yaml'
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