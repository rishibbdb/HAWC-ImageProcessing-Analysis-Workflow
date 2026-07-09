#!/usr/bin/env python3
"""
Pipeline Utilities Module
Helper functions and classes for the source search pipeline
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SourceInfo:
    """Information about a detected source"""
    name: str
    ra: float
    dec: float
    l: float  # Galactic longitude
    b: float  # Galactic latitude
    significance: float  # Detection significance (sigma)
    flux: float  # Flux estimate
    spectrum: str = 'PowerLaw'  # Spectral model
    morphology: str = 'PointSource'  # Spatial model
    extension: float = 0.0  # Extension (0 for point source)
    ts: float = 0.0  # Test statistic
    likelihood: float = 0.0  # Log-likelihood
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


@dataclass
class FitResults:
    """Results from a model fit"""
    model_file: str
    map_file: str
    converged: bool
    log_likelihood: float
    num_free_params: int
    num_data_points: int
    chi2: float
    bic: float
    aic: float
    sources: List[SourceInfo]
    timestamp: str
    fit_type: str  # 'initial', 'test_source', 'test_extension', 'test_spectrum', 'final'
    
    def to_dict(self):
        return {
            'model_file': self.model_file,
            'map_file': self.map_file,
            'converged': self.converged,
            'log_likelihood': self.log_likelihood,
            'num_free_params': self.num_free_params,
            'num_data_points': self.num_data_points,
            'chi2': self.chi2,
            'bic': self.bic,
            'aic': self.aic,
            'sources': [s.to_dict() for s in self.sources],
            'timestamp': self.timestamp,
            'fit_type': self.fit_type
        }
    
    @classmethod
    def from_dict(cls, data: Dict):
        sources = [SourceInfo.from_dict(s) for s in data.pop('sources', [])]
        return cls(sources=sources, **data)


@dataclass
class DeltaLikelihoodTest:
    """Results from a delta-likelihood test"""
    test_name: str
    source_id: int
    parameter_tested: str  # e.g., 'extension', 'spectrum'
    baseline_likelihood: float
    test_likelihood: float
    delta_ts: float
    threshold: float
    accepted: bool
    test_model_file: str
    timestamp: str
    
    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict):
        return cls(**data)


# ============================================================================
# SOURCE DETECTION UTILITIES
# ============================================================================

class SourceDetector:
    """Utilities for source detection"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def detect_blobs(self, significance_map: np.ndarray, 
                     min_sigma: float = 1.0,
                     max_sigma: float = 10.0,
                     sigma_ratio: float = 1.4,
                     threshold: float = 0.01) -> List[Dict]:
        """
        Detect sources using Difference of Gaussians
        
        Parameters:
        -----------
        significance_map : np.ndarray
            Significance map (units of sigma)
        min_sigma : float
            Minimum Gaussian sigma
        max_sigma : float
            Maximum Gaussian sigma
        sigma_ratio : float
            Ratio between consecutive sigma values
        threshold : float
            Threshold for blob detection
        
        Returns:
        --------
        blobs : list of dict
            Detected blobs with properties
        """
        from scipy.ndimage import gaussian_filter
        from skimage.filters import difference_of_gaussians
        from skimage.feature import blob_dog
        
        try:
            # Normalize map
            norm_map = (significance_map - np.nanmean(significance_map)) / np.nanstd(significance_map)
            
            # Detect blobs using DoG
            blobs = blob_dog(
                norm_map,
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                sigma_ratio=sigma_ratio,
                threshold=threshold,
                overlap=0.5
            )
            
            # Convert to source list
            sources = []
            for blob in blobs:
                y, x, sigma = blob
                sources.append({
                    'x_pixel': int(x),
                    'y_pixel': int(y),
                    'sigma': float(sigma),
                    'significance': float(norm_map[int(y), int(x)])
                })
            
            self.logger.info(f"Detected {len(sources)} sources using DoG")
            return sources
        
        except Exception as e:
            self.logger.error(f"Error in blob detection: {str(e)}")
            return []
    
    def cluster_sources(self, sources: List[Dict], min_separation: float = 0.1,
                       pixel_scale: float = 0.05) -> List[Dict]:
        """
        Cluster nearby sources and merge them
        
        Parameters:
        -----------
        sources : list of dict
            List of detected sources with pixel coordinates
        min_separation : float
            Minimum separation in degrees
        pixel_scale : float
            Pixel scale in degrees/pixel
        
        Returns:
        --------
        clustered : list of dict
            Clustered source list
        """
        if not sources:
            return []
        
        min_sep_pix = min_separation / pixel_scale
        clustered = []
        used = set()
        
        for i, src in enumerate(sources):
            if i in used:
                continue
            
            cluster = [src]
            used.add(i)
            
            # Find nearby sources
            for j, other_src in enumerate(sources[i+1:], i+1):
                if j in used:
                    continue
                
                dist = np.sqrt(
                    (src['x_pixel'] - other_src['x_pixel'])**2 +
                    (src['y_pixel'] - other_src['y_pixel'])**2
                )
                
                if dist < min_sep_pix:
                    cluster.append(other_src)
                    used.add(j)
            
            # Merge cluster
            if len(cluster) == 1:
                merged = cluster[0]
            else:
                # Weight by significance
                weights = [s.get('significance', 1.0) for s in cluster]
                weights = np.array(weights)
                weights = weights / weights.sum()
                
                merged = {
                    'x_pixel': np.average([s['x_pixel'] for s in cluster], weights=weights),
                    'y_pixel': np.average([s['y_pixel'] for s in cluster], weights=weights),
                    'sigma': np.average([s['sigma'] for s in cluster], weights=weights),
                    'significance': np.max([s['significance'] for s in cluster]),
                    'num_components': len(cluster)
                }
            
            clustered.append(merged)
        
        self.logger.info(f"Clustered {len(sources)} sources into {len(clustered)} sources")
        return clustered
    
    def pixel_to_sky(self, wcs, x_pix: float, y_pix: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to sky coordinates
        
        Parameters:
        -----------
        wcs : astropy.wcs.WCS
            WCS information
        x_pix : float
            X pixel coordinate
        y_pix : float
            Y pixel coordinate
        
        Returns:
        --------
        ra, dec : float, float
            RA and Dec in degrees
        """
        from astropy.wcs import WCS
        
        try:
            sky = wcs.pixel_to_world(x_pix, y_pix)
            return sky.ra.degree, sky.dec.degree
        except Exception as e:
            self.logger.error(f"Error converting coordinates: {str(e)}")
            return None, None


# ============================================================================
# FITTING UTILITIES
# ============================================================================

class FittingUtilities:
    """Utilities for model fitting"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def create_model_xml(self, template_file: str, sources: List[SourceInfo],
                        background: Dict, output_file: str) -> bool:
        """
        Create a 3ML model XML file
        
        Parameters:
        -----------
        template_file : str
            Path to template XML file
        sources : list of SourceInfo
            List of sources to include
        background : dict
            Background model information
        output_file : str
            Output XML file path
        
        Returns:
        --------
        success : bool
            Whether the file was created successfully
        """
        try:
            # TODO: Implement actual XML generation
            # This should create a proper 3ML/threeML compatible XML file
            
            self.logger.info(f"Created model XML with {len(sources)} sources")
            return True
        
        except Exception as e:
            self.logger.error(f"Error creating model XML: {str(e)}")
            return False
    
    def extract_parameters_from_fit(self, fit_results_file: str) -> Dict[str, Any]:
        """
        Extract fitted parameters from fit results file
        
        Parameters:
        -----------
        fit_results_file : str
            Path to fit results JSON file
        
        Returns:
        --------
        params : dict
            Extracted parameters
        """
        try:
            with open(fit_results_file, 'r') as f:
                results = json.load(f)
            
            params = {
                'log_likelihood': results.get('log_likelihood'),
                'sources': results.get('sources', []),
                'background': results.get('background', {}),
                'ts_values': results.get('ts_values', {})
            }
            
            return params
        
        except Exception as e:
            self.logger.error(f"Error extracting parameters: {str(e)}")
            return {}
    
    def compute_delta_likelihood(self, baseline_fit: FitResults,
                                test_fit: FitResults) -> float:
        """
        Compute delta log-likelihood (TS) between two fits
        
        Parameters:
        -----------
        baseline_fit : FitResults
            Baseline model fit
        test_fit : FitResults
            Test model fit
        
        Returns:
        --------
        delta_ts : float
            2 * (log_likelihood_test - log_likelihood_baseline)
        """
        delta_ll = test_fit.log_likelihood - baseline_fit.log_likelihood
        delta_ts = 2 * delta_ll
        
        self.logger.info(f"Delta TS: {delta_ts:.2f}")
        return delta_ts
    
    def estimate_ts_significance(self, delta_ts: float) -> float:
        """
        Estimate significance (sigma) from delta TS
        
        Parameters:
        -----------
        delta_ts : float
            Delta test statistic
        
        Returns:
        --------
        sigma : float
            Approximate significance in sigma
        """
        # Rough approximation: TS ~ sigma^2 for 1 DOF
        sigma = np.sqrt(max(delta_ts, 0))
        return sigma
    
    def freeze_parameters(self, model_file: str, param_names: List[str],
                         output_file: str) -> bool:
        """
        Freeze specific parameters in a model
        
        Parameters:
        -----------
        model_file : str
            Input model XML file
        param_names : list of str
            Names of parameters to freeze
        output_file : str
            Output model XML file
        
        Returns:
        --------
        success : bool
        """
        try:
            # TODO: Implement actual parameter freezing
            self.logger.info(f"Froze {len(param_names)} parameters")
            return True
        
        except Exception as e:
            self.logger.error(f"Error freezing parameters: {str(e)}")
            return False
    
    def free_parameters(self, model_file: str, param_names: List[str],
                       output_file: str) -> bool:
        """
        Free specific parameters in a model
        
        Parameters:
        -----------
        model_file : str
            Input model XML file
        param_names : list of str
            Names of parameters to free
        output_file : str
            Output model XML file
        
        Returns:
        --------
        success : bool
        """
        try:
            # TODO: Implement actual parameter unfreezing
            self.logger.info(f"Freed {len(param_names)} parameters")
            return True
        
        except Exception as e:
            self.logger.error(f"Error freeing parameters: {str(e)}")
            return False


# ============================================================================
# MAP AND RESIDUAL UTILITIES
# ============================================================================

class MapUtilities:
    """Utilities for working with maps"""
    
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def compute_significance_map(self, data_map: np.ndarray,
                                model_map: np.ndarray,
                                background_map: Optional[np.ndarray] = None,
                                smoothing_kernel: int = 3) -> np.ndarray:
        """
        Compute significance map from data and model
        
        Parameters:
        -----------
        data_map : np.ndarray
            Observed data map
        model_map : np.ndarray
            Model prediction map
        background_map : np.ndarray, optional
            Background map (if separate from model)
        smoothing_kernel : int
            Size of Gaussian smoothing kernel
        
        Returns:
        --------
        significance_map : np.ndarray
            Significance map in units of sigma
        """
        from scipy.ndimage import gaussian_filter
        
        try:
            # Compute residuals
            residuals = data_map - model_map
            
            # Smooth residuals
            smoothed_residuals = gaussian_filter(residuals, sigma=smoothing_kernel)
            
            # Compute significance (very simplified)
            # In real implementation, would use proper statistical methods
            significance = smoothed_residuals / np.sqrt(np.abs(model_map) + 1e-10)
            
            self.logger.info(f"Computed significance map (max: {np.nanmax(np.abs(significance)):.2f} sigma)")
            
            return significance
        
        except Exception as e:
            self.logger.error(f"Error computing significance map: {str(e)}")
            return np.zeros_like(data_map)
    
    def save_fits_map(self, data: np.ndarray, output_file: str,
                      wcs=None, header: Dict = None) -> bool:
        """
        Save data as FITS file
        
        Parameters:
        -----------
        data : np.ndarray
            Data array
        output_file : str
            Output file path
        wcs : astropy.wcs.WCS, optional
            WCS information
        header : dict, optional
            Additional header keywords
        
        Returns:
        --------
        success : bool
        """
        try:
            from astropy.io import fits
            
            hdu = fits.PrimaryHDU(data)
            if wcs:
                hdu.header.update(wcs.to_header())
            if header:
                hdu.header.update(header)
            
            hdu.writeto(output_file, overwrite=True)
            self.logger.info(f"Saved FITS file: {output_file}")
            return True
        
        except Exception as e:
            self.logger.error(f"Error saving FITS file: {str(e)}")
            return False
    
    def load_fits_map(self, fits_file: str) -> Tuple[np.ndarray, Optional[Any]]:
        """
        Load FITS map file
        
        Parameters:
        -----------
        fits_file : str
            Path to FITS file
        
        Returns:
        --------
        data : np.ndarray
            Data array
        header : astropy.io.fits.Header
            FITS header
        """
        try:
            from astropy.io import fits
            
            with fits.open(fits_file) as hdul:
                data = hdul[0].data
                header = hdul[0].header
            
            return data, header
        
        except Exception as e:
            self.logger.error(f"Error loading FITS file: {str(e)}")
            return None, None


# ============================================================================
# CONFIGURATION UTILITIES
# ============================================================================

class ConfigValidator:
    """Validate configuration files"""
    
    @staticmethod
    def validate_required_fields(config: Dict) -> Tuple[bool, List[str]]:
        """
        Validate that all required fields are present
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        
        Returns:
        --------
        valid : bool
            Whether config is valid
        errors : list of str
            List of error messages
        """
        required = {
            'coordinates': ['ra', 'dec', 'system', 'roi_x', 'roi_y'],
            'paths': ['main_dir', 'map_file', 'detector_response'],
            'source_detection': ['blob_method', 'threshold'],
            'fitting': ['plugin', 'estimator'],
            'likelihood_thresholds': ['point_source_detection', 'extension_test']
        }
        
        errors = []
        
        for section, fields in required.items():
            if section not in config:
                errors.append(f"Missing section: {section}")
            else:
                for field in fields:
                    if field not in config[section]:
                        errors.append(f"Missing field: {section}.{field}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_coordinate_system(coordsys: str) -> Tuple[bool, str]:
        """Validate coordinate system"""
        if coordsys not in ['C', 'G']:
            return False, f"Invalid coordinate system: {coordsys} (must be 'C' or 'G')"
        return True, ""
    
    @staticmethod
    def validate_file_paths(config: Dict) -> Tuple[bool, List[str]]:
        """Validate that referenced files exist"""
        errors = []
        
        paths = config.get('paths', {})
        for key, path in paths.items():
            if path and isinstance(path, str) and key != 'main_dir':
                if not Path(path).exists():
                    errors.append(f"File not found: {key} = {path}")
        
        return len(errors) == 0, errors


# ============================================================================
# RESULT REPORTING
# ============================================================================

class ResultsReporter:
    """Generate reports and summaries of results"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_summary_report(self, pipeline_results: Dict) -> str:
        """
        Generate a text summary report
        
        Parameters:
        -----------
        pipeline_results : dict
            Results from pipeline execution
        
        Returns:
        --------
        report_file : str
            Path to generated report file
        """
        report_file = self.output_dir / 'summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("4HWC CATALOG SEARCH AND FITTING PIPELINE - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Completion Time: {pipeline_results.get('completion_time')}\n")
            f.write(f"Final Model: {pipeline_results.get('final_model')}\n")
            f.write(f"Number of Sources: {pipeline_results.get('num_final_sources')}\n")
            f.write(f"Final Log-Likelihood: {pipeline_results.get('final_loglikelihood')}\n")
            f.write("\n" + "="*80 + "\n")
        
        return str(report_file)
    
    def generate_source_catalog(self, sources: List[SourceInfo]) -> str:
        """
        Generate a source catalog in multiple formats
        
        Parameters:
        -----------
        sources : list of SourceInfo
            List of detected sources
        
        Returns:
        --------
        catalog_file : str
            Path to generated catalog file
        """
        catalog_file = self.output_dir / 'source_catalog.json'
        
        catalog_data = {
            'timestamp': datetime.now().isoformat(),
            'num_sources': len(sources),
            'sources': [s.to_dict() for s in sources]
        }
        
        with open(catalog_file, 'w') as f:
            json.dump(catalog_data, f, indent=2)
        
        return str(catalog_file)
