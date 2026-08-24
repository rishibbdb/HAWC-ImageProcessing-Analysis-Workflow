"""
HAWC I/O Module: HDF5 ↔ FITS Conversion

Extracted from: pipeline_hd5.py
Handles conversion between HDF5 map files and FITS format for compatibility.
"""

from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import logging

import numpy as np
import healpy as hp
from astropy.io import fits
import os
try:
    from hawc_hal.maptree import map_tree_factory
except ImportError:
    raise ImportError("hawc_hal package required. Install via: pip install hawc_hal")


logger = logging.getLogger('hawc_analysis.io')


class HDF5Handler:
    """Handle HDF5 map file conversions to/from FITS format
    
    Provides utilities for:
    - Converting HAWC HDF5 map trees to FITS files
    - Managing FITS headers with HAWC-specific metadata
    - Per-energy-bin map conversion
    """
    
    # Default FITS header metadata
    DEFAULT_START_MJD = 56987.9286332451
    DEFAULT_STOP_MJD = 58107.2396848326
    
    FITS_COMMENT = ("FITS (Flexible Image Transport System) format is defined in "
                   "'Astronomy and Astrophysics', volume 376, page 359; "
                   "bibcode: 2001A&A...376..359H")
    
    @staticmethod
    def convert_hd5_to_fits(
        input_dir: str,
        hd5_filename: str,
        output_basename: str,
        logger: Optional[logging.Logger] = None
    ) -> List[Path]:
        """Convert HAWC HDF5 map tree to FITS format
        
        Converts per-energy-bin HDF5 maps to individual FITS files with proper
        headers and metadata.
        
        Parameters:
        -----------
        input_dir : str
            Directory containing input HDF5 file
        hd5_filename : str
            Filename of HDF5 map tree (e.g., 'model_fit.hd5')
        output_basename : str
            Basename for output FITS files (e.g., 'model')
            Creates: '{output_basename}_binB0C0.fits.gz', etc.
        logger : logging.Logger, optional
            Logger instance for messages
        
        Returns:
        --------
        list of Path
            List of created FITS file paths
        
        Raises:
        -------
        FileNotFoundError
            If input HDF5 file not found
        ImportError
            If hawc_hal not installed
        
        Examples:
        ---------
        >>> files = HDF5Handler.convert_hd5_to_fits(
        ...     './results', 'model_fit.hd5', 'model'
        ... )
        >>> print(f"Created {len(files)} FITS files")
        Created 22 FITS files
        """
        if logger is None:
            logger = logging.getLogger('hawc_analysis.io')
        
        input_dir = Path(input_dir)
        input_filepath = input_dir / hd5_filename
        
        # Validate input file
        if not input_filepath.exists():
            raise FileNotFoundError(f"HDF5 file not found: {input_filepath}")
        
        logger.info(f"Converting HDF5 to FITS: {hd5_filename}")
        
        try:
            # Load map tree from HDF5
            maptree = map_tree_factory(str(input_filepath), None)
        except Exception as e:
            logger.error(f"Failed to load HDF5 map tree: {e}")
            raise
        
        # Prepare FITS header metadata
        creation_time = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
        
        fits_header_keys = [
            'COMMENT', 'COMMENT', 'DATE', 'STARTMJD', 'STOPMJD',
            'NEVENTS', 'TOTDUR', 'DURATION', 'MAPTYPE', 'MAXDUR',
            'MINDUR', 'EPOCH', 'MAPFILETYPE'
        ]
        
        fits_header_values = [
            HDF5Handler.FITS_COMMENT,
            HDF5Handler.FITS_COMMENT,
            creation_time,
            HDF5Handler.DEFAULT_START_MJD,
            HDF5Handler.DEFAULT_STOP_MJD,
            -1.0,
            24412.9020670185,
            1.9943578604616,
            'duration',
            -1.0,
            -1.0,
            'unknown',
            'duration'
        ]
        
        fits_header_comments = [
            "file conforms to FITS standard",
            "number of bits per data pixel",
            "file creation date (YYYY-MM-DDThh:mm:ss UT)",
            "MJD of first event",
            "MJD of last event",
            "Number of events in map",
            "Total integration time [hours]",
            "Avg integration time [hours]",
            "e.g. Skymap, Moonmap",
            "Max integration time [hours]",
            "Min integration time [hours]",
            "e.g. J2000, current, J2016, B1950, etc.",
            "e.g. standard, duration, integration"
        ]
        
        # Column metadata
        column_names = ['data map', 'background map', 'exposure map']
        column_formats = [np.float64 for _ in column_names]
        column_units = ['unknown' for _ in column_names]
        
        created_files = []
        
        # Convert each energy bin
        for analysis_bin in maptree.analysis_bins_labels:
            map_bin = maptree[analysis_bin]
            
            # Extract map properties
            nside = map_bin.nside
            npix = map_bin.npix
            transits = map_bin.n_transits
            scheme = map_bin.scheme
            
            # Determine HEALPix scheme
            use_nested = scheme.lower() == 'nested'
            
            # Extract map data
            try:
                data = map_bin.observation_map.as_dense()
                background = map_bin.background_map.as_dense()
            except Exception as e:
                logger.error(f"Failed to extract maps from bin {analysis_bin}: {e}")
                raise
            
            # Create dummy exposure map (all pixels set to large value)
            exposure = np.full(npix, 9e9, dtype=np.float64)
            
            # Output filename
            output_filename = f"{output_basename}_bin{analysis_bin}.fits.gz"
            output_filepath = input_dir / 'fits'/ output_filename
            os.makedirs(output_filepath.parent, exist_ok=True)
            
            try:
                # Write FITS file with healpy
                hp.fitsfunc.write_map(
                    str(output_filepath),
                    (data, background, exposure),
                    column_names=column_names,
                    column_units=column_units,
                    dtype=column_formats,
                    partial=False,
                    fits_IDL=True,
                    overwrite=True,
                    nest=use_nested
                )
                
                # Update FITS headers with metadata
                with fits.open(str(output_filepath), 'update') as hdul:
                    header = hdul[0].header
                    
                    for key, value, comment in zip(
                        fits_header_keys,
                        fits_header_values,
                        fits_header_comments
                    ):
                        # Update dynamic values based on actual data
                        if key == 'TOTDUR':
                            value = 24.0 * transits
                        elif key == 'STOPMJD':
                            value = HDF5Handler.DEFAULT_START_MJD + transits
                        
                        header[key] = (value, comment)
                
                logger.info(f"Created FITS file: {output_filepath}")
                created_files.append(output_filepath)
                
            except Exception as e:
                logger.error(f"Failed to write FITS file {output_filepath}: {e}")
                raise
        
        logger.info(f"Successfully converted {len(created_files)} energy bins to FITS")
        return created_files
    
    @staticmethod
    def read_fits_map(
        fits_filepath: str,
        hdu_index: int = 0,
        logger: Optional[logging.Logger] = None
    ) -> Tuple[np.ndarray, dict]:
        """Read map data from FITS file
        
        Parameters:
        -----------
        fits_filepath : str
            Path to FITS file
        hdu_index : int, optional
            HDU index to read (default: 0 - primary HDU)
        logger : logging.Logger, optional
            Logger instance
        
        Returns:
        --------
        tuple
            (map_data, header_dict)
            - map_data : numpy array of shape (ncolumns, npix)
            - header_dict : FITS header as dictionary
        """
        if logger is None:
            logger = logging.getLogger('hawc_analysis.io')
        
        fits_path = Path(fits_filepath)
        
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")
        
        try:
            with fits.open(str(fits_path)) as hdul:
                hdu = hdul[hdu_index]
                data = hdu.data
                header = dict(hdu.header)
            
            logger.debug(f"Read FITS map from {fits_path}")
            return data, header
        
        except Exception as e:
            logger.error(f"Failed to read FITS file {fits_path}: {e}")
            raise
    
    @staticmethod
    def write_fits_map(
        fits_filepath: str,
        data: np.ndarray,
        header: Optional[dict] = None,
        column_names: Optional[List[str]] = None,
        logger: Optional[logging.Logger] = None,
        overwrite: bool = False
    ) -> Path:
        """Write map data to FITS file
        
        Parameters:
        -----------
        fits_filepath : str
            Output FITS file path
        data : np.ndarray
            Map data (2D array with shape (ncolumns, npix))
        header : dict, optional
            Header keywords and values
        column_names : list of str, optional
            Names for data columns
        logger : logging.Logger, optional
            Logger instance
        overwrite : bool, optional
            Overwrite existing file (default: False)
        
        Returns:
        --------
        Path
            Path to written FITS file
        """
        if logger is None:
            logger = logging.getLogger('hawc_analysis.io')
        
        output_path = Path(fits_filepath)
        
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"FITS file already exists: {output_path}\n"
                f"Use overwrite=True to replace"
            )
        
        try:
            # Create primary HDU
            primary_hdu = fits.PrimaryHDU(data=data[0] if len(data.shape) > 1 else data)
            
            # Add header cards if provided
            if header:
                for key, value in header.items():
                    primary_hdu.header[key] = value
            
            # Create HDU list and write
            hdul = fits.HDUList([primary_hdu])
            hdul.writeto(str(output_path), overwrite=overwrite)
            
            logger.info(f"Wrote FITS map to {output_path}")
            return output_path
        
        except Exception as e:
            logger.error(f"Failed to write FITS file {output_path}: {e}")
            raise


# Convenience functions for backward compatibility with original code

def convert_hd5_to_fits(
    dir: str,
    filename: str,
    outfile: str,
    logger: Optional[logging.Logger] = None
) -> List[Path]:
    """Convenience wrapper for HDF5Handler.convert_hd5_to_fits
    
    Maintains backward compatibility with original pipeline_hd5.py API.
    
    Parameters:
    -----------
    dir : str
        Directory containing HDF5 file
    filename : str
        HDF5 filename
    outfile : str
        Output basename for FITS files
    logger : logging.Logger, optional
        Logger instance
    
    Returns:
    --------
    list of Path
        List of created FITS files
    """
    return HDF5Handler.convert_hd5_to_fits(dir, filename, outfile, logger)