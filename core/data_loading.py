"""
Data Loading Utilities

Extracted from: pipeline_helpers.py
Handles loading and manipulating HAWC significance maps and coordinate systems.
"""

from pathlib import Path
from typing import Tuple, Optional, List
import logging

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle
import astropy.units as u
import astropy.wcs.utils as astropy_utils
from reproject import reproject_from_healpix
from astropy.io.fits import Header

class DataLoader:
    """Load and process HAWC observation data"""
    
    @staticmethod
    def loadmap(filename, coord_sys, coords, *args, logger=None):
        # print("Coords=",coords)
        with fits.open(filename) as ihdu:
            if 'xyrange' in args:
                e1, e2, e3 , e4 = coords
                cX, cY = (e1+e2)/2, (e3+e4)/2
                xR = int(np.abs(e1-e2)/(1/360))
                yR = int(np.abs(e3-e4)/(1/360))
            if 'origin' in args:
                cX, cY, xR, yR = coords
                xR = int(xR/(1/360))
                yR = int(yR/(1/360))
            
            if coord_sys == 'C':   ###Celestial Coordinate System
                target_header = Header()
                target_header['NAXIS'] = 2
                target_header['NAXIS1'] = xR
                target_header['NAXIS2'] = yR
                target_header['CTYPE1'] = 'RA---MOL'
                target_header['CRPIX1'] = xR/2
                target_header['CRVAL1'] = cX
                target_header['CDELT1'] = -2./360
                target_header['CUNIT1'] = 'deg     '
                target_header['CTYPE2'] = 'DEC--MOL'
                target_header['CRPIX2'] = yR/2
                target_header['CRVAL2'] = cY
                target_header['CDELT2'] = 2./360
                target_header['CUNIT2'] = 'deg     '
                target_header['COORDSYS'] = 'icrs    '
                print("Loading Celestial Map")
            if coord_sys == 'G':  ###Galactic Coordinate System
                target_header = Header()
                target_header['NAXIS'] = 2
                target_header['NAXIS1'] = xR
                target_header['NAXIS2'] = yR
                target_header['CTYPE1'] = 'GLON-AIT'
                target_header['CRPIX1'] = xR/2
                target_header['CRVAL1'] = cX
                target_header['CDELT1'] = -2./360
                target_header['CUNIT1'] = 'deg     '
                target_header['CTYPE2'] = 'GLAT-AIT'
                target_header['CRPIX2'] = yR/2
                target_header['CRVAL2'] = cY
                target_header['CDELT2'] = 2./360
                target_header['CUNIT2'] = 'deg     '
                target_header['COORDSYS'] = 'galactic    '
                print("Loading Galactic Map")
            
            skymap_data = ihdu[1].data["significance"]
            ihdu[1].header['COORDSYS'] = 'icrs    '
            wcs = WCS(target_header)
            array, footprint = reproject_from_healpix(ihdu[1], target_header)
            print("Fits File loaded")
            
        return array, footprint, wcs
    
    @staticmethod
    def load_hawc_data(
        filename: str,
        center_x: float,
        center_y: float,
        x_length: float,
        y_length: float,
        coord_sys: str,
        logger: Optional[object] = None
    ) -> Tuple[np.ndarray, np.ndarray, WCS, int, int, float]:
        """Load HAWC data from FITS file
        
        Parameters:
        -----------
        filename : str
            Path to FITS significance map
        center_x : float
            Center RA (C) or L (G) in degrees
        center_y : float
            Center Dec (C) or B (G) in degrees
        x_length : float
            Width of ROI in degrees (RA or L)
        y_length : float
            Height of ROI in degrees (Dec or B)
        coord_sys : str
            Coordinate system: 'C' (celestial) or 'G' (galactic)
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        tuple
            (array, footprint, wcs, xnum, ynum, pixel_size)
            - array : numpy array of significance map
            - footprint : footprint array
            - wcs : WCS projection object
            - xnum : number of pixels in X direction
            - ynum : number of pixels in Y direction
            - pixel_size : pixel size in degrees
        """
        if logger:
            if coord_sys == 'C':
                logger.info(f"Loading HAWC data: Celestial RA={center_x}, Dec={center_y}")
            else:
                logger.info(f"Loading HAWC data: Galactic L={center_x}, B={center_y}")
        
        origin = [center_x, center_y, x_length, y_length]
        
        # Load the map
        array, footprint, wcs = DataLoader.loadmap(
            filename,
            coord_sys,
            origin,
            'origin',
            logger=logger,
        )
        
        xnum = array.shape[1]
        ynum = array.shape[0]
        pixel_size = wcs.wcs.cdelt[1]
        
        if logger:
            logger.info(f"Loaded map dimensions: {xnum}x{ynum} pixels")
            logger.info(f"Pixel size: {pixel_size:.6f} degrees/pixel")
        
        return array, footprint, wcs, xnum, ynum, pixel_size
    
    @staticmethod
    def find_peak(
        array: np.ndarray,
        wcs: WCS,
        logger: Optional[object] = None
    ) -> Tuple[np.ndarray, tuple, tuple, float]:
        """Find peak (maximum) in significance map
        
        Parameters:
        -----------
        array : np.ndarray
            Significance map
        wcs : WCS
            WCS projection object
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        tuple
            (peak_value, pixel_location, sky_location, significance)
            - peak_value : maximum value in array
            - pixel_location : (x, y) pixel coordinates
            - sky_location : SkyCoord object
            - significance : significance in sigma
        """
        peak_index = np.unravel_index(np.argmax(array), array.shape)
        peak_value = array[peak_index]
        
        # Convert to sky coordinates
        sky_coord = astropy_utils.pixel_to_skycoord(peak_index[1], peak_index[0], wcs)
        
        if logger:
            logger.info(f"Peak pixel location: {peak_index}")
            logger.info(f"Peak sky location: {sky_coord}")
            logger.info(f"Peak significance: {peak_value:.2f} sigma")
        
        return peak_value, peak_index, sky_coord, peak_value
    
    @staticmethod
    def find_minimum(
        array: np.ndarray,
        wcs: WCS,
        logger: Optional[object] = None
    ) -> Tuple[np.ndarray, tuple, tuple, float]:
        """Find minimum (well) in significance map
        
        Parameters:
        -----------
        array : np.ndarray
            Significance map
        wcs : WCS
            WCS projection object
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        tuple
            (min_value, pixel_location, sky_location, significance)
        """
        min_index = np.unravel_index(np.argmin(array), array.shape)
        min_value = array[min_index]
        
        # Convert to sky coordinates
        sky_coord = astropy_utils.pixel_to_skycoord(min_index[1], min_index[0], wcs)
        
        if logger:
            logger.info(f"Minimum pixel location: {min_index}")
            logger.info(f"Minimum sky location: {sky_coord}")
            logger.info(f"Minimum significance: {min_value:.2f} sigma")
        
        return min_value, min_index, sky_coord, min_value
    
    @staticmethod
    def extract_roi(
        array: np.ndarray,
        wcs: WCS,
        roi_x_deg: float,
        roi_y_deg: float,
        logger: Optional[object] = None
    ) -> Tuple[np.ndarray, WCS]:
        """Extract rectangular ROI from map
        
        Parameters:
        -----------
        array : np.ndarray
            Full significance map
        wcs : WCS
            WCS projection object
        roi_x_deg : float
            ROI width in degrees
        roi_y_deg : float
            ROI height in degrees
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        tuple
            (roi_array, roi_wcs)
        """
        # Get center of map
        center_y, center_x = array.shape[0] // 2, array.shape[1] // 2
        
        # Convert degrees to pixels (assuming ~0.01 deg/pixel)
        pixel_scale = np.abs(wcs.wcs.cdelt[0])
        roi_x_pix = int(roi_x_deg / pixel_scale)
        roi_y_pix = int(roi_y_deg / pixel_scale)
        
        # Extract ROI
        y_start = max(0, center_y - roi_y_pix // 2)
        y_end = min(array.shape[0], center_y + roi_y_pix // 2)
        x_start = max(0, center_x - roi_x_pix // 2)
        x_end = min(array.shape[1], center_x + roi_x_pix // 2)
        
        roi_array = array[y_start:y_end, x_start:x_end]
        
        # Update WCS for extracted ROI
        roi_wcs = wcs.deepcopy()
        roi_wcs.wcs.crpix[0] -= x_start
        roi_wcs.wcs.crpix[1] -= y_start
        
        if logger:
            logger.info(f"Extracted ROI: {roi_array.shape} pixels")
        
        return roi_array, roi_wcs
    
    @staticmethod
    def clip_negative_values(
        array: np.ndarray,
        floor_value: float = -5.0,
        logger: Optional[object] = None
    ) -> np.ndarray:
        """Soft floor negative values in significance map
        
        Parameters:
        -----------
        array : np.ndarray
            Input array
        floor_value : float, optional
            Floor value for clipping (default: -5.0)
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        np.ndarray
            Clipped array
        """
        clipped = np.clip(array, floor_value, None)
        
        if logger:
            logger.debug(f"Clipped array to floor: {floor_value}")
        
        return clipped
