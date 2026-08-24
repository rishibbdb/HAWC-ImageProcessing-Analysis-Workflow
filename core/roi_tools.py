"""
ROI Tools

Extracted from: pipeline_helpers.py
Handles Region of Interest (ROI) operations for HEALPix maps.
"""

from typing import List, Tuple, Optional
import logging

import numpy as np
import healpy as hp
from astropy.coordinates import SkyCoord
import astropy.units as u


class ROITools:
    """Region of Interest utilities for HEALPix maps"""
    
    @staticmethod
    def convert_to_icrs(
        region_galactic: List[Tuple[float, float]],
        logger: Optional[object] = None
    ) -> List[Tuple[float, float]]:
        """Convert galactic coordinates to ICRS (celestial)
        
        Parameters:
        -----------
        region_galactic : list of tuple
            List of (l, b) coordinates in degrees (galactic)
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        list of tuple
            List of (ra, dec) coordinates in degrees (ICRS)
        
        Examples:
        ---------
        >>> gal_coords = [(265.0, -28.5), (270.0, -25.0)]
        >>> icrs_coords = ROITools.convert_to_icrs(gal_coords)
        >>> print(icrs_coords)
        [(83.63, 22.51), (88.45, 24.23)]
        """
        region_icrs = []
        
        for lon_gal, lat_gal in region_galactic:
            # Create SkyCoord in galactic frame
            coord_gal = SkyCoord(
                l=lon_gal * u.degree,
                b=lat_gal * u.degree,
                frame='galactic'
            )
            
            # Convert to ICRS
            coord_icrs = coord_gal.icrs
            
            region_icrs.append((coord_icrs.ra.deg, coord_icrs.dec.deg))
        
        if logger:
            logger.debug(f"Converted {len(region_galactic)} galactic to ICRS coordinates")
        
        return region_icrs
    
    @staticmethod
    def convert_to_galactic(
        region_icrs: List[Tuple[float, float]],
        logger: Optional[object] = None
    ) -> List[Tuple[float, float]]:
        """Convert ICRS (celestial) coordinates to galactic
        
        Parameters:
        -----------
        region_icrs : list of tuple
            List of (ra, dec) coordinates in degrees (ICRS)
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        list of tuple
            List of (l, b) coordinates in degrees (galactic)
        """
        region_galactic = []
        
        for ra, dec in region_icrs:
            # Create SkyCoord in ICRS frame
            coord_icrs = SkyCoord(
                ra=ra * u.degree,
                dec=dec * u.degree,
                frame='icrs'
            )
            
            # Convert to galactic
            coord_gal = coord_icrs.galactic
            
            region_galactic.append((coord_gal.l.deg, coord_gal.b.deg))
        
        if logger:
            logger.debug(f"Converted {len(region_icrs)} ICRS to galactic coordinates")
        
        return region_galactic
    
    @staticmethod
    def coords_to_healpix_vectors(
        region_coords: List[Tuple[float, float]],
        coordinate_system: str = 'galactic',
        logger: Optional[object] = None
    ) -> List[np.ndarray]:
        """Convert coordinates to HEALPix unit vectors
        
        Parameters:
        -----------
        region_coords : list of tuple
            List of (lon, lat) coordinates in degrees
        coordinate_system : str, optional
            'galactic' or 'icrs' (default: 'galactic')
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        list of np.ndarray
            List of 3D unit vectors (HEALPix convention)
        
        Examples:
        ---------
        >>> coords = [(265.0, -28.5), (270.0, -25.0)]
        >>> vectors = ROITools.coords_to_healpix_vectors(coords, 'galactic')
        >>> len(vectors)
        2
        """
        region_vectors = []
        deg_to_rad = np.pi / 180.0
        
        for lon, lat in region_coords:
            # Convert to radians
            lon_rad = lon * deg_to_rad
            lat_rad = lat * deg_to_rad
            
            # HEALPix uses co-latitude (theta = 90 - lat)
            theta = np.pi / 2 - lat_rad
            
            # Convert to HEALPix vector
            vec = hp.ang2vec(theta, lon_rad, lonlat=False)
            region_vectors.append(vec)
        
        if logger:
            logger.debug(f"Converted {len(region_coords)} coordinates to HEALPix vectors")
        
        return region_vectors
    
    @staticmethod
    def healpix_vectors_to_pixels(
        vectors: List[np.ndarray],
        nside: int,
        logger: Optional[object] = None
    ) -> List[int]:
        """Convert HEALPix vectors to pixel indices
        
        Parameters:
        -----------
        vectors : list of np.ndarray
            List of 3D unit vectors
        nside : int
            HEALPix NSIDE parameter
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        list of int
            List of HEALPix pixel indices
        """
        pixels = []
        
        for vec in vectors:
            pix = hp.vec2pix(nside, vec[0], vec[1], vec[2])
            pixels.append(pix)
        
        if logger:
            logger.debug(f"Converted {len(vectors)} vectors to HEALPix pixels (NSIDE={nside})")
        
        return pixels
    
    @staticmethod
    def pixels_to_healpix_map(
        pixels: List[int],
        nside: int,
        logger: Optional[object] = None
    ) -> np.ndarray:
        """Create HEALPix map with specified pixels marked
        
        Parameters:
        -----------
        pixels : list of int
            List of HEALPix pixel indices
        nside : int
            HEALPix NSIDE parameter
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        np.ndarray
            HEALPix map array with specified pixels set to 1
        """
        npix = hp.nside2npix(nside)
        healpix_map = np.zeros(npix)
        healpix_map[pixels] = 1
        
        if logger:
            logger.debug(f"Created HEALPix map with {len(pixels)} pixels marked (NSIDE={nside})")
        
        return healpix_map
    
    @staticmethod
    def create_roi_template(
        region_coords: List[Tuple[float, float]],
        output_file: str,
        nside: int = 256,
        coordinate_system: str = 'galactic',
        logger: Optional[object] = None
    ) -> Optional[str]:
        """Create ROI template FITS file
        
        Parameters:
        -----------
        region_coords : list of tuple
            List of (lon, lat) coordinates defining ROI polygon
        output_file : str
            Output FITS filename
        nside : int, optional
            HEALPix NSIDE parameter (default: 256)
        coordinate_system : str, optional
            'galactic' or 'icrs' (default: 'galactic')
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        str or None
            Path to created FITS file, or None if failed
        
        Examples:
        ---------
        >>> roi_coords = [(265.0, -28.5), (270.0, -25.0), (270.0, -28.5), (265.0, -25.0)]
        >>> roi_file = ROITools.create_roi_template(
        ...     roi_coords,
        ...     'roi_template.fits',
        ...     nside=512
        ... )
        """
        try:
            if logger:
                logger.info(f"Creating ROI template: {output_file}")
                logger.info(f"Coordinates: {len(region_coords)} points")
                logger.info(f"HEALPix NSIDE: {nside}")
            
            # Convert coordinates to vectors
            vectors = ROITools.coords_to_healpix_vectors(
                region_coords,
                coordinate_system=coordinate_system,
                logger=logger
            )
            
            # Convert vectors to pixels
            pixels = ROITools.healpix_vectors_to_pixels(vectors, nside, logger=logger)
            
            # Create HEALPix map
            healpix_map = ROITools.pixels_to_healpix_map(pixels, nside, logger=logger)
            
            # Write to FITS file
            hp.write_map(
                output_file,
                healpix_map,
                nest=False,
                coord='C' if coordinate_system == 'icrs' else 'G',
                partial=False,
                overwrite=True
            )
            
            if logger:
                logger.info(f"ROI template created: {output_file}")
            
            return output_file
        
        except Exception as e:
            if logger:
                logger.error(f"Failed to create ROI template: {e}")
            return None
    
    @staticmethod
    def validate_roi_coordinates(
        region_coords: List[Tuple[float, float]],
        logger: Optional[object] = None
    ) -> bool:
        """Validate ROI coordinates
        
        Parameters:
        -----------
        region_coords : list of tuple
            List of (lon, lat) coordinates
        logger : object, optional
            Logger instance
        
        Returns:
        --------
        bool
            True if coordinates are valid
        """
        if not region_coords:
            if logger:
                logger.error("No coordinates provided")
            return False
        
        if len(region_coords) < 3:
            if logger:
                logger.error("Need at least 3 coordinates to define ROI polygon")
            return False
        
        # Check coordinate ranges
        for lon, lat in region_coords:
            if not (-180 <= lon <= 360):
                if logger:
                    logger.error(f"Invalid longitude: {lon}")
                return False
            
            if not (-90 <= lat <= 90):
                if logger:
                    logger.error(f"Invalid latitude: {lat}")
                return False
        
        if logger:
            logger.info(f"Validated {len(region_coords)} ROI coordinates")
        
        return True
