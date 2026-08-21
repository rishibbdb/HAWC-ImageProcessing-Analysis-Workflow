"""
Plotting Utilities

Extracted from: pipeline_helpers.py
Handles visualization of HAWC significance maps and detection results.
"""

from typing import Optional, Dict, Tuple, List
import logging

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.wcs.utils as astropy_utils
from astropy.wcs.utils import pixel_to_skycoord


class PlottingUtilities:
    """Plotting utilities for HAWC significance maps and detection results"""
    
    @staticmethod
    def set_axis_labels(ax: plt.Axes, coord_sys: str) -> None:
        """Set axis labels based on coordinate system
        
        Parameters:
        -----------
        ax : plt.Axes
            Matplotlib axis object
        coord_sys : str
            Coordinate system: 'C' (celestial) or 'G' (galactic)
        """
        if coord_sys == 'C':
            ax.set_xlabel(r"RA ($^\circ$)")
            ax.set_ylabel(r"Dec ($^\circ$)")
        elif coord_sys == 'G':
            ax.set_xlabel(r"$l$ ($^\circ$)")
            ax.set_ylabel(r"$b$ ($^\circ$)")
        else:
            raise ValueError(f"Invalid coordinate system: {coord_sys}")
    
    @staticmethod
    def create_circular_mask(
        height: int,
        width: int,
        center: Optional[Tuple[int, int]] = None,
        radius: Optional[float] = None
    ) -> np.ndarray:
        """Create circular mask for image
        
        Parameters:
        -----------
        height : int
            Image height in pixels
        width : int
            Image width in pixels
        center : tuple of int, optional
            (x, y) center position (default: image center)
        radius : float, optional
            Mask radius in pixels (default: min(height,width)/4)
        
        Returns:
        --------
        np.ndarray
            Boolean mask array
        """
        if center is None:
            center = (width // 2, height // 2)
        if radius is None:
            radius = min(height, width) / 4
        
        Y, X = np.ogrid[:height, :width]
        dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
        
        return dist_from_center <= radius
    
    @staticmethod
    def ultimet_colormap(
        vmin: float,
        vmax: float,
        threshold: float,
        color_map: str = 'turbo',
        n_colors: int = 256,
        blend_fraction: float = 0.05
    ) -> LinearSegmentedColormap:
        """Create HAWC-style colormap with special handling for threshold
        
        Parameters:
        -----------
        vmin : float
            Minimum value
        vmax : float
            Maximum value
        threshold : float
            Threshold value (e.g., 5 sigma)
        color_map : str, optional
            Base colormap name (default: 'turbo')
        n_colors : int, optional
            Number of colors (default: 256)
        blend_fraction : float, optional
            Blend fraction for transition (default: 0.05)
        
        Returns:
        --------
        LinearSegmentedColormap
            Colormap object
        """
        # Get base colormap
        base_cmap = plt.cm.get_cmap(color_map, n_colors)
        
        # Normalize threshold to [0, 1]
        threshold_norm = (threshold - vmin) / (vmax - vmin)
        threshold_norm = np.clip(threshold_norm, 0, 1)
        
        # Create color list with emphasis at threshold
        colors = []
        for i in range(n_colors):
            norm_pos = i / n_colors
            
            # Blend smoothly around threshold
            if abs(norm_pos - threshold_norm) < blend_fraction:
                # Smooth transition
                blend_factor = 1 - abs(norm_pos - threshold_norm) / blend_fraction
                base_color = base_cmap(norm_pos)
                # Brighten colors near threshold
                color = tuple(min(1.0, c + 0.3 * blend_factor) for c in base_color[:3]) + (1.0,)
            else:
                color = base_cmap(norm_pos)
            
            colors.append(color)
        
        return LinearSegmentedColormap.from_list('ultimet', colors)
    
    @staticmethod
    def make_plots(
        array: np.ndarray,
        wcs: WCS,
        coord_sys: str,
        npix: int = 1,
        threshold: float = 4,
        vmin: float = -5,
        vmax: float = 15,
        blobs: Optional[Dict] = None,
        contour: bool = False,
        title: Optional[str] = None,
        hotspots: Optional[Dict] = None,
        save_path: Optional[str] = None,
        cmap: str = 'inferno',
        figsize: Tuple[float, float] = (10, 6),
        logger: Optional[object] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot significance map with optional overlays
        
        Parameters:
        -----------
        array : np.ndarray
            2D significance map
        wcs : WCS
            WCS projection object
        coord_sys : str
            Coordinate system: 'C' (celestial) or 'G' (galactic)
        npix : int, optional
            HEALPix NSIDE parameter (default: 1)
        threshold : float, optional
            Significance threshold for colormap (default: 4)
        vmin : float, optional
            Minimum value for colormap (default: -5)
        vmax : float, optional
            Maximum value for colormap (default: 15)
        blobs : dict, optional
            Blob data with 'psblobs' and/or 'extblobs' keys
        contour : bool, optional
            Whether to draw contours (default: False)
        title : str, optional
            Plot title
        hotspots : dict, optional
            Hotspot data with 'Name', 'ra', 'dec', 'ext' keys
        save_path : str, optional
            Path to save figure
        cmap : str, optional
            Colormap name (default: 'inferno')
        figsize : tuple, optional
            Figure size (default: (10, 6))
        logger : object, optional
            Logger instance
        **kwargs
            Additional options (e.g., 'labels' for catalog overlays)
        
        Returns:
        --------
        tuple
            (fig, ax) matplotlib figure and axis objects
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=wcs)
        
        # Create colormap
        if cmap == 'ult':
            colormap = PlottingUtilities.ultimet_colormap(vmin, vmax, threshold)
        else:
            colormap = plt.cm.get_cmap(cmap)
        
        # Plot map
        im = ax.imshow(array, cmap=colormap, vmin=vmin, vmax=vmax, origin='lower')
        plt.colorbar(im, ax=ax, label=r'Significance ($\sigma$)', fraction=0.046, pad=0.04)
        
        # Add contours
        if contour:
            max_val = np.max(array)
            if max_val > 15:
                levels = [7, 9, 12, 13, 14, 15]
            elif max_val > 12:
                levels = [5, 7, 9, 11]
            elif max_val > 5:
                levels = [5, 6, 7]
            else:
                levels = None
            
            if levels:
                hi_transform = ax.get_transform(wcs)
                ax.contour(array, levels=levels, transform=hi_transform, colors='black', linewidths=0.5)
                if logger:
                    logger.debug(f"Plotted contours at levels: {levels}")
        
        # Add hotspots
        if hotspots is not None:
            PlottingUtilities._plot_hotspots(ax, wcs, hotspots)
        
        # Add blobs
        if blobs:
            PlottingUtilities._plot_blobs(ax, wcs, blobs)
        
        # Set title and labels
        if title:
            ax.set_title(title)
        
        PlottingUtilities.set_axis_labels(ax, coord_sys)
        
        # Set limits
        xnum, ynum = array.shape[1], array.shape[0]
        ax.set_xlim(0, xnum)
        ax.set_ylim(0, ynum)
        ax.coords[0].set_format_unit('deg')
        ax.coords[1].set_format_unit('deg')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            if logger:
                logger.info(f"Saved figure to {save_path}")
        
        return fig, ax
    
    @staticmethod
    def _plot_hotspots(
        ax: plt.Axes,
        wcs: WCS,
        hotspots: Dict
    ) -> None:
        """Plot hotspot sources with error circles
        
        Parameters:
        -----------
        ax : plt.Axes
            Matplotlib axis
        wcs : WCS
            WCS projection
        hotspots : dict
            Dictionary with 'Name', 'ra', 'dec', 'ext' keys
        """
        names = hotspots.get('Name', [])
        ras = hotspots.get('ra', [])
        decs = hotspots.get('dec', [])
        exts = hotspots.get('ext', [])
        
        for name, ra, dec, ext in zip(names, ras, decs, exts):
            # Plot point source position
            sky_pos = SkyCoord(ra=ra*u.degree, dec=dec*u.degree)
            px, py = astropy_utils.skycoord_to_pixel(sky_pos, wcs)
            
            ax.plot(px, py, 'r+', markersize=10, markeredgewidth=2, label=name)
            
            # Plot extension circle if applicable
            if ext > 0:
                ext_pix = ext / np.abs(wcs.wcs.cdelt[0])  # Convert deg to pixels
                circle = patches.Circle((px, py), ext_pix, fill=False, edgecolor='r', linewidth=1)
                ax.add_patch(circle)
    
    @staticmethod
    def _plot_blobs(
        ax: plt.Axes,
        wcs: WCS,
        blobs: Dict
    ) -> None:
        """Plot blob detection results
        
        Parameters:
        -----------
        ax : plt.Axes
            Matplotlib axis
        wcs : WCS
            WCS projection
        blobs : dict
            Dictionary with 'psblobs' and/or 'extblobs' keys
        """
        if 'psblobs' in blobs:
            ps_blobs = blobs['psblobs']
            # Plot point source blobs
            for blob in ps_blobs:
                # blob should have (y, x, sigma) structure
                pass
        
        if 'extblobs' in blobs:
            ext_blobs = blobs['extblobs']
            # Plot extended blobs
            for blob in ext_blobs:
                pass
    
    @staticmethod
    def make_logplots(
        array: np.ndarray,
        wcs: WCS,
        coord_sys: str,
        npix: int = 1,
        threshold: Optional[float] = None,
        vmin: float = 0.01,
        vmax: Optional[float] = None,
        blobs: Optional[Dict] = None,
        title: Optional[str] = None,
        hotspots: Optional[Dict] = None,
        save_path: Optional[str] = None,
        cmap: str = 'inferno',
        figsize: Tuple[float, float] = (10, 6),
        logger: Optional[object] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot significance map with logarithmic scale
        
        Parameters:
        -----------
        array : np.ndarray
            2D significance map
        wcs : WCS
            WCS projection object
        coord_sys : str
            Coordinate system: 'C' (celestial) or 'G' (galactic)
        npix : int, optional
            HEALPix NSIDE parameter (default: 1)
        threshold : float, optional
            Threshold for visualization
        vmin : float, optional
            Minimum value (default: 0.01)
        vmax : float, optional
            Maximum value (default: max(array))
        blobs : dict, optional
            Blob data
        title : str, optional
            Plot title
        hotspots : dict, optional
            Hotspot data
        save_path : str, optional
            Path to save figure
        cmap : str, optional
            Colormap name (default: 'inferno')
        figsize : tuple, optional
            Figure size (default: (10, 6))
        logger : object, optional
            Logger instance
        **kwargs
            Additional options
        
        Returns:
        --------
        tuple
            (fig, ax) matplotlib figure and axis objects
        """
        if vmax is None:
            vmax = np.max(array)
        
        if threshold is None:
            threshold = np.min(array) + (np.max(array) - np.min(array)) / 8
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=wcs)
        
        # Use logarithmic normalization
        im = ax.imshow(
            array,
            cmap=cmap,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            origin='lower'
        )
        plt.colorbar(im, ax=ax, label=r'log(Significance ($\sigma$))', fraction=0.046, pad=0.04)
        
        # Add hotspots
        if hotspots is not None:
            PlottingUtilities._plot_hotspots(ax, wcs, hotspots)
        
        # Add blobs
        if blobs:
            PlottingUtilities._plot_blobs(ax, wcs, blobs)
        
        # Set title and labels
        if title:
            ax.set_title(title)
        
        PlottingUtilities.set_axis_labels(ax, coord_sys)
        
        # Set limits
        xnum, ynum = array.shape[1], array.shape[0]
        ax.set_xlim(0, xnum)
        ax.set_ylim(0, ynum)
        ax.coords[0].set_format_unit('deg')
        ax.coords[1].set_format_unit('deg')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            if logger:
                logger.info(f"Saved log plot to {save_path}")
        
        return fig, ax
