from core.config import ConfigManager
from core.logger import PipelineLogger
from core.checkpoint import CheckpointManager
from core.directory_manager import DirectoryManager
from core.data_loading import DataLoader
from core.plotting import PlottingUtilities
from core.map_tools import MapGenerator
from core.roi_tools import ROITools
from core.model_generator import ModelGenerator
from core.hdf5_handler import HDF5Handler

config = ConfigManager('config.yaml')
method = config.get('fitting_procedure')
ra = config.get('coordinates.ra', 0.0)
output_path = config.get('output_dir', './')
output_dir_name = config.get('fit_name', 'fit_results')



# Test logger
logger = PipelineLogger('./logs')
logger.info("Pipeline test")

# Test checkpoint
checkpoint = CheckpointManager('./checkpoints')
print("✓ All systems ready!")

dm = DirectoryManager(output_path, output_dir_name, logger=logger)
dm.create_structure()

# Get paths
model_path = dm.get_model_file_path('Step0-Allpoint-sources')
results_dir = dm.get_step_results_dir('Step0-Allpoint-sources')
sig_map = dm.get_significance_map_path()

# List steps
steps = dm.list_steps()

# Main directories
datamap_dir = dm.get_datamap_dir()      # fit_name/DataMap/
logs_dir = dm.get_logs_dir()            # fit_name/Logs/
models_dir = dm.get_models_dir()        # fit_name/Models/
fit_results_dir = dm.get_fit_results_dir()  # fit_name/FitResults/

# Step-specific directories
step_dir = dm.get_step_results_dir('Step0-Allpoint-sources')
# Returns: fit_name/FitResults/Step0-Allpoint-sources/

model_path = dm.get_model_file_path('Step0-Allpoint-sources')
# Returns: fit_name/Models/Step0-Allpoint-sources.model

# Fit results YAML
results_path = dm.get_fit_results_file_path('Step0-Allpoint-sources')
# Returns: fit_name/FitResults/Step0-Allpoint-sources/fit_results.yaml

# HDF5 maps
model_hdf5 = dm.get_hdf5_file_path('Step0-Allpoint-sources', 'model')
residual_hdf5 = dm.get_hdf5_file_path('Step0-Allpoint-sources', 'residual')
# Returns: fit_name/FitResults/Step0-Allpoint-sources/model_fit.hd5
#          fit_name/FitResults/Step0-Allpoint-sources/residual_fit.hd5

# Significance map
sig_map = dm.get_significance_map_path()

step_name = 'Step1-Testsource1extension'
step_dir = dm.get_step_results_dir(step_name)

config = ConfigManager('config.yaml')

sigmap = config.get('coordinates.sig_map_path')
ra = config.get('coordinates.ra')
dec = config.get('coordinates.dec')
roi_x = config.get('coordinates.roi_x')
roi_y = config.get('coordinates.roi_y')
coord_sys = config.get('coordinates.coord_sys')
print(f"Loading HAWC data from {sigmap} with center RA={ra}, Dec={dec}, ROI size=({roi_x}, {roi_y}), coordinate system={coord_sys}")
array, footprint, wcs, xnum, ynum, pixel_size = DataLoader.load_hawc_data(
    filename=sigmap,
    center_x=ra,
    center_y=dec,
    x_length=roi_x,
    y_length=roi_y,
    coord_sys=coord_sys,
    logger=logger
)

# Find peak significance
peak_value, peak_index, sky_coord, significance = DataLoader.find_peak(
    array, wcs, logger=logger
)

# Extract ROI
roi_array, roi_wcs = DataLoader.extract_roi(
    array, wcs, roi_x_deg=2.0, roi_y_deg=2.0
)

fig, ax = PlottingUtilities.make_plots(
    array=array,
    wcs=wcs,
    coord_sys='C',
    threshold=4,
    vmin=-5,
    vmax=15,
    title='Significance Map',
    contour=True,
    save_path='sig_map.png',
    cmap='ult'
)

# Create log-scale plot
fig, ax = PlottingUtilities.make_logplots(
    array=array,
    wcs=wcs,
    coord_sys='C',
    title='Log Significance Map',
    save_path='sig_map_log.png'
)


countmap_dir = config.get('coordinates.count_map_dir')
image_bins = config.get('coordinates.image_bins')
detector_response = config.get('coordinates.detector_response', 'detRes.root')
print(f"Finding FITS files in {countmap_dir} for energy bins: {image_bins}")

fits_mapping = MapGenerator.find_fits_files_by_bins(
    data_directory=countmap_dir,
    energy_bins=image_bins,
    logger=logger
)

# # Create HEALPix map
output_map = MapGenerator.create_healpix_map(
    input_fits_files=list(fits_mapping.values()),
    energy_bins=list(fits_mapping.keys()),
    detector_response=detector_response,
    ra_center=83.5,
    dec_center=22.0,
    roi_x=10.0,
    roi_y=10.0,
    output_file='significance_map.fits',
    spectral_index=2.6,
    logger=logger
)
end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"EXXXEAAD time: {execution_time:.6f} seconds")
print(f"Created HEALPix significance map: {output_map}")

array, footprint, wcs, xnum, ynum, pixel_size = DataLoader.load_hawc_data(
    filename=output_map,
    center_x=ra,
    center_y=dec,
    x_length=roi_x,
    y_length=roi_y,
    coord_sys=coord_sys,
    logger=logger
)

fig, ax = PlottingUtilities.make_plots(
    array=array,
    wcs=wcs,
    coord_sys='C',
    threshold=4,
    vmin=-5,
    vmax=15,
    title='Significance Map',
    contour=True,
    save_path='sig_map-2.png',
    cmap='ult'
)