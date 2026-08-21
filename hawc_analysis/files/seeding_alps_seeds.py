"""
ALPS Seeder (Automated Likelihood Pipeline Search) — Part 1

Extracted from: testalps.py (AutomatedLikelihoodPipelineSearch class)

This module contains the VERBATIM-extracted portion of ALPS: config reading,
model-file I/O (yml / fits / .model parsing and writing), the source database
builders, and the freeze/free parameter helpers. All method bodies are copied
unchanged from testalps.py; the only additions are type hints and docstrings.

NOT YET INCLUDED (added in Part 2 — the fit-integration layer):
  - run_single_fit / run_multi_fit / _retry_multi_fit  (subprocess -> in-process fitmodel)
  - _run_command_with_live_log / _run_multiple_commands (subprocess runners)
  - _run_point_source_adding_phase / _after_run_accouting (iterative loop)
  - run_alt_hypothesis                                    (extension/spectrum tests)
  - make_map / _find_next_hotpsot / _convert_hd5_to_fits / _get_residual_fixed_model
  - the SeedingModule.run() wrapper

EXCLUDED entirely (per project rules): _get_sloppy_TS
"""

import ast
import glob
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

import threeML
import astromodels

# ----------------------------------------------------------------------------
# Module-level source/spectrum databases and markers — extracted verbatim
# ----------------------------------------------------------------------------

begin_source_marker = '#----------BEGIN_SOURCE----------#\n'
end_source_marker   = '#-----------END_SOURCE-----------#\n'

source_types_db = pd.DataFrame(columns=['model', 'params_of_interest', 'default_param_values'])

source_types_db['model'] = ['Latitude_galactic_diffuse', 'Gaussian_on_sphere', 'Asymm_Gaussian_on_sphere', 'Disk_on_sphere', 'Ellipse_on_sphere', 'SpatialTemplate_2D', 'Power_law_on_sphere', 'Continuous_injection_diffusion_ellipse', 'Continuous_injection_diffusion', 'Continuous_injection_diffusion_legacy', 'GalPropTemplate_3D', 'Hermes', 'PointSource']

source_types_db['params_of_interest'] = ['K, sigma_b, l_min, l_max', 'lon0, lat0, sigma', 'lon0, lat0, a, e, theta', 'lon0, lat0, radius', 'lon0, lat0, a, e, theta', 'K, hash, ihdu, fits_file, frame', 'lon0, lat0, index', 'lon0, lat0, rdiff0, delta, b, piv, piv2, incl, elongation', 'lon0, lat0, rdiff0, rinj, delta, b, piv, piv2', 'lon0, lat0, rdiff0, delta, uratio, piv, piv2', ' K, hash', 'N, hash, ihdu, fits_file, frame', 'ra, dec']

source_types_db['default_param_values'] = [{"K": [1, None, None, False], "sigma_b": [1, None, None, False], "l_min": [0, None, None, False], "l_max": [0, None, None, False]},
                                           {"lon0": [0.0, 0.0, 360.0, True], "lat0": [0.0, -90.0, 90.0, True], "sigma": [1, 0, 5, True]},
                                           {"lon0": [0.0, 0.0, 360.0, True], "lat0": [0.0, -90.0, 90.0, True], "a": [1, 0, 5, True], "e": [0.9, 0, 1, True], "theta": [0.0, -90.0, 90.0, True]},
                                           {"lon0": [0.0, 0.0, 360.0, True], "lat0": [0.0, -90.0, 90.0, True], "radius": [1, 0, 5, True]},
                                           {"lon0": [0.0, 0.0, 360.0, True], "lat0": [0.0, -90.0, 90.0, True], "a": [1, 0, 5, True], "e": [0.9, 0, 1, True], "theta": [0.0, -90.0, 90.0, True]},
                                           {"K": [1, 0.01, 100, True], "hash": [None, None, None, False], "ihdu": [0, None, None, False], "properties": [None, None, None, False], "fits_file": [None, None, None, False], "frame": ["icrs", None, None, False]},
                                           {"lon0": [0.0, 0.0, 360.0, True], "lat0": [0.0, -90.0, 90.0, True], "index": [-2.0, -5.0, -1.0, True], "maxr": [20.0, None, None, False], "minr": [0.05, None, None, False]},
                                           {"lon0": [0.0, 0.0, 360.0, True], "lat0": [0.0, -90.0, 90.0, True], "rdiff0": [1.0, 0, 20, True], "delta": [0.33, 0.3, 0.6, False], "b": [3, 1, 10.0, False], "piv": [2e10, 0, None, False], "piv2": [1e9, 0, None, False], "incl": [0.0, -90.0, 90.0, True], "elongation": [1.0, 0.1, 10.0, True]},
                                           {"lon0": [0.0, 0.0, 360.0, True], "lat0": [0.0, -90.0, 90.0, True], "rdiff0": [1.0, 0, 20, True], "rinj": [100.0, 0, 200, False], "delta": [0.33, 0.3, 0.6, False], "b": [3, 1, 10.0, False], "piv": [2e10, 0, None, False], "piv2": [1e9, 0, None, False]},
                                           {"lon0": [0.0, 0.0, 360.0, True], "lat0": [0.0, -90.0, 90.0, True], "rdiff0": [1.0, 0, 20, True], "delta": [0.33, 0.3, 0.6, False], "uratio": [0.5, 0.01, 100.0, False], "piv": [2e10, 0, None, False], "piv2": [1e9, 0, None, False]},
                                           {"K": [1, None, None, False], "hash": [1, None, None, False]},
                                           {"N": [1, 0.01, 100, True], "hash": [None, None, None, False], "ihdu": [0, None, None, False], "properties": [None, None, None, False], "fits_file": [None, None, None, False], "frame": ["icrs", None, None, False]},
                                           {"ra": [0.0, 0.0, 360.0, True], "dec": [0.0, -90.0, 90.0, True]}]


spectrum_types_db = pd.DataFrame(columns=['model', 'params_of_interest', 'default_param_values'])

spectrum_types_db['model'] = ['Broken_powerlaw', 'Cutoff_powerlaw', 'Exponential_cutoff', 'Log_parabola', 'Powerlaw', 'Cutoff_powerlaw_Ep']

spectrum_types_db['params_of_interest'] = ['K xb alpha beta piv', 'K piv index xc', 'K xc', 'K piv alpha beta', 'K piv index', 'K piv index xp']

spectrum_types_db['default_param_values'] = [{'K': [1e-23, 1e-29, 1e-19, True], 'xb': [10, 1, 1000, False], 'alpha': [-2, -4, -1, True], 'beta': [-3, -4, -1, True], 'piv': [2e9, None, None, False]},
                                             {'K': [1e-23, 1e-29, 1e-19, True], 'xc': [10, 1, 1000, False], 'index': [-2.5, -4, -1, True], 'piv': [2e9, None, None, False]},
                                             {'K': [1e-23, 1e-29, 1e-19, True], 'xc': [10, 1, 1000, False]},
                                             {'K': [1e-23, 1e-29, 1e-19, True], 'alpha': [-2, -4, -1, True], 'beta': [0, -1, 1, True], 'piv': [2e9, None, None, False]},
                                             {'K': [1e-23, 1e-29, 1e-19, True], 'index': [-2.5, -4, -1, True], 'piv': [2e9, None, None, False]},
                                             {'K': [1e-23, 1e-29, 1e-19, True], 'xp': [500, 10, 10000, False], 'index': [-2.5, -4, -1, True], 'piv': [2e9, None, None, False]}]


class ALPSSeederBase:
    """Verbatim-extracted ALPS methods: config, model I/O, source-db builders.

    This is the base layer for the full ALPSSeeder (assembled in Part 2). It
    holds every method that is a pure/self-contained extraction from
    AutomatedLikelihoodPipelineSearch, unchanged except for type hints and
    docstrings. The __init__ reads ALPS's own config schema via
    _load_config_value (space-delimited nested keys), exactly as the original.

    Part 2 subclasses this to add the fit-integration layer (in-process
    fitmodel calls, the iterative point-source loop, and alternate-hypothesis
    testing) plus the SeedingModule.run() wrapper.
    """

    def __init__(self, initial_config: str):
        """Initialize ALPS from its own YAML config (verbatim from testalps.py).

        Parameters:
        -----------
        initial_config : str
            Path to the ALPS config YAML (space-delimited nested key schema).
        """
        self.config_path = initial_config
        with open(self.config_path) as config_file:
            self.config_yml_string = yaml.safe_load(config_file)

        self.source_info_db = pd.DataFrame(columns=['source', 'morphology_type', 'spectrum_type', 'morphology_params', 'spectrum_params'])
        '''Required Params'''
        self.parent_dir = self._load_config_value(self.config_yml_string, 'required_arguments parent_folder', None)
        self.coord_1    = self._load_config_value(self.config_yml_string, 'required_arguments coord_1', None)
        self.coord_2    = self._load_config_value(self.config_yml_string, 'required_arguments coord_2', None)
        self.coord_sys  = self._load_config_value(self.config_yml_string, 'required_arguments coordinate_system', None)
        self.map_tree   = self._load_config_value(self.config_yml_string, 'required_arguments map_tree_file_path', None)
        self.det_res    = self._load_config_value(self.config_yml_string, 'required_arguments detector_response_file_path', None)
        self.bin_list   = self._load_config_value(self.config_yml_string, 'required_arguments bin_list', None)
        self.estimator  = self._load_config_value(self.config_yml_string, 'required_arguments estimator', None)
        self.fit_name   = self._load_config_value(self.config_yml_string, 'required_arguments fit_name', None)
        self.pixi_path  = self._load_config_value(self.config_yml_string, 'required_arguments pixi_aerie_folder', None)

        self.fit_model_path = '/lustre/hawcz01/scratch/userspace/sgroetsch/fitmodel-rework/fitModel.py'
        self.draw_maps_path = '/lustre/hawcz01/scratch/userspace/sgroetsch/fitmodel-rework/Drawmaps.py'

        log_local_path    = self._load_config_value(self.config_yml_string, 'optional_path_arguments log_local_path', 'Logs')
        self.log_abs_path = os.path.join(self.parent_dir, log_local_path)

        self.logger = ALPSLogger(str(self.log_abs_path), self._load_config_value(self.config_yml_string, 'pipeline_control_arguments logging_level', 'INFO'))

        required_arg_list = [self.parent_dir, self.coord_1, self.coord_2, self.coord_sys, self.map_tree, self.det_res, self.bin_list, self.estimator, self.fit_name, self.pixi_path]
        required_arg_name_list = ['parent_folder', 'coord_1', 'coord_2', 'coordinate_system', 'map_tree_file_path', 'detector_response_file_path', 'bin_list', 'estimator', 'fit_name', 'pixi_aerie_folder']
        none_arg_name = [name for arg, name in zip(required_arg_list, required_arg_name_list) if arg is None]
        if(len(none_arg_name) > 0):
            for arg in none_arg_name:
                self.logger.error(f'Arguement {arg} was not correctly specified in the config file {initial_config}. Please double check if all required args are present and correctly defined')

        fit_results_local_path = self._load_config_value(self.config_yml_string, 'optional_path_arguments fit_results_local_path', 'FitResults')
        self.fit_results_abs_path = os.path.join(self.parent_dir, fit_results_local_path)

        raw_sig_map_local_path = self._load_config_value(self.config_yml_string, 'optional_path_arguments raw_sig_map_local_path', 'DataMap')
        self.raw_sig_map_abs_path = os.path.join(self.parent_dir, raw_sig_map_local_path)

        prev_models_local_path = self._load_config_value(self.config_yml_string, 'optional_path_arguments prev_models_local_path', 'Models')
        self.prev_models_abs_path = os.path.join(self.parent_dir, prev_models_local_path)

        if(not os.path.isdir(self.parent_dir)):
            os.mkdir(self.parent_dir)
        if(not os.path.isdir(self.log_abs_path)):
            os.mkdir(self.log_abs_path)
        if(not os.path.isdir(self.fit_results_abs_path)):
            os.mkdir(self.fit_results_abs_path)
        if(not os.path.isdir(self.raw_sig_map_abs_path)):
            os.mkdir(self.raw_sig_map_abs_path)
        if(not os.path.isdir(self.prev_models_abs_path)):
            os.mkdir(self.prev_models_abs_path)

        self.roi_radius   = float(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments roi_radius', '5'))
        self.minimum_point_source_TS   = float(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments minimum_point_source_TS', '25'))
        self.minimum_morphology_TS_improvement   = float(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments minimum_morphology_TS_improvement', '16'))
        self.minimum_spectral_TS_improvement   = float(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments minimum_spectral_TS_improvement', '16'))
        self.point_source_coord_range   = float(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments point_source_coord_range', '1'))
        self.extended_source_coord_range   = float(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments extended_source_coord_range', '1'))
        self.custom_parameter_boundaries_dict   = self._load_config_value(self.config_yml_string, 'optional_fitting_arguments custom_parameter_boundaries_dict', None)
        self.include_old_diffuse_model   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments include_old_diffuse_model', 'False'))
        self.old_diffuse_model_min_l   = self._load_config_value(self.config_yml_string, 'optional_fitting_arguments old_diffuse_model_min_l', None)
        self.old_diffuse_model_max_l   = self._load_config_value(self.config_yml_string, 'optional_fitting_arguments old_diffuse_model_max_l', None)
        self.old_diffuse_model_sigma_b   = float(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments old_diffuse_model_sigma_b', '1'))
        self.include_hermes_diffuse_model   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments include_hermes_diffuse_model', 'False'))
        self.hermes_template_path   = self._load_config_value(self.config_yml_string, 'optional_fitting_arguments hermes_template_path', None)
        self.free_diffuse   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments free_diffuse_norm', 'False'))
        self.roi_template_path   = self._load_config_value(self.config_yml_string, 'optional_fitting_arguments roi_template_path', None)
        self.compute_optional_TS   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments compute_optional_TS', 'True'))
        self.compute_optional_uncertainties   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments compute_optional_uncertainties', 'True'))
        self.use_fast_TS_estimate   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments use_fast_TS_estimate', 'False'))
        self.max_param_retries   = int(self._load_config_value(self.config_yml_string, 'optional_fitting_arguments max_param_retries', '10'))

        if(self.include_hermes_diffuse_model):
            if(self.hermes_template_path is None):
                self.logger.error(f'Argument hermes_template_path was not correctly specified in the config file {initial_config} and include_hermes_diffuse_model is set to True. Please double check if all required args are present and correctly defined')

        if(self.include_old_diffuse_model):
            required_arg_list = [self.old_diffuse_model_min_l, self.old_diffuse_model_max_l]
            required_arg_name_list = ['old_diffuse_model_min_l', 'old_diffuse_model_max_l']
            none_arg_name = [name for arg, name in zip(required_arg_list, required_arg_name_list) if arg is None]
            if(len(none_arg_name) > 0):
                for arg in none_arg_name:
                    self.logger.error(f'Argument {arg} was not correctly specified in the config file {initial_config} and include_old_diffuse_model is set to True. Please double check if all required args are present and correctly defined')

        self.start_from_raw_map   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments start_from_raw_map', 'True'))
        self.start_from_existing_yml_file_model   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments start_from_existing_yml_file_model', 'False'))
        self.yml_file_model_path   = self._load_config_value(self.config_yml_string, 'pipeline_control_arguments yml_file_model_path', None)
        self.start_from_existing_model_file_model   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments start_from_existing_model_file_model', 'False'))
        self.model_file_model_path   = self._load_config_value(self.config_yml_string, 'pipeline_control_arguments model_file_model_path', None)
        self.start_from_existing_fits_file_model   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments start_from_existing_fits_file_model', 'False'))
        self.fits_file_model_path   = self._load_config_value(self.config_yml_string, 'pipeline_control_arguments fits_file_model_path', None)
        self.trusted_source_list   = self._load_config_value(self.config_yml_string, 'pipeline_control_arguments trusted_source_list', None)
        self.trust_all_alterante_morphologies   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments trust_all_alterante_morphologies', 'False'))
        self.trust_all_alternate_spectra   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments trust_all_alternate_spectra', 'False'))
        self.perform_point_source_adding_phase   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments perform_point_source_adding_phase', 'True'))
        self.perform_morpholgy_testing_phase   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments perform_morpholgy_testing_phase', 'True'))
        self.alternate_morpholgy_model_list   = self._load_config_value(self.config_yml_string, 'pipeline_control_arguments alternate_morpholgy_model_list', None)
        self.perform_spectrum_testing_phase   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments perform_spectrum_testing_phase', 'True'))
        self.alternate_spectrum_model_list   = self._load_config_value(self.config_yml_string, 'pipeline_control_arguments alternate_spectrum_model_list', None)
        self.perform_final_fitting_phase   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'pipeline_control_arguments perform_final_fitting_phase', 'True'))

        if(self.perform_spectrum_testing_phase):
            if(self.alternate_spectrum_model_list is None):
                self.logger.error(f'Argument alternate_spectrum_model_list was not correctly specified in the config file {initial_config} and perform_spectrum_testing_phase is set to True. Please provide a correctly formatted list of alternate spectrum models to test if you wish to perform this testing phase')
        if(self.perform_morpholgy_testing_phase):
            if(self.alternate_morpholgy_model_list is None):
                self.logger.error(f'Argument alternate_morpholgy_model_list was not correctly specified in the config file {initial_config} and perform_morpholgy_testing_phase is set to True. Please provide a correctly formatted list of alternate morpholgy models to test if you wish to perform this testing phase')

        if(self.start_from_existing_yml_file_model):
            if(self.yml_file_model_path is None):
                self.logger.error(f'Argument yml_file_model_path was not correctly specified in the config file {initial_config} and start_from_existing_yml_file_model is set to True. Please provide a correctly formatted path to use this argument')

        if(self.start_from_existing_model_file_model):
            if(self.model_file_model_path is None):
                self.logger.error(f'Argument model_file_model_path was not correctly specified in the config file {initial_config} and start_from_existing_model_file_model is set to True. Please provide a correctly formatted path to use this argument')

        if(self.start_from_existing_fits_file_model):
            if(self.fits_file_model_path is None):
                self.logger.error(f'Argument fits_file_model_path was not correctly specified in the config file {initial_config} and start_from_existing_fits_file_model is set to True. Please provide a correctly formatted path to use this argument')

        if(self.start_from_raw_map ^ self.start_from_existing_yml_file_model ^ self.start_from_existing_model_file_model ^ self.start_from_existing_fits_file_model):
            pass
        else:
            self.logger.error(f'Either no starting point or multiple starting points has been specified in {initial_config}. Please double check and select exactly one starting point.')

        # map_making_control_arguments make_raw_data_map
        self.make_raw_data_map   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'map_making_control_arguments make_raw_data_map', 'True'))
        # map_making_control_arguments make_model_map
        self.make_model_map   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'map_making_control_arguments make_model_map', 'True'))
        # map_making_control_arguments make_optional_residual_map
        self.make_optional_residual_map   = ast.literal_eval(self._load_config_value(self.config_yml_string, 'map_making_control_arguments make_optional_residual_map', 'True'))
        # map_making_control_arguments plot_size_coord_1
        self.plot_size_coord_1   = float(self._load_config_value(self.config_yml_string, 'map_making_control_arguments plot_size_coord_1', str(10)))
        # map_making_control_arguments plot_size_coord_2
        self.plot_size_coord_2   = float(self._load_config_value(self.config_yml_string, 'map_making_control_arguments plot_size_coord_2', str(10)))

    # ------------------------------------------------------------------------
    # Config reading
    # ------------------------------------------------------------------------

    def _load_config_value(self, config_yml_string: dict, key: str, default_val: str):
        """Read a space-delimited nested key from the ALPS config dict.

        Parameters:
        -----------
        config_yml_string : dict
            Parsed YAML config.
        key : str
            Space-delimited nested key path, e.g. 'required_arguments coord_1'.
        default_val : str
            Value returned if the key path resolves to None.
        """
        keys = key.split(' ')
        target_value = config_yml_string
        for key in keys:
            if(type(target_value) == dict):
                target_value = target_value.get(key)
        if(target_value == None):
            target_value = default_val
        return target_value

    # ------------------------------------------------------------------------
    # Model file loading (yml / fits / .model -> source_info_db)
    # ------------------------------------------------------------------------

    def _load_model_from_valid_file(self, model_path: str) -> pd.DataFrame:
        """Dispatch model loading by file extension (.yml / .fits / .model)."""
        model_path = os.path.abspath(model_path)
        _, model_type = os.path.splitext(model_path)
        if(model_type == '.yml'):
            source_db = self._load_from_yml_file(model_path)
        elif(model_type == '.fits'):
            source_db = self._load_from_fits_file(model_path)
        elif(model_type == '.model'):
            source_db = self._load_from_model_file(model_path)
        else:
            self.logger.error(f'Could not load from provided model path {model_path}')
        return source_db

    def _load_from_yml_file(self, model_yml_loc: str) -> pd.DataFrame:
        '''Convert model in yml format to pandas dataframe'''
        with open(model_yml_loc, 'r') as yaml_file:
            model_yml_string = yaml.safe_load(yaml_file)
            return self._parse_yml_string(model_yml_string)

    def _load_from_fits_file(self, fits_file_loc: str) -> pd.DataFrame:
        '''convert model from .fits format to pandas dataframe'''
        fits_file = fits.open(fits_file_loc)
        model_yml_string = yaml.safe_load(fits_file[1].header['MODEL'].replace('_NEWLINE_', '\n'))
        return self._parse_yml_string(model_yml_string)

    def _parse_yml_string(self, model_yml_string: str) -> pd.DataFrame:
        '''Read yml string and convert to dataframe'''
        source_list = list(model_yml_string.keys())

        source_info_db = pd.DataFrame(columns=['source', 'morphology_type', 'spectrum_type', 'morphology_params', 'spectrum_params'])

        #Iterate through all sources found in the yml
        for source in source_list:
            params_to_get = None

            source_type = list(model_yml_string[source].keys())[0]
            #Handle the fact that point sources don't have the model name in the yml file
            if(source_type == 'position'):
                source_type = 'PointSource'

            #attempt to get the list of relevant morphology parameters for the given source morphology. If none found print a fail
            try:
                params_to_get = source_types_db.loc[source_types_db['model'] == source_type, 'params_of_interest'].values[0]
            except:
                if('value' == source_type):
                    continue
                print(f'Could not find matching params for \"{source_type}\" this may be a bug or the model you are using is unsupported')

            #In the case that the relevant params were found begin to extract them from the yml
            if(params_to_get != None):
                #Grab the dict of parameters for the morpholgy
                model_dict = model_yml_string[source][list(model_yml_string[source].keys())[0]]

                #Initialize dicts for later updates
                morph_dict = {}
                spectrum_dict = {}

                #For each param grab the value, lower bound, upper bound, and freedom status
                for key in model_dict.keys():
                    if(key in params_to_get):
                        try:
                            morph_dict.update({key: [model_dict[key]['value'], model_dict[key]['min_value'], model_dict[key]['max_value'], model_dict[key]['free']]})
                        except:
                            morph_dict.update({key: [model_dict[key]['value'], None, None, None]})

                #Repeat above but for the spectrum params
                spectrum_type = list(model_yml_string[source]['spectrum']['main'].keys())[0]
                try:
                    params_to_get = spectrum_types_db.loc[spectrum_types_db['model'] == spectrum_type, 'params_of_interest'].values[0]
                except:
                    print(f'Could not find matching params for \"{spectrum_type}\" this may be a bug or the model you are using is unsupported')
                spectrum_dict = model_yml_string[source]['spectrum']['main'][spectrum_type]
                for key in spectrum_dict.keys():
                    if(key in params_to_get):
                        spectrum_dict.update({key: [spectrum_dict[key]['value'], spectrum_dict[key]['min_value'], spectrum_dict[key]['max_value'], spectrum_dict[key]['free']]})
            #Add entry to pandas dataframe for source
            source_info_db = pd.concat([source_info_db, pd.DataFrame([[source.split('(')[0].strip(), source_type.strip(), spectrum_type.strip(), morph_dict, spectrum_dict]], columns=source_info_db.columns)], ignore_index=True)

        #return dataframe
        return source_info_db

    def _load_from_model_file(self, model_file_loc: str) -> pd.DataFrame:
        '''convert model from .model format to pandas dataframe'''
        namespace = {'threeML': threeML}
        exec(open(model_file_loc).read(), namespace)
        try:
            model = namespace['model']
        except:
            print("Error occurred while loading model")
        source_list = list(model.to_dict_with_types().keys())
        model_odict = model.to_dict_with_types()
        source_info_db = pd.DataFrame(columns=['source', 'morphology_type', 'spectrum_type', 'morphology_params', 'spectrum_params'])
        # #Iterate through all sources found in the dict
        for source in source_list:
            params_to_get = None

            source_type = list(model_odict[source].keys())[0]
            #Handle the fact that point sources don't have the model name in the yml file
            if(source_type == 'position'):
                source_type = 'PointSource'

            #attempt to get the list of relevant morphology parameters for the given source morphology. If none found print a fail
            try:
                params_to_get = source_types_db.loc[source_types_db['model'] == source_type, 'params_of_interest'].values[0]
            except:
                if('value' == source_type):
                    continue
                print(f'Could not find matching params for \"{source_type}\" this may be a bug or the model you are using is unsupported')

            #In the case that the relevant params were found begin to extract them from the yml
            if(params_to_get != None):
                #Grab the dict of parameters for the morpholgy
                model_dict = model_odict[source][list(model_odict[source].keys())[0]]

                #Initialize dicts for later updates
                morph_dict = {}
                spectrum_dict = {}

                #For each param grab the value, lower bound, upper bound, and freedom status
                for key in model_dict.keys():
                    if(key in params_to_get):
                        try:
                            morph_dict.update({key: [model_dict[key]['value'], model_dict[key]['min_value'], model_dict[key]['max_value'], model_dict[key]['free']]})
                        except:
                            morph_dict.update({key: [model_dict[key]['value'], None, None, None]})

                #Repeat above but for the spectrum params
                spectrum_type = list(model_odict[source]['spectrum']['main'].keys())[0]
                try:
                    params_to_get = spectrum_types_db.loc[spectrum_types_db['model'] == spectrum_type, 'params_of_interest'].values[0]
                except:
                    print(f'Could not find matching params for \"{spectrum_type}\" this may be a bug or the model you are using is unsupported')
                spectrum_dict = model_odict[source]['spectrum']['main'][spectrum_type]
                for key in spectrum_dict.keys():
                    if(key in params_to_get):
                        spectrum_dict.update({key: [spectrum_dict[key]['value'], spectrum_dict[key]['min_value'], spectrum_dict[key]['max_value'], spectrum_dict[key]['free']]})
            #Add entry to pandas dataframe for source
            source_info_db = pd.concat([source_info_db, pd.DataFrame([[source.split('(')[0].strip(), source_type.strip(), spectrum_type.strip(), morph_dict, spectrum_dict]], columns=source_info_db.columns)], ignore_index=True)

        return source_info_db

    # ------------------------------------------------------------------------
    # Source database builder + model file writer
    # ------------------------------------------------------------------------

    def add_model_to_source_db(self, source_name: str, source_morphology_type: str, source_spectrum_type: str, source_info_db: pd.DataFrame, source_morphology_params={}, source_spectrum_params={}) -> pd.DataFrame:
        """Append a source (with morphology + spectrum params) to source_info_db."""
        if((source_morphology_params == {}) or (source_spectrum_params == {})):
            if(not (source_morphology_type in source_types_db['model'].array)):
                self.logger.error(f'Unable to add source of morpholgy type {source_morphology_type} as it was not found in the supported models. Please add this source to the model database or double check your model to make sure it is correct')
                return source_info_db
            if(not (source_spectrum_type in spectrum_types_db['model'].array)):
                self.logger.error(f'Unable to add source of spectrum type {source_spectrum_type} as it was not found in the supported models. Please add this source to the model database or double check your model to make sure it is correct')
                return source_info_db
        if(source_morphology_params == {}):
            self.logger.warning(f'It is not recommended to use the default morphology parameters if you have not set them yourself in {self.config_path}')
            source_morphology_params = source_types_db.loc[source_types_db['model'] == source_morphology_type, 'default_param_values'].values[0]
        if(source_spectrum_params == {}):
            # self.logger.warning(f'It is not recommended to use the default parameters if you have not set them yourself in {self.config_path}')
            source_spectrum_params = spectrum_types_db.loc[spectrum_types_db['model'] == source_spectrum_type, 'default_param_values'].values[0]
        return pd.concat([source_info_db, pd.DataFrame([[source_name.split('(')[0].strip(), source_morphology_type.strip(), source_spectrum_type.strip(), source_morphology_params, source_spectrum_params]], columns=source_info_db.columns)], ignore_index=True)

    def initialize_model_file(self, model_file_loc: str, source_info_db: pd.DataFrame) -> None:
        '''If model not already in existence make a new one with sources specified by dataframe'''
        with open(model_file_loc, 'w') as model_file:
            for source in source_info_db['source'].array:
                model_file.write(f'{begin_source_marker}\n')
                model_file.write(f'source_name = \'{source}\'\n\n')

                morphology_type = source_info_db.loc[source_info_db['source'] == source, 'morphology_type'].values[0]
                morphology_params = source_info_db.loc[source_info_db['source'] == source, 'morphology_params'].values[0]

                spectrum_type = source_info_db.loc[source_info_db['source'] == source, 'spectrum_type'].values[0]
                spectrum_params = source_info_db.loc[source_info_db['source'] == source, 'spectrum_params'].values[0]

                #Add source location params if needed
                if('ra' in morphology_params.keys()):
                    model_file.writelines([f'source_pos_1 = {morphology_params["ra"][0]}\n', f'source_pos_2 = {morphology_params["dec"][0]}\n', '\n'])
                elif('lon0' in morphology_params.keys()):
                    model_file.writelines([f'source_pos_1 = {morphology_params["lon0"][0]}\n', f'source_pos_2 = {morphology_params["lat0"][0]}\n', '\n'])
                else:
                    print(f'No location for source = {source}. If this is not a template source this is an error.')

                model_file.write(f'spectrum = threeML.{spectrum_type}()\n')

                if(morphology_type == 'PointSource'):
                    model_file.write(f'{source} = threeML.PointSource(source_name,ra=source_pos_1,dec=source_pos_2, spectral_shape=spectrum)\n')
                elif(morphology_type == 'Hermes'):
                    model_file.write(f'shape = threeML.{morphology_type}(fits_file=\'{morphology_params["fits_file"][0]}\',ihdu= {morphology_params["ihdu"][0]})\n{source} = threeML.ExtendedSource(source_name,spatial_shape=shape,spectral_shape=spectrum)\n')
                else:
                    model_file.write(f'shape = threeML.{morphology_type}()\n{source} = threeML.ExtendedSource(source_name,spatial_shape=shape,spectral_shape=spectrum)\n')

                model_file.write('fluxUnit = 1. / (threeML.u.keV * threeML.u.cm ** 2 * threeML.u.s)\n')

                #loop through spectrum params to define values
                for spectrum_param in spectrum_params.keys():
                    unit_mult = ''
                    if(spectrum_param == 'K'):
                        unit_mult = '* fluxUnit'

                    model_file.writelines(['\n', f'spectrum.{spectrum_param} = {spectrum_params[spectrum_param][0]} {unit_mult}\n', f'spectrum.{spectrum_param}.fix = {not spectrum_params[spectrum_param][3]}\n', f'spectrum.{spectrum_param}.bounds = ({spectrum_params[spectrum_param][1]}, {spectrum_params[spectrum_param][2]}) {unit_mult}\n'])
                #loop through spectrum params to define values
                for morphology_param in morphology_params.keys():
                    unit_mult = ''
                    if(morphology_param in ['ra', 'dec', 'lon0', 'lat0']):
                        unit_mult = '* threeML.u.degree'
                    if(morphology_param in ['hash', 'ihdu', 'fits_file', 'frame']):
                        continue
                    if(morphology_type == 'PointSource'):
                        model_file.writelines(['\n', f'{source}.position.{morphology_param}.bounds = ({morphology_params[morphology_param][1]}, {morphology_params[morphology_param][2]}) {unit_mult}\n', f'{source}.position.{morphology_param}.free = {morphology_params[morphology_param][3]}\n'])
                    else:
                        model_file.writelines(['\n', f'shape.{morphology_param} = {morphology_params[morphology_param][0]} {unit_mult}\n', f'shape.{morphology_param}.fix = {not morphology_params[morphology_param][3]}\n', f'shape.{morphology_param}.bounds = ({morphology_params[morphology_param][1]}, {morphology_params[morphology_param][2]}) {unit_mult}\n'])
                model_file.write(f'\n{end_source_marker}\n')

            model_file.write(f'model = threeML.Model(')
            for source in source_info_db['source'].array:
                if(not source == source_info_db['source'].array[-1]):
                    model_file.write(f'{source}, ')
                else:
                    model_file.write(f'{source})')

        return

    # ------------------------------------------------------------------------
    # Freeze / free morphology + spectrum parameters
    # ------------------------------------------------------------------------

    def freeze_morphology(self, sources_to_freeze: list, source_info_db: pd.DataFrame, params_to_freeze=[]) -> pd.DataFrame:
        """Set morphology params of given sources to fixed (free flag False)."""
        for source in sources_to_freeze:
            morph_dict = source_info_db.loc[source_info_db['source'] == source, 'morphology_params'].values[0]
            for key in morph_dict.keys():
                if(not morph_dict[key][3] is None):
                    if(params_to_freeze == []):
                        morph_dict[key][3] = False
                    elif(key in params_to_freeze):
                        morph_dict[key][3] = False
            source_info_db.loc[source_info_db['source'] == source, 'morphology_params'] = [morph_dict]
        return source_info_db

    def freeze_spectrum(self, sources_to_freeze: list, source_info_db: pd.DataFrame, params_to_freeze=[]) -> pd.DataFrame:
        """Set spectrum params of given sources to fixed (free flag False)."""
        for source in sources_to_freeze:
            spectrum_dict = source_info_db.loc[source_info_db['source'] == source, 'spectrum_params'].values[0]
            for key in spectrum_dict.keys():
                print(key)
                if(not spectrum_dict[key][3] is None):
                    if(params_to_freeze == []):
                        spectrum_dict[key][3] = False
                        print(f'freezing {key}')
                    elif(key in params_to_freeze):
                        spectrum_dict[key][3] = False
                        print(f'freezing {key} 2')
            source_info_db.loc[source_info_db['source'] == source, 'spectrum_params'] = [spectrum_dict]
        return source_info_db

    def free_morphology(self, sources_to_free: list, source_info_db: pd.DataFrame, params_to_free=[]) -> pd.DataFrame:
        """Set morphology params of given sources to free (True)."""
        if(not params_to_free == []):
            for source in sources_to_free:
                morph_dict = source_info_db.loc[source_info_db['source'] == source, 'morphology_params'].values[0]
                for key in params_to_free:
                    morph_dict[key][3] = True
                source_info_db.loc[source_info_db['source'] == source, 'morphology_params'] = [morph_dict]
        else:
            for source in sources_to_free:

                morph_dict = source_info_db.loc[source_info_db['source'] == source, 'morphology_params'].values[0]
                default_dict = source_types_db.loc[source_types_db['model'] == source_info_db.loc[source_info_db['source'] == source, 'morphology_type'].values[0], 'default_param_values'].values[0]
                for key in morph_dict.keys():
                    if(default_dict[key][3] == True):
                        morph_dict[key][3] = True
                source_info_db.loc[source_info_db['source'] == source, 'morphology_params'] = [morph_dict]
        return source_info_db

    def free_spectrum(self, sources_to_free: list, source_info_db: pd.DataFrame, params_to_free=[]) -> pd.DataFrame:
        """Set spectrum params of given sources to free (True)."""
        if(not params_to_free == []):
            for source in sources_to_free:
                spectrum_dict = source_info_db.loc[source_info_db['source'] == source, 'spectrum_params'].values[0]
                for key in params_to_free:
                    spectrum_dict[key][3] = True
                source_info_db.loc[source_info_db['source'] == source, 'spectrum_params'] = [spectrum_dict]
        else:
            for source in sources_to_free:

                spectrum_dict = source_info_db.loc[source_info_db['source'] == source, 'spectrum_params'].values[0]
                default_dict = spectrum_types_db.loc[spectrum_types_db['model'] == source_info_db.loc[source_info_db['source'] == source, 'spectrum_type'].values[0], 'default_param_values'].values[0]
                for key in spectrum_dict.keys():
                    if(default_dict[key][3] == True):
                        spectrum_dict[key][3] = True
                source_info_db.loc[source_info_db['source'] == source, 'spectrum_params'] = [spectrum_dict]
        return source_info_db

    # ------------------------------------------------------------------------
    # Small utility helpers (perturb, TS dict, terminal string cleaning)
    # ------------------------------------------------------------------------

    def _perturb_free_morphology_params(self, source_to_perturb: list, source_info_db: pd.DataFrame) -> pd.DataFrame:
        """Randomly nudge free position params (used on fit retry)."""
        for source in source_to_perturb:
            morph_dict = source_info_db.loc[source_info_db['source'] == source, 'morphology_params'].values[0]
            for key in morph_dict.keys():
                if(key in ['ra', 'dec', 'lon0', 'lat0'] and morph_dict[key][3] == True):
                    morph_dict[key][0] = morph_dict[key][0] + (np.random.rand(1)[0] - 0.5) * 0.05
            source_info_db.loc[source_info_db['source'] == source, 'morphology_params'] = [morph_dict]
        return source_info_db

    def _update_source_TS_dict(self, source_db: pd.DataFrame, accepted_step_path: str) -> dict:
        """Read SourceTS.txt from a step dir and map source -> TS value."""
        source_TS_path = glob.glob(f'{accepted_step_path}/SourceTS.txt')[0]
        TS_dict = {}
        with open(source_TS_path, 'r') as TS_file:
            TS_lines = TS_file.readlines()
            for line in TS_lines:
                source_name = line.split(' ')[0]
                if(source_name in source_db['source'].array):
                    TS_val = float(line.split(' ')[1])
                    TS_dict.update({source_name: TS_val})
        return TS_dict

    def remove_repeat(self, match):
        """Collapse a doubled terminal-hyperlink substring to a single copy."""
        text = match.group(1)
        half = len(text) // 2
        if text[:half] == text[half:]:
            return text[:half]
        return text

    def _clean_out_terminal_string(self, out_string: str, to_terminal=False) -> str:
        """Strip ANSI escape codes / OSC-8 hyperlinks from captured output."""
        if(to_terminal):
            cleaned_out_string = re.sub(r'\x1b\]8;.*?\x1b\\', '', out_string)
        else:
            cleaned_out_string = out_string
            cleaned_out_string = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', out_string)
            cleaned_out_string = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-9?]*[ -/]*[@-~])', '', cleaned_out_string)
            cleaned_out_string = re.sub(r'\[[0-9;]+m', '', cleaned_out_string)

            cleaned_out_string = re.sub(r'8;id=\d+;file://[^#;]+/([^/;#]+)8;;', self.remove_repeat, cleaned_out_string)
            cleaned_out_string = re.sub(r'8;id=\d+;file://[^#;]+#(\d+)8;;', self.remove_repeat, cleaned_out_string)

        return cleaned_out_string

    def _natural_sort(self, string: str):
        """Key for natural (human) ordering of strings containing numbers."""
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]


class ALPSLogger:
    '''Centralized logging system with separate pipeline and full logs'''
    '''Thanks to Rishi and Ramiro for this code that is a copy of the
       DRIPS logging code'''
    def __init__(self, log_dir: str, log_level: str = 'INFO'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create Pipeline logger (only Pipeline messages)
        self.logger = logging.getLogger('Pipeline')
        self.logger.setLevel(getattr(logging, log_level))
        self.logger.propagate = False  # Don't propagate to root logger
        if not self.logger.handlers:

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
