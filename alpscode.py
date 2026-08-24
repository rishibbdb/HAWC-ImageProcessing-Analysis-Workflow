import pandas as pd
import yaml
import threeML
import astromodels
from astropy.io import fits
import os
import subprocess
import numpy as np
import healpy as hp
from hawc_hal.maptree import map_tree_factory
from datetime import datetime
import logging
from pathlib import Path
import ast
import re
import glob
from hawc_hal import HAL, HealpixConeROI, HealpixMapROI
import time


begin_source_marker = '#----------BEGIN_SOURCE----------#\n'
end_source_marker   = '#-----------END_SOURCE-----------#\n'

source_types_db = pd.DataFrame(columns=['model','params_of_interest','default_param_values'])

source_types_db['model'] = ['Latitude_galactic_diffuse','Gaussian_on_sphere','Asymm_Gaussian_on_sphere','Disk_on_sphere','Ellipse_on_sphere','SpatialTemplate_2D','Power_law_on_sphere','Continuous_injection_diffusion_ellipse','Continuous_injection_diffusion','Continuous_injection_diffusion_legacy','GalPropTemplate_3D','Hermes','PointSource']

source_types_db['params_of_interest'] = ['K, sigma_b, l_min, l_max','lon0, lat0, sigma','lon0, lat0, a, e, theta','lon0, lat0, radius','lon0, lat0, a, e, theta','K, hash, ihdu, fits_file, frame','lon0, lat0, index','lon0, lat0, rdiff0, delta, b, piv, piv2, incl, elongation','lon0, lat0, rdiff0, rinj, delta, b, piv, piv2','lon0, lat0, rdiff0, delta, uratio, piv, piv2',' K, hash','N, hash, ihdu, fits_file, frame','ra, dec']

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


spectrum_types_db = pd.DataFrame(columns=['model','params_of_interest','default_param_values'])

spectrum_types_db['model'] = ['Broken_powerlaw','Cutoff_powerlaw','Exponential_cutoff','Log_parabola','Powerlaw','Cutoff_powerlaw_Ep']

spectrum_types_db['params_of_interest'] = ['K xb alpha beta piv','K piv index xc','K xc','K piv alpha beta','K piv index','K piv index xp']

spectrum_types_db['default_param_values'] = [{'K': [1e-23,1e-29,1e-19, True],'xb': [10,1,1000, False],'alpha': [-2,-4,-1, True],'beta': [-3,-4,-1, True],'piv':[2e9,None,None, False]},
                                             {'K': [1e-23,1e-29,1e-19, True],'xc': [10,1,1000, False],'index': [-2.5,-4,-1, True],'piv':[2e9,None,None, False]},
                                             {'K': [1e-23,1e-29,1e-19, True],'xc': [10,1,1000, False]},
                                             {'K': [1e-23,1e-29,1e-19, True],'alpha': [-2,-4,-1, True],'beta': [0,-1,1, True],'piv':[2e9,None,None, False]},
                                             {'K': [1e-23,1e-29,1e-19, True],'index': [-2.5,-4,-1, True],'piv':[2e9,None,None, False]},
                                             {'K': [1e-23,1e-29,1e-19, True],'xp': [500,10,10000, False],'index': [-2.5,-4,-1, True],'piv':[2e9,None,None, False]}]


begin_source_marker = '#----------BEGIN_SOURCE----------#\n'
end_source_marker   = '#-----------END_SOURCE-----------#\n'

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
        
class AutomatedLikelihoodPipelineSearch:
    '''Main control and intialization for the ALPS analysis'''
    
    def __init__(self,initial_config: str):
        self.config_path = initial_config
        with open(self.config_path) as config_file:
            self.config_yml_string = yaml.safe_load(config_file)
        
        self.source_info_db = pd.DataFrame(columns=['source','morphology_type','spectrum_type','morphology_params','spectrum_params'])
        '''Required Params'''
        self.parent_dir = self._load_config_value(self.config_yml_string,'required_arguments parent_folder', None)
        self.coord_1    = self._load_config_value(self.config_yml_string,'required_arguments coord_1', None)
        self.coord_2    = self._load_config_value(self.config_yml_string,'required_arguments coord_2', None)
        self.coord_sys  = self._load_config_value(self.config_yml_string,'required_arguments coordinate_system', None)
        self.map_tree   = self._load_config_value(self.config_yml_string,'required_arguments map_tree_file_path', None)
        self.det_res    = self._load_config_value(self.config_yml_string,'required_arguments detector_response_file_path', None)
        self.bin_list   = self._load_config_value(self.config_yml_string,'required_arguments bin_list', None)
        self.estimator  = self._load_config_value(self.config_yml_string,'required_arguments estimator', None)
        self.fit_name   = self._load_config_value(self.config_yml_string,'required_arguments fit_name', None)
        self.pixi_path  = self._load_config_value(self.config_yml_string,'required_arguments pixi_aerie_folder', None)
        
        
        self.fit_model_path = '/lustre/hawcz01/scratch/userspace/sgroetsch/fitmodel-rework/fitModel.py'
        self.draw_maps_path = '/lustre/hawcz01/scratch/userspace/sgroetsch/fitmodel-rework/Drawmaps.py'
        
        log_local_path    = self._load_config_value(self.config_yml_string,'optional_path_arguments log_local_path', 'Logs')
        self.log_abs_path = os.path.join(self.parent_dir,log_local_path)
        
        self.logger = ALPSLogger(str(self.log_abs_path), self._load_config_value(self.config_yml_string,'pipeline_control_arguments logging_level', 'INFO'))
        
        required_arg_list = [self.parent_dir,self.coord_1,self.coord_2,self.coord_sys,self.map_tree,self.det_res,self.bin_list,self.estimator,self.fit_name,self.pixi_path]
        required_arg_name_list = ['parent_folder','coord_1','coord_2','coordinate_system','map_tree_file_path','detector_response_file_path','bin_list','estimator','fit_name','pixi_aerie_folder']
        none_arg_name = [name for arg, name in zip(required_arg_list,required_arg_name_list) if arg is None]
        if(len(none_arg_name) > 0):
            for arg in none_arg_name:
                self.logger.error(f'Arguement {arg} was not correctly specified in the config file {initial_config}. Please double check if all required args are present and correctly defined')
        
        fit_results_local_path = self._load_config_value(self.config_yml_string,'optional_path_arguments fit_results_local_path', 'FitResults')
        self.fit_results_abs_path = os.path.join(self.parent_dir,fit_results_local_path)
        
        raw_sig_map_local_path = self._load_config_value(self.config_yml_string,'optional_path_arguments raw_sig_map_local_path', 'DataMap')
        self.raw_sig_map_abs_path = os.path.join(self.parent_dir,raw_sig_map_local_path)
        
        prev_models_local_path = self._load_config_value(self.config_yml_string,'optional_path_arguments prev_models_local_path', 'Models')
        self.prev_models_abs_path = os.path.join(self.parent_dir,prev_models_local_path)
        
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
        
        self.roi_radius   = float(self._load_config_value(self.config_yml_string,'optional_fitting_arguments roi_radius', '5'))
        self.minimum_point_source_TS   = float(self._load_config_value(self.config_yml_string,'optional_fitting_arguments minimum_point_source_TS', '25'))
        self.minimum_morphology_TS_improvement   = float(self._load_config_value(self.config_yml_string,'optional_fitting_arguments minimum_morphology_TS_improvement', '16'))
        self.minimum_spectral_TS_improvement   = float(self._load_config_value(self.config_yml_string,'optional_fitting_arguments minimum_spectral_TS_improvement', '16'))
        self.point_source_coord_range   = float(self._load_config_value(self.config_yml_string,'optional_fitting_arguments point_source_coord_range', '1'))
        self.extended_source_coord_range   = float(self._load_config_value(self.config_yml_string,'optional_fitting_arguments extended_source_coord_range', '1'))
        self.custom_parameter_boundaries_dict   = self._load_config_value(self.config_yml_string,'optional_fitting_arguments custom_parameter_boundaries_dict', None)
        self.include_old_diffuse_model   = ast.literal_eval(self._load_config_value(self.config_yml_string,'optional_fitting_arguments include_old_diffuse_model', 'False'))
        self.old_diffuse_model_min_l   = self._load_config_value(self.config_yml_string,'optional_fitting_arguments old_diffuse_model_min_l', None)
        self.old_diffuse_model_max_l   = self._load_config_value(self.config_yml_string,'optional_fitting_arguments old_diffuse_model_max_l', None)
        self.old_diffuse_model_sigma_b   = float(self._load_config_value(self.config_yml_string,'optional_fitting_arguments old_diffuse_model_sigma_b', '1'))
        self.include_hermes_diffuse_model   = ast.literal_eval(self._load_config_value(self.config_yml_string,'optional_fitting_arguments include_hermes_diffuse_model', 'False'))
        self.hermes_template_path   = self._load_config_value(self.config_yml_string,'optional_fitting_arguments hermes_template_path', None)
        self.free_diffuse   = ast.literal_eval(self._load_config_value(self.config_yml_string,'optional_fitting_arguments free_diffuse_norm', 'False'))
        self.roi_template_path   = self._load_config_value(self.config_yml_string,'optional_fitting_arguments roi_template_path', None)
        self.compute_optional_TS   = ast.literal_eval(self._load_config_value(self.config_yml_string,'optional_fitting_arguments compute_optional_TS', 'True'))
        self.compute_optional_uncertainties   = ast.literal_eval(self._load_config_value(self.config_yml_string,'optional_fitting_arguments compute_optional_uncertainties', 'True'))
        self.use_fast_TS_estimate   = ast.literal_eval(self._load_config_value(self.config_yml_string,'optional_fitting_arguments use_fast_TS_estimate', 'False'))
        self.max_param_retries   = int(self._load_config_value(self.config_yml_string,'optional_fitting_arguments max_param_retries', '10'))
        
        if(self.include_hermes_diffuse_model):
            if(self.hermes_template_path is None):
                self.logger.error(f'Argument hermes_template_path was not correctly specified in the config file {initial_config} and include_hermes_diffuse_model is set to True. Please double check if all required args are present and correctly defined')
                    
        if(self.include_old_diffuse_model):
            required_arg_list = [self.old_diffuse_model_min_l,self.old_diffuse_model_max_l]
            required_arg_name_list = ['old_diffuse_model_min_l','old_diffuse_model_max_l']
            none_arg_name = [name for arg, name in zip(required_arg_list,required_arg_name_list) if arg is None]
            if(len(none_arg_name) > 0):
                for arg in none_arg_name:
                    self.logger.error(f'Argument {arg} was not correctly specified in the config file {initial_config} and include_old_diffuse_model is set to True. Please double check if all required args are present and correctly defined')
                    
        self.start_from_raw_map   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments start_from_raw_map', 'True'))
        self.start_from_existing_yml_file_model   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments start_from_existing_yml_file_model', 'False'))
        self.yml_file_model_path   = self._load_config_value(self.config_yml_string,'pipeline_control_arguments yml_file_model_path', None)
        self.start_from_existing_model_file_model   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments start_from_existing_model_file_model', 'False'))
        self.model_file_model_path   = self._load_config_value(self.config_yml_string,'pipeline_control_arguments model_file_model_path', None)
        self.start_from_existing_fits_file_model   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments start_from_existing_fits_file_model', 'False'))
        self.fits_file_model_path   = self._load_config_value(self.config_yml_string,'pipeline_control_arguments fits_file_model_path', None)
        self.trusted_source_list   = self._load_config_value(self.config_yml_string,'pipeline_control_arguments trusted_source_list', None)
        self.trust_all_alterante_morphologies   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments trust_all_alterante_morphologies', 'False'))
        self.trust_all_alternate_spectra   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments trust_all_alternate_spectra', 'False'))
        self.perform_point_source_adding_phase   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments perform_point_source_adding_phase', 'True'))
        self.perform_morpholgy_testing_phase   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments perform_morpholgy_testing_phase', 'True'))
        self.alternate_morpholgy_model_list   = self._load_config_value(self.config_yml_string,'pipeline_control_arguments alternate_morpholgy_model_list', None)
        self.perform_spectrum_testing_phase   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments perform_spectrum_testing_phase', 'True'))
        self.alternate_spectrum_model_list   = self._load_config_value(self.config_yml_string,'pipeline_control_arguments alternate_spectrum_model_list', None)
        self.perform_final_fitting_phase   = ast.literal_eval(self._load_config_value(self.config_yml_string,'pipeline_control_arguments perform_final_fitting_phase', 'True'))
        
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
                
        if(self.start_from_raw_map ^ self.start_from_existing_yml_file_model ^ self.start_from_existing_model_file_model ^  self.start_from_existing_fits_file_model):
            pass
        else:
           self.logger.error(f'Either no starting point or multiple starting points has been specified in {initial_config}. Please double check and select exactly one starting point.')
            
        # map_making_control_arguments make_raw_data_map
        self.make_raw_data_map   = ast.literal_eval(self._load_config_value(self.config_yml_string,'map_making_control_arguments make_raw_data_map', 'True'))
        # map_making_control_arguments make_model_map
        self.make_model_map   = ast.literal_eval(self._load_config_value(self.config_yml_string,'map_making_control_arguments make_model_map', 'True'))
        # map_making_control_arguments make_optional_residual_map
        self.make_optional_residual_map   = ast.literal_eval(self._load_config_value(self.config_yml_string,'map_making_control_arguments make_optional_residual_map', 'True'))
        # map_making_control_arguments plot_size_coord_1
        self.plot_size_coord_1   = float(self._load_config_value(self.config_yml_string,'map_making_control_arguments plot_size_coord_1', str(10)))
        # map_making_control_arguments plot_size_coord_2
        self.plot_size_coord_2   = float(self._load_config_value(self.config_yml_string,'map_making_control_arguments plot_size_coord_2', str(10)))
    
    def _run_command_with_live_log(self,command: list):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,text=True,bufsize=1)#,shell=True,env=os.environ)
        
        full_output = []
        
        for line in process.stdout:
            line = self._clean_out_terminal_string(line,to_terminal=True)
            print(line, end='')
            full_output.append(line)
            
        process.wait()
        
        exit_code = process.returncode
        
        return "".join(full_output),exit_code
    
    def _run_multiple_commands(self, commands: list):
        
        processes = [subprocess.Popen(command,stdout=subprocess.PIPE, stderr=subprocess.STDOUT) for command in commands]
        
        exit_info = [process.communicate() for process in processes]
        
        return_info = []
        for output,process in zip(exit_info,processes):
            output = list(output)
            output.append(process.returncode)
            return_info.append(output)
        
        return return_info

    def _update_source_TS_dict(self,source_db:pd.DataFrame,accepted_step_path: str):
        source_TS_path = glob.glob(f'{accepted_step_path}/SourceTS.txt')[0]
        TS_dict = {}
        with open(source_TS_path,'r') as TS_file:
            TS_lines = TS_file.readlines()
            for line in TS_lines:
                source_name = line.split(' ')[0]
                if(source_name in source_db['source'].array):
                    TS_val = float(line.split(' ')[1])
                    TS_dict.update({source_name:TS_val})
        return TS_dict
    def run_alt_hypothesis(self,accepted_step_model_path: str ,alt_hypothesis_models: list,alt_test_type,sources_to_skip = [],morph_params_to_freeze = [],spectrum_params_to_freeze = [], compute_err = True, use_default_TS_calc = True):
        prev_model_path = os.path.abspath(accepted_step_model_path)
        source_db = self._load_model_from_valid_file(prev_model_path)
        try:
            TS_dict = self._update_source_TS_dict(source_db,os.path.pardir(prev_model_path))
        except:
            TS_dict = {}
            for source in source_db['source'].array:
                TS_dict.update({source:-1})
        source_db['Tested'] = False
        if(self.trust_all_alterante_morphologies):
            for source in source_db['source'].array:
                morphology = source_db.loc[source_db['source'] == source,'morphology_type'].values[0]
                if(not morphology == 'PointSource'):
                    source_db.loc[source_db['source'] == source,'Tested'] = True
        if(self.trust_all_alternate_spectra):
            for source in source_db['source'].array:
                spectrum = source_db.loc[source_db['source'] == source,'spectrum_type'].values[0]
                if(not spectrum == 'Powerlaw'):
                    source_db.loc[source_db['source'] == source,'Tested'] = True
        if(self.trusted_source_list is not None):
            for source in self.trusted_source_list:
                source_db.loc[source_db['source'] == source,'Tested'] = True  
        print(prev_model_path)
        TS_list = self._update_source_TS_dict(source_db,os.path.dirname(prev_model_path))
        source_db['TS'] = TS_list
        print(source_db)
        
    def _load_model_from_valid_file(self,model_path: str):
        model_path = os.path.abspath(model_path)
        _,model_type = os.path.splitext(model_path)
        if(model_type == '.yml'):
            source_db = self._load_from_yml_file(model_path)
        elif(model_type == '.fits'):
            source_db = self._load_from_fits_file(model_path)
        elif(model_type == '.model'):
            source_db = self._load_from_model_file(model_path)
        else:
            self.logger.error(f'Could not load from provided model path {model_path}')
        return source_db
        
    def run_multi_fit(self,source_db_list: list, step_name: str, run_local: bool = True, compute_err = True, compute_TS = True, use_default_TS_calc = True):
        start_time = time.perf_counter()
        command_list = []
        step_dirs = []
        step_models = []
        num_tries = []
        for i,source_db in enumerate(source_db_list):
            step_dir = os.path.join(self.fit_results_abs_path,f'{step_name}_{i}')
            if(not os.path.isdir(step_dir)):
                os.mkdir(step_dir)
            step_dirs.append(step_dir)
            model_loc = os.path.join(self.prev_models_abs_path,f'{step_name}_{i}.model')
            step_models.append(model_loc)
            self.initialize_model_file(model_loc,source_db)
            command = list(map(str,['pixi','run','-e','threeml','--manifest-path',self.pixi_path,'python',self.fit_model_path,'--estimator',self.estimator,'--model',model_loc,'--Name',self.fit_name,'--map-tree',self.map_tree,'--det-res',self.det_res,'--out-dir',step_dir,'--use-bins',self.bin_list,'--ROI-radius',self.roi_radius,'--ROI-center',self.coord_1,self.coord_2,'--like']))
            if(not compute_err):
                command.append('--NoError')
            if(not compute_TS or not use_default_TS_calc):
                command.append('--noTS')
            command_list.append(command)
            num_tries.append(1)
        results = self._run_multiple_commands(command_list)
        good_outputs = []
        models_to_rerun = []
        num_deleted = 0
        for j,result in enumerate(results):
            if(not result[2] == 0):
                models_to_rerun.append(source_db_list[j-num_deleted])
                num_tries[j-num_deleted] = 2
            else:
                self.logger.info(f'Fit for {step_dirs[j-num_deleted]}  finished successful after {num_tries[j-num_deleted]} attempts')
                del step_dirs[j-num_deleted]
                del step_models[j-num_deleted]
                del num_tries[j-num_deleted]
                num_deleted = num_deleted+1
                good_outputs.append(result)
        if(models_to_rerun == []):
            good_outputs = results
        else:
            good_outputs.append(self._retry_multi_fit(models_to_rerun,step_dirs,step_models,num_tries,run_local,compute_err,compute_TS,use_default_TS_calc))
            
        return good_outputs
    
    def _retry_multi_fit(self,source_db_list: list, step_dirs: list, step_models: list,num_tries: list, run_local: bool = True, compute_err = True, compute_TS = True, use_default_TS_calc = True):
        command_list = []
        good_outputs = []
        for i,source_db in enumerate(source_db_list):
            if(num_tries[i] > self.max_param_retries):
                self.logger.error(f'Steps {step_dirs} have reached maximum retries and cannot be completed')
                for step in step_dirs:
                    good_outputs.append([None,None,1])
                return good_outputs
            model_loc = step_models[i]
            self.initialize_model_file(model_loc,source_db)
            command = list(map(str,['pixi','run','-e','threeml','--manifest-path',self.pixi_path,'python',self.fit_model_path,'--estimator',self.estimator,'--model',model_loc,'--Name',self.fit_name,'--map-tree',self.map_tree,'--det-res',self.det_res,'--out-dir',step_dirs[i],'--use-bins',self.bin_list,'--ROI-radius',self.roi_radius,'--ROI-center',self.coord_1,self.coord_2,'--like']))
            if(not compute_err):
                command.append('--NoError')
            if(not compute_TS or not use_default_TS_calc):
                command.append('--noTS')
            command_list.append(command)
        results = self._run_multiple_commands(command_list)
        
        models_to_rerun = []
        num_deleted = 0
        for j,result in enumerate(results):
            
            if(not result[2] == 0):
                num_tries[j-num_deleted] = num_tries[j-num_deleted]+1
            else:
                self.logger.info(f'Fit for {step_dirs[j-num_deleted]}  finished successful after {num_tries[j]} attempts')
                del models_to_rerun[j-num_deleted]
                del step_dirs[j-num_deleted]
                del step_models[j-num_deleted]
                del num_tries[j-num_deleted]
                num_deleted = num_deleted+1
                good_outputs.append(result)
        if(models_to_rerun == []):
            good_outputs = results
        else:
            good_outputs.append(self._retry_multi_fit(models_to_rerun,step_dirs,step_models,num_tries,run_local,compute_err,compute_TS,use_default_TS_calc))   
        return good_outputs
            
    def run_single_fit(self,source_db: pd.DataFrame, step_name: str, run_local: bool = True, compute_err = True, compute_TS = True, use_default_TS_calc = True):
        start_time = time.perf_counter()
        step_dir = os.path.join(self.fit_results_abs_path,step_name)
        if(not os.path.isdir(step_dir)):
            os.mkdir(step_dir)
        model_loc = os.path.join(self.prev_models_abs_path,f'{step_name}.model')
        self.initialize_model_file(model_loc,source_db)
        
        retries = 0
        while(retries < self.max_param_retries):
            command = list(map(str,['pixi','run','-e','threeml','--manifest-path',self.pixi_path,'python',self.fit_model_path,'--estimator',self.estimator,'--model',model_loc,'--Name',self.fit_name,'--map-tree',self.map_tree,'--det-res',self.det_res,'--out-dir',step_dir,'--use-bins',self.bin_list,'--ROI-radius',self.roi_radius,'--ROI-center',self.coord_1,self.coord_2,'--like']))
            if(not compute_err):
                command.append('--NoError')
            if(not compute_TS or not use_default_TS_calc):
                command.append('--noTS')
            run_terminal_out,exit_code  = self._run_command_with_live_log(command)
            with open(f'{step_dir}/log_fit_{retries}.log','w') as fit_log:
                fit_log.write(self._clean_out_terminal_string(run_terminal_out)) 
            retries+=1
            if(exit_code == 0):
                self.logger.info(f'Successfully fit after {retries} attempt(s)')
                break
            else:
                retries += 1
                source_db = self._perturb_free_morphology_params(self.target_sources,source_db)
                self.initialize_model_file(model_loc,source_db)    
                # pass
        if(not use_default_TS_calc):
            updated_source_db = self._load_model_from_valid_file(glob.glob(f'{step_dir}/*_modelFit.yml')[0])
            self._get_sloppy_TS(updated_source_db,updated_source_db['source'].array,step_dir)
        command = list(map(str,['pixi','run','-e','threeml','--manifest-path',self.pixi_path,'python',self.draw_maps_path,'--Name',self.fit_name,'--map-tree',self.map_tree,'--det-res',self.det_res,'--in-file',glob.glob(f'{step_dir}/*_likeResults.fits')[0],'--out-dir',step_dir,'--use-bins',self.bin_list,'--ROI-radius',self.roi_radius,'--ROI-center',self.coord_1,self.coord_2,'--estimator',self.estimator]))
        maps_terminal_out,exit_code = self._run_command_with_live_log(command)
        with open(f'{step_dir}/log_map.log','w') as map_log:
            map_log.write(self._clean_out_terminal_string(maps_terminal_out))
        
        end_time = time.perf_counter()
        self.logger.info(f'Fit {step_name} completed in {round((end_time-start_time)/60.0,2)} minutes')
        return
    
    def _perturb_free_morphology_params(self,source_to_perturb: list,source_info_db: pd.DataFrame):
        for source in source_to_perturb:
            morph_dict = source_info_db.loc[source_info_db['source'] == source,'morphology_params'].values[0]
            for key in morph_dict.keys():
                if(key in ['ra','dec','lon0','lat0'] and morph_dict[key][3] == True):
                    morph_dict[key][0] = morph_dict[key][0] + (np.random.rand(1)[0]-0.5)*0.05
            source_info_db.loc[source_info_db['source'] == source,'morphology_params'] = [morph_dict]
        return source_info_db
            
    def _run_point_source_adding_phase(self,morph_params_to_freeze = [],spectrum_params_to_freeze = [], compute_err = True, compute_TS = True, use_default_TS_calc = True):
        add_PS_first_step = True
        added_fixed_diffuse = False
        if(self.start_from_raw_map):
            if(self.include_hermes_diffuse_model):
                self.source_info_db = self.add_model_to_source_db('DBE','Latitude_galactic_diffuse','Powerlaw',self.source_info_db,{"K": [1, None, None, False], "sigma_b": [self.old_diffuse_model_sigma_b, None, None, False], "l_min": [self.old_diffuse_model_min_l, None, None, False], "l_max": [self.old_diffuse_model_max_l, None, None, False]},{'K': [1e-23,1e-29,1e-19, self.free_diffuse],'index': [-2.65,-4,-1, self.free_diffuse],'piv':[2e9,None,None, False]})
                if(self.free_diffuse):
                    self.target_sources = ['DBE']
                    add_PS_first_step = False
                else:
                    added_fixed_diffuse = True
            elif(self.include_old_diffuse_model):
                self.source_info_db = self.add_model_to_source_db('Hermes','Hermes','Powerlaw',self.source_info_db,{"N": [1, 0.01, 100, self.free_diffuse], "hash": [None, None, None, False], "ihdu": [0, None, None, False], "properties": [None, None, None, False], "fits_file": [None, None, None, False], "frame": ["icrs", None, None, False]},{'K': [1,1,1, False],'index': [0,0,0, False],'piv':[0,None,None, False]})
                if(self.free_diffuse):
                    self.target_sources = ['Hermes']
                    add_PS_first_step = False
                else:
                    added_fixed_diffuse = True
            if(add_PS_first_step):
                if(added_fixed_diffuse):
                    residual_map = self._get_residual_fixed_model(self.source_info_db, f'{self.fit_name}_step_0')
                    next_ra,next_dec = self._find_next_hotpsot(residual_map)
                elif(self.make_raw_data_map):
                    self.make_map(self.map_tree,self.raw_sig_map_abs_path,'map')
                    next_ra,next_dec = self._find_next_hotpsot(f'{self.raw_sig_map_abs_path}/map.fits')
                else:
                    next_ra,next_dec = (self.coord_1,self.coord_2)
                self.source_info_db = self.add_model_to_source_db('Source_1','PointSource','Powerlaw',self.source_info_db,{"ra": [next_ra, next_ra - self.point_source_coord_range, next_ra + self.point_source_coord_range, True], "dec": [next_dec, next_dec - self.point_source_coord_range, next_dec + self.point_source_coord_range, True]})
                self.target_sources = ['Source_1']
        elif(self.start_from_existing_yml_file_model):
            self.source_info_db = self._load_model_from_valid_file(self.yml_file_model_path)
            self.target_sources = self.source_info_db['source'].array
        elif(self.start_from_existing_model_file_model):
            self.source_info_db = self._load_model_from_valid_file(self.model_file_model_path)
            self.target_sources = self.source_info_db['source'].array
        elif(self.start_from_existing_fits_file_model):
            self.source_info_db = self._load_model_from_valid_file(self.fits_file_model_path)
            self.target_sources = self.source_info_db['source'].array
        else:
            self.logger.error(f'You have to start somewhere. None of the starting conditions for the point source adding phase are set to True. Please adjust the config file to have one allowed')
            
        self.run_single_fit(self.source_info_db,f'{self.fit_name}_step_1', compute_err = compute_err, compute_TS = compute_TS, use_default_TS_calc = use_default_TS_calc)
        print(f'{self.fit_name}_step_1')
        self.source_info_db = self._after_run_accouting(f'{self.fit_name}_step_1',self.target_sources,morph_params_to_freeze=morph_params_to_freeze,spectrum_params_to_freeze=spectrum_params_to_freeze,need_residual=True,need_stats=True)
        
        self.last_delta_TS = 1000000
        i = 2
        while(self.last_delta_TS > self.minimum_point_source_TS):
            next_ra,next_dec = self._find_next_hotpsot(self.current_residual_fits_map)
            list_of_used_names = self.source_info_db['source'].array
            if(f'Source_{i}' in list_of_used_names):
                self.source_info_db = self.add_model_to_source_db(f'Source_{i}_ALPS','PointSource','Powerlaw',self.source_info_db,{"ra": [next_ra, next_ra - self.point_source_coord_range, next_ra + self.point_source_coord_range, True], "dec": [next_dec, next_dec - self.point_source_coord_range, next_dec + self.point_source_coord_range, True]})
                self.target_sources = [f'Source_{i}_ALPS']
            else:
                self.source_info_db = self.add_model_to_source_db(f'Source_{i}','PointSource','Powerlaw',self.source_info_db,{"ra": [next_ra, next_ra - self.point_source_coord_range, next_ra + self.point_source_coord_range, True], "dec": [next_dec, next_dec - self.point_source_coord_range, next_dec + self.point_source_coord_range, True]})
                self.target_sources = [f'Source_{i}']
            self.run_single_fit(self.source_info_db,f'{self.fit_name}_step_{i}', compute_err = compute_err, compute_TS = compute_TS, use_default_TS_calc = use_default_TS_calc)
            self.prev_log_like = self.current_log_like
            self.prev_AIC = self.current_AIC
            self.source_info_db = self._after_run_accouting(f'{self.fit_name}_step_{i}',self.target_sources,morph_params_to_freeze=morph_params_to_freeze,spectrum_params_to_freeze=spectrum_params_to_freeze,need_residual=True,need_stats=True)
            self.last_delta_TS = 2*(self.prev_log_like - self.current_log_like)
            i+=1
                    
    def _after_run_accouting(self,step_name: str,target_sources: list,morph_params_to_freeze = [],spectrum_params_to_freeze = [],need_residual = False, need_stats = False):
        step_dir = os.path.join(self.fit_results_abs_path,step_name)
        if(self.make_model_map):
            self.make_map(glob.glob(f'{step_dir}/*_modelMap.hd5')[0],f'{step_dir}/maps','model')
        if(self.make_optional_residual_map or need_residual):
            self.current_residual_fits_map = self.make_map(glob.glob(f'{step_dir}/*_residualMap.hd5')[0],f'{step_dir}/maps','residual')
        if(need_stats):
            self.current_fits_results = glob.glob(f'{step_dir}/*_likeResults.fits')[0]
            current_fits_file = fits.open(self.current_fits_results)
            self.current_log_like = current_fits_file[1].header['STAT0']
            self.current_AIC = current_fits_file[1].header['MV0']
        source_info_db = self._load_model_from_valid_file(glob.glob(f'{step_dir}/*_likeResults.fits')[0])
        self.freeze_morphology(target_sources,source_info_db,params_to_freeze=morph_params_to_freeze)
        self.freeze_spectrum(target_sources,source_info_db,params_to_freeze=spectrum_params_to_freeze)
        return source_info_db
    
    def _get_sloppy_TS(self,source_db: pd.DataFrame,sources: list,step_dir: str):
        roi_radius_model = np.max([5,self.roi_radius+2])
        if self.roi_template_path is not None:
            new_ROI = HealpixMapROI(data_radius=self.roi_radius, model_radius=roi_radius_model, ra=self.coord_1, dec=self.coord_2, roifile=self.roi_template_path, threshold=0.5)
        else:
            new_ROI = HealpixConeROI(data_radius=self.roi_radius, model_radius=roi_radius_model, ra=self.coord_1, dec=self.coord_2)
        like = HAL("HAWC", self.map_tree, self.det_res, new_ROI, 0.1)
        like.set_active_measurements(bin_list=self.bin_list.split(" "))
        
        main_model_loc = f'{step_dir}/temp_fitted_model.model'
        self.initialize_model_file(main_model_loc,source_db)
        namespace = {'threeML': threeML}
        exec(open(main_model_loc).read(),namespace)
        try:
            model = namespace['model']
        except:
            print("Error occurred while loading model")
            
        like.set_model(model)
        null_LL = like.get_log_like()
        with open(f"{step_dir}/SourceTS.txt", "w+") as TSout:
            TSout.write(f"Individual Source TS for model in {step_dir}\n")
            for source in sources:
                cloned_model = threeML.clone_model(model)
                cloned_model.remove_source(source)
                
                like.set_model(cloned_model)
                alt_LL = like.get_log_like()
                TS = 2*(null_LL - alt_LL)
                TSout.write(f"{source} {TS}\n")
                print(TS)
        return f"{step_dir}/SourceTS.txt"

    def _get_residual_fixed_model(self,source_db: pd.DataFrame,step_name: str):
        step_dir = os.path.join(self.fit_results_abs_path,step_name)
        if(not os.path.isdir(step_dir)):
            os.mkdir(step_dir)
        model_file_loc = f'{step_dir}/step0.model'
        self.initialize_model_file(model_file_loc,source_db)
        namespace = {'threeML': threeML}
        exec(open(model_file_loc).read(),namespace)
        try:
            model = namespace['model']
        except:
            print("Error occurred while loading model")
        
        new_ROI = HealpixConeROI(data_radius=self.roi_radius, model_radius=(self.roi_radius+2) , ra=self.coord_1, dec=self.coord_2)
        like = HAL("HAWC", self.map_tree, self.det_res, new_ROI, 0.1)
        like.set_active_measurements(bin_list=self.bin_list.split(" "))
        like.set_model(model)
        likelihood_object = like
        likelihood_object.write_model_map(f"{step_dir}/{step_name}_modelMap.hd5")
        likelihood_object.write_residual_map(f"{step_dir}/{step_name}_residualMap.hd5")
        return f"{step_dir}/{step_name}_residualMap.hd5"    
    
    def _convert_hd5_to_fits(self, hd5_loc: str,fits_out_dir: str, map_prefix: str):
        '''Copied over from hal_hdf5_to_fits.py. Converts '''
        hd5_path = os.path.expandvars(hd5_loc)
        hd5_dir = os.path.dirname(hd5_path)
        fits_out_dir_path = os.path.expandvars(fits_out_dir)
        map_file_names = []
        
        if(not os.path.isdir(fits_out_dir_path)):
            os.mkdir(fits_out_dir_path)
        
        if(os.path.isfile(hd5_path)):
            maptree = map_tree_factory(hd5_path, None)
            now=datetime.now()
            startMJD=56987.9286332
            
            #FIRST HEADER
            '''
            COMMENT   FITS (Flexible Image Transport System) format is defined in 'Astronomy
            COMMENT   and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H 
            DATE    = '2018-12-01T02:31:14' / file creation date (YYYY-MM-DDThh:mm:ss UT)   
            STARTMJD=     56987.9286332451 / MJD of first event                             
            STOPMJD =     58107.2396848326 / MJD of last event                              
            NEVENTS =                  -1. / Number of events in map                        
            TOTDUR  =     24412.9020670185 / Total integration time [hours]                 
            DURATION=      1.9943578604616 / Avg integration time [hours]                   
            MAPTYPE = 'duration'           / e.g. Skymap, Moonmap                           
            MAXDUR  =                  -1. / Max integration time [hours]                   
            MINDUR  =                  -1. / Min integration time [hours]                   
            EPOCH   = 'unknown '           / e.g. J2000, current, J2016, B1950, etc.        
            HIERARCH MAPFILETYPE = 'duration' / e.g. standard, duration, integration   
            '''

            FITS_COMMENT="FITS (Flexible Image Transport System) format is defined in 'Astronomy and Astrophysics', volume 376, page 359; bibcode: 2001A&A...376..359H"

            primary_keys=['COMMENT', 'COMMENT', 'DATE', 'STARTMJD', 'STOPMJD',
                        'NEVENTS', 'TOTDUR', 'DURATION', 'MAPTYPE', 'MAXDUR', 
                        'MINDUR', 'EPOCH', 'MAPFILETYPE']

            primary_values=[FITS_COMMENT  ,   FITS_COMMENT, "{0}".format(now), 56987.9286332451, 58107.2396848326,
                        -1.0, 24412.9020670185, 1.9943578604616, 'duration', -1.0, -1.0, 'unknown', 'duration']

            primary_comments=["file does conform to FITS standard",
                            "number of bits per data pixel",
                            "number of data axes",
                            "FITS dataset may contain extension",
                            "MJD of first event", "MJD of last event",
                            "Number of events in map",
                            "Total integration time [hours]",
                            "Avg integration time [hours]",
                            "e.g. Skymap, Moonmap",
                            "Max integration time [hours]",
                            "Min integration time [hours]",
                            "e.g. J2000, current, J2016, B1950, etc.",
                            "e.g. standard, duration, integration"]


            labels=['data map', 'background map', 'exposure map']
            label_format=[ np.float64 for i in range(len(labels)) ]
            label_units=[ 'unknown' for i in range(len(labels)) ]
            
            for i, analysis_bin in enumerate(maptree.analysis_bins_labels):
                map_bin    = maptree[analysis_bin]
                #properties
                nside      = map_bin.nside
                npix       = map_bin.npix
                see_pixels = map_bin.observation_map._pixels_ids
                transits   = map_bin.n_transits
                scheme     = map_bin.scheme

                nest_scheme=False
                if scheme.lower()=='nested':
                    nest_scheme=True

                #what we want
                data  = map_bin.observation_map.as_dense()
                bkg   = map_bin.background_map.as_dense()

                zeros = np.empty(npix)
                zeros.fill(9e9)

                out_file_name=f'{fits_out_dir_path}/{map_prefix}_bin{analysis_bin}.fits.gz'
                map_file_names.append(out_file_name)
                #I probably need to add some header info, but yeah
                hp.fitsfunc.write_map(out_file_name, (data, bkg, zeros), 
                                    column_names=labels, column_units=label_units, dtype=label_format, 
                                    partial=False, fits_IDL=True, overwrite=True, nest=nest_scheme)


                #add the cards to the header
                with fits.open(out_file_name,'update') as hdu1:
                    hdr=hdu1[0].header

                    for i,key in enumerate(primary_keys):
                        val=primary_values[i]
                        comment=primary_comments[i]

                        if key=='TOTDUR':
                            val=24.0*transits
                    
                        if key=='STOPMJD':
                            val=startMJD+transits

                        entry=(val,comment)

                        hdr[key]=entry

                    #File is now closed
                print("File Written: {0}".format(out_file_name))
            return map_file_names        
            
    def _find_next_hotpsot(self,path_to_map_to_search: str):
        terminal_out,exit_code = self._run_command_with_live_log(list(map(str,['pixi','run','-e','threeml','--manifest-path',self.pixi_path,'aerie-apps-get-local-extremum', '--ra',self.coord_1,'--dec',self.coord_2,'--windowRadius',self.roi_radius*2,'--input',path_to_map_to_search])))
        hotspot_ra,hotspot_dec = re.findall(r'\d+\.\d{3}', terminal_out.split("\n")[-2])
        return (float(hotspot_ra),float(hotspot_dec))
    
    def remove_repeat(self,match):
        text = match.group(1)
        half = len(text) // 2
        if text[:half] == text[half:]:
            return text[:half]
        return text
    
    def _clean_out_terminal_string(self, out_string: str,to_terminal=False):
        if(to_terminal):
            cleaned_out_string = re.sub(r'\x1b\]8;.*?\x1b\\', '', out_string)
        else:
            cleaned_out_string = out_string
            cleaned_out_string = re.sub(r'\x1b\[[0-9;]*[a-zA-Z]', '', out_string)
            cleaned_out_string = re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-9?]*[ -/]*[@-~])', '', cleaned_out_string)
            cleaned_out_string = re.sub(r'\[[0-9;]+m', '', cleaned_out_string)
        
            cleaned_out_string = re.sub(r'8;id=\d+;file://[^#;]+/([^/;#]+)8;;', self.remove_repeat , cleaned_out_string)
            cleaned_out_string = re.sub(r'8;id=\d+;file://[^#;]+#(\d+)8;;', self.remove_repeat, cleaned_out_string)
        
        
        return cleaned_out_string
      
    def _natural_sort(self,string: str):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]
    
    def make_map(self,map_path: str, map_save_dir: str, map_name: str, index = 2.6,pivot = 2):
        start_time = time.perf_counter()
        _,map_type = os.path.splitext(map_path)
        if(not os.path.isdir(map_save_dir)):
            os.mkdir(map_save_dir)
        if('hd5' in map_type):
            fits_file_list = self._convert_hd5_to_fits(map_path,map_save_dir,map_name)
        if('root' in map_type):
            self._run_command_with_live_log(list(map(str,['pixi','run','-e','threeml','--manifest-path',self.pixi_path,'skymaps-maptree2fits','-i',map_path,'-o',f'{map_save_dir}/{map_name}','-N']+self.bin_list.split(" "))))
            fits_file_list = sorted(glob.iglob(f'{map_save_dir}/{map_name}*.fits.gz'),key=self._natural_sort)    
        self._run_command_with_live_log(list(map(str,['pixi','run','-e','threeml','--manifest-path',self.pixi_path,'aerie-apps-HealpixSigFluxMap', '-i']+fits_file_list+['-b']+self.bin_list.split(" ")+['--index',index,'--pivot',pivot,'-d',self.det_res,'--nthreads','5','--window',self.coord_1,self.coord_2,self.roi_radius,self.roi_radius,'--negFlux', '--negSignif', '-o',f'{map_save_dir}/{map_name}.fits'])))
        end_time = time.perf_counter()
        self.logger.info(f'Map {map_save_dir}/{map_name}.fits created in {round((end_time-start_time)/60.0,2)} minutes')
        return f'{map_save_dir}/{map_name}.fits'
                     
    def _load_config_value(self,config_yml_string: dict,key: str, default_val: str):
        keys = key.split(' ')
        target_value = config_yml_string
        for key in keys:
            if(type(target_value) == dict):
                target_value = target_value.get(key)
        if(target_value == None):
            target_value = default_val
        return target_value
    
    def _load_from_yml_file(self,model_yml_loc: str):
        '''Convert model in yml format to pandas dataframe'''
        with open(model_yml_loc,'r') as yaml_file:
            model_yml_string = yaml.safe_load(yaml_file)
            return self._parse_yml_string(model_yml_string)
    
    def _load_from_fits_file(self,fits_file_loc: str):
        '''convert model from .fits format to pandas dataframe'''
        fits_file = fits.open(fits_file_loc)
        model_yml_string = yaml.safe_load(fits_file[1].header['MODEL'].replace('_NEWLINE_','\n'))
        return self._parse_yml_string(model_yml_string)
       
    def _parse_yml_string(self,model_yml_string: str):
        '''Read yml string and convert to dataframe'''
        source_list = list(model_yml_string.keys())

        source_info_db = pd.DataFrame(columns=['source','morphology_type','spectrum_type','morphology_params','spectrum_params'])

        #Iterate through all sources found in the yml
        for source in source_list:
            params_to_get = None
            
            source_type = list(model_yml_string[source].keys())[0]
            #Handle the fact that point sources don't have the model name in the yml file
            if(source_type == 'position'):
                source_type = 'PointSource'

            #attempt to get the list of relevant morphology parameters for the given source morphology. If none found print a fail
            try:
                params_to_get = source_types_db.loc[source_types_db['model'] == source_type,'params_of_interest'].values[0]
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
                spectrum_dict={}
                
                #For each param grab the value, lower bound, upper bound, and freedom status
                for key in model_dict.keys():
                    if(key in params_to_get):
                        try:
                            morph_dict.update({key: [model_dict[key]['value'],model_dict[key]['min_value'],model_dict[key]['max_value'],model_dict[key]['free']]})
                        except:
                            morph_dict.update({key: [model_dict[key]['value'],None,None,None]})
                        
                
                #Repeat above but for the spectrum params        
                spectrum_type = list(model_yml_string[source]['spectrum']['main'].keys())[0]
                try:
                    params_to_get = spectrum_types_db.loc[spectrum_types_db['model'] == spectrum_type,'params_of_interest'].values[0]
                except:
                    print(f'Could not find matching params for \"{spectrum_type}\" this may be a bug or the model you are using is unsupported')
                spectrum_dict = model_yml_string[source]['spectrum']['main'][spectrum_type]
                for key in spectrum_dict.keys():
                    if(key in params_to_get):
                        spectrum_dict.update({key: [spectrum_dict[key]['value'],spectrum_dict[key]['min_value'],spectrum_dict[key]['max_value'],spectrum_dict[key]['free']]})
            #Add entry to pandas dataframe for source  
            source_info_db = pd.concat([source_info_db,pd.DataFrame([[source.split('(')[0].strip(),source_type.strip(),spectrum_type.strip(),morph_dict,spectrum_dict]],columns = source_info_db.columns)],ignore_index=True)

        #return dataframe
        return source_info_db
    
    def _load_from_model_file(self,model_file_loc: str):
        '''convert model from .model format to pandas dataframe'''
        namespace = {'threeML': threeML}
        exec(open(model_file_loc).read(),namespace)
        try:
            model = namespace['model']
        except:
            print("Error occurred while loading model")
        source_list = list(model.to_dict_with_types().keys())
        model_odict = model.to_dict_with_types()
        source_info_db = pd.DataFrame(columns=['source','morphology_type','spectrum_type','morphology_params','spectrum_params'])
        # #Iterate through all sources found in the dict
        for source in source_list:
            params_to_get = None
            
            source_type = list(model_odict[source].keys())[0]
            #Handle the fact that point sources don't have the model name in the yml file
            if(source_type == 'position'):
                source_type = 'PointSource'

            #attempt to get the list of relevant morphology parameters for the given source morphology. If none found print a fail
            try:
                params_to_get = source_types_db.loc[source_types_db['model'] == source_type,'params_of_interest'].values[0]
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
                spectrum_dict={}
                
                #For each param grab the value, lower bound, upper bound, and freedom status
                for key in model_dict.keys():
                    if(key in params_to_get):
                        try:
                            morph_dict.update({key: [model_dict[key]['value'],model_dict[key]['min_value'],model_dict[key]['max_value'],model_dict[key]['free']]})
                        except:
                            morph_dict.update({key: [model_dict[key]['value'],None,None,None]})
                        
                
                #Repeat above but for the spectrum params        
                spectrum_type = list(model_odict[source]['spectrum']['main'].keys())[0]
                try:
                    params_to_get = spectrum_types_db.loc[spectrum_types_db['model'] == spectrum_type,'params_of_interest'].values[0]
                except:
                    print(f'Could not find matching params for \"{spectrum_type}\" this may be a bug or the model you are using is unsupported')
                spectrum_dict = model_odict[source]['spectrum']['main'][spectrum_type]
                for key in spectrum_dict.keys():
                    if(key in params_to_get):
                        spectrum_dict.update({key: [spectrum_dict[key]['value'],spectrum_dict[key]['min_value'],spectrum_dict[key]['max_value'],spectrum_dict[key]['free']]})
            #Add entry to pandas dataframe for source  
            source_info_db = pd.concat([source_info_db,pd.DataFrame([[source.split('(')[0].strip(),source_type.strip(),spectrum_type.strip(),morph_dict,spectrum_dict]],columns = source_info_db.columns)],ignore_index=True)
            
        return source_info_db
    
    def add_model_to_source_db(self,source_name: str, source_morphology_type: str,source_spectrum_type: str,source_info_db: pd.DataFrame, source_morphology_params = {}, source_spectrum_params = {}):
        
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
        return pd.concat([source_info_db,pd.DataFrame([[source_name.split('(')[0].strip(),source_morphology_type.strip(),source_spectrum_type.strip(),source_morphology_params,source_spectrum_params]],columns = source_info_db.columns)],ignore_index=True)
        
    def initialize_model_file(self,model_file_loc: str, source_info_db: pd.DataFrame):
        '''If model not already in existence make a new one with sources specified by dataframe'''
        with open(model_file_loc,'w') as model_file:
            for source in source_info_db['source'].array:
                model_file.write(f'{begin_source_marker}\n')
                model_file.write(f'source_name = \'{source}\'\n\n')
                
                morphology_type = source_info_db.loc[source_info_db['source'] == source,'morphology_type'].values[0]
                morphology_params = source_info_db.loc[source_info_db['source'] == source,'morphology_params'].values[0]
                
                spectrum_type = source_info_db.loc[source_info_db['source'] == source,'spectrum_type'].values[0]
                spectrum_params = source_info_db.loc[source_info_db['source'] == source,'spectrum_params'].values[0]
                
                
                #Add source location params if needed
                if('ra' in morphology_params.keys()):
                    model_file.writelines([f'source_pos_1 = {morphology_params['ra'][0]}\n',f'source_pos_2 = {morphology_params['dec'][0]}\n','\n'])
                elif('lon0' in morphology_params.keys()):
                    model_file.writelines([f'source_pos_1 = {morphology_params['lon0'][0]}\n',f'source_pos_2 = {morphology_params['lat0'][0]}\n','\n'])
                else:
                    print(f'No location for source = {source}. If this is not a template source this is an error.')
                    
                model_file.write(f'spectrum = threeML.{spectrum_type}()\n')
                
                if(morphology_type == 'PointSource'):
                    model_file.write(f'{source} = threeML.PointSource(source_name,ra=source_pos_1,dec=source_pos_2, spectral_shape=spectrum)\n')
                elif(morphology_type == 'Hermes'):
                    model_file.write(f'shape = threeML.{morphology_type}(fits_file=\'{morphology_params['fits_file'][0]}\',ihdu= {morphology_params['ihdu'][0]})\n{source} = threeML.ExtendedSource(source_name,spatial_shape=shape,spectral_shape=spectrum)\n')
                else:
                    model_file.write(f'shape = threeML.{morphology_type}()\n{source} = threeML.ExtendedSource(source_name,spatial_shape=shape,spectral_shape=spectrum)\n')
                    
                model_file.write('fluxUnit = 1. / (threeML.u.keV * threeML.u.cm ** 2 * threeML.u.s)\n')
                
                #loop through spectrum params to define values
                for spectrum_param in spectrum_params.keys():
                    unit_mult = ''
                    if(spectrum_param == 'K'):
                        unit_mult = '* fluxUnit'
                        
                    model_file.writelines(['\n',f'spectrum.{spectrum_param} = {spectrum_params[spectrum_param][0]} {unit_mult}\n',f'spectrum.{spectrum_param}.fix = {not spectrum_params[spectrum_param][3]}\n',f'spectrum.{spectrum_param}.bounds = ({spectrum_params[spectrum_param][1]}, {spectrum_params[spectrum_param][2]}) {unit_mult}\n'])
                #loop through spectrum params to define values
                for morphology_param in morphology_params.keys():
                    unit_mult = ''
                    if(morphology_param in ['ra','dec','lon0','lat0']):
                        unit_mult = '* threeML.u.degree'
                    if(morphology_param in ['hash','ihdu','fits_file','frame']):
                        continue
                    if(morphology_type == 'PointSource'):
                        model_file.writelines(['\n',f'{source}.position.{morphology_param}.bounds = ({morphology_params[morphology_param][1]}, {morphology_params[morphology_param][2]}) {unit_mult}\n',f'{source}.position.{morphology_param}.free = {morphology_params[morphology_param][3]}\n'])
                    else:
                        model_file.writelines(['\n',f'shape.{morphology_param} = {morphology_params[morphology_param][0]} {unit_mult}\n',f'shape.{morphology_param}.fix = {not morphology_params[morphology_param][3]}\n',f'shape.{morphology_param}.bounds = ({morphology_params[morphology_param][1]}, {morphology_params[morphology_param][2]}) {unit_mult}\n'])
                model_file.write(f'\n{end_source_marker}\n')
            
            model_file.write(f'model = threeML.Model(')
            for source in source_info_db['source'].array:
                if(not source == source_info_db['source'].array[-1]):
                    model_file.write(f'{source}, ')
                else:
                    model_file.write(f'{source})')
            
        
        return
    
    def freeze_morphology(self,sources_to_freeze: list, source_info_db: pd.DataFrame, params_to_freeze = []):
        for source in sources_to_freeze:
            morph_dict = source_info_db.loc[source_info_db['source'] == source,'morphology_params'].values[0]
            for key in morph_dict.keys():
                if(not morph_dict[key][3] is None):
                    if(params_to_freeze == []):
                        morph_dict[key][3] = False
                    elif(key in params_to_freeze):
                        morph_dict[key][3] = False
            source_info_db.loc[source_info_db['source'] == source,'morphology_params'] = [morph_dict]
        return source_info_db
    
    def freeze_spectrum(self,sources_to_freeze: list, source_info_db: pd.DataFrame, params_to_freeze = []):
        for source in sources_to_freeze:
            spectrum_dict = source_info_db.loc[source_info_db['source'] == source,'spectrum_params'].values[0]
            for key in spectrum_dict.keys():
                print(key)
                if(not spectrum_dict[key][3] is None):
                    if(params_to_freeze == []):
                        spectrum_dict[key][3] = False
                        print(f'freezing {key}')
                    elif(key in params_to_freeze):
                        spectrum_dict[key][3] = False
                        print(f'freezing {key} 2')    
            source_info_db.loc[source_info_db['source'] == source,'spectrum_params'] = [spectrum_dict]
        return source_info_db
    
    def free_morphology(self,sources_to_free: list, source_info_db: pd.DataFrame, params_to_free = []):
        if(not params_to_free == []):
            for source in sources_to_free:
                morph_dict = source_info_db.loc[source_info_db['source'] == source,'morphology_params'].values[0]
                for key in params_to_free:
                    morph_dict[key][3] = True
                source_info_db.loc[source_info_db['source'] == source,'morphology_params'] = [morph_dict]
        else:
            for source in sources_to_free:
                
                morph_dict = source_info_db.loc[source_info_db['source'] == source,'morphology_params'].values[0]
                default_dict = source_types_db.loc[source_types_db['model'] == source_info_db.loc[source_info_db['source'] == source,'morphology_type'].values[0],'default_param_values'].values[0]
                for key in morph_dict.keys():
                    if(default_dict[key][3] == True):
                        morph_dict[key][3] = True
                source_info_db.loc[source_info_db['source'] == source,'morphology_params'] = [morph_dict]
        return source_info_db
    
    def free_spectrum(self,sources_to_free: list, source_info_db: pd.DataFrame, params_to_free = []):
        if(not params_to_free == []):
            for source in sources_to_free:
                spectrum_dict = source_info_db.loc[source_info_db['source'] == source,'spectrum_params'].values[0]
                for key in params_to_free:
                    spectrum_dict[key][3] = True
                source_info_db.loc[source_info_db['source'] == source,'spectrum_params'] = [spectrum_dict]
        else:
            for source in sources_to_free:
                
                spectrum_dict = source_info_db.loc[source_info_db['source'] == source,'spectrum_params'].values[0]
                default_dict = spectrum_types_db.loc[spectrum_types_db['model'] == source_info_db.loc[source_info_db['source'] == source,'spectrum_type'].values[0],'default_param_values'].values[0]
                for key in spectrum_dict.keys():
                    if(default_dict[key][3] == True):
                        spectrum_dict[key][3] = True
                source_info_db.loc[source_info_db['source'] == source,'spectrum_params'] = [spectrum_dict]
        return source_info_db
                    
def main():
    testing_pipeline_obj = AutomatedLikelihoodPipelineSearch('/lustre/hawcz01/scratch/userspace/sgroetsch/PixiALPS/config.yml')
    # testing_pipeline_obj = AutomatedLikelihoodPipelineSearch('/lustre/hawcz01/scratch/userspace/sgroetsch/PixiALPS/config_full_freeze_default.yml')
    print(testing_pipeline_obj._run_multiple_commands([['ls'],['echo', '\'hello world\'']]))
    test_db_1 = testing_pipeline_obj._load_model_from_valid_file('/lustre/hawcz01/scratch/userspace/sgroetsch/PixiALPS/fresh_start_test_full_freeze/FitResults/fresh_crab_step_2/fresh_crab_modelFit.yml')
    test_db_2 = testing_pipeline_obj._load_model_from_valid_file('/lustre/hawcz01/scratch/userspace/sgroetsch/PixiALPS/fresh_start_test_index_freeze/FitResults/fresh_crab_step_2/fresh_crab_modelFit.yml')
    print(test_db_1)
    testing_pipeline_obj.run_alt_hypothesis('/lustre/hawcz01/scratch/userspace/sgroetsch/PixiALPS/fresh_start_test_full_freeze_default_no_err/FitResults/fresh_crab_step_8/fresh_crab_modelFit.yml',['Gaussian_on_sphere'],'Extension',sources_to_skip=['Source_2'])
    # print(testing_pipeline_obj.run_multi_fit([test_db_1,test_db_2],'multi_test',compute_err=False,compute_TS=False))
    # testing_pipeline_obj._run_point_source_adding_phase(compute_err=False)
    # testing_pipeline_obj = AutomatedLikelihoodPipelineSearch('/lustre/hawcz01/scratch/userspace/sgroetsch/PixiALPS/config_full_freeze_sloppy.yml')
    # testing_pipeline_obj._run_point_source_adding_phase(use_default_TS_calc=False,compute_err=False)
    # testing_pipeline_obj = AutomatedLikelihoodPipelineSearch('/lustre/hawcz01/scratch/userspace/sgroetsch/PixiALPS/config_index_freeze.yml')
    # testing_pipeline_obj._run_point_source_adding_phase(spectrum_params_to_freeze=['index'])
    # testing_pipeline_obj = AutomatedLikelihoodPipelineSearch('/lustre/hawcz01/scratch/userspace/sgroetsch/PixiALPS/config_no_freeze.yml')
    # testing_pipeline_obj._run_point_source_adding_phase(spectrum_params_to_freeze=[''])
    


if __name__ == "__main__":
    main()