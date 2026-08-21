"""
ALPS Seeder (Part 2b) — fit-integration layer + SeedingModule wrapper

Subclasses ALPSSeederBase (Part 1, verbatim extraction) and adds the parts
that had to change to run fitting in-process instead of via subprocess:

  - _db_from_model_obj      : rebuild source_info_db from a fitted model (Q2)
  - run_single_fit          : rewired onto ALPSFitAdapter (was subprocess)
  - _after_run_accouting     : reads log-like/AIC from the fit result (was FITS)
  - _find_next_hotpsot       : routed through MapGenerator/get-local-extremum
  - _run_point_source_adding_phase : structurally faithful; PS seed is pluggable
  - run_alt_hypothesis       : extension/spectrum testing on the in-process fit
  - run()                    : SeedingModule interface -> SeedingOutput

The morphology/spectrum *decision logic* (TS thresholds, freeze/free, add
source, ΔTS convergence) is preserved from testalps.py. What changed is only
the fit transport (in-process) and the result-reading (from objects, not disk).
_get_sloppy_TS remains excluded.
"""

import os
import glob
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from seeding.base import SeedingModule, SeedingOutput
from seeding.alps_seeds import ALPSSeederBase, source_types_db, spectrum_types_db
from seeding.alps_fit_adapter import ALPSFitAdapter


class ALPSSeeder(ALPSSeederBase, SeedingModule):
    """ALPS iterative source-finding seeder with in-process fitting.

    ADAPTED (per explicit human instruction): both ALPS's own control flow
    and the in-process fitter now read from ONE unified dot-notation config
    (config.yaml) instead of two separate config schemas. See
    ALPSSeederBase.__init__ docstring and backups_pre_config_unify/ for the
    prior two-config version.

    Parameters:
    -----------
    config : str or object
        Path to the unified pipeline config YAML, or an already-constructed
        config object exposing `.get(key, default)` / `.config_file` (e.g.
        core.config.ConfigManager). Drives both the ALPS control flow
        (thresholds, phases) and the in-process fitter (map_tree, det_res,
        bins, ROI).
    roi_template : str, optional
        ROI template forwarded to threeMLFit. Defaults to `roi.roi_template_path`
        from the config if not given.
    """

    def __init__(self, config, roi_template: str = None):
        # Part 1 __init__ reads the unified config and sets up dirs + self.logger.
        ALPSSeederBase.__init__(self, config)

        self.fit_config_path = self.config_path
        self.roi_template = roi_template if roi_template is not None else self.roi_template_path

        # In-process fit adapter replaces the subprocess fitModel.py calls.
        self.fit_adapter = ALPSFitAdapter(
            config_path=self.fit_config_path,
            logger=self.logger,
            db_from_model_fn=self._db_from_model_obj,
            roi_template=self.roi_template,
        )

    # ------------------------------------------------------------------------
    # Q2: rebuild source_info_db from a fitted model object (no file round-trip)
    # Mirrors _load_from_model_file's walk, but on a live model_obj.
    # ------------------------------------------------------------------------

    def _db_from_model_obj(self, model) -> pd.DataFrame:
        """Rebuild source_info_db from a fitted threeML model object.

        Same extraction logic as _load_from_model_file, applied to the live
        fitted model so fitted parameter values propagate to the next step.
        """
        source_list = list(model.to_dict_with_types().keys())
        model_odict = model.to_dict_with_types()
        source_info_db = pd.DataFrame(columns=['source', 'morphology_type', 'spectrum_type', 'morphology_params', 'spectrum_params'])
        for source in source_list:
            params_to_get = None

            source_type = list(model_odict[source].keys())[0]
            if(source_type == 'position'):
                source_type = 'PointSource'

            try:
                params_to_get = source_types_db.loc[source_types_db['model'] == source_type, 'params_of_interest'].values[0]
            except:
                if('value' == source_type):
                    continue
                print(f'Could not find matching params for \"{source_type}\" this may be a bug or the model you are using is unsupported')

            if(params_to_get != None):
                model_dict = model_odict[source][list(model_odict[source].keys())[0]]

                morph_dict = {}
                spectrum_dict = {}

                for key in model_dict.keys():
                    if(key in params_to_get):
                        try:
                            morph_dict.update({key: [model_dict[key]['value'], model_dict[key]['min_value'], model_dict[key]['max_value'], model_dict[key]['free']]})
                        except:
                            morph_dict.update({key: [model_dict[key]['value'], None, None, None]})

                spectrum_type = list(model_odict[source]['spectrum']['main'].keys())[0]
                try:
                    params_to_get = spectrum_types_db.loc[spectrum_types_db['model'] == spectrum_type, 'params_of_interest'].values[0]
                except:
                    print(f'Could not find matching params for \"{spectrum_type}\" this may be a bug or the model you are using is unsupported')
                spectrum_dict = model_odict[source]['spectrum']['main'][spectrum_type]
                for key in spectrum_dict.keys():
                    if(key in params_to_get):
                        spectrum_dict.update({key: [spectrum_dict[key]['value'], spectrum_dict[key]['min_value'], spectrum_dict[key]['max_value'], spectrum_dict[key]['free']]})
            source_info_db = pd.concat([source_info_db, pd.DataFrame([[source.split('(')[0].strip(), source_type.strip(), spectrum_type.strip(), morph_dict, spectrum_dict]], columns=source_info_db.columns)], ignore_index=True)

        return source_info_db

    # ------------------------------------------------------------------------
    # Fit transport: in-process instead of subprocess.
    # run_single_fit keeps the retry/perturb structure from testalps.py, but
    # each attempt calls the adapter instead of building a pixi command.
    # ------------------------------------------------------------------------

    def run_single_fit(self, source_db: pd.DataFrame, step_name: str, run_local: bool = True, compute_err=True, compute_TS=True, use_default_TS_calc=True):
        """Fit one model in-process; retry with perturbed positions on failure.

        Returns the FitStepResult from the successful attempt. Side effects:
        writes the .model file and (via the adapter) the model/residual .hd5
        maps under the step directory.
        """
        import time
        start_time = time.perf_counter()
        step_dir = os.path.join(self.fit_results_abs_path, step_name)
        if(not os.path.isdir(step_dir)):
            os.mkdir(step_dir)
        model_loc = os.path.join(self.prev_models_abs_path, f'{step_name}.model')
        self.initialize_model_file(model_loc, source_db)

        retries = 0
        last_result = None
        while(retries < self.max_param_retries):
            try:
                last_result = self.fit_adapter.fit(
                    model_file_loc=model_loc,
                    step_dir=step_dir,
                    compute_err=compute_err,
                    make_maps=True,
                )
                self.logger.info(f'Successfully fit after {retries + 1} attempt(s)')
                break
            except Exception as e:
                self.logger.warning(f'Fit attempt {retries + 1} failed: {e}')
                retries += 1
                source_db = self._perturb_free_morphology_params(self.target_sources, source_db)
                self.initialize_model_file(model_loc, source_db)

        end_time = time.perf_counter()
        self.logger.info(f'Fit {step_name} completed in {round((end_time - start_time) / 60.0, 2)} minutes')
        return last_result

    # ------------------------------------------------------------------------
    # After-run accounting: pull stats from the FitStepResult (not FITS header).
    # Residual/model maps already written by the adapter; freeze accepted sources.
    # ------------------------------------------------------------------------

    def _after_run_accouting(self, step_name: str, target_sources: list, fit_result, morph_params_to_freeze=[], spectrum_params_to_freeze=[], need_residual=False, need_stats=False):
        """Post-fit bookkeeping, reading from the in-process FitStepResult.

        Mirrors testalps.py _after_run_accouting: records log-like/AIC, exposes
        the residual map path for the next hotspot search, reloads the fitted
        source_info_db, and freezes the just-fit sources.
        """
        if(self.make_optional_residual_map or need_residual):
            # Adapter wrote residual_fit.hd5; convert to fits for hotspot search.
            self.current_residual_fits_map = self._residual_hd5_to_fits(fit_result)
        if(need_stats):
            self.current_log_like = fit_result.log_like
            self.current_AIC = fit_result.aic

        source_info_db = fit_result.fitted_db
        self.freeze_morphology(target_sources, source_info_db, params_to_freeze=morph_params_to_freeze)
        self.freeze_spectrum(target_sources, source_info_db, params_to_freeze=spectrum_params_to_freeze)
        return source_info_db

    def _residual_hd5_to_fits(self, fit_result) -> Optional[str]:
        """Convert the step's residual .hd5 to a significance .fits for hotspot search.

        Uses the project's HDF5Handler + MapGenerator (core utilities). Returns
        the path to the residual significance fits, or None if unavailable.
        """
        if fit_result.residual_map_hd5 is None:
            self.logger.warning('No residual .hd5 available to convert')
            return None
        try:
            from core.hdf5_handler import HDF5Handler
            from core.map_tools import MapGenerator
        except Exception:
            self.logger.warning('core HDF5Handler/MapGenerator not importable; skipping residual fits conversion')
            return str(fit_result.residual_map_hd5)

        results_dir = fit_result.step_dir / 'results'
        # Convert hd5 -> per-bin fits
        HDF5Handler.convert_hd5_to_fits(str(results_dir), 'residual_fit.hd5', 'residual', logger=self.logger)
        # Build significance map from per-bin fits
        bin_list = self.bin_list.split(' ') if isinstance(self.bin_list, str) else self.bin_list
        fits_mapping = MapGenerator.find_fits_files_by_bins(str(results_dir), bin_list, logger=self.logger)
        if not fits_mapping:
            return None
        out_fits = str(results_dir / 'residual.fits')
        created = MapGenerator.create_healpix_map(
            input_fits_files=list(fits_mapping.values()),
            energy_bins=list(fits_mapping.keys()),
            detector_response=self.det_res,
            ra_center=float(self.coord_1),
            dec_center=float(self.coord_2),
            roi_x=float(self.roi_radius),
            roi_y=float(self.roi_radius),
            output_file=out_fits,
            logger=self.logger,
            pixi_manifest_path=self.pixi_path,
        )
        if created is None:
            self.logger.warning(f'HealpixSigFluxMap did not produce {out_fits}')
            return None
        return out_fits

    # ------------------------------------------------------------------------
    # Hotspot finding: still uses the aerie extremum finder, but wrapped so the
    # subprocess concern is isolated to one method.
    # ------------------------------------------------------------------------

    def _find_next_hotpsot(self, path_to_map_to_search: str):
        """Find the brightest residual pixel (ra, dec) via aerie extremum app.

        Kept as a subprocess call to aerie-apps-get-local-extremum (there is no
        in-process equivalent); only the command construction is preserved.
        """
        import subprocess
        if not path_to_map_to_search or not os.path.exists(path_to_map_to_search):
            raise RuntimeError(f'_find_next_hotpsot: no residual map to search ({path_to_map_to_search!r}); see _residual_hd5_to_fits log output above for the underlying failure')
        cmd = list(map(str, ['pixi', 'run', '-e', 'threeml', '--manifest-path', self.pixi_path, 'aerie-apps-get-local-extremum', '--ra', self.coord_1, '--dec', self.coord_2, '--windowRadius', self.roi_radius * 2, '--input', path_to_map_to_search]))
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        terminal_out = proc.stdout
        hotspot_ra, hotspot_dec = re.findall(r'\d+\.\d{3}', terminal_out.split("\n")[-2])
        return (float(hotspot_ra), float(hotspot_dec))

    # ------------------------------------------------------------------------
    # Point-source adding phase. Pluggable seed source: ALPS-native hotspot loop
    # (default) or DRIPS filtered_df converted to point sources.
    # ------------------------------------------------------------------------

    def _seed_db_from_drips(self, filtered_df: pd.DataFrame) -> pd.DataFrame:
        """Convert a DRIPS filtered_df into an ALPS source_info_db of point sources.

        Each DRIPS row (ra, dec) becomes a PointSource/Powerlaw entry with the
        ALPS point-source coordinate range applied as bounds.
        """
        db = pd.DataFrame(columns=['source', 'morphology_type', 'spectrum_type', 'morphology_params', 'spectrum_params'])
        for i, row in filtered_df.reset_index(drop=True).iterrows():
            ra = float(row['ra'])
            dec = float(row['dec'])
            db = self.add_model_to_source_db(
                f'Source_{i + 1}', 'PointSource', 'Powerlaw', db,
                {"ra": [ra, ra - self.point_source_coord_range, ra + self.point_source_coord_range, True],
                 "dec": [dec, dec - self.point_source_coord_range, dec + self.point_source_coord_range, True]},
            )
        return db

    def _run_point_source_adding_phase(self, drips_filtered_df: pd.DataFrame = None, morph_params_to_freeze=[], spectrum_params_to_freeze=[], compute_err=True, compute_TS=True, use_default_TS_calc=True):
        """Iteratively add point sources until ΔTS falls below threshold.

        Faithful to testalps.py's loop structure. Two seed modes:
          - drips_filtered_df is None : ALPS-native — find hotspot, add PS, fit,
            repeat while last_delta_TS > minimum_point_source_TS.
          - drips_filtered_df given   : seed all DRIPS point sources up front,
            fit once, then continue the ALPS residual loop for anything missed.
        """
        if drips_filtered_df is not None:
            # DRIPS-seeded start: build the whole PS db, fit once.
            self.source_info_db = self._seed_db_from_drips(drips_filtered_df)
            self.target_sources = list(self.source_info_db['source'].array)
            first_result = self.run_single_fit(self.source_info_db, f'{self.fit_name}_step_1', compute_err=compute_err, compute_TS=compute_TS, use_default_TS_calc=use_default_TS_calc)
            self.source_info_db = self._after_run_accouting(f'{self.fit_name}_step_1', self.target_sources, first_result, morph_params_to_freeze=morph_params_to_freeze, spectrum_params_to_freeze=spectrum_params_to_freeze, need_residual=True, need_stats=True)
        else:
            # ALPS-native start: seed one source at the brightest hotspot.
            if(self.make_raw_data_map):
                # Note: raw data map creation goes through the same map utility;
                # here we assume the raw significance map is already available at
                # raw_sig_map_abs_path/map.fits or via config significance map.
                raw_map = os.path.join(self.raw_sig_map_abs_path, 'map.fits')
                if os.path.exists(raw_map):
                    next_ra, next_dec = self._find_next_hotpsot(raw_map)
                else:
                    next_ra, next_dec = (self.coord_1, self.coord_2)
            else:
                next_ra, next_dec = (self.coord_1, self.coord_2)
            self.source_info_db = self.add_model_to_source_db('Source_1', 'PointSource', 'Powerlaw', self.source_info_db, {"ra": [next_ra, next_ra - self.point_source_coord_range, next_ra + self.point_source_coord_range, True], "dec": [next_dec, next_dec - self.point_source_coord_range, next_dec + self.point_source_coord_range, True]})
            self.target_sources = ['Source_1']
            first_result = self.run_single_fit(self.source_info_db, f'{self.fit_name}_step_1', compute_err=compute_err, compute_TS=compute_TS, use_default_TS_calc=use_default_TS_calc)
            self.source_info_db = self._after_run_accouting(f'{self.fit_name}_step_1', self.target_sources, first_result, morph_params_to_freeze=morph_params_to_freeze, spectrum_params_to_freeze=spectrum_params_to_freeze, need_residual=True, need_stats=True)

        # Iterative residual loop (identical convergence rule to testalps.py)
        self.last_delta_TS = 1000000
        i = 2
        while(self.last_delta_TS > self.minimum_point_source_TS):
            next_ra, next_dec = self._find_next_hotpsot(self.current_residual_fits_map)
            list_of_used_names = self.source_info_db['source'].array
            if(f'Source_{i}' in list_of_used_names):
                self.source_info_db = self.add_model_to_source_db(f'Source_{i}_ALPS', 'PointSource', 'Powerlaw', self.source_info_db, {"ra": [next_ra, next_ra - self.point_source_coord_range, next_ra + self.point_source_coord_range, True], "dec": [next_dec, next_dec - self.point_source_coord_range, next_dec + self.point_source_coord_range, True]})
                self.target_sources = [f'Source_{i}_ALPS']
            else:
                self.source_info_db = self.add_model_to_source_db(f'Source_{i}', 'PointSource', 'Powerlaw', self.source_info_db, {"ra": [next_ra, next_ra - self.point_source_coord_range, next_ra + self.point_source_coord_range, True], "dec": [next_dec, next_dec - self.point_source_coord_range, next_dec + self.point_source_coord_range, True]})
                self.target_sources = [f'Source_{i}']
            result = self.run_single_fit(self.source_info_db, f'{self.fit_name}_step_{i}', compute_err=compute_err, compute_TS=compute_TS, use_default_TS_calc=use_default_TS_calc)
            self.prev_log_like = self.current_log_like
            self.prev_AIC = self.current_AIC
            self.source_info_db = self._after_run_accouting(f'{self.fit_name}_step_{i}', self.target_sources, result, morph_params_to_freeze=morph_params_to_freeze, spectrum_params_to_freeze=spectrum_params_to_freeze, need_residual=True, need_stats=True)
            self.last_delta_TS = 2 * (self.prev_log_like - self.current_log_like)
            i += 1

    # ------------------------------------------------------------------------
    # Alternate-hypothesis testing (extension / spectrum). Faithful to the
    # ALPS decision logic; fitting is in-process.
    # ------------------------------------------------------------------------

    def run_alt_hypothesis(self, accepted_step_source_db: pd.DataFrame, alt_hypothesis_models: list, alt_test_type: str, sources_to_skip=[], morph_params_to_freeze=[], spectrum_params_to_freeze=[], compute_err=True):
        """Test alternate morphologies/spectra source-by-source against a baseline.

        For each eligible source, swap in each alternate model, refit in-process,
        and accept the alternate when the improvement exceeds the configured TS
        threshold (morphology or spectrum). Returns the updated source_info_db.

        alt_test_type : 'Extension' (morphology) or 'Spectrum'.
        """
        source_db = accepted_step_source_db.copy()
        source_db['Tested'] = False

        # Honor trust flags exactly as testalps.py did.
        if(self.trust_all_alterante_morphologies):
            for source in source_db['source'].array:
                morphology = source_db.loc[source_db['source'] == source, 'morphology_type'].values[0]
                if(not morphology == 'PointSource'):
                    source_db.loc[source_db['source'] == source, 'Tested'] = True
        if(self.trust_all_alternate_spectra):
            for source in source_db['source'].array:
                spectrum = source_db.loc[source_db['source'] == source, 'spectrum_type'].values[0]
                if(not spectrum == 'Powerlaw'):
                    source_db.loc[source_db['source'] == source, 'Tested'] = True
        if(self.trusted_source_list is not None):
            for source in self.trusted_source_list:
                source_db.loc[source_db['source'] == source, 'Tested'] = True

        if alt_test_type == 'Extension':
            ts_threshold = self.minimum_morphology_TS_improvement
        else:
            ts_threshold = self.minimum_spectral_TS_improvement

        # Baseline fit for reference likelihood.
        baseline_db = source_db.drop(columns=['Tested'])
        baseline_result = self.run_single_fit(baseline_db, f'{self.fit_name}_{alt_test_type}_baseline', compute_err=False, compute_TS=False)
        if baseline_result is None:
            # Known failure mode (needs human review, see TASKS.md Task 4): sources
            # accepted by the point-source-adding phase have ALL morphology+spectrum
            # params frozen by _after_run_accouting, so this baseline re-fit can run
            # on a model with zero free parameters, which threeML's results-table
            # construction does not handle. Skip alt-hypothesis testing rather than
            # crash the whole run() with an AttributeError on None.
            self.logger.error(f'{alt_test_type} baseline fit failed (see TASKS.md Task 4); skipping {alt_test_type} testing for this call')
            return accepted_step_source_db
        baseline_log_like = baseline_result.log_like
        accepted_db = baseline_result.fitted_db

        step_counter = 0
        for source in list(source_db['source'].array):
            if source in sources_to_skip:
                continue
            if bool(source_db.loc[source_db['source'] == source, 'Tested'].values[0]):
                continue

            best_log_like = baseline_log_like
            best_db = accepted_db
            for alt_model in alt_hypothesis_models:
                step_counter += 1
                trial_db = accepted_db.copy()
                if alt_test_type == 'Extension':
                    trial_db = self._swap_morphology(trial_db, source, alt_model)
                else:
                    trial_db = self._swap_spectrum(trial_db, source, alt_model)

                self.target_sources = [source]
                trial_result = self.run_single_fit(trial_db, f'{self.fit_name}_{alt_test_type}_{source}_{step_counter}', compute_err=compute_err, compute_TS=False)
                delta_TS = 2 * (best_log_like - trial_result.log_like)
                self.logger.info(f'{alt_test_type} test {source} -> {alt_model}: delta_TS={delta_TS:.2f} (threshold {ts_threshold})')
                if delta_TS > ts_threshold:
                    best_log_like = trial_result.log_like
                    best_db = trial_result.fitted_db
                    self.logger.info(f'Accepted alternate {alt_model} for {source}')

            accepted_db = best_db
            baseline_log_like = best_log_like

        return accepted_db

    def _swap_morphology(self, source_db: pd.DataFrame, source: str, new_morphology: str) -> pd.DataFrame:
        """Replace a source's morphology with a new type using default params.

        Position seed is carried over from the source's current location.
        """
        cur_morph = source_db.loc[source_db['source'] == source, 'morphology_params'].values[0]
        # carry position seed
        if 'ra' in cur_morph:
            lon, lat = cur_morph['ra'][0], cur_morph['dec'][0]
        elif 'lon0' in cur_morph:
            lon, lat = cur_morph['lon0'][0], cur_morph['lat0'][0]
        else:
            lon, lat = self.coord_1, self.coord_2
        new_params = {k: list(v) for k, v in source_types_db.loc[source_types_db['model'] == new_morphology, 'default_param_values'].values[0].items()}
        if 'lon0' in new_params:
            new_params['lon0'][0] = lon
            new_params['lat0'][0] = lat
        spectrum_type = source_db.loc[source_db['source'] == source, 'spectrum_type'].values[0]
        spectrum_params = source_db.loc[source_db['source'] == source, 'spectrum_params'].values[0]
        source_db = source_db[source_db['source'] != source]
        source_db = self.add_model_to_source_db(source, new_morphology, spectrum_type, source_db, new_params, spectrum_params)
        return source_db

    def _swap_spectrum(self, source_db: pd.DataFrame, source: str, new_spectrum: str) -> pd.DataFrame:
        """Replace a source's spectrum with a new type using default params."""
        morph_type = source_db.loc[source_db['source'] == source, 'morphology_type'].values[0]
        morph_params = source_db.loc[source_db['source'] == source, 'morphology_params'].values[0]
        new_spec = {k: list(v) for k, v in spectrum_types_db.loc[spectrum_types_db['model'] == new_spectrum, 'default_param_values'].values[0].items()}
        source_db = source_db[source_db['source'] != source]
        source_db = self.add_model_to_source_db(source, morph_type, new_spectrum, source_db, morph_params, new_spec)
        return source_db

    # ------------------------------------------------------------------------
    # SeedingModule interface
    # ------------------------------------------------------------------------

    def run(self, drips_filtered_df: pd.DataFrame = None) -> SeedingOutput:
        """Run ALPS seeding and return a standardized SeedingOutput.

        Parameters:
        -----------
        drips_filtered_df : pd.DataFrame, optional
            If provided, ALPS is seeded from DRIPS point sources instead of the
            native hotspot search. The extension/spectrum testing phases run the
            same either way.

        Returns:
        --------
        SeedingOutput
        """
        self.log_seeding_start('ALPS')

        # Phase 1: point-source adding (pluggable seed)
        if self.perform_point_source_adding_phase:
            self._run_point_source_adding_phase(drips_filtered_df=drips_filtered_df)

        # Phase 2: morphology (extension) testing
        if self.perform_morpholgy_testing_phase and self.alternate_morpholgy_model_list is not None:
            alt_morph = self.alternate_morpholgy_model_list if isinstance(self.alternate_morpholgy_model_list, list) else [self.alternate_morpholgy_model_list]
            self.source_info_db = self.run_alt_hypothesis(self.source_info_db, alt_morph, 'Extension')

        # Phase 3: spectrum testing
        if self.perform_spectrum_testing_phase and self.alternate_spectrum_model_list is not None:
            alt_spec = self.alternate_spectrum_model_list if isinstance(self.alternate_spectrum_model_list, list) else [self.alternate_spectrum_model_list]
            self.source_info_db = self.run_alt_hypothesis(self.source_info_db, alt_spec, 'Spectrum')

        num_sources = len(self.source_info_db)
        baseline_like = getattr(self, 'current_log_like', float('nan'))
        model_path = Path(self.prev_models_abs_path) / f'{self.fit_name}_final.model'
        self.initialize_model_file(str(model_path), self.source_info_db)

        residual_path = Path(getattr(self, 'current_residual_fits_map', self.raw_sig_map_abs_path))

        output = SeedingOutput(
            source_info_db=self.source_info_db,
            baseline_model_path=model_path,
            baseline_likelihood=float(baseline_like),
            baseline_params={},
            ts_values={},
            residual_map_path=residual_path,
            checkpoint_data={
                'method': 'ALPS',
                'num_sources': num_sources,
                'seeded_from': 'DRIPS' if drips_filtered_df is not None else 'ALPS_hotspots',
            },
            num_sources=num_sources,
            num_iterations=num_sources,
            method='ALPS',
        )

        self.log_seeding_complete('ALPS', num_sources, num_sources, output.baseline_likelihood)
        return output
