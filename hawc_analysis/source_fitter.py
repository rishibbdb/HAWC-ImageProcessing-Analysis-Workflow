"""
Source fitter: DRIPS-seeded joint fit + extension/spectrum testing + refit.

"""

from pathlib import Path
from typing import List
import os
from astropy.coordinates import SkyCoord
import pandas as pd
import astromodels
from fit_runner import FitRunner, FitResult
from model_generator import ModelGenerator
from seeding.base import SeedingOutput
from core.hdf5_handler import HDF5Handler
from core.map_tools import MapGenerator
from pipeline_helpers import load_hawc_data, find_peak, make_plots

def _as_list(value) -> List[str]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]

def check_hotspots(path, fit_output, config, logger):
    mapname = path
    l = config.get('coordinates.l')
    b = config.get('coordinates.b')
    x_length = config.get('coordinates.roi_x')
    y_length = config.get('coordinates.roi_y')
    coord_sys = config.get('coordinates.coord_sys')
    array, _, wcs, _, _, pixel_size = load_hawc_data( mapname, l, b, x_length, y_length, coord_sys )
    max_value = find_peak(array, wcs)
    print(f"path.parent: {path.parent.parent}")
    if max_value > 5:
        name = []
        ra = []
        dec = []
        ext = []
        for source in fit_output.model.sources:
            logger.info(f"Source {source}")
            if source == 'URM':
                continue
            try:
                logger.info(f"Source position: RA={fit_output.model[source].position.ra.value}, Dec={fit_output.model[source].position.dec.value}")
                name.append(source)
                ra.append(fit_output.model[source].position.ra.value)
                dec.append(fit_output.model[source].position.dec.value)
                ext.append(0.01)
            except:
                logger.info(f"Source position: RA={fit_output.model[source].spatial_shape.lon0.value}, Dec={fit_output.model[source].spatial_shape.lat0.value}")
                name.append(source)
                ra.append(fit_output.model[source].spatial_shape.lon0.value)
                dec.append(fit_output.model[source].spatial_shape.lat0.value)
                ext.append(fit_output.model[source].spatial_shape.sigma.value)
        df = {'Name': name, 'ra': ra, 'dec': dec, 'ext': ext}
        make_plots(array, wcs, pixel_size, coord_sys, save_dir = str(path.parent), cmap='ult', hotspots=df)
    logger.info(f"Max value in residual map: {max_value}")

def _build_fit_maps(config, logger, directory_manager, path, name, checkpoint=None):
    """Create significance maps."""

    ra = config.get('coordinates.ra')
    dec = config.get('coordinates.dec')
    if ra is None or dec is None:
        l = config.get('coordinates.l')
        b = config.get('coordinates.b')
        skycoord = SkyCoord(l, b, frame='galactic', unit='deg')
        ra = skycoord.icrs.ra.deg
        dec = skycoord.icrs.dec.deg
        logger.info(f"Converted galactic coordinates (l={l}, b={b}) to equatorial (RA={ra}, Dec={dec})")
        config.set('coordinates.ra', ra)
        config.set('coordinates.dec', dec) 
    
    
    created_files = HDF5Handler.convert_hd5_to_fits(
        input_dir=str(path),
        hd5_filename='residual_fit.hd5',
        output_basename='residual',
        logger=logger
    )
    print(f"Created FITS files: {created_files}")
    bins = config.get('fitting.bins')
    detector_response = config.get('coordinates.detector_response')
    print(f"Path: {path}")

    if os.path.exists(str(path / 'fits' / f'{name}.fits')):
        logger.info(f"Output FITS file {path / 'fits' / f'{name}.fits'} already exists, skipping map generation")
        return path / 'fits' / f'{name}.fits' 
    output_path = MapGenerator.create_healpix_map(
        input_fits_files=list(created_files),
        energy_bins=list(bins),
        detector_response=detector_response,
        ra_center=float(config.get('coordinates.ra')),
        dec_center=float(config.get('coordinates.dec')),
        roi_x=float(config.get('coordinates.roi_x', 4.0)*2.5),
        roi_y=float(config.get('coordinates.roi_y', 4.0)*2),
        output_file=str(path / 'fits' / f'{name}.fits'),
        logger=logger,
        pixi_manifest_path=config.get('alps.pixi_aerie_folder'),
    )

    return output_path

def _model_summary_df(model) -> pd.DataFrame:
    """Descriptive (not round-trippable) summary of a live model's sources,
    for the SeedingOutput.source_info_db contract."""
    rows = []
    for name, source in model.sources.items():
        if hasattr(source, 'position'):
            ra, dec = source.position.ra.value, source.position.dec.value
        else:
            ra, dec = source.spatial_shape.lon0.value, source.spatial_shape.lat0.value
        rows.append({
            'source': name,
            'ra': ra,
            'dec': dec,
            'spatial_model': type(source.spatial_shape).__name__ if hasattr(source, 'spatial_shape') else 'PointSource',
            'spectral_model': type(source.spectrum.main.shape).__name__,
        })
    return pd.DataFrame(rows, columns=['source', 'ra', 'dec', 'spatial_model', 'spectral_model'])

def _build_fit_maps(config, logger, directory_manager, path, name, checkpoint=None):
    """Create significance maps."""

    ra = config.get('coordinates.ra')
    dec = config.get('coordinates.dec')
    if ra is None or dec is None:
        l = config.get('coordinates.l')
        b = config.get('coordinates.b')
        skycoord = SkyCoord(l, b, frame='galactic', unit='deg')
        ra = skycoord.icrs.ra.deg
        dec = skycoord.icrs.dec.deg
        logger.info(f"Converted galactic coordinates (l={l}, b={b}) to equatorial (RA={ra}, Dec={dec})")
        config.set('coordinates.ra', ra)
        config.set('coordinates.dec', dec) 
    
    
    created_files = HDF5Handler.convert_hd5_to_fits(
        input_dir=str(path),
        hd5_filename='residual_fit.hd5',
        output_basename='residual',
        logger=logger
    )
    print(f"Created FITS files: {created_files}")
    bins = config.get('fitting.bins')
    detector_response = config.get('coordinates.detector_response')
    print(f"Path: {path}")

    if os.path.exists(str(path / 'fits' / f'{name}.fits')):
        logger.info(f"Output FITS file {path / 'fits' / f'{name}.fits'} already exists, skipping map generation")
        return path / 'fits' / f'{name}.fits' 
    output_path = MapGenerator.create_healpix_map(
        input_fits_files=list(created_files),
        energy_bins=list(bins),
        detector_response=detector_response,
        ra_center=float(config.get('coordinates.ra')),
        dec_center=float(config.get('coordinates.dec')),
        roi_x=float(config.get('coordinates.roi_x', 4.0)*2.5),
        roi_y=float(config.get('coordinates.roi_y', 4.0)*2),
        output_file=str(path / 'fits' / f'{name}.fits'),
        logger=logger,
        pixi_manifest_path=config.get('alps.pixi_aerie_folder'),
    )

    return output_path

def run_joint_fit(drip_model_path: Path, config, logger, directory_manager) -> FitResult:
    """One in-process joint fit of DRIPS's full seed model, all sources free.

    This is the replacement for ALPS's entire iterative point-source-adding
    loop: no hotspot search, no per-source add/freeze cycle. DRIPS's blob
    detection already found every excess.
    """
    runner = FitRunner(
        config_path=str(config.config_file),
        logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )
    step_dir = directory_manager.get_step_results_dir('Step1-JointFit')
    logger.info(f'Running joint fit on DRIPS seed model ({drip_model_path}) in {step_dir}')
    result = runner.fit(
        model_file=str(drip_model_path),
        step_dir=str(step_dir),
        compute_err=False,
        compute_TS=True,
        make_maps=True,
    )
    resmap = _build_fit_maps(config, logger, directory_manager, result.step_dir, 'residual')
    check_hotspots(resmap, result, config, logger)

    return result

def _check_and_remove_low_ts(trial_result: FitResult, source_names_to_protect: List[str],
                              ts_threshold: float, step_label: str, config, logger,
                              directory_manager) -> FitResult:
    ts_by_source = trial_result.ts
    if not isinstance(ts_by_source, dict):
        logger.warning('trial_result.ts is not a dict; skipping low-TS check')
        return None

    low_ts_sources = [
        n for n, ts in ts_by_source.items()
        if n != 'URM' and n not in source_names_to_protect and ts < ts_threshold
    ]
    if not low_ts_sources:
        return None

    logger.info(f'Sources dropped below TS threshold {ts_threshold} during {step_label}: {low_ts_sources}; removing and refitting')
    pruned_model = ModelGenerator.remove_sources(trial_result.model, low_ts_sources, logger=logger)
    remaining_names = list(pruned_model.sources.keys())

    ModelGenerator.set_free(pruned_model, remaining_names, kind='spatial', free=True, free_diffuse=True, logger=logger)
    ModelGenerator.set_free(pruned_model, remaining_names, kind='spectral', free=True, free_diffuse=True, logger=logger)

    prune_step_name = f'{step_label}-Pruned_{"_".join(low_ts_sources)}'
    prune_step_dir = directory_manager.get_step_results_dir(prune_step_name)
    yml_path = f'{prune_step_dir}/curModel.yml'
    model_path = f'{prune_step_dir}/curModel.model'
    pruned_model.save(yml_path, overwrite=True)
    ModelGenerator.write_model_file_from_yaml(yml_path, model_path, logger=logger)

    runner = FitRunner(
        config_path=str(config.config_file), logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )
    return runner.fit(
        model_file=str(model_path),
        step_dir=str(prune_step_dir),
        compute_err=config.get('error_and_TS.error_point', True),
        compute_TS=True,
        make_maps=True,
    )

def run_extension_test(fit_result: FitResult, config, logger, directory_manager) -> FitResult:
    """Test alternate spatial models per source; accept if TS improvement
    exceeds likelihood_thresholds.extension_test. Other sources are frozen
    during each trial to isolate the tested source's effect; nothing is
    permanently frozen going into the next phase -- run_final_refit unfreezes
    everything.
    """
    alt_models = _as_list(config.get('fitting.alternate_spatial_models'))
    if not alt_models:
        logger.info('No fitting.alternate_spatial_models configured; skipping extension test')
        return fit_result
    free_dbe = config.get('diffuse.free_diffuse_norm', False)
    logger.info(f"Diffuse background normalization status during spectrum test: {free_dbe}")
    source_ts_threshold = config.get('likelihood_thresholds.point_source_detection', 16)
    extension_ts_threshold = config.get('likelihood_thresholds.extension_test', 16)
    coord_range = config.get('fitting.extended_source_coord_range', 1.0)

    # force_low_ts_source = config.get('testing.force_low_ts_source', None)
    force_low_ts_source = 'Source3' 
    # force_low_ts_source = True
    if force_low_ts_source:
        logger.info(
            f'TESTING OVERRIDE ACTIVE: forcing TS for source {force_low_ts_source!r} '
            f'below threshold {source_ts_threshold} after every trial fit. '
            f'Unset testing.force_low_ts_source for production runs.'
        )
    runner = FitRunner(
        config_path=str(config.config_file),
        logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )

    model = fit_result.model
    baseline_log_like = fit_result.log_like
    source_names = list(model.sources.keys())
    i=0
    for source_name in source_names:
        i += 1
        if source_name == 'URM':
            logger.info(f'Skipping extension test for {source_name} (URM source)')
            continue

        other_sources = [n for n in model.sources.keys() if n != source_name]
        best_log_like = baseline_log_like
        logger.info(f'Current best log-likelihood: {best_log_like:.3f}')
        best_model = model

        for alt_shape in alt_models:

            source = model.sources[source_name]
            current_spatial_model = list(source._children.keys())[0]
            if current_spatial_model == alt_shape:
                logger.info(f"Source {source_name} already has spatial shape {alt_shape}, skipping swap")
                continue

            trial_model = ModelGenerator.swap_spatial_shape(
                model, source_name, alt_shape, coord_range=coord_range, logger=logger,
            )
            ModelGenerator.set_free(trial_model, other_sources, kind='spatial', free=True, free_diffuse=free_dbe, param_names=['sigma', 'e', 'theta'], logger=logger)
            ModelGenerator.set_free(trial_model, other_sources, kind='spectral', free=True, free_diffuse=free_dbe, param_names=['K', 'index', 'alpha', 'beta'], logger=logger)

            step_name = f'Step2-{source_name}-Extension-{alt_shape}'
            step_dir = directory_manager.get_step_results_dir(step_name)
            trial_model.save("{1}/{0}.yml".format('curModel', step_dir), overwrite=True)
            ModelGenerator.write_model_file_from_yaml("{1}/{0}.yml".format('curModel', step_dir), "{1}/{0}.model".format('curModel', step_dir), logger=logger)
            model_file = "{1}/{0}.model".format('curModel', step_dir)
            trial_result = runner.fit(
                model_file=str(model_file),
                step_dir=str(step_dir),
                compute_err=config.get('error_and_TS.error_extension', True),
                make_maps=True,
            )

            if force_low_ts_source and isinstance(trial_result.ts, dict) and force_low_ts_source in trial_result.ts:
                real_ts = trial_result.ts[force_low_ts_source]
                trial_result.ts[force_low_ts_source] = source_ts_threshold - 1.0
                logger.warning(
                    f'TESTING OVERRIDE: {force_low_ts_source} real TS={real_ts:.2f} '
                    f'-> forced to {trial_result.ts[force_low_ts_source]:.2f}'
                )

            pruned_result = _check_and_remove_low_ts(
                trial_result, source_names_to_protect=[source_name],
                ts_threshold=source_ts_threshold, step_label=f'{step_name}',
                config=config, logger=logger, directory_manager=directory_manager,
            )
            if pruned_result is not None:
                trial_result = pruned_result
                best_log_like = trial_result.log_like
                best_model = trial_result.model
                fit_result = trial_result
                model = best_model
                baseline_log_like = best_log_like
                low_ts_dropped = set(source_names) - set(model.sources.keys())
                source_names = [n for n in source_names if n not in low_ts_dropped]
                other_sources = [n for n in other_sources if n not in low_ts_dropped]

                if force_low_ts_source in low_ts_dropped:
                    force_low_ts_source = None
                continue

            delta_ts = 2 * (best_log_like - trial_result.log_like)
            logger.info(f'Extension test {source_name} -> {alt_shape}: delta_TS={delta_ts:.2f} (threshold {extension_ts_threshold})')
            if delta_ts > extension_ts_threshold:
                best_log_like = trial_result.log_like
                best_model = trial_result.model
                logger.info(f'Accepted alternate spatial model {alt_shape} for {source_name}')
                resmap = _build_fit_maps(config, logger, directory_manager, trial_result.step_dir, 'residual')
                fit_result = trial_result
        model = best_model
        baseline_log_like = best_log_like

    return FitResult(
        model=model, log_like=baseline_log_like, aic=fit_result.aic,
        model_map_path=fit_result.model_map_path, residual_map_path=fit_result.residual_map_path,
        step_dir=fit_result.step_dir, fitter=fit_result.fitter,
    )

# def run_extension_test(fit_result: FitResult, config, logger, directory_manager) -> FitResult:
#     """Test alternate spatial models per source; accept if TS improvement
#     exceeds likelihood_thresholds.extension_test. Other sources are frozen
#     during each trial to isolate the tested source's effect; nothing is
#     permanently frozen going into the next phase -- run_final_refit unfreezes
#     everything.
#     """
#     alt_models = _as_list(config.get('fitting.alternate_spatial_models'))
#     if not alt_models:
#         logger.info('No fitting.alternate_spatial_models configured; skipping extension test')
#         return fit_result
#     free_dbe = config.get('diffuse.free_diffuse_norm', False)
#     logger.info(f"Diffuse background normalization status during spectrum test: {free_dbe}")
#     source_ts_threshold = config.get('likelihood_thresholds.point_source_detection', 16)
#     extension_ts_threshold = config.get('likelihood_thresholds.extension_test', 16)
#     coord_range = config.get('fitting.extended_source_coord_range', 1.0)
#     runner = FitRunner(
#         config_path=str(config.config_file),
#         logger=logger,
#         roi_template=config.get('roi.roi_template_path'),
#     )

#     model = fit_result.model
#     baseline_log_like = fit_result.log_like
#     source_names = list(model.sources.keys())
#     i=0
#     for source_name in source_names:
#         i += 1
#         if source_name == 'URM':
#             logger.info(f'Skipping extension test for {source_name} (URM source)')
#             continue

#         other_sources = [n for n in model.sources.keys() if n != source_name]
#         best_log_like = baseline_log_like
#         logger.info(f'Current best log-likelihood: {best_log_like:.3f}')
#         best_model = model

#         for alt_shape in alt_models:

#             source = model.sources[source_name]
#             current_spatial_model = list(source._children.keys())[0]
#             if current_spatial_model == alt_shape:
#                 logger.info(f"Source {source_name} already has spatial shape {alt_shape}, skipping swap")
#                 continue

#             trial_model = ModelGenerator.swap_spatial_shape(
#                 model, source_name, alt_shape, coord_range=coord_range, logger=logger,
#             )
#             ModelGenerator.set_free(trial_model, other_sources, kind='spatial', free=True, free_diffuse=free_dbe, param_names=['sigma', 'e', 'theta'], logger=logger)
#             ModelGenerator.set_free(trial_model, other_sources, kind='spectral', free=True, free_diffuse=free_dbe, param_names=['K', 'index', 'alpha', 'beta'], logger=logger)

#             step_name = f'Step2-{source_name}-Extension-{alt_shape}'
#             step_dir = directory_manager.get_step_results_dir(step_name)
#             trial_model.save("{1}/{0}.yml".format('curModel', step_dir), overwrite=True)
#             ModelGenerator.write_model_file_from_yaml("{1}/{0}.yml".format('curModel', step_dir), "{1}/{0}.model".format('curModel', step_dir), logger=logger)
#             model_file = "{1}/{0}.model".format('curModel', step_dir)
#             trial_result = runner.fit(
#                 model_file=str(model_file),
#                 step_dir=str(step_dir),
#                 compute_err=config.get('error_and_TS.error_extension', True),
#                 make_maps=True,
#             )
#             pruned_result = _check_and_remove_low_ts(
#                 trial_result, source_names_to_protect=[source_name],
#                 ts_threshold=source_ts_threshold, step_label=f'{step_name}',
#                 config=config, logger=logger, directory_manager=directory_manager,
#             )
#             if pruned_result is not None:
#                 trial_result = pruned_result
#                 best_log_like = trial_result.log_like
#                 best_model = trial_result.model
#                 fit_result = trial_result
#                 model = best_model
#                 baseline_log_like = best_log_like
#                 low_ts_dropped = set(source_names) - set(model.sources.keys())
#                 source_names = [n for n in source_names if n not in low_ts_dropped]
#                 other_sources = [n for n in other_sources if n not in low_ts_dropped]
#                 continue

#             delta_ts = 2 * (best_log_like - trial_result.log_like)
#             logger.info(f'Extension test {source_name} -> {alt_shape}: delta_TS={delta_ts:.2f} (threshold {extension_ts_threshold})')
#             if delta_ts > extension_ts_threshold:
#                 best_log_like = trial_result.log_like
#                 best_model = trial_result.model
#                 logger.info(f'Accepted alternate spatial model {alt_shape} for {source_name}')
#                 resmap = _build_fit_maps(config, logger, directory_manager, trial_result.step_dir, 'residual')
#                 fit_result = trial_result
#         model = best_model
#         baseline_log_like = best_log_like
#         if i >=1:
#                 return FitResult(
#                 model=model, log_like=baseline_log_like, aic=fit_result.aic, ts = fit_result.ts,
#                 model_map_path=fit_result.model_map_path, residual_map_path=fit_result.residual_map_path,
#                 step_dir=fit_result.step_dir, fitter=fit_result.fitter,
#             )


#     return FitResult(
#         model=model, log_like=baseline_log_like, aic=fit_result.aic, ts= fit_result.ts,
#         model_map_path=fit_result.model_map_path, residual_map_path=fit_result.residual_map_path,
#         step_dir=fit_result.step_dir, fitter=fit_result.fitter,
#     )


def run_spectrum_test(fit_result: FitResult, config, logger, directory_manager) -> FitResult:
    """Test alternate spectral models per source; accept if TS improvement
    exceeds likelihood_thresholds.spectrum_test. Same freeze-others-per-trial
    approach as run_extension_test.
    """
    alt_models = _as_list(config.get('fitting.alternate_spectral_models'))
    if not alt_models:
        logger.info('No fitting.alternate_spectral_models configured; skipping spectrum test')
        return fit_result

    threshold = config.get('likelihood_thresholds.spectrum_test', 16)
    runner = FitRunner(
        config_path=str(config.config_file),
        logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )
    free_dbe = config.get('fitting.free_diffuse_norm', False)
    logger.info(f"Diffuse background normalization status during spectrum test: {free_dbe}")
    model = fit_result.model
    baseline_log_like = fit_result.log_like
    source_names = list(model.sources.keys())

    for source_name in source_names:
        other_sources = [n for n in model.sources.keys() if n != source_name]
        best_log_like = baseline_log_like
        best_model = model

        for alt_spectrum in alt_models:
            trial_model = ModelGenerator.swap_spectral_shape(
                model, source_name, alt_spectrum, logger=logger,
            )
            ModelGenerator.set_free(trial_model, other_sources, kind='spatial', free=False, free_diffuse=free_dbe, param_names=['lon0', 'lat0', 'ra', 'dec'], logger=logger)
            ModelGenerator.set_free(trial_model, other_sources, kind='spectral', free=True, free_diffuse=free_dbe, param_names=['piv'], logger=logger)

            step_name = f'Step3-{source_name}-Spectrum-{alt_spectrum}'
            step_dir = directory_manager.get_step_results_dir(step_name)
            trial_model.save("{1}/{0}.yml".format('curModel', step_dir), overwrite=True)
            ModelGenerator.write_model_file_from_yaml("{1}/{0}.yml".format('curModel', step_dir), "{1}/{0}.model".format('curModel', step_dir), logger=logger)
            model_file = "{1}/{0}.model".format('curModel', step_dir)
            trial_result = runner.fit(
                model_file=str(model_file),
                step_dir=str(step_dir),
                compute_err=config.get('error_and_TS.error_spectrum', True),
                compute_TS=True,
                make_maps=False,
            )

            pruned_result = _check_and_remove_low_ts(
                trial_result, source_names_to_protect=[source_name],
                ts_threshold=25, step_label=f'{step_name}',
                config=config, logger=logger, directory_manager=directory_manager,
            )
            if pruned_result is not None:
                trial_result = pruned_result
                best_log_like = trial_result.log_like
                best_model = trial_result.model
                fit_result = trial_result
                model = best_model
                baseline_log_like = best_log_like
                low_ts_dropped = set(source_names) - set(model.sources.keys())
                source_names = [n for n in source_names if n not in low_ts_dropped]
                other_sources = [n for n in other_sources if n not in low_ts_dropped]
                continue

            delta_ts = 2 * (best_log_like - trial_result.log_like)
            logger.info(f'Spectrum test {source_name} -> {alt_spectrum}: delta_TS={delta_ts:.2f} (threshold {threshold})')
            if delta_ts > threshold:
                best_log_like = trial_result.log_like
                best_model = trial_result.model
                logger.info(f'Accepted alternate spectral model {alt_spectrum} for {source_name}')

        model = best_model
        baseline_log_like = best_log_like

    return FitResult(
        model=model, log_like=baseline_log_like, aic=fit_result.aic,
        model_map_path=fit_result.model_map_path, residual_map_path=fit_result.residual_map_path,
        step_dir=fit_result.step_dir, fitter=fit_result.fitter,
    )


def run_final_refit(fit_result: FitResult, config, logger, directory_manager) -> FitResult:
    """Unfreeze every source's parameters and do one more joint fit.

    Genuinely new: neither alpscode.py nor seeding/alps_seeder.py has a
    working final-refit method (perform_final_fitting_phase is read but never
    consumed anywhere in either).
    """
    model = fit_result.model
    all_sources = list(model.sources.keys())
    ModelGenerator.set_free(model, all_sources, kind='spatial', free=True)
    ModelGenerator.set_free(model, all_sources, kind='spectral', free=True)

    step_name = 'Step4-FinalRefit'
    step_dir = directory_manager.get_step_results_dir(step_name)
    model_file = ModelGenerator.write_model_from_live(
        model, str(directory_manager.get_model_file_path(step_name)), logger=logger,
    )
    runner = FitRunner(
        config_path=str(config.config_file),
        logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )
    logger.info('Running final joint refit with all parameters free')
    return runner.fit(
        model_file=str(model_file),
        step_dir=str(step_dir),
        compute_err=config.get('error_and_TS.error_point', True),
        make_maps=True,
    )




def run(drip_model_path, config, logger, directory_manager, checkpoint=None) -> SeedingOutput:
    """Run joint fit -> extension test -> spectrum test -> final refit (each
    gated by config) and package the result as a SeedingOutput.
    """
    logger.info('Starting source_fitter (DRIPS-seeded in-process fit)')

    result = run_joint_fit(drip_model_path, config, logger, directory_manager)
    checkpoint.save_step('drips_joint_fit', 0, 'completed', result, metadata={'num_sources': len(result.model.sources)})
    resmap = _build_fit_maps(config, logger, directory_manager, result.step_dir, 'residual')

    if config.get('fitting.run_extension_test', True):
        result = run_extension_test(result, config, logger, directory_manager)

    if config.get('fitting.run_spectrum_test', True):
        result = run_spectrum_test(result, config, logger, directory_manager)

    if config.get('fitting.run_final_refit', True):
        result = run_final_refit(result, config, logger, directory_manager)

    num_sources = len(result.model.sources)
    model_path = directory_manager.get_model_file_path('Final')
    ModelGenerator.write_model_from_live(result.model, str(model_path), logger=logger)

    output = SeedingOutput(
        source_info_db=_model_summary_df(result.model),
        baseline_model_path=model_path,
        baseline_likelihood=result.log_like,
        baseline_params={},
        ts_values={},
        residual_map_path=Path(result.residual_map_path) if result.residual_map_path else seeding_output.residual_map_path,
        checkpoint_data={
            'method': 'DripsFit',
            'num_sources': num_sources,
            'seeded_from': 'DRIPS',
            'aic': result.aic,
        },
        num_sources=num_sources,
        num_iterations=1,
        method='DripsFit',
    )
    logger.info(f'source_fitter complete: {num_sources} sources, -logL={result.log_like:.3f}')
    return output
