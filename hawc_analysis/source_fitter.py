"""
Source fitter: DRIPS-seeded joint fit + extension/spectrum testing + refit.

"""

from pathlib import Path
from typing import List
from astropy.coordinates import SkyCoord
import pandas as pd

from fit_runner import FitRunner, FitResult
from model_generator import ModelGenerator
from seeding.base import SeedingOutput
from core.hdf5_handler import HDF5Handler
from core.map_tools import MapGenerator

def _as_list(value) -> List[str]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


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
        compute_TS=False,
        make_maps=True,
    )
    return result


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
    free_dbe = config.get('fitting.free_diffuse_norm', False)
    threshold = config.get('likelihood_thresholds.extension_test', 16)
    coord_range = config.get('fitting.extended_source_coord_range', 1.0)
    runner = FitRunner(
        config_path=str(config.config_file),
        logger=logger,
        roi_template=config.get('roi.roi_template_path'),
    )

    model = fit_result.model
    baseline_log_like = fit_result.log_like
    source_names = list(model.sources.keys())
    for source_name in source_names:
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
            ModelGenerator.set_free(trial_model, other_sources, kind='spatial', free=False, free_diffuse=free_dbe, logger=logger)
            ModelGenerator.set_free(trial_model, other_sources, kind='spectral', free=False, free_diffuse=free_dbe, logger=logger)

            step_name = f'Step2-{source_name}-Extension-{alt_shape}'
            step_dir = directory_manager.get_step_results_dir(step_name)
            # model_file = ModelGenerator.write_model_from_live(
            #     trial_model, str(directory_manager.get_model_file_path(step_name)), logger=logger,
            # )
            trial_model.save("{1}/{0}.yml".format('curModel', step_dir), overwrite=True)
            ModelGenerator.write_model_file_from_yaml("{1}/{0}.yml".format('curModel', step_dir), "{1}/{0}.model".format('curModel', step_dir), logger=logger)
            model_file = "{1}/{0}.model".format('curModel', step_dir)
            trial_result = runner.fit(
                model_file=str(model_file),
                step_dir=str(step_dir),
                compute_err=config.get('error_and_TS.error_extension', True),
                make_maps=True,
            )

            delta_ts = 2 * (best_log_like - trial_result.log_like)
            logger.info(f'Extension test {source_name} -> {alt_shape}: delta_TS={delta_ts:.2f} (threshold {threshold})')
            if delta_ts > threshold:
                best_log_like = trial_result.log_like
                best_model = trial_result.model
                logger.info(f'Accepted alternate spatial model {alt_shape} for {source_name}')

        model = best_model
        baseline_log_like = best_log_like

    return FitResult(
        model=model, log_like=baseline_log_like, aic=fit_result.aic,
        model_map_path=fit_result.model_map_path, residual_map_path=fit_result.residual_map_path,
        step_dir=fit_result.step_dir, fitter=fit_result.fitter,
    )


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
            ModelGenerator.set_free(trial_model, other_sources, kind='spatial', free=False)
            ModelGenerator.set_free(trial_model, other_sources, kind='spectral', free=False)

            step_name = f'Step3-{source_name}-Spectrum-{alt_spectrum}'
            step_dir = directory_manager.get_step_results_dir(step_name)
            model_file = ModelGenerator.write_model_from_live(
                trial_model, str(directory_manager.get_model_file_path(step_name)), logger=logger,
            )
            trial_result = runner.fit(
                model_file=str(model_file),
                step_dir=str(step_dir),
                compute_err=config.get('error_and_TS.error_spectrum', True),
                make_maps=False,
            )

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
