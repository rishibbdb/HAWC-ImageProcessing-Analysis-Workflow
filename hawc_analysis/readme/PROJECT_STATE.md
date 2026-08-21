# HAWC Analysis Pipeline — Project State & Handoff

This document orients an AI coding agent (Claude Code) working on the HAWC
gamma-ray source-detection pipeline refactor. Read this first, then `RULES.md`,
then `FILE_MAP.md`, then pick up work from `TASKS.md`.

## What this project is

A production refactor of a working-but-monolithic HAWC point-source / extended-source
detection and fitting pipeline. The science: given a HAWC significance map, detect
source seeds, fit them with threeML/HAL, then iteratively test whether sources are
extended (morphology test) and whether alternate spectra fit better (spectrum test).

Two seeding algorithms exist and must both be supported behind one interface:
- **DRIPS** — image-processing blob detection on the significance map (`SourceSeedDetector`).
- **ALPS** — iterative likelihood search: add a point source at the brightest
  residual hotspot, fit, repeat until the TS gain drops below threshold; then run
  morphology and spectrum tests.

The refactor splits the monolith into `core/` utilities and a `seeding/` package,
config-driven, with each algorithm producing an identical `SeedingOutput`.

## Origin files (the working code being refactored)

These are the ORIGINAL scripts. They are the source of truth for algorithm
behavior. When refactored code and these disagree on logic, these win unless a
change is explicitly listed in `RULES.md` or `TASKS.md`.

- `pipeline_sourcedetector.py` — DRIPS `SourceSeedDetector` class.
- `pipeline_helpers.py` — DRIPS helper functions (blob filters, plotting, loadmap, model builders).
- `pipeline_fitmodel.py` — `threeMLFit`: in-process HAL/threeML fitter. THIS is the fitter to use.
- `pipeline_map_maker.py`, `pipeline_utilities.py`, `pipeline_hd5.py` — map making, utilities, hd5→fits.
- `testalps.py` — ALPS `AutomatedLikelihoodPipelineSearch` class (its own config schema).
- `main.py` — early orchestrator (`SourceSearchPipeline`); note its extension/spectrum
  test methods are PLACEHOLDERS (`pass`). The real alt-hypothesis logic lives in ALPS.
- `config.yaml` — the dot-notation pipeline config (`threeMLFit` reads this).
- `testalpsconfig.yaml` — ALPS's own space-delimited-key config.

## Architecture (target)

```
hawc_analysis/
  core/
    config.py            ConfigManager (dot-notation .get())
    logger.py            PipelineLogger
    checkpoint.py        CheckpointManager
    hdf5_handler.py      HDF5Handler (hd5 <-> fits)
    directory_manager.py DirectoryManager (fit_name/ dir tree)
    data_loading.py      DataLoader (loadmap, load_hawc_data, find_peak)
    plotting.py          PlottingUtilities
    map_tools.py         MapGenerator (HealpixSigFluxMap wrapper)
    roi_tools.py         ROITools
    model_generator.py   ModelGenerator (threeML models from source df)
  seeding/
    base.py              SeedingModule (ABC) + SeedingOutput (dataclass)
    image_seeds.py       DRIPSSeeder(SeedingModule)
    alps_seeds.py        ALPSSeederBase (verbatim ALPS extraction) + ALPSLogger
    alps_fit_adapter.py  ALPSFitAdapter (runs threeMLFit in-process) + FitStepResult
    alps_seeder.py       ALPSSeeder(ALPSSeederBase, SeedingModule)
    pipeline_helpers.py  (copied here; DRIPS seeder imports from it)
  pipeline.py            (TODO) orchestrator
  cli.py                 (TODO) CLI
```

Imports are `from core.xxx import Yyy` and `from seeding.xxx import Yyy`.
No pip install / no setup.py is used — the user runs inside a `pixi` environment
and imports by module path. (A `setup.py` may exist in outputs but is not used.)

## Key interfaces

### SeedingOutput (seeding/base.py)
Both seeders return this. Fields: `source_info_db` (DataFrame), `baseline_model_path`
(Path), `baseline_likelihood` (float), `baseline_params` (dict), `ts_values` (dict),
`residual_map_path` (Path), `checkpoint_data` (dict), `num_sources` (int),
`num_iterations` (int), `method` (str). Has `.summary()` and `.to_dict()`.

### threeMLFit (pipeline_fitmodel.py) — the fitter
Constructed with `(config_path, model, save_dir, roiTemplate, logger)` where `model`
is a path to a `.model` file. After `.hal_fit()` or `.hal_fit_with_covariance()`:
- `self.params` — DataFrame indexed by e.g. `Source0.position.ra`, column `value`.
- `self.statistics` — DataFrame; total −logL at `.loc['total','-log(likelihood)']`.
- `self.jl.results` — threeML `MLEResults`; AIC via `get_statistic_measure('AIC')`.
- `self.model_obj` — the fitted threeML model.
- `.make_maps()` — writes `save_dir/results/model_fit.hd5` and `residual_fit.hd5`.
- `.get_TS()` — returns `(source_names, ts)`.

NOTE: `statistics` stores **−log(likelihood)**, so ALPS's STAT0 maps directly with
no sign flip. AIC = ALPS MV0.

## How ALPS was integrated (important context)

Original ALPS shelled out to `fitModel.py` per fit and read results back off disk
(`*_likeResults.fits` for STAT0/MV0, `*_modelFit.yml` for the refit model,
`*.hd5` for maps). The refactor removes subprocess fitting:

- `ALPSFitAdapter.fit()` runs `threeMLFit` in-process and returns a `FitStepResult`
  carrying `log_like`, `aic`, `fitted_db`, and the two `.hd5` map paths.
- `ALPSSeeder.run_single_fit` calls the adapter (keeping the retry/perturb structure).
- `ALPSSeeder._after_run_accouting` reads stats from the `FitStepResult`, not FITS.
- `ALPSSeeder._db_from_model_obj` rebuilds the fitted `source_info_db` from the live
  fitted model (same walk as `_load_from_model_file`).

The point-source phase is pluggable: `ALPSSeeder.run(drips_filtered_df=...)` seeds from
DRIPS point sources; `run(drips_filtered_df=None)` uses ALPS's native hotspot search
(which still shells out to `aerie-apps-get-local-extremum` in `_find_next_hotpsot`).

## Where things stand

DONE and syntax-verified: all of `core/`, `seeding/base.py`, `seeding/image_seeds.py`
(DRIPS), and the three ALPS files. NOT yet runtime-tested end to end.

See `TASKS.md` for the remaining work and the known-risky spots to validate first.
