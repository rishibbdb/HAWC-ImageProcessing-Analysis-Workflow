# TASKS — Remaining work & validation backlog

Work top-to-bottom. Tasks 0–3 are validation of what already exists (do these
before building anything new). Tasks 4+ are new construction. Each task says how
to know it's done. Respect `RULES.md` throughout.

---

## Task 0 — Place files & import smoke test

Per `FILE_MAP.md`, move files into `core/` and `seeding/`, add `__init__.py`s.

Done when this runs with no ImportError:
```python
from core.config import ConfigManager
from core.logger import PipelineLogger
from core.directory_manager import DirectoryManager
from core.data_loading import DataLoader
from core.hdf5_handler import HDF5Handler
from core.map_tools import MapGenerator
from seeding.base import SeedingModule, SeedingOutput
from seeding.image_seeds import DRIPSSeeder
from seeding.alps_seeds import ALPSSeederBase
from seeding.alps_fit_adapter import ALPSFitAdapter
from seeding.alps_seeder import ALPSSeeder
```
Fix import paths only — do not change logic to make imports pass.

---

## Task 1 — DRIPS seeder runtime test

`DRIPSSeeder` wraps `SourceSeedDetector` verbatim. Validate it still detects the
same sources as the origin `pipeline_sourcedetector.py` on one significance map.

Steps:
- Build a `DRIPSSeeder(config, logger, directory_manager)` and call `.run()`.
- Compare `output.source_info_db` (the `filtered_df`) against running the origin
  `SourceSeedDetector(...).run()` on the same input — same source count, same
  ra/dec within floating tolerance.

Done when: DRIPS output matches origin on at least one real map. Report any diff.

Known-risky:
- `seeding/image_seeds.py` imports helpers from `seeding.pipeline_helpers`. Confirm
  that file is present and exports every name in the import list.
- `_run()` (the original `SourceSeedDetector.run()`) writes `curModel.model`; the
  wrapper's `run()` packages it. Confirm the model file path in `SeedingOutput`
  points to the real written file.

---

## Task 2 — ALPS fit adapter: single fit + AIC accessor

The adapter is the load-bearing new code. Validate ONE in-process fit end to end.

Steps:
- Construct `ALPSSeeder(initial_config=ALPS_CONFIG, fit_config_path=FIT_CONFIG)`.
- Hand-build a 2 point-source `source_info_db` (see the notebook test the human has).
- `seeder.target_sources = [...]; result = seeder.run_single_fit(db, 'adaptertest',
  compute_err=False, compute_TS=False)`.
- Assert `result.log_like` is finite and `result.fitted_db` has the sources with
  shifted positions.

CRITICAL sub-task — pin the AIC accessor. `ALPSFitAdapter._extract_aic` guesses at
`jl.results.get_statistic_measure('AIC')` with fallbacks. On the real threeML
version, determine the correct accessor and make `_extract_aic` return a finite AIC.
Check: `result.fitter.jl.results.get_statistic_measure('AIC')` — if that errors,
inspect `type(result.fitter.jl.results)` (expected `MLEResults`) and find the right
call. Update `_extract_aic` to use it directly. AIC feeds `MV0`/`prev_AIC` bookkeeping;
the ΔTS convergence uses `log_like`, not AIC, so a NaN AIC won't break the loop but
should still be fixed.

Done when: one fit returns finite `log_like` AND finite `aic`, and `fitted_db` round-trips.

---

## Task 3 — ALPS residual→fits + one PS-loop iteration

Validate the residual conversion and a single iteration of the point-source loop.

Steps:
- After Task 2's fit, call `seeder._residual_hd5_to_fits(result)`; confirm a
  `residual.fits` is produced (needs `core.HDF5Handler` + `core.MapGenerator`, and
  `aerie-apps-HealpixSigFluxMap` on PATH).
- Run `seeder.run(drips_filtered_df=<small df>)` (DRIPS-seeded, avoids the native
  hotspot subprocess) and confirm it completes Phase 1 and returns a `SeedingOutput`.

Known-risky:
- `_residual_hd5_to_fits` assumes `HDF5Handler.convert_hd5_to_fits` writes per-bin
  fits that `MapGenerator.find_fits_files_by_bins` can then find. Verify the filename
  patterns line up; adjust the glob/prefix if not (this is integration glue, fixable).
- Native hotspot mode (`run(drips_filtered_df=None)`) needs
  `aerie-apps-get-local-extremum`; only test on the cluster.

---

## Task 4 — ALPS alt-hypothesis review (NOT a verbatim extraction)

`ALPSSeeder.run_alt_hypothesis` / `_swap_morphology` / `_swap_spectrum` implement the
extension and spectrum tests. The origin `testalps.py::run_alt_hypothesis` was
incomplete (printed a db, returned nothing), so this logic was written to the
documented intent, NOT copied. It needs a human-informed review:

- Confirm the accept rule: an alternate morphology/spectrum is accepted when
  `2*(best_log_like - trial_log_like) > threshold` (morphology threshold vs spectrum
  threshold from config). Confirm this matches how the science wants acceptance decided.
- Confirm default-parameter seeding for swapped models is acceptable (position carried
  over; other params from `source_types_db` / `spectrum_types_db` defaults).
- Confirm per-source iteration order and the trust flags
  (`trust_all_alterante_morphologies`, `trust_all_alternate_spectra`,
  `trusted_source_list`) behave as intended.

Done when: the human signs off on the accept logic, or specifies the correct rule and
you implement it.

---

## Task 5 — Pipeline orchestrator (`pipeline.py`)

Build `HAWCAnalysisPipeline` that ties it together, config-driven:
- Read `fitting_procedure` from config (`'Drips'` or `'Alps'`).
- Build the significance map if needed (via `MapGenerator`) — mirror `main.py`'s
  `make_maps` behavior.
- Run the chosen seeder → `SeedingOutput`.
- If DRIPS: optionally hand `output.source_info_db` to `ALPSSeeder.run(drips_filtered_df=...)`
  for the morphology/spectrum testing phases (DRIPS does detection only, no alt tests).
- Persist results into the `DirectoryManager` tree (DataMap/Logs/Models/FitResults).
- Checkpoint each phase via `CheckpointManager`.

Reference: `main.py::SourceSearchPipeline` shows the intended step/dir structure and
the seed→fit→residual→model-map flow. Reuse that structure; swap its subprocess bits
for `core`/adapter equivalents.

Done when: `HAWCAnalysisPipeline(config).run()` executes DRIPS→fit end to end on one
ROI and writes the expected directory tree.

---

## Task 6 — CLI (`cli.py`)

Thin CLI (argparse or click) exposing: run full pipeline from a config path; run
seeding only; choose procedure override. Keep it a thin wrapper over
`HAWCAnalysisPipeline` — no logic in the CLI itself.

Done when: `python -m cli --config config.yaml` runs the pipeline.

---

## Task 7 — Directory structure conformance

Confirm outputs land in the agreed tree (from the origin design):
```
{fit_name}/
  DataMap/      sky_map.fits (if generated)
  Logs/         pipeline_*.log, full_log_*.log
  Models/       {step}.model
  FitResults/
    {step}/     fit_results.yaml, parameters.yaml, model_fit.hd5, residual_fit.hd5
```
Step naming differs by method (DRIPS: `Step0-Allpoint-sources`, ...; ALPS:
`{fit_name}_step_1`, ...). Preserve each method's own naming; don't force a shared
scheme.

---

## Reporting expectations

For each task, report: what you changed, what you ran, what passed, what you could
NOT verify (e.g. cluster-only tools), and any origin-vs-refactor discrepancies you
found. Do not mark a task done on syntax alone if it has a runtime check.
