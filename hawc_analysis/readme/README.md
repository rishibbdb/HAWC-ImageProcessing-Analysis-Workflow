# HAWC Analysis Pipeline

Refactored HAWC gamma-ray source-detection pipeline. Two seeding/fitting
methods behind one interface, driven by one config file.

See `PROJECT_STATE.md` / `RULES.md` / `FILE_MAP.md` / `TASKS.md` for the
original design docs. This README documents what has actually been built and
validated on top of them.

## Status

| Task | What | Status |
|---|---|---|
| 0 | File placement, import smoke test | Done |
| 1 | DRIPS verbatim-extraction validation | Done |
| 2 | ALPS in-process fit via `ALPSFitAdapter`, AIC accessor pinned | Done |
| 3 | ALPS residual-map conversion + point-source loop | Done |
| — | Config unification (see below) | Done |
| 5 | `pipeline.py` orchestrator | Done, validated end-to-end on real Crab data |
| 6 | `cli.py` | Done, validated |
| 7 | Directory structure conformance | Checked, matches intended per-method shape |
| 4 | `ALPSSeeder.run_alt_hypothesis` accept-rule (extension/spectrum testing) | **Open — not verbatim, needs your review** |

## What changed from the original design docs

`RULES.md` originally called for keeping the DRIPS config (`config.yaml`,
dot-notation) and the ALPS config (`alspconfig.yaml`, space-delimited)
separate. That was overridden in-session: ALPS now reads the same
`config.yaml` as DRIPS, via a new `alps:` section for ALPS-only settings
(everything else — coordinates, `fitting.*`, `roi.*`,
`likelihood_thresholds.*`, `diffuse.*` — is shared). `alspconfig.yaml` is no
longer read by anything; it's dead but left in place.

Code touched to make this work: `seeding/alps_seeds.py`
(`ALPSSeederBase.__init__`), `seeding/alps_seeder.py`
(`ALPSSeeder.__init__`), `core/map_tools.py` (`MapGenerator.create_healpix_map`
gained a `pixi_manifest_path` param so `pixi run aerie-apps-*` resolves
without depending on cwd). Originals are backed up in
`backups_pre_config_unify/` before any rewrite.

Two real pre-existing bugs were found and fixed along the way (not design
changes — genuine wrong config keys):
- `pipeline_fitmodel.py`: `threeMLFit` read `paths.map_tree` /
  `paths.detector_response`, keys that never existed in `config.yaml`. Fixed
  to `fitting.map_tree` / `fitting.detector_response`.
- `seeding/image_seeds.py`: `DRIPSSeeder` read `paths.use_dbe` for the
  diffuse-background flag; fixed to `diffuse.use_diffuse_background`. This
  had been silently masking `diffuse.use_diffuse_background: true` with no
  `diffuse_template_path` configured — see **Known gaps** below.

## How to run it

Use the real project interpreter, not system `python3`:

```
PYBIN=/Users/rishi/Documents/Analysis/aerie/.pixi/envs/threeml/bin/python
```

From `hawc_analysis/`:

```bash
# Full run, method picked by config.yaml's top-level `fitting_procedure` key
$PYBIN -m cli --config config.yaml

# Force a method regardless of what's in the config
$PYBIN -m cli --config config.yaml --procedure Drips
$PYBIN -m cli --config config.yaml --procedure Alps

# DRIPS detection only, no fit (fast smoke test)
$PYBIN -m cli --config config.yaml --seed-only
```

Or from Python directly:

```python
from pipeline import HAWCAnalysisPipeline

pipeline = HAWCAnalysisPipeline("config.yaml")
output = pipeline.run()
print(output.summary())
```

`fitting_procedure: Drips` runs DRIPS blob-detection for seeding, then hands
the detected sources to `ALPSSeeder` for the in-process fit (unless
`coordinates.generate_seed_only: true`, in which case it stops after
detection). `fitting_procedure: Alps` runs ALPS's own native hotspot-driven
search + fit instead of DRIPS.

Output lands under `fitting.output_dir/fitting.fit_name/` (currently
`crab_fit/` next to this directory). DRIPS writes into
`Results/{step_name}/`; ALPS keeps its own `FitResults/{fit_name}_step_N/`,
`Models/`, `DataMap/`, `Logs/` at the same root — the two methods were kept
in their own native shapes rather than unified (matches `TASKS.md`'s intent).
Checkpoints are written to `.checkpoints/` inside that same root; note
`pipeline.py` does not currently skip already-completed steps on rerun — a
second `run()` against the same `fit_name` redoes everything.

## Config

Single file: `config.yaml`. Top-level `fitting_procedure` picks the method.
Shared sections: `coordinates`, `fitting`, `roi`, `diffuse`,
`likelihood_thresholds`, `error_and_TS`. ALPS-only settings live under
`alps:` (see comments in the file for each key). `alps.pixi_aerie_folder`
must point at the aerie pixi project root — it's what lets `pixi run
aerie-apps-*` resolve `aerie-apps-HealpixSigFluxMap` /
`aerie-apps-get-local-extremum` regardless of cwd.

## Known gaps / things to check before trusting results

- **Task 4 — yours to review.** `ALPSSeeder.run_alt_hypothesis` (extension
  and spectrum alt-hypothesis testing) was not a verbatim extraction — the
  origin (`testalps.py`) had this as a stub, so it was written to intent.
  I fixed a real crash in it (baseline re-fit returning `None` when threeML/
  pandas chokes on a fully-frozen zero-free-parameter model — happens
  because `_after_run_accouting` freezes just-fit sources before this phase
  re-fits them as a baseline) but did **not** touch the underlying accept-rule
  logic or decide whether target sources should be re-freed before that
  baseline re-fit. It's disabled by default:
  `alps.alternate_spectrum_model_list: null`,
  `alps.alternate_morpholgy_model_list: null` in `config.yaml`. Set either to
  a real model list (e.g. `[Log_parabola]` / `[Gaussian_on_sphere]`) once
  you've reviewed and fixed the accept rule.
- **`diffuse.use_diffuse_background` is `false`.** It was `true` in the
  config but silently inert due to the `DRIPSSeeder` bug above. Once that
  bug was fixed, turning it on raised `hermes_path must be provided when
  hermes_present=True` — there's no `diffuse.diffuse_template_path`
  configured. Left `false` with a comment in `config.yaml:40`; flip it once
  you have a real template path.
- **`_get_sloppy_TS` remains excluded**, per `RULES.md`.
- Fitting is in-process via `threeMLFit`/`ALPSFitAdapter`
  (`HAL(..., n_workers=4)`, a multiprocessing pool) — any script that
  imports `pipeline.py` or `ALPSSeeder` for a full fit run must guard its
  entry point with `if __name__ == "__main__":` or macOS spawn-mode will
  re-execute top-level script code in every worker.
