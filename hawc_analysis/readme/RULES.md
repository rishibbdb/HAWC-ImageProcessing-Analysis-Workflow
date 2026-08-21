# RULES — Constraints for the HAWC Pipeline Refactor

These are hard constraints the human has enforced throughout. Follow them unless
the human explicitly overrides in the current session. When in doubt, ask rather
than assume.

## 1. Do not change algorithm logic when extracting

The refactor moves working functions into classes/modules. When you extract a
function from an origin file (`pipeline_sourcedetector.py`, `pipeline_helpers.py`,
`testalps.py`, etc.):

- **Copy the body verbatim.** Do not "improve", simplify, re-vectorize, rename
  local variables, change thresholds, reorder operations, or fix perceived bugs.
- The ONLY additions allowed during extraction are: type hints, docstrings, an
  optional `logger` parameter, and reading inputs from config via `config.get()`.
- If you believe there is a real bug in origin logic, DO NOT silently fix it.
  Flag it to the human and let them decide.

Rationale: this code is scientifically validated. Behavior changes invalidate
prior results. Faithfulness beats elegance here.

## 2. Fitting is in-process — no subprocess for fits

Use `threeMLFit` from `pipeline_fitmodel.py` (via `ALPSFitAdapter`). Do NOT
reintroduce `subprocess` calls to `fitModel.py` / `Drawmaps.py`. The one place
subprocess is still acceptable is external map/hotspot tools that have no Python
API: `aerie-apps-get-local-extremum` (`_find_next_hotpsot`) and
`aerie-apps-HealpixSigFluxMap` (in `MapGenerator`). Keep those isolated in a
single method each; don't scatter shell-outs through the code.

## 3. ALPS exclusion: `_get_sloppy_TS`

The `_get_sloppy_TS` method from `testalps.py` is intentionally excluded. Do not
re-add it. If a code path references it, route through the default TS calculation
(`use_default_TS_calc=True`) instead.

## 4. No packaging / standalone modules

The user runs inside a `pixi` environment and imports by module path
(`from core.x import Y`, `from seeding.x import Y`). Do not add a build step,
do not require `pip install -e`, do not turn this into an installable distribution
unless explicitly asked. A `setup.py` in outputs is vestigial and unused.

## 5. Config-driven, not hardcoded

Read paths, coordinates, bins, thresholds from config via `.get()` — never
hardcode file paths, RA/Dec, or bin lists in the modules. There are TWO configs:
- Dot-notation pipeline config (`config.yaml`) — read by `threeMLFit` / `core`.
- ALPS space-delimited config (`testalpsconfig.yaml`) — read by ALPS via
  `_load_config_value`. Keep them separate; do not attempt to unify them unless
  the human asks for that as an explicit task.

## 6. Two source-of-truth configs, one loader each

`ConfigManager`/`PipelineConfig` use dot-notation. ALPS uses its own
space-delimited `_load_config_value`. Don't cross-wire them.

## 7. Preserve the SeedingOutput contract

Both DRIPS and ALPS must return a valid `SeedingOutput` (see `seeding/base.py`).
If you add fields, add them as optional with defaults so existing construction
sites keep working. Run `SeedingModule.validate_output()` conceptually — the
required fields must always be populated.

## 8. Style

- Terse, technically precise. Ready-to-run code, minimal doc overhead.
- Match the existing module style (type hints + short docstrings).
- Don't add heavyweight frameworks. Stdlib + the scientific stack already in use
  (numpy, pandas, astropy, healpy, scipy, skimage, threeML, hawc_hal).

## 9. When origin code is half-implemented

Some origin methods are placeholders or incomplete (notably the extension/spectrum
tests in `main.py`, and `run_alt_hypothesis` in `testalps.py` which printed a db and
returned nothing). Where you must implement real logic that the origin only stubbed,
say so explicitly in comments and in your message to the human, and implement to the
documented *intent* (TS thresholds from config). Do not present invented logic as a
verbatim extraction.

## 10. Verify before declaring done

Parse every file you write (`python -c "import ast; ast.parse(open(f).read())"`).
For runtime changes, prefer to run the smallest possible test (a single fit, a single
db round-trip) over claiming success. Report honestly what was and wasn't tested.
