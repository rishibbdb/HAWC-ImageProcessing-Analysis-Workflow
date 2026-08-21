# CLAUDE.md — HAWC Pipeline Refactor (agent entry point)

You are picking up a partially-complete refactor of a HAWC gamma-ray
source-detection pipeline. Read these docs in order before acting:

1. `PROJECT_STATE.md` — what the project is, architecture, current state.
2. `RULES.md` — hard constraints (verbatim extraction, in-process fitting,
   exclusions). Follow these unless the human overrides in-session.
3. `FILE_MAP.md` — where each delivered file goes; import contracts.
4. `TASKS.md` — the work backlog. Start at Task 0.

## TL;DR of the non-negotiables

- **Extract logic verbatim** from the origin scripts (`pipeline_sourcedetector.py`,
  `pipeline_helpers.py`, `testalps.py`). Add only type hints, docstrings, `logger`,
  and `config.get()`. Never silently change thresholds or fix perceived bugs — flag them.
- **Fitting is in-process** via `threeMLFit` (`pipeline_fitmodel.py`) through
  `ALPSFitAdapter`. No subprocess for fits. Subprocess only for `aerie-apps-*`
  map/hotspot tools, isolated to one method each.
- **`_get_sloppy_TS` stays excluded.**
- **No packaging.** Import by module path inside the `pixi` env.
- **Config-driven.** Two configs (dot-notation pipeline config; ALPS space-delimited).
  Don't unify them unless asked.

## The one thing most likely to need your attention first

`ALPSFitAdapter._extract_aic` guesses the threeML AIC accessor. On the real
threeML build, confirm `jl.results.get_statistic_measure('AIC')` (or find the right
call) and hard-wire it. See `TASKS.md` Task 2.

## The one piece that is NOT a verbatim extraction

`ALPSSeeder.run_alt_hypothesis` (extension/spectrum tests) — the origin was a stub,
so this was written to intent and needs human review of the accept rule. See
`TASKS.md` Task 4. Don't treat it as validated.

## Working style the human expects

Terse and technically precise. Ready-to-run code. Verify by running the smallest
real check, not by asserting success. Report honestly what you did and didn't test.
