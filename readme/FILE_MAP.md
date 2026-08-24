# FILE_MAP — Flat delivery names → real package paths

The refactored files were delivered with flat names (no directories). Before an
agent works on them, place each at its real path in the package. If the user has
already moved them, use the "Target path" column to locate the live file.

## core/

| Delivered file            | Target path                       | Contents |
|---------------------------|-----------------------------------|----------|
| `core_config.py`          | `core/config.py`                  | `ConfigManager` |
| `core_logger.py`          | `core/logger.py`                  | `PipelineLogger` |
| `core_checkpoint.py`      | `core/checkpoint.py`              | `CheckpointManager` |
| `core_init.py` / `hawc_analysis_init.py` | `core/__init__.py`   | package init |
| `hdf5_handler.py` / `week2_io_hdf5_handler.py` | `core/hdf5_handler.py` | `HDF5Handler` |
| `week2_io_init.py`        | (io init, if used)                | package init |
| `directory_manager.py`    | `core/directory_manager.py`       | `DirectoryManager` |
| `data_loading.py`         | `core/data_loading.py`            | `DataLoader` |
| `plotting.py`             | `core/plotting.py`                | `PlottingUtilities` |
| `map_tools.py`            | `core/map_tools.py`               | `MapGenerator` |
| `roi_tools.py`            | `core/roi_tools.py`               | `ROITools` |
| `model_generator.py`      | `core/model_generator.py`         | `ModelGenerator` |

## seeding/

| Delivered file                 | Target path                    | Contents |
|--------------------------------|--------------------------------|----------|
| `seeding_init.py`              | `seeding/__init__.py`          | exports SeedingModule, SeedingOutput, DRIPSSeeder, ALPSSeeder |
| `seeding_base.py`              | `seeding/base.py`              | `SeedingModule` (ABC), `SeedingOutput` |
| `seeding_image_seeds.py`       | `seeding/image_seeds.py`       | `DRIPSSeeder` |
| `seeding_alps_seeds.py`        | `seeding/alps_seeds.py`        | `ALPSSeederBase`, `ALPSLogger`, module dbs |
| `seeding_alps_fit_adapter.py`  | `seeding/alps_fit_adapter.py`  | `ALPSFitAdapter`, `FitStepResult` |
| `seeding_alps_seeder.py`       | `seeding/alps_seeder.py`       | `ALPSSeeder` |

## Origin / support files (copied alongside, imported as-is)

- `pipeline_helpers.py` → `seeding/pipeline_helpers.py` (DRIPS seeder imports many helpers from it).
- `pipeline_fitmodel.py` → importable as `pipeline_fitmodel` (the adapter does `from pipeline_fitmodel import threeMLFit`).
- `pipeline_hd5.py`, `pipeline_map_maker.py`, `pipeline_utilities.py` → as referenced.

## Import contracts to keep working

- `seeding/image_seeds.py` does `from seeding.pipeline_helpers import (...)`
  (a specific list of DRIPS helpers). If helpers live elsewhere, fix this import,
  not the call sites.
- `seeding/alps_fit_adapter.py` does `from pipeline_fitmodel import threeMLFit`.
  Ensure `pipeline_fitmodel.py` is on the path.
- `seeding/alps_seeder.py` does:
  - `from seeding.base import SeedingModule, SeedingOutput`
  - `from seeding.alps_seeds import ALPSSeederBase, source_types_db, spectrum_types_db`
  - `from seeding.alps_fit_adapter import ALPSFitAdapter`
- `ALPSSeeder._residual_hd5_to_fits` does lazy `from core.hdf5_handler import HDF5Handler`
  and `from core.map_tools import MapGenerator` inside the method (so ALPS imports
  don't hard-depend on core at module load).

## First action for the agent

1. Create `core/` and `seeding/` dirs, move files to Target paths, rename to the
   stem shown (drop the flat prefixes).
2. Add empty/real `__init__.py` files where missing.
3. Run a syntax pass: `python -c "import ast,glob; [ast.parse(open(f).read()) for f in glob.glob('**/*.py', recursive=True)]"`.
4. Then attempt the import smoke test in `TASKS.md` Task 0.
