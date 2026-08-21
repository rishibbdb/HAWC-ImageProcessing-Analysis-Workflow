"""
CLI (TASKS.md Task 6)

Thin wrapper over HAWCAnalysisPipeline. No pipeline logic lives here --
only argument parsing and config overrides.

Usage:
    python -m cli --config config.yaml
    python -m cli --config config.yaml --procedure Alps
    python -m cli --config config.yaml --seed-only
"""

import argparse
import sys

from core.config import ConfigManager
from pipeline import HAWCAnalysisPipeline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the HAWC analysis pipeline")
    parser.add_argument("--config", required=True, help="Path to the pipeline config YAML")
    parser.add_argument(
        "--procedure", choices=["Drips", "Alps"], default=None,
        help="Override fitting_procedure from the config",
    )
    parser.add_argument(
        "--seed-only", action="store_true",
        help="Override coordinates.generate_seed_only to True (DRIPS detection only, no fit)",
    )
    args = parser.parse_args(argv)

    config = ConfigManager(args.config)
    if args.procedure is not None:
        config.config["fitting_procedure"] = args.procedure
    if args.seed_only:
        config.config.setdefault("coordinates", {})["generate_seed_only"] = True

    pipeline = HAWCAnalysisPipeline(config)
    output = pipeline.run()
    print(output.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
