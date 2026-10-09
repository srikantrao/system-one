"""Thin CLI boundary for configuration validation and stage preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from system_one.config import STAGES, load_config
from system_one.runs import prepare_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible decision-model experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate YAML without creating a run")
    validate.add_argument("--config", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Record a stage; does not execute it")
    prepare.add_argument("stage", choices=STAGES)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--seed", type=int, help="Override experiment.seed in the YAML")
    prepare.add_argument("--results-dir", "--run-root", type=Path, default=Path("results"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, seed=getattr(args, "seed", None))
        if args.command == "validate":
            print(f"Valid: {config.experiment.name}")
        else:
            run_dir = prepare_run(
                config,
                stage=args.stage,
                run_root=args.results_dir,
            )
            print(f"Prepared {args.stage}: {run_dir}")
            print("Stage not executed; model and dataset adapters arrive in later slices.")
    except (ValueError, OSError, yaml.YAMLError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
