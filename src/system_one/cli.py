"""Thin CLI boundary for configuration validation and stage preparation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from system_one.config import STAGES, ConfigError, load_config
from system_one.runs import prepare_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reproducible decision-model experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate", help="Validate YAML without creating a run")
    validate.add_argument("--config", type=Path, required=True)
    prepare = commands.add_parser("prepare", help="Record a stage; does not execute it")
    prepare.add_argument("stage", choices=STAGES)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--run-root", type=Path, default=Path("runs"))
    args = parser.parse_args(argv)
    try:
        loaded = load_config(args.config)
        if args.command == "validate":
            print(f"Valid: {loaded.config.experiment.name}")
            print(f"Config SHA-256: {loaded.config.sha256()}")
        else:
            run_dir = prepare_run(
                loaded,
                stage=args.stage,
                seed=args.seed,
                run_root=args.run_root,
                project_root=Path(__file__).resolve().parents[2],
            )
            print(f"Prepared {args.stage}: {run_dir}")
            print("Stage not executed; model and dataset adapters arrive in later slices.")
    except (ConfigError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0
