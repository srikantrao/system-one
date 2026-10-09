"""Minimal dated JSON records shared by preparation and execution scripts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from system_one.config import ExperimentSettings, RunConfig, Stage


def prepare_run(
    config: RunConfig,
    *,
    stage: Stage,
    run_root: Path,
) -> Path:
    return create_run(config, stage=stage, run_root=run_root)


def create_run(config: ExperimentSettings, *, stage: str, run_root: Path) -> Path:
    """Save parameters before executing a script; each invocation gets its own folder."""
    now = datetime.now(UTC)
    run_id = f"{now:%Y%m%dT%H%M%S%fZ}-{stage}-s{config.experiment.seed}"
    results = {
        "run_id": run_id,
        "created_at": now.isoformat(),
        "status": "prepared",
        "stage": stage,
        "parameters": config.model_dump(mode="json"),
        "metrics": None,  # Filled after the experiment runs.
    }
    date = now.astimezone(ZoneInfo("America/Los_Angeles")).date()
    experiment_dir = run_root.resolve() / f"{date}-{config.experiment.name}"
    run_dir = experiment_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return run_dir


def finish_run(run_dir: Path, *, status: str = "completed", **evidence: object) -> None:
    """Add executed-stage evidence to the existing JSON record."""
    path = run_dir / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    results.update(status=status, **evidence)
    path.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
