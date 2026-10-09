"""Create a stage preparation record without executing the stage."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import yaml

from system_one.config import STAGES, ConfigError, LoadedConfig, Stage, digest_json
from system_one.provenance import (
    environment_snapshot,
    file_sha256,
    hardware_snapshot,
    source_snapshot,
)


def prepare_run(
    loaded: LoadedConfig,
    *,
    stage: Stage,
    seed: int,
    run_root: Path,
    project_root: Path,
) -> Path:
    config = loaded.config
    if stage not in STAGES:
        raise ConfigError(f"unknown stage: {stage}")
    if type(seed) is not int or seed not in config.experiment.training_seeds:
        raise ConfigError("seed must be one of experiment.training_seeds")

    split = {"path": None, "sha256": None, "status": "not_prepared"}
    if config.dataset.split_manifest is not None:
        path = (loaded.path.parent / config.dataset.split_manifest).resolve()
        if not path.is_file():
            raise ConfigError(f"dataset.split_manifest does not exist: {path}")
        split = {"path": str(path), "sha256": file_sha256(path), "status": "fingerprinted"}

    source = source_snapshot(project_root)
    environment = environment_snapshot()
    hardware = hardware_snapshot()
    identity = {
        "config_sha256": config.sha256(),
        "stage": stage,
        "training_seed": seed,
        "source_sha256": source["sha256"],
        "split_manifest_sha256": split["sha256"],
        "environment_sha256": digest_json(environment),
        "hardware_sha256": digest_json(hardware),
    }
    unresolved = []
    if config.model.revision is None:
        unresolved.append("model revision has not been pinned")
    if config.dataset.version is None:
        unresolved.append("dataset version has not been pinned")
    if split["status"] == "not_prepared":
        unresolved.append("dataset and split contents have not been fingerprinted")
    else:
        unresolved.append("split manifest fingerprinted; referenced dataset contents not verified")
    unresolved.append("model adapter, requested device, and training strategy not checked")

    now = datetime.now(UTC)
    run_id = f"{now.strftime('%Y%m%dT%H%M%S%fZ')}-{stage}-s{seed}-{uuid4().hex[:12]}"
    manifest = {
        "manifest_version": 1,
        "run_id": run_id,
        "created_at": now.isoformat(),
        "status": "prepared",
        "stage": stage,
        "execution": {"implemented": False, "executed": False},
        "experiment_fingerprint": digest_json(identity),
        "identity": identity,
        "config": config.resolved(),
        "config_source": {
            "path": str(loaded.path),
            "sha256": hashlib.sha256(loaded.source).hexdigest(),
        },
        "seed_plan": {
            "training": seed,
            "split": config.dataset.split_seed,
            "applied": False,
            "deterministic_requested": config.runtime.deterministic,
        },
        "dataset": {"split_manifest": split},
        "source": source,
        "environment": environment,
        "hardware": hardware,
        "reproducibility": {"ready_for_execution": False, "unresolved": unresolved},
        "artifacts": {"original_config": "source.yaml", "resolved_config": "config.yaml"},
    }
    run_dir = run_root.resolve() / config.experiment.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        (run_dir / "source.yaml").write_bytes(loaded.source)
        (run_dir / "config.yaml").write_text(
            yaml.safe_dump(config.resolved(), sort_keys=False),
            encoding="utf-8",
        )
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except Exception:
        shutil.rmtree(run_dir)
        raise
    return run_dir
