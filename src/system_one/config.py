"""Validate YAML once and pass typed settings to experiment scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonemptyString = Annotated[str, Field(min_length=1)]
Stage = Literal["baseline", "finetune", "calibrate", "evaluate"]
STAGES: tuple[Stage, ...] = ("baseline", "finetune", "calibrate", "evaluate")


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class ExperimentConfig(StrictConfig):
    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
    seed: Seed


class DatasetConfig(StrictConfig):
    adapter: NonemptyString
    version: NonemptyString | None = None
    split_seed: Seed
    split_manifest: NonemptyString | None = None
    split_id: NonemptyString | None = None
    evidence_template: NonemptyString


class ModelConfig(StrictConfig):
    adapter: Literal["laya", "clef"]
    checkpoint: NonemptyString
    revision: Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")] | None = None
    max_length: PositiveInt


class TrainingConfig(StrictConfig):
    objective: Literal["cross_entropy", "rlcd"]
    strategy: Literal["full", "head_only", "lora"]
    epochs: PositiveInt
    batch_size: PositiveInt
    gradient_accumulation_steps: PositiveInt = 1
    learning_rate: Annotated[float, Field(gt=0)]
    weight_decay: Annotated[float, Field(ge=0)] = 0.0


class CalibrationConfig(StrictConfig):
    method: Literal["temperature_scaling", "none"]


class EvaluationConfig(StrictConfig):
    selection_metric: Literal["average_precision", "log_loss", "brier_score"]
    review_budget_fraction: Annotated[float, Field(gt=0, le=1)]


class RuntimeConfig(StrictConfig):
    device: Literal["cpu", "cuda", "mps"]
    precision: Literal["float32", "float16", "bfloat16"]
    deterministic: bool = True


class RunConfig(StrictConfig):
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig


def load_config(path: Path, *, seed: int | None = None) -> RunConfig:
    config = RunConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if seed is not None:
        experiment = ExperimentConfig(**{**config.experiment.model_dump(), "seed": seed})
        config = config.model_copy(update={"experiment": experiment})
    return config
