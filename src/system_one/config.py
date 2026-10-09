"""Strict, versioned experiment configuration; no dataset or model loading."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

Seed = Annotated[int, Field(ge=0, le=2**32 - 1)]
PositiveInt = Annotated[int, Field(gt=0)]
NonemptyString = Annotated[str, Field(min_length=1, pattern=r"\S")]
Stage = Literal["baseline", "finetune", "calibrate", "evaluate"]
STAGES: tuple[Stage, ...] = ("baseline", "finetune", "calibrate", "evaluate")


class ConfigError(ValueError):
    """A configuration cannot be interpreted without ambiguity."""


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)


class ExperimentConfig(StrictConfig):
    name: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
    training_seeds: Annotated[list[Seed], Field(min_length=1)]

    @field_validator("training_seeds")
    @classmethod
    def distinct_seeds(cls, seeds: list[int]) -> list[int]:
        if len(set(seeds)) != len(seeds):
            raise ValueError("training_seeds must be unique")
        return seeds


class DatasetConfig(StrictConfig):
    adapter: NonemptyString
    version: NonemptyString | None = None
    split_seed: Seed
    split_manifest: NonemptyString | None = None
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
    learning_rate: Annotated[float, Field(gt=0, le=1)]
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
    schema_version: Literal[1]
    experiment: ExperimentConfig
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    calibration: CalibrationConfig
    evaluation: EvaluationConfig
    runtime: RuntimeConfig

    @field_validator("schema_version", mode="before")
    @classmethod
    def integer_version(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    def resolved(self) -> dict:
        return self.model_dump(mode="json")

    def sha256(self) -> str:
        return digest_json(self.resolved())


def digest_json(value: object) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class UniqueKeyLoader(yaml.SafeLoader):
    """Reject duplicate keys rather than silently keeping their last value."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        result = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("YAML mapping keys must be strings")
            if key in result:
                raise ConfigError(f"duplicate YAML key: {key!r}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    source: bytes
    config: RunConfig


def load_config(path: Path) -> LoadedConfig:
    path = path.resolve()
    try:
        source = path.read_bytes()
        raw = yaml.load(source.decode("utf-8"), Loader=UniqueKeyLoader)
        if not isinstance(raw, dict):
            raise ConfigError("configuration must be a YAML mapping")
        config = RunConfig.model_validate(raw)
    except (OSError, UnicodeError, yaml.YAMLError, ValidationError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc
    return LoadedConfig(path=path, source=source, config=config)
