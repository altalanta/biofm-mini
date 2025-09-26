"""Configuration schema and loader for BioFM."""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

ConfigProfile = Literal["toy", "real"]
DeviceOption = Literal["auto", "cpu", "cuda"]
AmpOption = Literal["auto", "on", "off"]


class ProjectConfig(BaseModel):
    """Top-level project metadata and run controls."""

    seed: int = Field(
        7, ge=0, description="Deterministic seed applied across libraries."
    )


class PathsConfig(BaseModel):
    """Filesystem layout for inputs and outputs."""

    model_config = ConfigDict(validate_assignment=True)

    data_dir: Path = Field(default_factory=lambda: Path("data"))
    output_dir: Path = Field(default_factory=lambda: Path("outputs"))

    @model_validator(mode="after")
    def _normalise(self) -> PathsConfig:
        self.data_dir = self.data_dir.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()
        return self

    @computed_field(return_type=Path)
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @computed_field(return_type=Path)
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @computed_field(return_type=Path)
    def artifacts_dir(self) -> Path:
        return self.output_dir / "artifacts"


class DataConfig(BaseModel):
    """Settings controlling data preparation and loading."""

    profile: ConfigProfile = Field("toy")
    samples: int | None = Field(32, ge=2)
    image_size: int = Field(128, ge=32)
    select_hvg: int | None = Field(64, ge=2)
    n_genes: int = Field(256, ge=16)
    num_workers: int = Field(0, ge=0)


class TrainConfig(BaseModel):
    """Hyperparameters for the multimodal training loop."""

    batch_size: int = Field(4, ge=1)
    epochs: int = Field(2, ge=1)
    learning_rate: float = Field(5e-4, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    grad_clip: float = Field(1.0, ge=0)
    amp: AmpOption = Field("auto")
    augment: bool = Field(True)
    embedding_dim: int = Field(128, ge=8)
    rna_hidden_dim: int = Field(256, ge=8)
    dropout: float = Field(0.1, ge=0.0, le=1.0)
    temperature: float = Field(0.07, gt=0)
    pretrained: bool = Field(False)
    use_lora: bool = Field(False)
    lora_rank: int = Field(4, ge=1)
    lora_alpha: float = Field(8.0, gt=0)


class EvalConfig(BaseModel):
    """Evaluation configuration for embedding-based probes."""

    method: Literal["linear_probe", "knn"] = Field("linear_probe")
    batch_size: int = Field(8, ge=1)
    bootstrap_samples: int = Field(256, ge=16)
    checkpoint: Path | None = Field(default=None)
    knn: dict[str, Any] = Field(default_factory=lambda: {"k": 5, "metric": "cosine"})


class ReportConfig(BaseModel):
    """Reporting configuration for downstream summaries."""

    bootstrap_samples: int = Field(256, ge=16)


class BioFMConfig(BaseModel):
    """Fully validated configuration object consumed throughout the stack."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    def model_dump_json(self, **kwargs: Any) -> str:  # pragma: no cover - convenience
        return super().model_dump_json(**kwargs)


def _read_yaml_resource(profile: ConfigProfile) -> dict[str, Any]:
    resource_path = resources.files("biofm.configs").joinpath(f"{profile}.yaml")
    if not resource_path.is_file():  # pragma: no cover - defensive
        raise FileNotFoundError(f"Missing packaged config for profile '{profile}'")
    raw = resource_path.read_text(encoding="utf-8")
    return yaml.safe_load(raw) or {}


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            base[key] = _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    profile: ConfigProfile = "toy",
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> BioFMConfig:
    """Load configuration either from a packaged profile or a user-provided file."""

    if config_path is not None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    else:
        payload = _read_yaml_resource(profile)
    if overrides:
        payload = _deep_merge(payload, overrides)
    return BioFMConfig.model_validate(payload)


def config_to_dict(config: BioFMConfig) -> dict[str, Any]:
    """Serialise a config object into built-in Python structures."""

    return json.loads(config.model_dump_json())


__all__ = [
    "AmpOption",
    "BioFMConfig",
    "ConfigProfile",
    "DeviceOption",
    "EvalConfig",
    "PathsConfig",
    "ProjectConfig",
    "ReportConfig",
    "TrainConfig",
    "config_to_dict",
    "load_config",
]
