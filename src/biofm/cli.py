"""Typer-based CLI entry point for the BioFM mini-stack."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import click
import typer
from rich.console import Console
from rich.table import Table

from biofm.configuration import (
    AmpOption,
    BioFMConfig,
    ConfigProfile,
    DeviceOption,
    config_to_dict,
    load_config,
)
from biofm.datamodels import DatasetBundle, validate_dataset_counts
from biofm.utils.seeds import seed_everything

app = typer.Typer(
    help="Train and evaluate the BioFM multimodal mini-stack.",
    no_args_is_help=True,
)
console = Console()

PROFILE_OPTION = typer.Option(
    "toy",
    "--profile",
    help="Packaged configuration profile to load.",
    show_default=True,
    click_type=click.Choice(["toy", "real"], case_sensitive=False),
)
CONFIG_OPTION = typer.Option(
    None,
    "--config",
    help="Path to a custom YAML config overriding packaged defaults.",
)
DATA_DIR_OPTION = typer.Option(
    None,
    "--data-dir",
    help="Override the data directory (defaults to config).",
)
OUT_DIR_OPTION = typer.Option(
    None,
    "--out",
    help="Override the output directory (defaults to config).",
)
DEVICE_OPTION = typer.Option(
    "auto",
    "--device",
    help="Execution device (auto, cpu, cuda).",
    show_default=True,
    click_type=click.Choice(["auto", "cpu", "cuda"], case_sensitive=False),
)
MIXED_PRECISION_OPTION = typer.Option(
    "auto",
    "--mixed-precision",
    help="Mixed precision policy (auto/on/off).",
    show_default=True,
    click_type=click.Choice(["auto", "on", "off"], case_sensitive=False),
)
SEED_OPTION = typer.Option(
    None,
    "--seed",
    help="Deterministic seed override for this run.",
)
VERBOSE_OPTION = typer.Option(False, "--verbose", help="Enable debug logging.")
TABLE_OPTION = typer.Option(False, "--table", help="Render output as a table.")
CHECKPOINT_OPTION = typer.Option(
    None,
    "--checkpoint",
    help="Path to checkpoint to load before running the command.",
)


@dataclass
class RuntimeState:
    config: BioFMConfig
    device_option: DeviceOption
    amp_option: AmpOption
    seed: int


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    for extractor in ("item", "tolist"):
        method = getattr(value, extractor, None)
        if callable(method):
            extracted = method()
            if isinstance(extracted, (int, float)):
                return extracted
    return value


def _ensure_output_dirs(config: BioFMConfig) -> None:
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    config.paths.artifacts_dir.mkdir(parents=True, exist_ok=True)


def _write_metrics(
    config: BioFMConfig,
    section: str,
    payload: Mapping[str, object],
) -> None:
    _ensure_output_dirs(config)
    metrics_path = config.paths.output_dir / "metrics.json"
    if metrics_path.exists():
        existing: MutableMapping[str, object] = json.loads(
            metrics_path.read_text(encoding="utf-8")
        )
    else:
        existing = {}
    existing[section] = dict(payload)
    metrics_path.write_text(
        json.dumps(existing, indent=2, default=_json_default),
        encoding="utf-8",
    )


def _write_artifact(
    config: BioFMConfig,
    name: str,
    payload: Mapping[str, object],
) -> Path:
    _ensure_output_dirs(config)
    artifact_path = config.paths.artifacts_dir / f"{name}.json"
    artifact_path.write_text(
        json.dumps(dict(payload), indent=2, default=_json_default),
        encoding="utf-8",
    )
    return artifact_path


def _emit(payload: Mapping[str, object], table: bool, title: str) -> None:
    if table:
        table_obj = Table(title=title)
        table_obj.add_column("Key", style="bold")
        table_obj.add_column("Value")
        for key, value in payload.items():
            rendered = (
                json.dumps(value, indent=2, default=_json_default)
                if isinstance(value, Mapping)
                else str(value)
            )
            table_obj.add_row(key, rendered)
        console.print(table_obj)
    else:
        typer.echo(json.dumps(dict(payload), indent=2, default=_json_default))


def _bundle_summary(bundle: DatasetBundle) -> Mapping[str, object]:
    from biofm.dataio.clinical import clinical_summary

    summary = validate_dataset_counts(bundle)
    summary["clinical"] = clinical_summary(bundle.clinical)
    return summary


def _default_checkpoint(config: BioFMConfig) -> Path | None:
    user_defined = config.eval.checkpoint
    if user_defined:
        return user_defined
    candidate = config.paths.output_dir / "checkpoints" / "last.pt"
    return candidate if candidate.exists() else None


def _prepare_training_summary(
    checkpoint: Path,
    losses: list[float],
    device: str,
    amp: bool,
    batch_size: int,
) -> Mapping[str, object]:
    return {
        "checkpoint": checkpoint,
        "epochs": len(losses),
        "loss": losses,
        "device": device,
        "mixed_precision": amp,
        "batch_size": batch_size,
    }


def get_state(ctx: typer.Context) -> RuntimeState:
    state = ctx.obj
    if not isinstance(state, RuntimeState):  # pragma: no cover - defensive
        raise RuntimeError(
            "CLI state is uninitialised. Call through typer entry point."
        )
    return state


@app.callback()
def main(
    ctx: typer.Context,
    profile: str = PROFILE_OPTION,
    config_file: Path | None = CONFIG_OPTION,
    data_dir: Path | None = DATA_DIR_OPTION,
    out_dir: Path | None = OUT_DIR_OPTION,
    device: str = DEVICE_OPTION,
    mixed_precision: str = MIXED_PRECISION_OPTION,
    seed_override: int | None = SEED_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    overrides: dict[str, object] = {}
    if data_dir is not None:
        overrides.setdefault("paths", {})["data_dir"] = str(data_dir)
    if out_dir is not None:
        overrides.setdefault("paths", {})["output_dir"] = str(out_dir)
    if seed_override is not None:
        overrides.setdefault("project", {})["seed"] = seed_override

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    config = load_config(
        profile=cast(ConfigProfile, profile.lower()),
        config_path=config_file,
        overrides=overrides,
    )
    state = RuntimeState(
        config=config,
        device_option=cast(DeviceOption, device.lower()),
        amp_option=cast(AmpOption, mixed_precision.lower()),
        seed=config.project.seed,
    )
    ctx.obj = state


@app.command("prep-data")
def prep_data(
    ctx: typer.Context,
    table: bool = TABLE_OPTION,
) -> None:
    state = get_state(ctx)
    config = state.config
    seed_everything(state.seed)
    from biofm.utils.pipeline import ensure_data, load_bundle

    ensure_data(config)
    bundle = load_bundle(config)
    payload = {
        "profile": config.data.profile,
        "data_dir": config.paths.data_dir,
        "summary": _bundle_summary(bundle),
    }
    _emit(payload, table, title="Data Preparation")


@app.command("train")
def train(
    ctx: typer.Context,
    table: bool = TABLE_OPTION,
) -> None:
    state = get_state(ctx)
    config = state.config
    seed_everything(state.seed)
    from biofm.models.lora import apply_lora_adapters
    from biofm.training.loop import LoopConfig, train_model
    from biofm.training.utils import (
        create_paired_dataloader,
        resolve_amp,
        select_device,
    )
    from biofm.utils.pipeline import build_model, ensure_data, load_bundle

    ensure_data(config)
    bundle = load_bundle(config)
    device = select_device(state.device_option)
    dataloader = create_paired_dataloader(
        bundle=bundle,
        batch_size=int(config.train.batch_size),
        image_size=int(config.data.image_size),
        augment=bool(config.train.augment),
        select_hvg=int(config.data.select_hvg) if config.data.select_hvg else None,
        num_workers=int(config.data.num_workers),
        pin_memory=device.type == "cuda",
    )
    rna_input_dim = dataloader.dataset.rna_dataset.matrix.shape[1]  # type: ignore[attr-defined]
    model = build_model(config, rna_input_dim)
    if config.train.use_lora:
        apply_lora_adapters(
            model,
            target_modules=["image_encoder.head", "rna_encoder.network"],
            rank=int(config.train.lora_rank),
            alpha=float(config.train.lora_alpha),
        )
    loop_config = LoopConfig(
        epochs=int(config.train.epochs),
        learning_rate=float(config.train.learning_rate),
        weight_decay=float(config.train.weight_decay),
        grad_clip=float(config.train.grad_clip),
        amp_mode=state.amp_option,
        checkpoint_dir=config.paths.output_dir / "checkpoints",
    )
    summary = train_model(
        model=model,
        dataloader=dataloader,
        config=loop_config,
        device=device,
    )
    amp_enabled = resolve_amp(state.amp_option, device)
    payload = _prepare_training_summary(
        summary.checkpoint_path,
        summary.losses,
        device=device.type,
        amp=amp_enabled,
        batch_size=int(config.train.batch_size),
    )
    _write_metrics(config, "training", payload)
    _write_artifact(
        config,
        "training_loss",
        {
            "epoch": list(range(1, len(summary.losses) + 1)),
            "loss": summary.losses,
        },
    )
    _emit(payload, table, title="Training")


@app.command("embed")
def embed(
    ctx: typer.Context,
    checkpoint: Path | None = CHECKPOINT_OPTION,
    table: bool = TABLE_OPTION,
) -> None:
    state = get_state(ctx)
    config = state.config
    seed_everything(state.seed)
    from biofm.training.utils import select_device
    from biofm.utils.embeddings import export_embeddings
    from biofm.utils.pipeline import ensure_data

    ensure_data(config)
    device = select_device(state.device_option)
    chosen_checkpoint = checkpoint or _default_checkpoint(config)
    image_df, rna_df = export_embeddings(
        config=config,
        device=device,
        checkpoint=chosen_checkpoint,
        batch_size=int(config.eval.batch_size),
        save=True,
    )
    payload = {
        "samples": int(len(image_df)),
        "image_embedding_dim": int(image_df.shape[1]),
        "rna_embedding_dim": int(rna_df.shape[1]),
        "data_dir": config.paths.data_dir,
    }
    _write_artifact(config, "embedding_summary", payload)
    _emit(payload, table, title="Embedding Export")


@app.command("eval")
def evaluate(
    ctx: typer.Context,
    checkpoint: Path | None = CHECKPOINT_OPTION,
    table: bool = TABLE_OPTION,
) -> None:
    state = get_state(ctx)
    config = state.config
    seed_everything(state.seed)
    try:
        import numpy as np  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("NumPy is required for evaluation.") from exc
    from biofm.eval.linear_probe import fit_linear_probe
    from biofm.training.utils import select_device
    from biofm.utils.embeddings import export_embeddings, load_embeddings_from_disk
    from biofm.utils.pipeline import ensure_data, load_bundle

    ensure_data(config)
    device = select_device(state.device_option)
    chosen_checkpoint = checkpoint or _default_checkpoint(config)
    try:
        image_df, rna_df = load_embeddings_from_disk(config.paths.data_dir)
    except FileNotFoundError:
        image_df, rna_df = export_embeddings(
            config=config,
            device=device,
            checkpoint=chosen_checkpoint,
            batch_size=int(config.eval.batch_size),
            save=True,
        )
    bundle = load_bundle(config)
    labels_map = {record.sample_id: record.label for record in bundle.clinical}
    common_ids = image_df.index.intersection(rna_df.index)
    if common_ids.empty:
        raise ValueError("No overlapping samples found between modalities")
    features = np.hstack(
        [
            image_df.loc[common_ids].to_numpy(),
            rna_df.loc[common_ids].to_numpy(),
        ]
    )
    labels = np.array([labels_map[sample] for sample in common_ids])
    probe = fit_linear_probe(features, labels, seed=state.seed)
    metrics_payload = {
        name: {
            "point_estimate": metric.point_estimate,
            "ci_low": metric.ci_low,
            "ci_high": metric.ci_high,
        }
        for name, metric in probe.metrics.items()
    }
    payload = {
        "n_samples": int(len(common_ids)),
        "metrics": metrics_payload,
        "checkpoint": chosen_checkpoint,
    }
    _write_metrics(config, "evaluation", payload)
    _write_artifact(config, "evaluation_metrics", payload)
    _emit(payload, table, title="Evaluation")


@app.command("report")
def report(
    ctx: typer.Context,
    table: bool = TABLE_OPTION,
) -> None:
    state = get_state(ctx)
    config = state.config
    report_payload: MutableMapping[str, object] = {
        "config": config_to_dict(config),
    }
    try:
        from biofm.utils.pipeline import load_bundle

        bundle = load_bundle(config)
        report_payload["dataset"] = _bundle_summary(bundle)
    except Exception as exc:  # pragma: no cover - best effort reporting
        report_payload["dataset_error"] = str(exc)
    metrics_path = config.paths.output_dir / "metrics.json"
    if metrics_path.exists():
        report_payload["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8"))
    else:
        report_payload["metrics"] = {}
    artifacts = sorted(config.paths.artifacts_dir.glob("*.json"))
    report_payload["artifacts"] = [artifact.name for artifact in artifacts]
    _emit(report_payload, table, title="Run Report")


def main_cli() -> None:  # pragma: no cover - Typer entry
    app()


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main_cli()
