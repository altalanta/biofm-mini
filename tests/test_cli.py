from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from biofm.cli import app
from biofm.configuration import load_config
from biofm.datamodels import (
    ClinicalRecord,
    MicroscopySample,
    ScrnaSample,
    build_dataset_bundle,
)
from biofm.training.loop import TrainingSummary


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


def _bundle(tmp_path: Path) -> Any:
    microscopy = [
        MicroscopySample(
            sample_id="sample-1",
            image_path=tmp_path / "sample-1.png",
            height=32,
            width=32,
            channels=3,
        )
    ]
    scrna = [
        ScrnaSample(
            sample_id="sample-1",
            expression_path=tmp_path / "sample-1.csv",
            num_genes=8,
            library_size=100,
            normalized=False,
        )
    ]
    clinical = [ClinicalRecord(sample_id="sample-1", label=1)]
    return build_dataset_bundle(microscopy=microscopy, scrna=scrna, clinical=clinical)


def test_prep_data_smoke(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setattr("biofm.utils.pipeline.ensure_data", lambda config: None)
    monkeypatch.setattr(
        "biofm.utils.pipeline.load_bundle", lambda config: _bundle(tmp_path)
    )
    result = runner.invoke(
        app,
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--out",
            str(tmp_path / "out"),
            "prep-data",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["profile"] == "toy"
    assert payload["summary"]["n_samples"] == 1


def test_train_wires_components(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr("biofm.utils.pipeline.ensure_data", lambda config: None)
    monkeypatch.setattr("biofm.utils.pipeline.load_bundle", lambda config: bundle)
    monkeypatch.setattr("biofm.utils.pipeline.build_model", lambda config, _: object())
    monkeypatch.setattr(
        "biofm.models.lora.apply_lora_adapters", lambda *args, **kwargs: 0
    )

    class DummyLoader:
        def __init__(self) -> None:
            self.dataset = SimpleNamespace(
                rna_dataset=SimpleNamespace(matrix=SimpleNamespace(shape=(1, 8)))
            )

        def __iter__(self):
            return iter([])

    monkeypatch.setattr(
        "biofm.training.utils.create_paired_dataloader",
        lambda **_: DummyLoader(),
    )
    monkeypatch.setattr(
        "biofm.training.utils.select_device",
        lambda option="auto": SimpleNamespace(type="cpu"),
    )
    monkeypatch.setattr(
        "biofm.training.utils.resolve_amp",
        lambda option, device: False,
    )
    monkeypatch.setattr(
        "biofm.training.loop.train_model",
        lambda **_: TrainingSummary(
            checkpoint_path=tmp_path / "chk.pt", losses=[0.8, 0.4]
        ),
    )

    result = runner.invoke(
        app,
        [
            "--data-dir",
            str(tmp_path / "data"),
            "--out",
            str(tmp_path / "out"),
            "train",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["epochs"] == 2
    assert payload["batch_size"] == load_config().train.batch_size


def test_report_reads_metrics(
    runner: CliRunner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = _bundle(tmp_path)
    monkeypatch.setattr("biofm.utils.pipeline.load_bundle", lambda config: bundle)
    metrics_path = tmp_path / "out" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps({"training": {"epochs": 1}}), encoding="utf-8")
    result = runner.invoke(
        app,
        [
            "--out",
            str(tmp_path / "out"),
            "report",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["metrics"]["training"]["epochs"] == 1
