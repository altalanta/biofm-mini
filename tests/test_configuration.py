from __future__ import annotations

from pathlib import Path

from biofm.configuration import load_config


def test_load_config_overrides(tmp_path: Path) -> None:
    cfg = load_config(profile="toy", overrides={"paths": {"data_dir": str(tmp_path)}})
    assert cfg.data.profile == "toy"
    assert cfg.paths.data_dir == tmp_path.resolve()
    assert cfg.paths.processed_dir == tmp_path.resolve() / "processed"


def test_load_config_from_file(tmp_path: Path) -> None:
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(
        """
project:
  seed: 42
paths:
  data_dir: custom_data
  output_dir: run_out
train:
  epochs: 3
  batch_size: 2
""",
        encoding="utf-8",
    )
    cfg = load_config(config_path=config_path)
    assert cfg.project.seed == 42
    assert cfg.train.epochs == 3
    assert cfg.paths.data_dir.name == "custom_data"
    assert cfg.paths.output_dir.name == "run_out"
