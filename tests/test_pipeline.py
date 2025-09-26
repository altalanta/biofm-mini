from __future__ import annotations

from pathlib import Path

from biofm.configuration import load_config
from biofm.utils.pipeline import ensure_data, load_bundle


def test_ensure_data_generates_toy(tmp_path: Path) -> None:
    cfg = load_config(profile="toy", overrides={"paths": {"data_dir": str(tmp_path)}})
    ensure_data(cfg)
    bundle = load_bundle(cfg)
    assert bundle.microscopy
    assert bundle.scrna
    assert bundle.clinical
    clinical_dir = cfg.paths.raw_dir / "clinical"
    assert (clinical_dir / "clinical.csv").exists()
