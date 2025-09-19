from pathlib import Path

from biofm.dataio.toydata import generate_toy_dataset
from biofm.datamodels import validate_dataset_counts


def test_generate_toy_dataset(tmp_path: Path) -> None:
    bundle = generate_toy_dataset(tmp_path, n_samples=6, image_size=32, n_genes=16, seed=1)
    summary = validate_dataset_counts(bundle)
    assert summary["n_samples"] == 6
    assert (tmp_path / "data" / "raw" / "microscopy").exists()
    assert (tmp_path / "data" / "raw" / "scrna").exists()
    assert (tmp_path / "data" / "raw" / "clinical" / "clinical.csv").exists()
