from pathlib import Path

from biofm.dataio.toydata import generate_toy_dataset
from biofm.datamodels import validate_dataset_counts


def test_generate_toy_dataset(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    bundle = generate_toy_dataset(
        data_dir=data_dir,
        n_samples=6,
        image_size=32,
        n_genes=16,
        seed=1,
    )
    summary = validate_dataset_counts(bundle)
    assert summary["n_samples"] == 6
    assert (data_dir / "raw" / "microscopy").exists()
    assert (data_dir / "raw" / "scrna").exists()
    assert (data_dir / "raw" / "clinical" / "clinical.csv").exists()
