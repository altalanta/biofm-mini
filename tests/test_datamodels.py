from pathlib import Path

import pytest

from biofm.datamodels import (
    ClinicalRecord,
    MicroscopySample,
    ScrnaSample,
    build_dataset_bundle,
)


def _build_samples(tmp_path: Path):
    microscopy = [
        MicroscopySample(
            sample_id="sample_0",
            image_path=tmp_path / "sample_0.png",
            height=64,
            width=64,
            channels=3,
        ),
        MicroscopySample(
            sample_id="sample_1",
            image_path=tmp_path / "sample_1.png",
            height=64,
            width=64,
            channels=3,
        ),
    ]
    scrna = [
        ScrnaSample(
            sample_id="sample_0",
            expression_path=tmp_path / "sample_0.csv",
            num_genes=10,
            library_size=1000,
            normalized=False,
        ),
        ScrnaSample(
            sample_id="sample_1",
            expression_path=tmp_path / "sample_1.csv",
            num_genes=10,
            library_size=1000,
            normalized=False,
        ),
    ]
    clinical = [
        ClinicalRecord(sample_id="sample_0", label=1, age=50.0, sex="female"),
        ClinicalRecord(sample_id="sample_1", label=0, age=45.0, sex="male"),
    ]
    return microscopy, scrna, clinical


def test_build_dataset_bundle(tmp_path: Path) -> None:
    microscopy, scrna, clinical = _build_samples(tmp_path)
    bundle = build_dataset_bundle(microscopy=microscopy, scrna=scrna, clinical=clinical)
    assert bundle.sample_ids == ["sample_0", "sample_1"]


def test_duplicate_ids_raise(tmp_path: Path) -> None:
    microscopy, scrna, clinical = _build_samples(tmp_path)
    clinical.append(ClinicalRecord(sample_id="sample_0", label=0))
    with pytest.raises(ValueError):
        build_dataset_bundle(microscopy=microscopy, scrna=scrna, clinical=clinical)
