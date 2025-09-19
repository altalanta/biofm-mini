"""Pydantic schemas describing the multimodal dataset layout."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

from pydantic import BaseModel, Field, ValidationError, model_validator


class MicroscopySample(BaseModel):
    """Description of a microscopy tile stored on disk."""

    sample_id: str
    image_path: Path
    height: int = Field(..., gt=0)
    width: int = Field(..., gt=0)
    channels: int = Field(3, gt=0)
    staining: Optional[str] = None

    @model_validator(mode="after")
    def _check_path(self) -> "MicroscopySample":
        if self.image_path.suffix.lower() not in {".png", ".tif", ".tiff"}:
            raise ValueError(f"Unsupported image format: {self.image_path}")
        return self


class ScrnaSample(BaseModel):
    """Description of an scRNA-seq profile."""

    sample_id: str
    expression_path: Path
    num_genes: int = Field(..., gt=0)
    library_size: int = Field(..., gt=0)
    normalized: bool = False
    row_index: Optional[int] = Field(default=None, ge=0)
    modality: str = Field("rna", frozen=True)

    @model_validator(mode="after")
    def _check_extension(self) -> "ScrnaSample":
        if self.expression_path.suffix.lower() not in {".h5ad", ".csv", ".parquet"}:
            raise ValueError("Expression data must be .h5ad, .csv, or .parquet")
        return self


class ClinicalRecord(BaseModel):
    """Clinical metadata associated with a sample."""

    sample_id: str
    label: int = Field(..., ge=0, le=1)
    age: Optional[float] = Field(None, ge=0.0, le=120.0)
    sex: Optional[str] = Field(default=None)
    metadata: Dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalise_sex(self) -> "ClinicalRecord":
        if self.sex is not None:
            value = self.sex.strip().lower()
            if value not in {"female", "male", "other", "unknown"}:
                raise ValueError("sex must be female, male, other, or unknown")
            self.sex = value
        return self


class DatasetBundle(BaseModel):
    """Container holding the aligned multimodal dataset."""

    microscopy: List[MicroscopySample]
    scrna: List[ScrnaSample]
    clinical: List[ClinicalRecord]

    @model_validator(mode="after")
    def _validate_alignment(self) -> "DatasetBundle":
        microscopy_ids = {s.sample_id for s in self.microscopy}
        scrna_ids = {s.sample_id for s in self.scrna}
        clinical_ids = {s.sample_id for s in self.clinical}
        _ensure_unique_ids("microscopy", self.microscopy)
        _ensure_unique_ids("scrna", self.scrna)
        _ensure_unique_ids("clinical", self.clinical)

        missing_rna = microscopy_ids - scrna_ids
        missing_clinical = microscopy_ids - clinical_ids
        if missing_rna:
            raise ValueError(f"Missing RNA records for samples: {sorted(missing_rna)}")
        if missing_clinical:
            raise ValueError(
                f"Missing clinical records for samples: {sorted(missing_clinical)}"
            )
        return self

    @property
    def sample_ids(self) -> List[str]:
        return [sample.sample_id for sample in self.microscopy]

    def summary(self) -> Dict[str, int]:
        """Return counts across modalities for quick reporting."""
        return {
            "n_samples": len(self.microscopy),
            "n_rna": len(self.scrna),
            "n_clinical": len(self.clinical),
        }


def _ensure_unique_ids(name: str, samples: Sequence[BaseModel]) -> None:
    seen: Set[str] = set()
    for sample in samples:
        sample_id = getattr(sample, "sample_id")
        if sample_id in seen:
            raise ValueError(f"Duplicate sample_id in {name}: {sample_id}")
        seen.add(sample_id)


def validate_dataset_counts(bundle: DatasetBundle) -> Dict[str, float]:
    """Perform dataset-level integrity checks and return simple summary stats."""

    n_samples = len(bundle.microscopy)
    if n_samples == 0:
        raise ValueError("Dataset contains no microscopy samples")

    label_counts = _count_labels(bundle.clinical)
    if len({label for label, count in label_counts.items() if count > 0}) < 2:
        raise ValueError("Clinical labels must include both classes for contrastive training")

    average_genes = sum(sample.num_genes for sample in bundle.scrna) / n_samples
    average_library = sum(sample.library_size for sample in bundle.scrna) / n_samples

    return {
        "n_samples": n_samples,
        "positive_labels": label_counts.get(1, 0),
        "negative_labels": label_counts.get(0, 0),
        "avg_genes": average_genes,
        "avg_library_size": average_library,
    }


def _count_labels(records: Iterable[ClinicalRecord]) -> Dict[int, int]:
    counts: Dict[int, int] = {0: 0, 1: 0}
    for record in records:
        counts[record.label] += 1
    return counts


def build_dataset_bundle(
    microscopy: Sequence[MicroscopySample],
    scrna: Sequence[ScrnaSample],
    clinical: Sequence[ClinicalRecord],
) -> DatasetBundle:
    """Helper that wraps validation errors with clearer context."""

    try:
        bundle = DatasetBundle(
            microscopy=list(microscopy),
            scrna=list(scrna),
            clinical=list(clinical),
        )
    except ValidationError as exc:  # pragma: no cover - defensive
        raise ValueError(f"Invalid dataset: {exc}") from exc
    validate_dataset_counts(bundle)
    return bundle


__all__ = [
    "MicroscopySample",
    "ScrnaSample",
    "ClinicalRecord",
    "DatasetBundle",
    "build_dataset_bundle",
    "validate_dataset_counts",
]
