"""Clinical CSV ingestion with schema validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from biofm.datamodels import ClinicalRecord

LOGGER = logging.getLogger(__name__)


def load_clinical_records(path: Path) -> List[ClinicalRecord]:
    """Load a CSV file into validated clinical records."""

    df = pd.read_csv(path)
    if "sample_id" not in df or "label" not in df:
        raise ValueError("Clinical CSV must contain sample_id and label columns")

    records: List[ClinicalRecord] = []
    for row in df.to_dict(orient="records"):
        metadata: Dict[str, str] = {
            key: str(value)
            for key, value in row.items()
            if key not in {"sample_id", "label", "age", "sex"} and pd.notnull(value)
        }
        try:
            record = ClinicalRecord(
                sample_id=str(row["sample_id"]),
                label=int(row["label"]),
                age=float(row["age"]) if pd.notnull(row.get("age")) else None,
                sex=row.get("sex"),
                metadata=metadata,
            )
            records.append(record)
        except ValueError as exc:
            LOGGER.warning("Skipping clinical row %s: %s", row.get("sample_id"), exc)
    return records


def clinical_summary(records: Iterable[ClinicalRecord]) -> Dict[str, float]:
    values = list(records)
    if not values:
        return {"n_records": 0}
    prevalence = sum(record.label for record in values) / len(values)
    ages = [record.age for record in values if record.age is not None]
    return {
        "n_records": len(values),
        "prevalence": prevalence,
        "age_mean": float(sum(ages) / len(ages)) if ages else float("nan"),
    }


__all__ = ["load_clinical_records", "clinical_summary"]
