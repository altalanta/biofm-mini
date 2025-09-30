# Dataset Sheet

## Overview
- **Dataset Name**: biofm-mini toy set (replace with real dataset name when supplied)
- **Modality Coverage**: Microscopy tiles, scRNA-seq profiles, binary clinical metadata
- **Default Size**: 16 paired samples with 64×64 RGB tiles and 64 genes

## Collection
- **Source**: Synthetic via `biofm.dataio.toydata.generate_toy_dataset`
- **Generation Notes**: Procedurally generated textures with label-specific perturbations; RNA features drawn from gamma distributions with class shifts
- **Intended Real-World Replacement**: Users should document acquisition pipelines, QC steps, and consent status here when swapping to real cohorts

## Preprocessing & Validation
- Microscopy: optional tiling and normalisation (`src/biofm/preprocess/microscopy_prep.py`)
- RNA: log-normalisation, highly variable gene selection (`src/biofm/dataio/scrna.py`)
- Clinical: schema-enforced CSV with missingness checks (`src/biofm/dataio/clinical.py`)
- Dataset-level validation through `DatasetBundle` and `validate_dataset_counts`

## Splits & Usage
- Default workflow uses all samples for contrastive pretraining; downstream probes operate on frozen embeddings
- For real datasets, document train/val/test splits and any cross-validation strategy

## License & Ethics
- Synthetic toy data is MIT licensed alongside the repo
- Replace with appropriate licenses, IRB approvals, and DUA information for real datasets
