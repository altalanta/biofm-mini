# biofm-mini

biofm-mini demonstrates a laptop-friendly multimodal foundation model workflow that pairs microscopy tiles, scRNA-seq profiles, and binary clinical labels. end-to-end data wrangling, FM-style contrastive pretraining, and reproducible evaluation across imaging and omics.

## Quickstart in Three Commands
```
make setup
make train
make eval
```
Each command runs in under ten minutes on CPU-only laptops by default. `make train` falls back to toy data and mixed precision automatically uses AMP when CUDA is present.

## Data Schema & Layout
- `data/raw/microscopy/`: PNG/TIF tiles (`sample_id.png`)
- `data/raw/scrna/`: per-sample CSV or `.h5ad` matrices with gene × count values
- `data/raw/clinical/clinical.csv`: tabular metadata with `sample_id`, `label`, `age`, `sex`, plus optional columns
- `data/processed/`: embeddings exported as Parquet; additional artefacts may be added alongside

All raw modalities are validated with pydantic schemas (`MicroscopySample`, `ScrnaSample`, `ClinicalRecord`) and combined via `DatasetBundle`. Schema counts and ranges are checked before training, and loaders gracefully degrade if `scanpy`/`anndata` are absent.

## Swap In Real Datasets
1. Drop microscopy tiles, scRNA-seq matrices, and clinical CSVs into `data/raw/...` matching the schema above.
2. Run `python scripts/prep_data.py --mode real --input-dir data` to validate shape, counts, and metadata.
3. Adjust configs (e.g. `configs/data/real.yaml`, `configs/train/lora.yaml`) to reflect image size, HVG cutoffs, or LoRA fine-tuning preferences and rerun the Make targets.

## Open Science Checklist
- ✅ MIT License
- ✅ Reproducible env (`pyproject.toml`, Dockerfile, Makefile)
- ✅ Automated lint/type/test via pre-commit + GitHub Actions
- ✅ Documented data schema, model card, dataset sheet
- ✅ Synthetic fallback data for transparent benchmarking
- ✅ Easy path to plug external datasets (no vendor lock-in)

## Repo Map & Allen Institute Alignment
- `src/biofm/datamodels.py`: strict schemas and dataset validation for imaging, scRNA, and clinical tables
- `src/biofm/dataio/`: modality-aware loaders (microscopy augmentation, Scanpy-aware RNA ingestion, clinical QA, synthetic toy data)
- `src/biofm/models/`: ResNet18 encoder, RNA MLP, CLIP head, optional LoRA stubs
- `src/biofm/training/`: reproducible training loop with AMP, grad clipping, checkpoints
- `src/biofm/eval/`: linear probe with AUROC/AUPRC + bootstrap CIs, decision curve utilities
- `src/biofm/preprocess/`: tiling/normalisation and scRNA filtering helpers
- `src/biofm/utils/`: registries, seeding, embedding export
- `scripts/`: `prep_data`, `train`, `embed`, `eval` Hydra-aware entrypoints
- `configs/`: hydra defaults plus data/train/eval overrides for toy vs. real workflows
- `tests/`: unit coverage for toy data, schemas, model step, and metrics

### How this maps to the role
- Contribute to ML models… (data wrangling/curation/validation)
- Lead data management, software infrastructure and AI/ML workflow best practices and policies
- Participate in institute-wide initiatives…
- Promote open science…
- Advance community standards for scalability…
- Foster a collaborative and inclusive work environment…

Further context and governance lives in `MODEL_CARD.md`, `DATASET_SHEET.md`, `CONTRIBUTING.md`, and the `CODE_OF_CONDUCT.md`.
