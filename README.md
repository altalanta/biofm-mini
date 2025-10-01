# biofm-mini

[![CI](https://github.com/altalanta/biofm-mini/actions/workflows/ci.yml/badge.svg)](https://github.com/altalanta/biofm-mini/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/badge/coverage-≥80%25-brightgreen.svg)](https://github.com/altalanta/biofm-mini)

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

## Docker (CPU)

Run the full pipeline in a containerized environment:

```bash
docker build -t biofm-mini:cpu -f Dockerfile.cpu .
docker run --rm biofm-mini:cpu
```

This builds a CPU-only container, installs dependencies, and runs `make train && make eval`.

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

## Governance

This project follows responsible AI and open science practices:

- **[MODEL_CARD.md](MODEL_CARD.md)**: Details model architecture, intended use, limitations, and risks
- **[DATASET_SHEET.md](DATASET_SHEET.md)**: Documents data sources, collection methods, and known biases

Additional governance documentation:
- [CONTRIBUTING.md](CONTRIBUTING.md): Contribution guidelines and development practices  
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md): Community standards and behavior expectations

## Deterministic Demo

Run a tiny, reproducible pipeline and view artifacts:

```bash
make demo
ls -1 artifacts/demo
```

This creates a deterministic end-to-end demonstration with:

### Generated Artifacts
- **`metrics.json`**: Core classification metrics (AUROC, AUPRC, accuracy, precision, recall, F1, ECE)
- **`roc_curve.png`**: Receiver Operating Characteristic curve plot
- **`pr_curve.png`**: Precision-Recall curve plot  
- **`calibration_curve.png`**: Model calibration analysis
- **`confusion_matrix.png`**: Confusion matrix visualization
- **`version.txt`**: Git SHA, package version, and seed for reproducibility

### Reproducibility Features
- **Fixed seed (1337)**: Ensures identical results across runs
- **Synthetic data**: 1000-sample binary classification with ~30% class imbalance
- **Deterministic splits**: Fixed 700/300 train/test division
- **Statistical validation**: Tests verify metrics meet quality thresholds

### Statistical Tests and Thresholds
The demo includes comprehensive endpoint tests with the following quality gates:

- **AUROC ≥ 0.70**: Model discrimination above random baseline
- **AUPRC ≥ prevalence + 0.10**: Precision-recall performance shows meaningful uplift  
- **Accuracy ≥ 0.70**: Overall classification performance threshold
- **ECE ≤ 0.10**: Expected Calibration Error for prediction reliability
- **Determinism**: Identical results within 1e-6 tolerance across runs

These thresholds reflect sanity floors for model performance over prevalence baselines and ensure calibration quality suitable for downstream decision-making.

### Usage
```bash
# Run the demo
make demo

# View artifacts
tree artifacts/demo

# Clean artifacts  
make clean-artifacts

# Run statistical tests
make test
```

The demo completes in under 30 seconds on CPU and produces identical metrics on every run, making it ideal for CI/CD validation and reproducibility verification.

### How this maps to the role
- Contribute to ML models… (data wrangling/curation/validation)
- Lead data management, software infrastructure and AI/ML workflow best practices and policies
- Participate in institute-wide initiatives…
- Promote open science…
- Advance community standards for scalability…
- Foster a collaborative and inclusive work environment…
