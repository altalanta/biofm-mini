# Model Card: biofm-mini CLIP

## Model Details
- **Architecture**: ResNet18 image encoder + MLP RNA encoder + two-layer projection heads trained with a CLIP-style symmetric cross-entropy loss.
- **Intended Use**: Demonstration of multimodal pretraining across microscopy and scRNA-seq with a binary clinical label for rapid experimentation on laptops.
- **Out-of-Scope**: Production deployment, diagnosis, treatment recommendations. The model is trained on synthetic toy data by default and should not be used for clinical decisions without further validation.

## Training Data
- Synthetic microscopy tiles with class-correlated textures
- Synthetic scRNA profiles with gene-level shifts tied to the clinical label
- Binary clinical labels sampled deterministically during generation

Users should replace toy data with curated datasets via `scripts/prep_data.py --mode real` and document provenance in `DATASET_SHEET.md`.

## Evaluation Data & Metrics
- Linear probe on concatenated modality embeddings
- Metrics: AUROC, AUPRC with 1000-sample bootstrap CIs, optional decision curve analysis

## Ethical Considerations
- Synthetic data avoids privacy concerns but does not reflect real-world biases
- When using real data, ensure appropriate consent, de-identification, and governance
- Keep clinical stakeholders in the loop and document evaluation under distributional shift

## Caveats and Recommendations
- Checkpoint shipping is disabled by default; run `make train` to produce a local model
- For GPU acceleration, enable `train.pretrained=true` and adjust batch sizes
- Consider LoRA adapters (`configs/train/lora.yaml`) for parameter-efficient fine-tuning on larger datasets
