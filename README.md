# Esophageal Endoscopy Contrastive-Attention Reproducibility Package

Flat-file PyTorch implementation for the manuscript **Advanced Deep Learning Approaches for Early Esophageal Cancer Detection with Attention and Contrastive Learning**.

The implementation preserves the raw dataset labels (`esophagus`, `no-esophagus`) unless a clinically validated mapping is explicitly supplied. It treats the study as single-modality endoscopic image analysis with pairwise/contrastive learning, performs splitting before augmentation, and uses a single-image classifier at inference.

## Files
`DATASET.md`, `requirements.txt`, `config.yaml`, `data_pipeline.py`, `model.py`, `train.py`, `evaluate.py`, `experiments.py`, `statistics.py`, `complexity.py`, `reproduce.py`

## Run
```bash
python -m pip install -r requirements.txt
python reproduce.py --config config.yaml
```

Edit `data.root` in `config.yaml` first.

## Expected dataset layout
```text
DATASET_ROOT/
  esophagus/
  no-esophagus/
```

The pipeline writes generated artifacts to `outputs/` at runtime. No numerical result is hard-coded.
