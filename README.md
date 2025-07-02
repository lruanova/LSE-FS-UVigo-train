# LSE-FS-UVigo Dataset and Fingerspelling Recognition

This repository contains the official codebase for the experiments presented in: [coming soon]


---

## Overview

This project provides:

- Training and evaluation code for continuous fingerspelling recognition using 3D hand keypoints
- Baseline and proposed models (AAGCN, EdgeConv+LSTM, BiLSTM)
- Data processing, augmentation, and evaluation tools
- Full configuration using [Hydra](https://hydra.cc/)

## Dataset

### LSE-FS-UVigo Download

The dataset and download instructions are available here:
**[👉 Dataset Download & Info (placeholder)]()**

### ChicagoFSWild+

See [ChicagoFSWild+](https://home.ttic.edu/~klivescu/ChicagoFSWild.htm) for download and more information about the ASL dataset.

---

## Installation

We recommend using [uv](https://uv.pycqa.org/) for fast and reproducible environment setup.

```bash
git clone https://github.com/tu_usuario/FingerspellingTrainer.git
cd FingerspellingTrainer
uv venv .venv
source .venv/bin/activate
uv pip install -e .
```

## Configuration

All experiment settings (dataset, model, augmentation, etc.) are defined in `fingerspelling_trainer/config/`, managed by Hydra.
Main entrypoint is `fingerspelling_trainer/main.py`.

### Structure
```
fingerspelling_trainer/
├── config/               # Hydra configuration files
├── data/                 # Data loading and transformations
├── training/             # Training loops, models, utilities
├── scripts/              # Other utilities (weights, scaler, etc.)
├── notebooks/            # Jupyter notebooks for data exploration
├── main.py               # Main training/evaluation entrypoint
├── infer.py              # Inference
├── ...
pyproject.toml            # Dependencies (uv, pip, custom wheels)
uv.lock                   # uv lockfile
```

### Train a model

```bash
uv run python fingerspelling_trainer/main.py dataset=fswild model=lstm
```

### Evaluate a model
```bash
uv run python fingerspelling_trainer/main.py mode=evaluate evaluation.checkpoint_path="path/to/ckpt.ckpt"
```

## Notes
-  **fast-ctc-decode** : This repo depends on a custom fork of fast-ctc-decode with support for multi-letter tokens (e.g., "LL", "RR", etc. in LSE). This is configured in pyproject.toml. You can use [this implementation](https://github.com/lruanova/fast-ctc-decode).
- **Keypoint Extraction**: The keypoints used here were extracted according to the protocol described in the article. You can use the provided: [keypoint extraction code](https://github.com/lruanova/LSE-FS-UVigo-extraction/). 
- Weights & Biases (See config) was used for tracking experiments, and Ray for distributed training. 

## How to cite

If you use this code, please cite:
```
[coming soon]
```

