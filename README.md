# DAS Preprocessing & Vehicle Detection

A Python toolkit for loading, preprocessing, and running ML on **Distributed Acoustic Sensing (DAS)** data — with a full pipeline from raw HDF5 files to vehicle detection and weight estimation.

---

## Overview

```
HDF5 Files → DAS.py (loading) → preprocessing.py (signal processing)
    → data_prep.ipynb (windowing + labeling) → dataset.py (PyTorch)
    → train.py (denoising → detection / weight regression)
```

**Tasks:**
- **Denoising** — UNet trained to map raw DAS → preprocessed (clean) signal
- **Detection** — Transformer predicting vehicle count + type (SUV / van / sedan / truck / mixed) per window
- **Weight** — CNN regressing vehicle weight from denoised DAS windows

---

## Installation

```bash
git clone https://github.com/UWGeoD/DAS_Preprocessing.git
cd DAS_Preprocessing

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

For ML training, also install:
```bash
pip install torch torchvision scikit-learn
```

---

## Configuration

Copy the example config and fill in your local paths:

```bash
cp config.example.py config.py
# Edit config.py with your HDF5 file path, video path, and label CSV path
```

`config.py` is gitignored — never committed.

---

## Usage

### 1. Prepare training data

Run `data_prep.ipynb` to window the DAS array and generate `data/raw/sample_XXXXXX.npy` + `data/labels.csv`.

### 2. Train models

```bash
# Step 1: train denoiser (also saves denoised samples to data/denoised/)
python train.py --task denoising

# Step 2a: train vehicle detector
python train.py --task detection

# Step 2b: train weight regressor
python train.py --task weight
```

Override config values from the CLI:
```bash
python train.py --task detection --epochs 100 --lr 5e-4
```

All hyperparameters live in `configs/<task>.yaml`. Trained models are saved to `results/<task>/best_model.pt`.

### 3. Visualize results

Open the corresponding notebook for a trained model:
- `train_denoising.ipynb` — compare raw / preprocessed / UNet output
- `train_detection.ipynb` — per-sample predictions with count & type metrics
- `train_weight_predicting.ipynb` — predicted vs. actual weight plots

---

## Project Structure

```
DAS_Preprocessing/
├── DAS.py                  # HDF5 loader (OptaSense, Silixa); single + multi-file
├── preprocessing.py        # Composable signal processing (detrend, bandpass, f-k, curvelet)
├── dataset.py              # PyTorch Datasets: DASSampleDataset, DASCountDataset, DASWeightDataset
├── Utilities.py            # Visualization helpers (plot_das_data, plot_single)
├── train.py                # Unified training CLI
├── configs/
│   ├── denoising.yaml
│   ├── detection.yaml
│   └── weight.yaml
├── models/
│   ├── unet.py             # UNet denoiser
│   ├── detection_cnn.py    # DASCountCNN
│   ├── detection_transformer.py  # DASCountTransformer
│   └── weight_cnn.py       # DASWeightCNN
├── data_prep.ipynb         # Label windows and export .npy samples
├── demo.ipynb              # Quickstart: load, inspect, and visualize DAS data
├── train_denoising.ipynb   # Visualization notebook for denoising model
├── train_detection.ipynb   # Visualization notebook for detection model
├── train_weight_predicting.ipynb  # Visualization notebook for weight model
├── config.example.py       # Template for local paths
└── requirements.txt
```

---

## Preprocessing Steps

Build a pipeline with `make_preprocess`:

```python
from preprocessing import make_preprocess

pp = make_preprocess(
    steps=[
        ("detrend",   {"axis": 1}),
        ("bandpass",  {"f_lo": 0.5, "f_hi": 30.0, "order": 2, "axis": 1}),
        ("fk_filter", {"vmin": 10.0, "vmax": 60.0, "taper": 0.05}),
    ],
    dx=1.0,   # meters per channel
    dt=5e-4,  # seconds per sample (1/fs)
)

clean = pp(raw_data, fs=2000)  # raw shape: (n_channels, n_time)
```

Available steps: `detrend`, `bandpass`, `fk_filter`, `curvelet`, `hilbert`.

---

## Loading DAS Data

```python
from DAS import DAS, MulDAS

# Single file
das = DAS("recording.h5", vendor="silixa")
# das.data: (channels, time) float32
# das.meta: {fs, dx, dt, start_time_dt, ...}

# Multiple files concatenated by timestamp
mul = MulDAS(file_list, vendor="silixa")
```

Supported vendors: `"optasense"`, `"silixa"`.
