# DAS Preprocessing & Vehicle Detection

A Python toolkit for loading, preprocessing, and running ML on **Distributed Acoustic Sensing (DAS)** data — with a full pipeline from raw HDF5 files to vehicle detection and weight estimation.

---

## Overview

```
HDF5 Files → DAS.py (loading) → preprocessing.py (signal processing)
    → data_prep.ipynb (windowing + labeling) → dataset.py (PyTorch)
    → train.py --task denoising  →  predict.py --task denoising  (→ data/denoised/)
    → train.py --task detection / weight
    → eval_*.ipynb (load saved model + test split → visualize)
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

### 2. Train and run inference

```bash
# Step 1: train denoiser
python train.py --task denoising

# Step 2: generate denoised samples (required before detection/weight training)
python predict.py --task denoising        # writes data/denoised/

# Step 3a: train vehicle detector
python train.py --task detection

# Step 3b: train weight regressor
python train.py --task weight
```

Override config values from the CLI:
```bash
python train.py --task detection --epochs 100 --lr 5e-4
```

All hyperparameters live in `configs/<task>.yaml`. Each run saves `results/<task>/best_model.pt` and `results/<task>/splits.json` (exact train/val/test sample IDs).

### 3. Run inference on new data

```bash
python predict.py --task detection --input data/denoised/denoised_sample_000042.npy
python predict.py --task weight    --input data/denoised/denoised_sample_000042.npy
```

### 4. Visualize results

Open the eval notebook for a trained model — it loads `splits.json` to use the exact same test set:
- `notebooks/eval_denoising.ipynb` — raw / preprocessed / UNet output comparison
- `notebooks/eval_detection.ipynb` — per-sample predictions with count & type metrics
- `notebooks/eval_weight.ipynb` — predicted vs. actual weight

---

## Project Structure

```
DAS_Preprocessing/
├── DAS.py                  # HDF5 loader (OptaSense, Silixa); single + multi-file
├── preprocessing.py        # Composable signal processing (detrend, bandpass, f-k, curvelet)
├── dataset.py              # PyTorch Datasets: DASSampleDataset, DASCountDataset, DASWeightDataset
├── Utilities.py            # Visualization helpers (plot_das_data, plot_single)
├── train.py                # Unified training CLI (saves model + splits.json)
├── predict.py              # Inference CLI (denoising batch / detection / weight)
├── configs/
│   ├── denoising.yaml
│   ├── detection.yaml
│   └── weight.yaml
├── models/
│   ├── unet.py             # UNet denoiser
│   ├── detection_cnn.py    # DASCountCNN
│   ├── detection_transformer.py  # DASCountTransformer
│   └── weight_cnn.py       # DASWeightCNN
├── notebooks/
│   ├── data_prep.ipynb     # Label windows and export .npy samples
│   ├── demo.ipynb          # Quickstart: load, inspect, and visualize DAS data
│   ├── eval_denoising.ipynb  # Load model + test split, visualize denoising
│   ├── eval_detection.ipynb  # Load model + test split, visualize detection
│   └── eval_weight.ipynb     # Load model + test split, visualize weight
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
