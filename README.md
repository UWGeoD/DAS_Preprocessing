# DAS Preprocessing & Denoising

A Python toolkit for loading, preprocessing, and denoising **Distributed Acoustic Sensing (DAS)** data, with a UNet-based ML pipeline for vehicle signal recovery from noisy fiber-optic recordings.

---

## Overview

The primary goal is **denoising**: train a UNetV2 to map raw DAS recordings to clean (preprocessed) signals, then evaluate SNR improvement across train/val/test splits.

```
HDF5 Files → DAS.py (loading) → preprocessing.py (signal processing)
    → prepare_data.py (windowing + labeling)
    → train.py --task denoising  →  predict.py --task denoising  (→ data/<dataset>/denoised/)
    → compute_metrics.py (SNR: raw vs preprocessed vs denoised → results/metrics.csv)
    → eval_denoising.ipynb / eval_metrics.ipynb (visualize)
```

---

## Installation

```bash
git clone https://github.com/UWGeoD/DAS_Preprocessing.git
cd DAS_Preprocessing

python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install torch torchvision scikit-learn
```

---

## Configuration

This project uses a two-level config system:

### 1. Dataset config — `configs/datasets/<name>.yaml`

Each dataset has its own YAML with sensor parameters, file paths, and windowing settings. Example (`configs/datasets/newville_apr2025.yaml`):

```yaml
# File paths
recording_dir: /path/to/h5/files/
file_pattern: "*Newville*.h5"
labeling_csv: /path/to/Vehicle_Labelings.csv

# DAS hardware
vendor: silixa          # or optasense
fs_das: 2000            # sampling rate (Hz)
dx: 1.02                # channel spacing (m)
channel_start: 26       # first channel to use
channel_end: 60

# Video / time-alignment
fps_video: 29.99
frame_ref: 115          # known reference frame number
sys_time_ref_hms: "10:27:09"   # system clock at frame_ref
das_start_hms: "10:24:47"      # DAS recording start time

# Bridge geometry (used to estimate when a vehicle exits the bridge)
estimate_speed_mph: 70
side_length_ft: 20
bridge_length_m: 35

# Windowing
window_length_s: 5.0
stride_s: 5.0           # non-overlapping by default
shuffle_samples: false

# Output paths
data_dir: data/newville_apr2025
```

Set the active dataset in `config.py` (copy from `config.example.py`, gitignored):

```python
ACTIVE_DATASET = "newville_apr2025"   # matches configs/datasets/<name>.yaml
```

Switch datasets by changing `ACTIVE_DATASET` — all scripts pick this up automatically.

### 2. Task config — `configs/denoising.yaml`

Model and training hyperparameters. Dataset-specific values (`fs`, `dx`, `data_dir`) are injected from the active dataset config at runtime.

```yaml
# Split strategy
train_frac: 0.6         # fraction of samples for training
overlap_gap: 0          # samples to skip between splits (use when stride < window)
split_mode: random      # random or temporal (temporal = first n_train sorted samples)
split_seed: 42

# Training
batch_size: 4
lr: 1.0e-4
epochs: 200
lr_patience: 20         # reduce LR after this many epochs without val improvement
include_background: false  # whether to train on count=0 windows

# Loss function — key in configs/losses.yaml
loss: mse               # mse | l1 | gan_nce

# Model
model: unet_v2          # unet_v2 (active) | unet (baseline)
save_dir: results/denoising

# Preprocessing pipeline (defines the clean target for training)
steps:
  - name: detrend
    axis: 1
  - name: bandpass
    f_lo: 0.5
    f_hi: 2.0
    order: 2
    zero_phase: true
  - name: fk_filter
    vmin: 10.0
    vmax: 80.0
    taper: 0.05
  # - name: curvelet     # uncomment to add curvelet shrinkage
  #   n_scales: 4
  #   n_angles: 16
  #   thresh: 3.5
```

### 3. Loss config — `configs/losses.yaml`

Registered loss functions for denoising. Set `loss: <name>` in `denoising.yaml` to select one:

| Loss | Description |
|------|-------------|
| `mse` | MSE reconstruction + optional L1 prediction regularizer |
| `l1` | L1 reconstruction + optional L1 prediction regularizer |
| `gan_nce` | PatchGAN adversarial + PatchNCE contrastive loss (much slower; needs `discriminator.py` / `patchnce.py`) |

All CLI flags override YAML values for any single run.

---

## Denoising Pipeline

### 1. Prepare training data

```bash
python prepare_data.py                              # uses ACTIVE_DATASET from config.py
python prepare_data.py --dataset newville_apr2025   # explicit dataset name
```

Windows the DAS array, maps vehicle labels to per-window signal regions, and writes:
- `data/<dataset>/raw/sample_XXXXXX.npy` — one window per file
- `data/<dataset>/labels.csv` — count, vehicle type, and `signal_rects` per window

Re-run whenever labels or windowing config change. Leftover `_tmp_*.npy` files from interrupted runs are cleaned up automatically.

### 2. Train

```bash
python train.py --task denoising
python train.py --task denoising --dataset newville_apr2025   # explicit dataset
```

Common overrides:

```bash
python train.py --task denoising --epochs 100 --lr 5e-5
python train.py --task denoising --loss l1
python train.py --task denoising --split_mode temporal --train_frac 0.7
python train.py --task denoising --include_background true
```

Any key in `denoising.yaml` can be overridden as a `--flag`. Saves to `results/denoising/`:
- `best_model.pt` — best validation checkpoint
- `splits.json` — exact sample IDs for train / val / test (used by eval notebooks)
- `losses.png` — training curve

### 3. Generate denoised outputs

```bash
python predict.py --task denoising
python predict.py --task denoising --dataset newville_apr2025
```

Runs the saved UNetV2 on all `data/<dataset>/raw/*.npy` and writes `data/<dataset>/denoised/denoised_sample_XXXXXX.npy`. Stale files from a previous run are cleared first.

### 4. Evaluate SNR

`compute_metrics.py` measures denoising quality by computing SNR (dB) on every window that contains a vehicle signal (`count > 0`). It compares three versions of each window:

| Column | What it measures |
|--------|-----------------|
| `snr_raw` | Raw DAS signal as recorded |
| `snr_pp` | After applying the preprocessing pipeline (the training target) |
| `snr_denoised` | After running the trained UNetV2 |

SNR is computed using `signal_rects` from `labels.csv` — time regions where a vehicle was present define the signal window; everything else is treated as noise.

```bash
# Minimal: just writes results/metrics.csv
python compute_metrics.py

# Full: also saves plots and per-split stats
python compute_metrics.py \
    --splits results/denoising/splits.json \
    --plot-dir results/denoising/SNR

# All flags
python compute_metrics.py \
    --dataset newville_apr2025 \           # explicit dataset (default: ACTIVE_DATASET)
    --out results/metrics.csv \            # output CSV path
    --splits results/denoising/splits.json \   # for per-split breakdown
    --plot-dir results/denoising/SNR \     # save plots here
    --denoised-dir data/newville_apr2025/denoised  # if not in default location
```

With `--plot-dir`, three figures and a summary CSV are saved:
- `snr_boxplot.png` — box plots of SNR distribution for all three methods, split by train/val/test
- `snr_histogram.png` — overlaid histograms with per-method median labeled
- `snr_gain.png` — per-sample bar chart of ΔSNR vs raw (how much each method improved over raw)
- `snr_stats.csv` — mean, median, std, min, max per method per split

### 5. Visualize

Open the eval notebooks — they load `splits.json` to use the same test set as training:

- `notebooks/eval_denoising.ipynb` — raw / preprocessed / UNet comparison with SNR panels per sample
- `notebooks/eval_metrics.ipynb` — SNR distributions and per-sample gain over raw (all signal windows)

---

## Hyperparameter Sweep

`run_sweep.py` runs a grid search over denoising hyperparameters. The grid is defined in `configs/sweep.yaml`; base training config comes from `configs/denoising.yaml` and the active dataset config.

### Configure the grid — `configs/sweep.yaml`

```yaml
sweep_dir: results/sweep        # where per-run artifacts are saved
sweep_data_root: data/sweep     # windowed .npy cache (reused across trials)

# Grid dimensions — all combinations are run
sample_size: [80]               # number of signal windows per trial
window_length_s: [5.0]          # window sizes to try

# Loss variants — each entry is a dict with 'name' + optional overrides
loss:
  - {name: mse, pred_reg_weight: 0}
  - {name: mse, pred_reg_weight: 0.0001}
  - {name: l1,  pred_reg_weight: 0}
  - {name: l1,  pred_reg_weight: 0.0001}
  # - {name: gan_nce, lambda_NCE: 1.0}   # uncomment to include (~2× slower)

# Preprocessing pipelines — each entry is a complete pipeline list
steps:
  - - {name: detrend}
    - {name: bandpass, f_lo: 0.1, f_hi: 2.0, order: 2, zero_phase: true}
    - {name: fk_filter, vmin: 5.0, vmax: 100.0, taper: 0.05}
```

To add a sweep dimension, add a new key with a list of values. Any key that also exists in `denoising.yaml` (e.g., `lr`, `epochs`, `batch_size`) is forwarded directly to the training config.

### Run the sweep

```bash
python run_sweep.py                                        # uses ACTIVE_DATASET
python run_sweep.py --dataset newville_apr2025             # explicit dataset
python run_sweep.py --sweep configs/sweep.yaml             # explicit sweep config
python run_sweep.py --mode reset                           # wipe DB before starting
python run_sweep.py --mode skip                            # skip already-completed trials
python run_sweep.py --force-rewindow                       # ignore cached windowed data
```

Each trial runs three stages: preprocess timing → train → predict + SNR. Results (SNR for raw / preprocessed / denoised across full / train / val / test splits, timing, status) are written to `results/sweep/experiments.db` (SQLite). Per-run artifacts (model checkpoint, loss history, representative sample arrays) are saved under `results/sweep/<run_id>/`.

Query results directly:

```bash
sqlite3 results/sweep/experiments.db \
  "SELECT params_json, snr_pred_mean, snr_raw_mean, time_train FROM runs WHERE status='ok' ORDER BY snr_pred_mean DESC LIMIT 10"
```

---

## Finding Results

After running the full pipeline, all outputs are in `results/`:

```
results/
├── metrics.csv                   # per-sample SNR: raw / preprocessed / denoised
├── denoising/
│   ├── best_model.pt             # trained UNetV2 weights (state_dict)
│   ├── splits.json               # exact sample IDs for train / val / test
│   ├── losses.png                # training and validation loss curves
│   ├── losses.json               # loss values by epoch (machine-readable)
│   └── SNR/
│       ├── snr_boxplot.png       # SNR distribution: raw vs pp vs denoised, per split
│       ├── snr_histogram.png     # overlaid histograms with median labels
│       ├── snr_gain.png          # per-sample ΔSNR vs raw (bar chart, per split)
│       └── snr_stats.csv         # mean/median/std/min/max per method per split
├── detection/
│   ├── best_model.pt
│   └── splits.json
└── sweep/
    ├── experiments.db            # SQLite: one row per trial (params, SNR, timing, status)
    └── <run_id>/
        ├── best_model.pt
        ├── splits.json
        ├── losses.json
        └── repr_count<N>_{raw,target,pred}.npy   # representative sample arrays per count value
```

**`results/metrics.csv`** — the main quantitative output. Columns: `sample_id`, `vehicle_type`, `count`, `snr_raw`, `snr_pp`, `snr_denoised`. One row per signal window (count > 0). Load it in `notebooks/eval_metrics.ipynb` for distribution plots, or query directly:

```python
import pandas as pd
df = pd.read_csv("results/metrics.csv")
df["gain"] = df["snr_denoised"] - df["snr_raw"]
print(df[["snr_raw", "snr_pp", "snr_denoised", "gain"]].describe().round(2))
```

**`results/denoising/splits.json`** — the exact sample IDs used for each split during training. Always load this in eval notebooks to reproduce the same test set, even if the dataset has grown since training:

```python
import json
with open("results/denoising/splits.json") as f:
    splits = json.load(f)
test_ids = splits["test"]   # list of sample_id strings
```

**`results/sweep/experiments.db`** — SQLite database with one row per sweep trial. Query to compare configurations:

```bash
# Top 10 trials by mean denoised SNR on the test set
sqlite3 -column -header results/sweep/experiments.db \
  "SELECT params_json, snr_pred_mean, snr_raw_mean, time_train
   FROM runs WHERE status='ok'
   ORDER BY snr_pred_mean DESC LIMIT 10"
```

---

## Preprocessing API

Build a composable pipeline with `make_preprocess`:

```python
from preprocessing import make_preprocess

pp = make_preprocess(
    steps=[
        ("detrend",   {"axis": 1}),
        ("bandpass",  {"f_lo": 0.5, "f_hi": 2.0, "order": 2, "zero_phase": True}),
        ("fk_filter", {"vmin": 10.0, "vmax": 80.0, "taper": 0.05}),
    ],
    dx=1.02,   # meters per channel (required for fk_filter)
)
clean = pp(raw_data, fs=2000)   # raw shape: (n_channels, n_time)
```

Available steps: `detrend`, `bandpass`, `fk_filter`, `curvelet`, `hilbert`.

---

## Loading DAS Data

```python
from DAS import DAS, MulDAS

# Single file
das = DAS("recording.h5", vendor="silixa")   # or vendor="optasense"
# das.data: (channels, time) float32
# das.meta: {fs, dx, dt, start_time_dt, ...}

# Multiple files concatenated by timestamp
mul = MulDAS(file_list, vendor="silixa")
```

---

## Project Structure

```
DAS_Preprocessing/
├── DAS.py                  # HDF5 loader (OptaSense, Silixa); single + multi-file
├── preprocessing.py        # Composable signal processing (detrend, bandpass, f-k, curvelet)
├── dataset.py              # PyTorch Datasets: DASSampleDataset, DASCountDataset, DASWeightDataset
├── Utilities.py            # Visualization helpers and compute_snr metric
├── prepare_data.py         # Data prep CLI → data/<dataset>/raw/ + labels.csv
├── train.py                # Unified training CLI (saves model + splits.json)
├── predict.py              # Inference CLI
├── compute_metrics.py      # SNR CLI → results/metrics.csv
├── run_sweep.py            # Grid search over denoising hyperparameters
├── configs/
│   ├── datasets/           # Per-dataset sensor params + file paths
│   ├── losses.yaml         # Loss function registry (mse, l1, gan_nce)
│   ├── denoising.yaml      # UNetV2 training hyperparameters
│   ├── sweep.yaml          # Grid search dimensions
│   ├── detection.yaml      # (secondary task)
│   └── weight.yaml         # (secondary task)
├── models/
│   ├── unet_v2.py          # UNetV2 denoiser (active) — SPP bottleneck, channel attention
│   ├── unet.py             # UNet denoiser (baseline reference)
│   ├── discriminator.py    # PatchGAN discriminator (for gan_nce loss)
│   ├── patchnce.py         # PatchNCE contrastive loss (for gan_nce loss)
│   ├── detection_cnn.py    # DASCountCNN (secondary)
│   ├── detection_transformer.py  # DASCountTransformer (secondary)
│   └── weight_cnn.py       # DASWeightCNN (secondary)
├── notebooks/
│   ├── demo.ipynb          # Quickstart: load, inspect, and visualize DAS data
│   ├── data_prep.ipynb     # Sanity-check: inspect labels.csv + visualize sample windows
│   ├── eval_denoising.ipynb  # Denoising results: raw / preprocessed / UNet + SNR
│   ├── eval_metrics.ipynb    # SNR distributions and per-sample gain (all signal windows)
│   ├── eval_detection.ipynb  # (secondary) detection results
│   └── eval_weight.ipynb     # (secondary) weight prediction results
├── config.example.py       # Template — copy to config.py and set ACTIVE_DATASET
└── requirements.txt
```

---

## Secondary Tasks: Detection & Weight

Detection (vehicle count + type) and weight estimation are exploratory tasks built on top of denoised outputs. Both require the denoising step to have run first.

```bash
# After predict.py --task denoising:
python train.py --task detection
python train.py --task weight

python predict.py --task detection --input data/<dataset>/denoised/denoised_sample_000042.npy
python predict.py --task weight    --input data/<dataset>/denoised/denoised_sample_000042.npy
```

Models: `DASCountTransformer` (count + vehicle type dual-head), `DASWeightCNN` (polarity-invariant weight regression). Results in `notebooks/eval_detection.ipynb` and `notebooks/eval_weight.ipynb`.
