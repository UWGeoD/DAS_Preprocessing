# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision scikit-learn
```

Copy `config.example.py` → `config.py` and fill in local paths (gitignored).

Run notebooks:
```bash
jupyter lab
```

---

## Architecture Overview

This is a **Distributed Acoustic Sensing (DAS)** data processing and ML toolkit. The pipeline flows:

```
HDF5 Files → DAS.py (loading) → preprocessing.py (signal processing)
    → prepare_data.py (windowing + labeling) → dataset.py (PyTorch datasets)
    → train.py --task denoising  →  predict.py --task denoising  (saves data/denoised/)
    → train.py --task detection / weight
    → compute_metrics.py (SNR: raw vs pp vs denoised → results/metrics.csv)
    → eval_*.ipynb (load saved model + splits.json → visualize)
```

Training order matters: denoising must run first, then `predict.py --task denoising` generates `data/denoised/` before detection/weight training.

### Core Modules

**[DAS.py](DAS.py)** — HDF5 file reader for OptaSense and Silixa vendors. `DAS` handles single files; `MulDAS` concatenates multiple files sorted by timestamp. All data is returned as `[channels, time]` numpy arrays with a `meta` dict containing `fs`, `dx`, `dt`, `start_time_dt`.

**[preprocessing.py](preprocessing.py)** — Composable signal processing pipeline. Build pipelines with `make_preprocess(steps)` which returns a callable `preprocess(x, fs, **ctx)`. Each step is a pure function on 2D `[channels, time]` arrays. Available steps: `detrend_linear`, `bandpass_sos`, `fk_filter`, `curvelet_like_denoise`, `hilbert_transform`. The `dx` and `dt` parameters are captured in the closure for spatial operations (f-k filter).

**[dataset.py](dataset.py)** — Three PyTorch Datasets, all reading from `labels.csv` + `.npy` sample files:
- `DASSampleDataset` — raw input → preprocessed target (for denoising training). Applies per-sample z-score normalization jointly to raw and clean (preserving the relative raw→clean relationship). Supports optional augmentation (`augment=True`): channel-flip and time-reversal, each with p=0.5.
- `DASCountDataset` — denoised input → (count, vehicle_type) labels (for detection)
- `DASWeightDataset` — denoised input → weight label (for regression)

`TYPE_MAP` / `IDX_TO_TYPE` constants define the vehicle type ↔ class index mapping shared across tasks.

**[prepare_data.py](prepare_data.py)** — Data preparation CLI. Reads HDF5 files via `MulDAS`, adds `frame_end_bridge` via physics-based estimate, slides a fixed-length window over the DAS array, and writes `data/raw/sample_XXXXXX.npy` + `data/labels.csv`. The CSV includes a `signal_rects` column: per-window list of `[frame_start-3, frame_start+27]` time rectangles (global video frames, clipped to window bounds) marking vehicle-induced signal regions. Supports `shuffle_samples` (default `false`) to renumber samples randomly after windowing — only safe when `stride_s >= window_length_s`; warns otherwise. Leftover `_tmp_*.npy` files from interrupted runs are cleaned up at start. Hyperparameters in `configs/data_prep.yaml`; local paths from `config.py`.

**[compute_metrics.py](compute_metrics.py)** — SNR evaluation CLI. For every window with `count > 0`, loads raw `.npy`, applies the preprocessing pipeline, loads the pre-generated denoised `.npy`, and computes SNR (dB) for all three versions using `signal_rects` from `labels.csv`. Writes `results/metrics.csv`. Optional `--splits` (path to `splits.json`) and `--plot-dir` flags generate per-split SNR boxplots and histograms saved as PNGs. Requires `prepare_data.py` and `predict.py --task denoising` to have run first.

**[train.py](train.py)** — Unified CLI entry point for all three training tasks. Loads YAML config from `configs/{task}.yaml`, applies CLI overrides, and saves `results/<task>/splits.json` and `results/<task>/losses.png` alongside the model checkpoint. Denoising training uses UNetV2, a combined MSE + correlation loss (`correlation_weight` config, prevents mean-collapse), and supports `split_mode: temporal` (default — first `train_frac` of sorted samples) or `split_mode: random` (shuffled by `split_seed`). `overlap_gap` skips samples between splits to prevent data contamination when windows overlap.

**[predict.py](predict.py)** — Inference-only CLI. `--task denoising` runs the saved UNetV2 on all `data/raw/*.npy`, applies the same per-sample z-score normalization used during training, and writes `data/denoised/denoised_sample_XXXXXX.npy`. Stale `denoised_*.npy` files from previous runs are cleared first. `--task detection/weight` runs inference on a single `--input` file.

**[models/unet_v2.py](models/unet_v2.py)** — UNetV2, the active denoising architecture: 4 encoder levels (32→64→128→256, bottleneck 512), SPP (Spatial Pyramid Pooling) at the bottleneck for multi-scale temporal context, SE-Net style channel attention on every skip connection to localize sparse vehicle signal. Input/output shape: `(1, channels, time)`.

**[models/unet.py](models/unet.py)** — UNet baseline: 3 encoder blocks → bottleneck → 3 decoder blocks with skip connections. Dropout2d(0.3) applied at bottleneck and first two decoder blocks. Uses bilinear upsampling. Not used by default — kept as a reference/baseline.

**[models/](models/)** — Task models: `detection_cnn.py` (`DASCountCNN`), `detection_transformer.py` (`DASCountTransformer` — count + vehicle type dual-head), `weight_cnn.py` (`DASWeightCNN` — absolute-value input for polarity-invariant weight regression).

**[Utilities.py](Utilities.py)** — Visualization (`plot_das_data`, `plot_single`, `plot_das_comparison`), helpers (`downsample_data`, `normalize`), and `compute_snr(window, signal_rects, win_start_frame, fps_video, fs_das)` which returns SNR in dB using signal/noise regions derived from `signal_rects`. `plot_das_comparison` renders a publication-quality figure: three stacked heatmaps (raw / preprocessed / denoised, shared axes) plus an SNR horizontal bar chart, with optional signal-rect shading and per-step pipeline labels.

### Config System

- **[config.example.py](config.example.py)** — template for local file paths (committed); copy to `config.py` (gitignored)
- **[configs/](configs/)** — per-task YAML hyperparameters; CLI flags override YAML values
  - `data_prep.yaml` — windowing params, fps, time-alignment, bridge geometry, channel range, `shuffle_samples`, `random_seed`
  - `denoising.yaml` — UNetV2 training: `train_frac`, `split_mode`, `overlap_gap`, `correlation_weight`, `lr_patience`, `include_background`, plus preprocessing steps
  - `detection.yaml`, `weight.yaml` — model training hyperparameters; `overlap_gap` controls split buffer for overlapping windows

### Training & Inference CLI

```bash
# Data preparation (run once; re-run if labels or config change)
python prepare_data.py                        # → data/raw/ + data/labels.csv

# Train (saves best_model.pt + splits.json to results/<task>/)
python train.py --task denoising
python train.py --task detection
python train.py --task weight
# Override any YAML value via flags:
python train.py --task denoising --epochs 50 --lr 1e-5

# Inference
python predict.py --task denoising                           # → data/denoised/
python predict.py --task detection --input sample.npy        # prints count + type
python predict.py --task weight    --input sample.npy        # prints weight lbs

# SNR evaluation (requires prepare_data.py + predict.py --task denoising first)
python compute_metrics.py                     # → results/metrics.csv
python compute_metrics.py --splits results/denoising/splits.json \
                          --plot-dir results/denoising/SNR   # + per-split SNR plots
```

### Notebooks

Notebooks are for **data preparation and visualization only** — never for training:

- **[notebooks/data_prep.ipynb](notebooks/data_prep.ipynb)** — Sanity-check only (run `prepare_data.py` first): inspects `labels.csv` and visualizes sample windows with signal rects overlaid.
- **[notebooks/demo.ipynb](notebooks/demo.ipynb)** — Quickstart: load, inspect, preprocess, and visualize DAS files.
- **[notebooks/eval_denoising.ipynb](notebooks/eval_denoising.ipynb)** — Loads trained UNetV2 + test split → shows raw / preprocessed / denoised comparisons with SNR panels using `plot_das_comparison` from `Utilities.py`.
- **[notebooks/eval_detection.ipynb](notebooks/eval_detection.ipynb)** — Loads trained DASCountTransformer + test split → shows per-sample predictions and metrics.
- **[notebooks/eval_weight.ipynb](notebooks/eval_weight.ipynb)** — Loads trained DASWeightCNN + test split → shows predicted vs. actual weight.
- **[notebooks/eval_metrics.ipynb](notebooks/eval_metrics.ipynb)** — Loads `results/metrics.csv` → SNR distribution boxplots/histograms and per-sample gain over raw for all signal windows.

### Key Patterns

**Loading DAS data:**
```python
from DAS import DAS, MulDAS
das = DAS("file.h5", vendor="silixa")   # or vendor="optasense"
# das.data: [channels, time], das.meta: {fs, dx, dt, start_time_dt, ...}
```

**Building a preprocessing pipeline:**
```python
from preprocessing import make_preprocess
steps = ["detrend", "bandpass", "fk_filter"]
pp = make_preprocess(steps, f_lo=1, f_hi=20, order=5, dx=1.0)
clean = pp(raw_data, fs=2000)
```

**Label aggregation logic** (in `prepare_data.py`): `count=0` → `background`, `count=1` → specific vehicle type, `count>1` → `mixed`.

---

## Code Quality Guidelines

### Design Principles

This project prioritizes **code reuse**, **readability**, and **SOLID** where practical:

- **Single Responsibility**: Each module has one job — `DAS.py` reads original DAS file, for now only h5 with different vendor, might need more in the future, `preprocessing.py` processes especially for physical filters, `Utilities.py` supporting functions like visualizations, might add more later. Keep new code scoped the same way.
- **Open/Closed**: Extend `preprocessing.py` by adding new step functions; don't modify `make_preprocess` itself. Add new model architectures under `models/` without touching existing ones.
- **Reuse over duplication**: Before writing new DSP logic, check `preprocessing.py`; before writing new plots, check `Utilities.py`. Extend existing abstractions rather than copying.
- **Readable over clever**: DAS data has physical meaning (channels = space, samples = time). Names should reflect domain (`fs`, `dx`, `channels`, not `n`, `m`, `arr`).

### Think Before Coding

Don't assume. Surface tradeoffs.

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so.

### Simplicity First

Minimum code that solves the problem. Nothing speculative.

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" that wasn't requested.
- If you write 200 lines and it could be 50, rewrite it.

### Surgical Changes

Touch only what you must. Clean up only your own mess.

- Don't "improve" adjacent code or formatting.
- Match existing style, even if you'd do it differently.
- Remove imports/variables made unused by **your** changes. Leave pre-existing dead code alone unless asked.

### Goal-Driven Execution

For multi-step tasks, state a brief plan before implementing:

```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## ML Project Conventions

This project follows standard ML engineering practices. Key references:
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) — canonical project structure (separate `train`, `predict`, `evaluate`)
- [PyTorch — Saving and Loading Models](https://pytorch.org/tutorials/beginner/saving_loading_models.html) — `state_dict` checkpoint format
- [PyTorch Lightning design](https://lightning.ai/docs/pytorch/stable/) — strict separation of `fit`, `predict`, `test` phases

Rules enforced in this project:

1. **prepare_data.py prepares only** — reads HDF5 files, windows the array, writes `.npy` samples and `labels.csv`. No training, no inference.
2. **train.py trains only** — no inference, no data generation side-effects. After training, it saves `best_model.pt` and `splits.json` to `results/<task>/` and exits.
3. **predict.py infers only** — loads checkpoint, runs forward pass, writes outputs. Never modifies model weights.
4. **compute_metrics.py computes metrics only** — reads pre-generated files (raw `.npy`, denoised `.npy`), writes `results/metrics.csv`. Optionally saves SNR distribution plots when `--plot-dir` is passed. No training, no inference.
5. **eval_*.ipynb visualize only** — load `splits.json` to get the exact test set used during training, load `best_model.pt`, run inference, plot. Never retrain or re-split inside a notebook.
6. **Persist train/val/test splits** — `results/<task>/splits.json` records the exact sample IDs for each split so notebooks always reproduce the same test set even if the dataset grows or order changes.
7. **Checkpoint format** — always save `model.state_dict()` (not the full model object). Load with `model.load_state_dict(torch.load(..., map_location=device))`.
8. **Config-driven** — all hyperparameters live in `configs/<task>.yaml`. Never hardcode LR, epochs, batch size in `.py` files.
