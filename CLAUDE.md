# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# For ML training notebooks, also install:
pip install torch torchvision scikit-learn
```

Run notebooks:
```bash
jupyter lab
```

---

## Architecture Overview

This is a **Distributed Acoustic Sensing (DAS)** data processing and ML toolkit. The pipeline flows:

```
HDF5 Files → DAS.py (loading) → preprocessing.py (signal processing)
    → data_prep.ipynb (windowing) → dataset.py (PyTorch) → training notebooks (1. train_denoising then 2a. train_detection, 2b. train_weight_predicting, 1 comes first then 2a, 2b )
```

### Core Modules

**[DAS.py](DAS.py)** — HDF5 file reader for OptaSense and Silixa vendors. `DAS` handles single files; `MulDAS` concatenates multiple files sorted by timestamp. All data is returned as `[channels, time]` numpy arrays with a `meta` dict containing `fs`, `dx`, `dt`, `start_time_dt`.

**[preprocessing.py](preprocessing.py)** — Composable signal processing pipeline. Build pipelines with `make_preprocess(steps)` which returns a callable `preprocess(x, fs, **ctx)`. Each step is a pure function on 2D `[channels, time]` arrays. Available steps: `detrend_linear`, `bandpass_sos`, `fk_filter`, `curvelet_like_denoise`, `hilbert_transform`. The `dx` and `dt` parameters are captured in the closure for spatial operations (f-k filter).

**[dataset.py](dataset.py)** — `DASSampleDataset` is a PyTorch Dataset that loads `.npy` sample files listed in a `labels.csv`. It applies the preprocessing pipeline to generate clean targets (raw input → preprocessed target for denoising training).

**[models/unet.py](models/unet.py)** — UNet denoising architecture: 3 encoder blocks → bottleneck → 3 decoder blocks with skip connections. Uses bilinear upsampling (not ConvTranspose2d). Input/output shape: `(1, channels, time)`.

**[models/](models/)** — Task models: `detection_cnn.py`/`detection_transformer.py` (`DASCountCNN`, `DASCountTransformer` — count + vehicle type), `unet.py` (denoising), `weight_cnn.py` (`DASWeightCNN` — weight regression).

**[Utilities.py](Utilities.py)** — Visualization (`plot_das_data`, `plot_single`) and helpers (`downsample_data`, `normalize`).

### Notebooks

- **[data_prep.ipynb](data_prep.ipynb)** — Creates labeled training data: maps video frame labels → DAS time windows → `sample_XXXXXX.npy` + `labels.csv`. The `labels.csv` columns are `sample_id`, `data_path`, `count`, `start_frame`, `end_frame`, `vehicle_type`.
- **[demo.ipynb](demo.ipynb)** — Quickstart: load, inspect, preprocess, and visualize DAS files.
- **[train_denoising.ipynb](train_denoising.ipynb)** — Train UNet: raw → preprocessed (denoised) signal. LR=6e-6, batch=2, MSELoss.
- **[train_detection.ipynb](train_detection.ipynb)** — Demo: vehicle detection (count + type) from denoised signals. Train with `python train.py --task detection`.
- **[train_weight_predicting.ipynb](train_weight_predicting.ipynb)** — Train vehicle weight regression.

### Key Patterns

**Loading DAS data:**
```python
from DAS import DAS, MulDAS
das = DAS("file.h5", vendor="silixa")   # or vendor="optasense"
# das.data: [channels, time], das.meta: {fs, dx, dt, start_time_dt, ...}

mul = MulDAS(file_list, vendor="silixa")
```

**Building a preprocessing pipeline:**
```python
from preprocessing import make_preprocess
steps = ["detrend", "bandpass", "fk_filter"]
pp = make_preprocess(steps, f_lo=1, f_hi=20, order=5, dx=1.0)
clean = pp(raw_data, fs=2000)
```

**Label aggregation logic** (in `data_prep.ipynb`): `count=0` → `background`, `count=1` → specific vehicle type, `count>1` → `mixed`.

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
