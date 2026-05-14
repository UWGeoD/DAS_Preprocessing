"""
Grid search over DAS denoising hyperparameters.

Non-grid parameters come from:
  configs/datasets/{dataset}.yaml  →  sensor params, paths, video timing
  configs/denoising.yaml           →  training hyperparameters (lr, epochs, etc.)
  configs/sweep.yaml               →  grid dimensions (sample_size, window_length_s, steps)

Usage:
    python run_sweep.py
    python run_sweep.py --sweep configs/sweep.yaml --dataset newville_apr2025
    python run_sweep.py --mode reset          # wipe runs table before starting
    python run_sweep.py --mode skip           # skip trials whose params already succeeded
    python run_sweep.py --force-rewindow      # ignore cached windowed data
"""

import warnings
warnings.filterwarnings("ignore", message="Pandas requires version", category=UserWarning)

import argparse
import gc
import glob
import itertools
import json
import os
import sqlite3
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from preprocessing import make_preprocess
from Utilities import compute_snr
from train import load_config, train_denoising_core


# ---------------------------------------------------------------------------
# Pipeline label helpers
# ---------------------------------------------------------------------------

def _loss_label(loss_spec):
    """Short human-readable label for a loss spec.

    Examples:
        {name: mse}                            → 'mse'
        {name: mse, pred_reg_weight: 0.001}    → 'mse[w=0.001]'
        {name: l1,  pred_reg_type: l2, ...}    → 'l1[reg=l2,w=0.001]'
    """
    if loss_spec is None:
        return "mse"
    if isinstance(loss_spec, str):
        return loss_spec
    name = loss_spec.get("name", "mse")
    # Short aliases for common keys to keep labels readable
    _ALIAS = {"pred_reg_weight": "w", "pred_reg_type": "reg", "lambda_NCE": "nce",
               "nce_tau": "tau", "d_depth": "d"}
    extras = {_ALIAS.get(k, k): v for k, v in loss_spec.items() if k != "name"}
    if not extras:
        return name
    parts = [f"{k}={v}" for k, v in sorted(extras.items())]
    return f"{name}[{','.join(parts)}]"


def _steps_label(steps_spec):
    """Short human-readable label for a pipeline spec, e.g. 'detrend+bp[0.5-2.0]+fk'."""
    parts = []
    for s in steps_spec:
        name = s["name"]
        if name == "detrend":
            parts.append("detrend")
        elif name == "bandpass":
            parts.append(f"bp[{s.get('f_lo','?')}-{s.get('f_hi','?')}]")
        elif name == "fk_filter":
            parts.append("fk")
        elif name == "curvelet":
            parts.append("curvelet")
        else:
            parts.append(name)
    return "+".join(parts)


def _params_label(params):
    """One-line description of a trial's params for the summary table."""
    label = f"sample_size={params['sample_size']}, window={params['window_length_s']}s"
    label += f", steps={_steps_label(params['steps'])}"
    if "loss" in params:
        label += f", loss={_loss_label(params['loss'])}"
    known = {"sample_size", "window_length_s", "steps", "loss"}
    extras = {k: v for k, v in params.items() if k not in known}
    if extras:
        label += ", " + ", ".join(f"{k}={v}" for k, v in extras.items())
    return label


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    timestamp        TEXT,
    params_json      TEXT,
    -- Summary columns over the full windowed set (for fast SQL comparison)
    snr_raw_mean     REAL,
    snr_raw_std      REAL,
    snr_target_mean  REAL,
    snr_target_std   REAL,
    snr_pred_mean    REAL,
    snr_pred_std     REAL,
    -- Full SNR value lists (JSON) for all 5 sets x 3 methods
    -- Schema: {"full":{raw:[...],target:[...],pred:[...]}, "used":{...}, "train":{...}, "val":{...}, "test":{...}}
    snr_json         TEXT,
    -- Split sizes
    n_train          INTEGER,
    n_val            INTEGER,
    n_test           INTEGER,
    n_full           INTEGER,
    -- Artifact path
    model_path       TEXT,
    -- Timing (seconds)
    time_preprocess  REAL,
    time_train       REAL,
    time_predict     REAL,
    time_total       REAL,
    -- Outcome
    status           TEXT,
    error            TEXT
)
"""


def setup_database(db_path, mode="append"):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    if mode == "reset":
        conn.execute("DROP TABLE IF EXISTS runs")
        print("DB: runs table cleared.")
    conn.execute(_SCHEMA)
    conn.commit()
    return conn


def _params_json_canonical(params):
    """Stable JSON string for params dict (sorted keys, no whitespace variation)."""
    def _make_serializable(obj):
        if isinstance(obj, list):
            return [_make_serializable(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in sorted(obj.items())}
        return obj
    return json.dumps(_make_serializable(params), sort_keys=True, separators=(",", ":"))


def _already_succeeded(conn, params):
    """Check if an identical params_json row exists with status='ok'."""
    canon = _params_json_canonical(params)
    row = conn.execute(
        "SELECT 1 FROM runs WHERE params_json = ? AND status = 'ok' LIMIT 1", (canon,)
    ).fetchone()
    return row is not None


def log_run(conn, run_id, params, snr_data, timing, model_path, status, error=None):
    """Insert or replace a row in the runs table."""
    ts     = datetime.now().isoformat(timespec="seconds")
    pjson  = _params_json_canonical(params)
    sjson  = json.dumps(snr_data) if snr_data else None

    def _mean(vals):
        v = [x for x in (vals or []) if x is not None and not (isinstance(x, float) and x != x)]
        return float(np.mean(v)) if v else None

    def _std(vals):
        v = [x for x in (vals or []) if x is not None and not (isinstance(x, float) and x != x)]
        return float(np.std(v)) if v else None

    full = (snr_data or {}).get("full", {})
    conn.execute(
        """INSERT OR REPLACE INTO runs VALUES
           (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id, ts, pjson,
            _mean(full.get("raw")),   _std(full.get("raw")),
            _mean(full.get("target")), _std(full.get("target")),
            _mean(full.get("pred")),  _std(full.get("pred")),
            sjson,
            (snr_data or {}).get("_n_train"),
            (snr_data or {}).get("_n_val"),
            (snr_data or {}).get("_n_test"),
            (snr_data or {}).get("_n_full"),
            model_path,
            (timing or {}).get("preprocess"),
            (timing or {}).get("train"),
            (timing or {}).get("predict"),
            (timing or {}).get("total"),
            status, error,
        ),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------

def build_grid(sweep_space):
    """
    itertools.product over all dimensions.
    Returns a list of param dicts, one per trial.
    Parameter-agnostic: works with any keys in sweep_space.
    """
    keys   = list(sweep_space.keys())
    values = list(sweep_space.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


# ---------------------------------------------------------------------------
# Window sample cache
# ---------------------------------------------------------------------------

_window_cache = {}  # {window_length_s: {"df_all": df, "df_pos": df, "repr_ids": dict, "data_dir": str}}


def _prepare_window_samples(window_length_s, das_array, vehicle_labels, ds_cfg, sweep_data_root, force=False):
    """
    Ensure windowed samples exist for `window_length_s`.
    Uses cached files on disk if present; re-windows only when forced or missing.
    stride_s is always set to window_length_s (non-overlapping) for clean splits.

    Returns a cache entry dict with keys: df_all, df_pos, repr_ids, data_dir.
    """
    if window_length_s in _window_cache and not force:
        return _window_cache[window_length_s]

    ws_tag   = f"ws_{window_length_s:.1f}".replace(".", "p")
    data_dir = os.path.abspath(os.path.join(sweep_data_root, ws_tag))
    raw_dir  = os.path.join(data_dir, "raw")
    csv_path = os.path.join(data_dir, "labels.csv")

    if os.path.exists(csv_path) and glob.glob(os.path.join(raw_dir, "sample_*.npy")) and not force:
        print(f"  [window cache] loading ws={window_length_s}s from {data_dir}/")
        df_all = pd.read_csv(csv_path)
    else:
        print(f"  [windowing] ws={window_length_s}s → {data_dir}/")
        from prepare_data import prepare_das_windows_and_labels, add_frame_end_bridge

        label_mode    = ds_cfg.get("label_mode", "estimated_end")
        veh_labels    = vehicle_labels.copy()

        if label_mode == "estimated_end":
            veh_labels = add_frame_end_bridge(
                df=veh_labels,
                estimate_speed=ds_cfg["estimate_speed_mph"],
                side_length_ft=ds_cfg["side_length_ft"],
                bridge_length_m=ds_cfg["bridge_length_m"],
                fps_video=ds_cfg["fps_video"],
                speed_unit="mph",
                frame_col="frame_start",
                out_col="frame_end_bridge",
            )
            end_frame_col = "frame_end_bridge"
        else:
            end_frame_col = "frame_end"

        df_all, _ = prepare_das_windows_and_labels(
            das_array=das_array,
            labels_csv_or_df=veh_labels,
            window_length_s=window_length_s,
            stride_s=window_length_s,           # always non-overlapping in sweep
            fs_das=ds_cfg["fs_das"],
            fps_video=ds_cfg["fps_video"],
            frame_ref=ds_cfg["frame_ref"],
            sys_time_ref_hms=ds_cfg["sys_time_ref_hms"],
            das_start_hms=ds_cfg["das_start_hms"],
            out_dir=raw_dir,
            csv_out="../labels.csv",
            start_frame_col="frame_start",
            end_frame_col=end_frame_col,
            label_mode=label_mode,
            multi_token=ds_cfg.get("multi_token", "mixed"),
            none_token=ds_cfg.get("none_token", "background"),
            shuffle_samples=False,
        )

    df_pos = df_all[df_all["count"] > 0].sort_values("sample_id").reset_index(drop=True)

    # Pick one representative sample per count value (stable across all trials with this window_length_s)
    repr_ids = {}
    for count_val in sorted(df_all["count"].unique()):
        first_row = df_all[df_all["count"] == count_val].iloc[0]
        repr_ids[int(count_val)] = str(first_row["sample_id"])

    entry = {"df_all": df_all, "df_pos": df_pos, "repr_ids": repr_ids, "data_dir": data_dir}
    _window_cache[window_length_s] = entry
    return entry


# ---------------------------------------------------------------------------
# In-memory predict
# ---------------------------------------------------------------------------

def _predict_one(model, raw, device):
    """Run denoising model on a single [channels, time] array. Returns pred array."""
    chan_mean = raw.mean(axis=1, keepdims=True)
    std       = (raw - chan_mean).std() + 1e-8
    raw_norm  = (raw - chan_mean) / std
    t         = torch.from_numpy(raw_norm).unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_norm = model(t).squeeze().cpu().numpy()
    return pred_norm * std


# ---------------------------------------------------------------------------
# SNR computation over a DataFrame subset
# ---------------------------------------------------------------------------

def _compute_snr_sets(df_pos_full, df_pos_used, train_ids, val_ids, test_ids,
                      model, pp, device, ds_cfg):
    """
    Compute raw/target/pred SNR for every signal sample in df_pos_full.
    Each sample is loaded and predicted exactly once; results are assigned
    to the appropriate sets (full / used / train / val / test) by sample_id.

    Returns a dict with keys: full, used, train, val, test
    (each containing raw, target, pred lists of floats) plus split-size metadata.
    """
    fs_das    = ds_cfg["fs_das"]
    fps_video = ds_cfg["fps_video"]

    used_ids  = set(df_pos_used["sample_id"].astype(str))
    train_set = set(str(i) for i in train_ids)
    val_set   = set(str(i) for i in val_ids)
    test_set  = set(str(i) for i in test_ids)

    snr_data = {
        split: {"raw": [], "target": [], "pred": []}
        for split in ("full", "used", "train", "val", "test")
    }

    # Only iterate signal windows (count > 0 AND non-empty signal_rects)
    df_sig = df_pos_full[
        df_pos_full["signal_rects"].apply(lambda x: bool(json.loads(x)))
    ].reset_index(drop=True)

    model.eval()
    for _, row in df_sig.iterrows():
        rects = json.loads(row["signal_rects"])
        raw   = np.load(row["data_path"]).astype(np.float32)
        tgt   = pp(raw, fs_das).astype(np.float32)
        pred  = _predict_one(model, raw, device)

        win_start = row["start_frame"]
        r_snr = compute_snr(raw,  rects, win_start, fps_video, fs_das)
        t_snr = compute_snr(tgt,  rects, win_start, fps_video, fs_das)
        p_snr = compute_snr(pred, rects, win_start, fps_video, fs_das)

        snr_data["full"]["raw"].append(r_snr)
        snr_data["full"]["target"].append(t_snr)
        snr_data["full"]["pred"].append(p_snr)

        sid = str(row["sample_id"])
        if sid in used_ids:
            snr_data["used"]["raw"].append(r_snr)
            snr_data["used"]["target"].append(t_snr)
            snr_data["used"]["pred"].append(p_snr)

            if sid in train_set:
                snr_data["train"]["raw"].append(r_snr)
                snr_data["train"]["target"].append(t_snr)
                snr_data["train"]["pred"].append(p_snr)
            elif sid in val_set:
                snr_data["val"]["raw"].append(r_snr)
                snr_data["val"]["target"].append(t_snr)
                snr_data["val"]["pred"].append(p_snr)
            elif sid in test_set:
                snr_data["test"]["raw"].append(r_snr)
                snr_data["test"]["target"].append(t_snr)
                snr_data["test"]["pred"].append(p_snr)

    # Attach split sizes as private metadata for log_run
    snr_data["_n_train"] = len(train_ids)
    snr_data["_n_val"]   = len(val_ids)
    snr_data["_n_test"]  = len(test_ids)
    snr_data["_n_full"]  = len(df_pos_full)
    return snr_data


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------

def _run_trial(params, trial_idx, n_trials, ds_cfg, den_cfg, conn,
               das_array, vehicle_labels, device, args):
    run_id  = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{trial_idx:03d}"
    label   = _params_label(params)
    divider = "─" * 70

    print(f"\n{divider}")
    print(f"Trial {trial_idx}/{n_trials}: {label}")
    print(f"run_id: {run_id}")
    print(divider)

    t_total_start = time.time()
    window_length_s = params["window_length_s"]
    sample_size     = params["sample_size"]
    steps_spec      = params["steps"]

    # ── Pre-windowing skip check ────────────────────────────────────────────
    est_total = das_array.shape[1] // int(window_length_s * ds_cfg["fs_das"])
    if est_total < sample_size:
        msg = f"est_total_windows={est_total} < sample_size={sample_size} — skip"
        print(f"  SKIP: {msg}")
        log_run(conn, run_id, params, None, None, None, status="skip", error=msg)
        return

    # ── Windowing (cached) ──────────────────────────────────────────────────
    cache       = _prepare_window_samples(
        window_length_s, das_array, vehicle_labels, ds_cfg,
        args.sweep_data_root, force=args.force_rewindow,
    )
    df_pos_full = cache["df_pos"]
    repr_ids    = cache["repr_ids"]

    # Post-windowing skip check
    if len(df_pos_full) < sample_size:
        msg = f"positive_samples={len(df_pos_full)} < sample_size={sample_size} — skip"
        print(f"  SKIP: {msg}")
        log_run(conn, run_id, params, None, None, None, status="skip", error=msg)
        return

    # ── Build per-trial config (base = denoising.yaml + dataset.yaml) ───────
    cfg = dict(den_cfg)
    cfg["data_dir"]  = cache["data_dir"]
    cfg["fs"]        = ds_cfg["fs_das"]
    cfg["dx"]        = ds_cfg.get("dx")
    cfg["dt"]        = ds_cfg.get("dt")
    cfg["steps"]     = steps_spec
    cfg["sample_size"] = sample_size

    # Scalar sweep params forwarded directly to training config
    _FORWARDED_PARAMS = {"lr", "epochs", "batch_size", "pred_reg_weight"}
    for k in _FORWARDED_PARAMS:
        if k in params:
            cfg[k] = params[k]

    # Loss spec: {name: mse/l1/gan_nce, <optional overrides>}
    # Priority: losses.yaml defaults < denoising.yaml < sweep spec overrides
    if "loss" in params:
        loss_spec = params["loss"]
        if isinstance(loss_spec, str):
            loss_spec = {"name": loss_spec}   # backward compat
        loss_name = loss_spec.get("name", "mse")
        cfg["loss"] = loss_name
        # Re-apply loss defaults for the chosen loss (den_cfg had defaults for a different loss)
        _losses_path = "configs/losses.yaml"
        if os.path.exists(_losses_path):
            with open(_losses_path) as _lf:
                _losses_raw = yaml.safe_load(_lf)
            for k, v in _losses_raw.get(loss_name, {}).items():
                cfg[k] = v
        # Apply explicit per-spec overrides (highest priority)
        for k, v in loss_spec.items():
            if k != "name":
                cfg[k] = v

    # ── Setup ───────────────────────────────────────────────────────────────
    run_dir = os.path.join(args.sweep_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    steps_for_pp = [(s["name"], {k: v for k, v in s.items() if k != "name"})
                    for s in steps_spec]
    pp = make_preprocess(steps=steps_for_pp, dx=cfg.get("dx"), dt=cfg.get("dt"))

    df_pos_used = df_pos_full.iloc[:sample_size].reset_index(drop=True)

    snr_data   = None
    timing     = {}
    model_path = None
    model      = None

    try:
        # ── Stage 1: Preprocess timing ──────────────────────────────────────
        # Apply pp once to all sample_size samples in memory to measure throughput.
        print(f"  [1/3] preprocessing {sample_size} samples…")
        t0 = time.time()
        for _, row in df_pos_used.iterrows():
            raw = np.load(row["data_path"]).astype(np.float32)
            _   = pp(raw, ds_cfg["fs_das"])
        timing["preprocess"] = time.time() - t0
        print(f"        done in {timing['preprocess']:.1f}s")

        # ── Stage 2: Train ──────────────────────────────────────────────────
        print(f"  [2/3] training…")
        t0 = time.time()
        model, history, train_ids, val_ids, test_ids = train_denoising_core(
            df_pos_used, cfg, run_dir, device=device
        )
        timing["train"] = time.time() - t0
        print(f"        done in {timing['train']:.1f}s")

        # Save model + loss history
        model_path = os.path.join(run_dir, "best_model.pt")
        torch.save(model.state_dict(), model_path)
        with open(os.path.join(run_dir, "losses.json"), "w") as f:
            json.dump(history, f, indent=2)

        # ── Stage 3: Predict + SNR ──────────────────────────────────────────
        print(f"  [3/3] predict + SNR over {len(df_pos_full)} signal windows…")
        t0 = time.time()
        model.eval()
        snr_data = _compute_snr_sets(
            df_pos_full, df_pos_used, train_ids, val_ids, test_ids,
            model, pp, device, ds_cfg,
        )
        timing["predict"] = time.time() - t0
        print(f"        done in {timing['predict']:.1f}s")

        # ── Representative samples (one per count value, same IDs across trials) ─
        for count_val, sid in sorted(repr_ids.items()):
            row = cache["df_all"][cache["df_all"]["sample_id"].astype(str) == sid]
            if row.empty:
                continue
            row = row.iloc[0]
            raw  = np.load(row["data_path"]).astype(np.float32)
            tgt  = pp(raw, ds_cfg["fs_das"]).astype(np.float32)
            pred = _predict_one(model, raw, device)
            pfx  = os.path.join(run_dir, f"repr_count{count_val}")
            np.save(f"{pfx}_raw.npy",    raw)
            np.save(f"{pfx}_target.npy", tgt)
            np.save(f"{pfx}_pred.npy",   pred)

        timing["total"] = time.time() - t_total_start
        log_run(conn, run_id, params, snr_data, timing, model_path, status="ok")

        # ── Print trial summary ─────────────────────────────────────────────
        full = snr_data.get("full", {})
        test = snr_data.get("test", {})

        def _fmt(vals):
            v = [x for x in (vals or []) if x is not None and x == x]
            if not v:
                return "n/a"
            return f"{np.mean(v):.2f}±{np.std(v):.2f}"

        print(f"\n  ✓ Trial {trial_idx}/{n_trials} completed in {timing['total']:.1f}s")
        print(f"    SNR full  — raw: {_fmt(full.get('raw'))}  "
              f"target: {_fmt(full.get('target'))}  "
              f"pred: {_fmt(full.get('pred'))} dB")
        print(f"    SNR test  — raw: {_fmt(test.get('raw'))}  "
              f"target: {_fmt(test.get('target'))}  "
              f"pred: {_fmt(test.get('pred'))} dB")
        print(f"    Timing    — pp: {timing['preprocess']:.1f}s  "
              f"train: {timing['train']:.1f}s  "
              f"predict: {timing['predict']:.1f}s")
        print(f"    Model     → {model_path}")

    except Exception:
        timing["total"] = time.time() - t_total_start
        err = traceback.format_exc()
        print(f"\n  ✗ Trial {trial_idx}/{n_trials} FAILED after {timing['total']:.1f}s")
        print(f"    {err.splitlines()[-1]}")
        log_run(conn, run_id, params, None, timing, None, status="error", error=err)

    finally:
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grid search over DAS denoising hyperparameters.")
    parser.add_argument("--sweep",   default="configs/sweep.yaml",
                        help="Path to sweep config YAML (default: configs/sweep.yaml)")
    parser.add_argument("--dataset", default=None,
                        help="Dataset profile name or path (default: ACTIVE_DATASET in config.py)")
    parser.add_argument("--db",      default=None,
                        help="SQLite DB path (default: <sweep_dir>/experiments.db)")
    parser.add_argument("--mode",    default="append", choices=["append", "reset", "skip"],
                        help="append: accumulate results (default); "
                             "reset: wipe runs table first; "
                             "skip: skip trials whose params already have a successful run")
    parser.add_argument("--force-rewindow", action="store_true",
                        help="Ignore cached windowed data and re-run windowing")
    args = parser.parse_args()

    # ── Load configs ────────────────────────────────────────────────────────
    with open(args.sweep) as f:
        sweep_cfg = yaml.safe_load(f)

    sweep_dir       = sweep_cfg.pop("sweep_dir",       "results/sweep")
    sweep_data_root = sweep_cfg.pop("sweep_data_root", "data/sweep")
    args.sweep_dir       = sweep_dir
    args.sweep_data_root = sweep_data_root

    # Base training config (denoising.yaml + dataset.yaml merged)
    den_cfg = load_config("denoising", dataset=args.dataset)
    from config import load_dataset_config
    ds_cfg  = load_dataset_config(args.dataset)

    db_path = args.db or os.path.join(sweep_dir, "experiments.db")
    os.makedirs(sweep_dir, exist_ok=True)

    # ── Build grid ──────────────────────────────────────────────────────────
    grid     = build_grid(sweep_cfg)
    n_trials = len(grid)

    print("=" * 70)
    print(f"  Sweep: {n_trials} trial{'s' if n_trials != 1 else ''}")
    print(f"  DB:    {db_path}  (mode={args.mode})")
    print("=" * 70)
    for i, params in enumerate(grid, 1):
        print(f"  #{i:>3d} | {_params_label(params)}")
    print("=" * 70)

    # ── Load DAS data ONCE (expensive HDF5 read) ────────────────────────────
    print("\nLoading DAS recordings…")
    import glob as _glob
    from DAS import MulDAS

    recording_dir = ds_cfg["recording_dir"]
    file_pattern  = ds_cfg.get("file_pattern", "*.h5")
    files         = sorted(_glob.glob(os.path.join(recording_dir, file_pattern)))
    if not files:
        raise FileNotFoundError(f"No DAS files matching '{file_pattern}' in {recording_dir}")

    channels  = np.arange(ds_cfg["channel_start"], ds_cfg["channel_end"])
    das_array = MulDAS(files, channels).data
    print(f"Loaded DAS array: {das_array.shape}  ({das_array.nbytes / 1e6:.0f} MB)")

    vehicle_labels = pd.read_csv(ds_cfg["labeling_csv"])
    print(f"Loaded vehicle labels: {len(vehicle_labels)} rows\n")

    # ── Setup DB ────────────────────────────────────────────────────────────
    conn   = setup_database(db_path, mode=args.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Run trials ──────────────────────────────────────────────────────────
    for i, params in enumerate(grid, 1):
        if args.mode == "skip" and _already_succeeded(conn, params):
            print(f"\nTrial {i}/{n_trials}: SKIP (already succeeded) — {_params_label(params)}")
            continue
        _run_trial(params, i, n_trials, ds_cfg, den_cfg, conn,
                   das_array, vehicle_labels, device, args)

    # ── Final summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  Sweep complete. Results in DB:")
    rows = conn.execute(
        "SELECT status, COUNT(*) FROM runs GROUP BY status"
    ).fetchall()
    for status, count in rows:
        print(f"    {status}: {count}")
    print(f"  DB: {db_path}")
    print("=" * 70)
    conn.close()


if __name__ == "__main__":
    main()
