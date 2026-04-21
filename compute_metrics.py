"""
Compute per-sample SNR for raw, preprocessed, and UNet-denoised DAS windows.

Prerequisites (run in order):
    python prepare_data.py                   # generates data/raw/ + data/labels.csv
    python train.py --task denoising         # trains the UNet
    python predict.py --task denoising       # generates data/denoised/

Usage:
    python compute_metrics.py
    python compute_metrics.py --out results/metrics.csv
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from config import DAS_FILE
from DAS import DAS
from preprocessing import make_preprocess
from Utilities import compute_snr


def main():
    parser = argparse.ArgumentParser(description="Compute per-sample SNR metrics.")
    parser.add_argument("--denoising-config", default="configs/denoising.yaml")
    parser.add_argument("--data-prep-config", default="configs/data_prep.yaml")
    parser.add_argument("--denoised-dir", default="data/denoised",
                        help="Directory containing denoised_sample_XXXXXX.npy files")
    parser.add_argument("--out", default="results/metrics.csv")
    args = parser.parse_args()

    with open(args.denoising_config) as f:
        den_cfg = yaml.safe_load(f)
    with open(args.data_prep_config) as f:
        dp_cfg = yaml.safe_load(f)

    # Sensor metadata (same resolution as used during training)
    _meta = DAS(DAS_FILE).meta
    dx = den_cfg.get("dx") or _meta["dx"]
    dt = den_cfg.get("dt") or _meta["dt"]

    # Preprocessing pipeline (same as training)
    steps = [(s["name"], {k: v for k, v in s.items() if k != "name"}) for s in den_cfg["steps"]]
    pp = make_preprocess(steps=steps, dx=dx, dt=dt)
    fs_das = den_cfg["fs"]
    fps_video = dp_cfg["fps_video"]

    # All samples that have a signal rect (count > 0 and non-empty signal_rects)
    data_dir = den_cfg.get("data_dir", "data")
    df = pd.read_csv(os.path.join(data_dir, den_cfg["labels_csv"]))
    df_signal = df[df["count"] > 0].reset_index(drop=True)

    rows = []
    skipped = 0
    for _, row in df_signal.iterrows():
        rects = json.loads(row["signal_rects"])
        if not rects:
            skipped += 1
            continue

        denoised_fname = f"denoised_sample_{int(row['sample_id']):06d}.npy"
        denoised_path = os.path.join(args.denoised_dir, denoised_fname)
        if not os.path.exists(denoised_path):
            print(f"  missing denoised file: {denoised_path} — skipping sample {row['sample_id']}")
            skipped += 1
            continue

        raw = np.load(row["data_path"]).astype(np.float32)
        clean = pp(raw, fs_das).astype(np.float32)
        denoised = np.load(denoised_path).astype(np.float32)

        win_start_frame = row["start_frame"]
        rows.append({
            "sample_id":    row["sample_id"],
            "vehicle_type": row["vehicle_type"],
            "count":        row["count"],
            "snr_raw":      compute_snr(raw,     rects, win_start_frame, fps_video, fs_das),
            "snr_pp":       compute_snr(clean,   rects, win_start_frame, fps_video, fs_das),
            "snr_denoised": compute_snr(denoised, rects, win_start_frame, fps_video, fs_das),
        })

    df_out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(f"\nProcessed {len(df_out)} samples ({skipped} skipped) → {out_path}")
    print(df_out[["snr_raw", "snr_pp", "snr_denoised"]].describe().round(2))


if __name__ == "__main__":
    main()
