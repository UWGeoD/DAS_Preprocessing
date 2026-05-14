"""
Compute per-sample SNR for raw, preprocessed, and UNet-denoised DAS windows.

Prerequisites (run in order):
    python prepare_data.py                   # generates data/raw/ + data/labels.csv
    python train.py --task denoising         # trains the UNet
    python predict.py --task denoising       # generates data/denoised/

Usage:
    python compute_metrics.py
    python compute_metrics.py --out results/metrics.csv
    python compute_metrics.py --splits results/denoising/splits.json --plot-dir results/denoising/SNR
"""

import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message="Pandas requires version '2.8.4' or newer of 'numexpr'",
    category=UserWarning,
)

import numpy as np
import pandas as pd
import yaml

from config import load_dataset_config
from DAS import DAS
from preprocessing import make_preprocess
from Utilities import compute_snr


def _save_snr_plots(df, splits_path, plot_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    methods = ["snr_raw", "snr_pp", "snr_denoised"]
    labels  = ["Raw", "Preprocessed", "UNet Denoised"]
    colors  = ["#d9534f", "#5bc0de", "#5cb85c"]
    splits_order = ["train", "val", "test"]

    df = df.copy()
    df["gain_pp"]       = df["snr_pp"]       - df["snr_raw"]
    df["gain_denoised"] = df["snr_denoised"] - df["snr_raw"]

    df["split"] = "unknown"
    if os.path.exists(splits_path):
        with open(splits_path) as f:
            splits = json.load(f)
        for split_name, ids in splits.items():
            ids_str = [f"{int(i):06d}" for i in ids]
            df.loc[df["sample_id"].isin(ids_str), "split"] = split_name

    split_data = {s: df[df["split"] == s] for s in splits_order}

    def _make_fig():
        fig = plt.figure(figsize=(14, 10))
        gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
        return fig, fig.add_subplot(gs[0, :]), [fig.add_subplot(gs[1, i]) for i in range(3)]

    def _boxplot(ax, sub, title):
        bp = ax.boxplot([sub[m].dropna().values for m in methods],
                        tick_labels=labels, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_ylabel("SNR (dB)")
        ax.set_title(title)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    def _histogram(ax, sub, title):
        for m, lbl, c in zip(methods, labels, colors):
            v = sub[m].dropna().values
            if len(v):
                ax.hist(v, bins=15, alpha=0.5, label=f"{lbl} (med={np.median(v):.1f} dB)", color=c)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    def _gain_bars(ax, sub, title):
        x = np.arange(len(sub))
        ax.bar(x - 0.2, sub["gain_pp"].values,       width=0.4, label="PP gain",       color="#5bc0de", alpha=0.8)
        ax.bar(x + 0.2, sub["gain_denoised"].values, width=0.4, label="Denoised gain", color="#5cb85c", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("ΔSNR vs Raw (dB)")
        ax.set_title(title)
        ax.legend(fontsize=7)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    def _fill_split_axes(plot_fn, axes_s):
        for ax, s in zip(axes_s, splits_order):
            sub = split_data[s]
            if len(sub):
                plot_fn(ax, sub, f"{s.capitalize()} (n={len(sub)})")
            else:
                ax.axis("off")
                ax.set_title(f"{s.capitalize()} — no samples")

    # Figure 1: Boxplot
    fig, ax_all, axes_s = _make_fig()
    _boxplot(ax_all, df, f"All (n={len(df)})")
    _fill_split_axes(_boxplot, axes_s)
    fig.suptitle("SNR Boxplot", fontsize=13)
    fig.savefig(os.path.join(plot_dir, "snr_boxplot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  snr_boxplot.png")

    # Figure 2: Histogram
    fig, ax_all, axes_s = _make_fig()
    _histogram(ax_all, df, f"All (n={len(df)})")
    _fill_split_axes(_histogram, axes_s)
    fig.suptitle("SNR Histogram", fontsize=13)
    fig.savefig(os.path.join(plot_dir, "snr_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  snr_histogram.png")

    # Figure 3: Per-sample gain
    fig, ax_all, axes_s = _make_fig()
    _gain_bars(ax_all, df, f"All (n={len(df)})")
    _fill_split_axes(_gain_bars, axes_s)
    fig.suptitle("Per-Sample SNR Gain vs Raw", fontsize=13)
    fig.savefig(os.path.join(plot_dir, "snr_gain.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  snr_gain.png")

    print(f"Plots saved → {plot_dir}/")

    # Summary stats CSV
    stats_rows = []
    for split_name in ["all", "train", "val", "test"]:
        sub = df if split_name == "all" else split_data[split_name]
        for m, lbl in zip(methods, labels):
            vals = sub[m].dropna()
            if len(vals) == 0:
                continue
            stats_rows.append({
                "split": split_name, "method": lbl, "n": len(vals),
                "mean": round(vals.mean(), 2), "median": round(vals.median(), 2),
                "std": round(vals.std(), 2), "min": round(vals.min(), 2), "max": round(vals.max(), 2),
            })
    stats_df = pd.DataFrame(stats_rows)
    stats_path = os.path.join(plot_dir, "snr_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"Stats saved → {stats_path}")

    # Print per-split summary to stdout
    print()
    for s in splits_order:
        sub = split_data[s]
        if len(sub) == 0:
            continue
        print(f"{s.upper()} (n={len(sub)}):")
        for m, lbl in zip(methods, labels):
            vals = sub[m].dropna()
            print(f"  {lbl:>15s}: median={vals.median():6.2f} dB  mean={vals.mean():6.2f} dB")


def main():
    parser = argparse.ArgumentParser(description="Compute per-sample SNR metrics.")
    parser.add_argument(
        "--dataset", default=None,
        help="Dataset profile name or path (default: ACTIVE_DATASET in config.py)",
    )
    parser.add_argument("--denoising-config", default="configs/denoising.yaml")
    parser.add_argument("--denoised-dir", default=None,
                        help="Directory containing denoised_sample_XXXXXX.npy files "
                             "(default: <data_dir>/denoised)")
    parser.add_argument("--out", default="results/metrics.csv")
    parser.add_argument("--splits", default="results/denoising/splits.json",
                        help="Path to splits.json for per-split breakdown")
    parser.add_argument("--plot-dir", default="results/denoising/SNR",
                        help="Directory to save SNR plots and stats")
    args = parser.parse_args()

    ds_cfg = load_dataset_config(args.dataset)
    with open(args.denoising_config) as f:
        den_cfg = yaml.safe_load(f)

    _meta = DAS(ds_cfg["das_file"]).meta
    dx = ds_cfg.get("dx") or _meta.get("dx")
    dt = ds_cfg.get("dt") or _meta.get("dt")

    steps = [(s["name"], {k: v for k, v in s.items() if k != "name"}) for s in den_cfg["steps"]]
    pp = make_preprocess(steps=steps, dx=dx, dt=dt)
    fs_das = ds_cfg["fs_das"]
    fps_video = ds_cfg["fps_video"]

    data_dir = ds_cfg["data_dir"]
    denoised_dir = args.denoised_dir or os.path.join(data_dir, "denoised")
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
        denoised_path = os.path.join(denoised_dir, denoised_fname)
        if not os.path.exists(denoised_path):
            print(f"  missing denoised file: {denoised_path} — skipping sample {row['sample_id']}")
            skipped += 1
            continue

        raw      = np.load(row["data_path"]).astype(np.float32)
        clean    = pp(raw, fs_das).astype(np.float32)
        denoised = np.load(denoised_path).astype(np.float32)

        win_start_frame = row["start_frame"]
        rows.append({
            "sample_id":    f"{int(row['sample_id']):06d}",
            "vehicle_type": row["vehicle_type"],
            "count":        row["count"],
            "snr_raw":      compute_snr(raw,      rects, win_start_frame, fps_video, fs_das),
            "snr_pp":       compute_snr(clean,    rects, win_start_frame, fps_video, fs_das),
            "snr_denoised": compute_snr(denoised, rects, win_start_frame, fps_video, fs_das),
        })

    df_out = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(f"\nProcessed {len(df_out)} samples ({skipped} skipped) → {out_path}")
    print(df_out[["snr_raw", "snr_pp", "snr_denoised"]].describe().round(2))
    print()

    _save_snr_plots(df_out, args.splits, args.plot_dir)


if __name__ == "__main__":
    main()
