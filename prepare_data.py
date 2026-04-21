"""
CLI for DAS data preparation.

Reads raw HDF5 files, slices into fixed-length windows, maps manual vehicle
labels to per-window counts/types/signal_rects, and writes:
  - data/raw/sample_XXXXXX.npy  (one per window)
  - data/labels.csv

Usage:
    python prepare_data.py
    python prepare_data.py --config configs/data_prep.yaml
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from DAS import MulDAS


# ---------------------------------------------------------------------------
# Time / frame helpers
# ---------------------------------------------------------------------------

def _hms_to_seconds(hms: str) -> int:
    return sum(int(x) * m for x, m in zip(hms.split(":"), [3600, 60, 1]))


def _frame_to_das_seconds(
    frame: float,
    fps_video: float,
    frame_ref: int,
    sys_time_ref_hms: str,
    das_start_hms: str,
) -> float:
    t_ref = _hms_to_seconds(sys_time_ref_hms)
    t_das0 = _hms_to_seconds(das_start_hms)
    t_frame_abs = t_ref + (frame - frame_ref) / fps_video
    return t_frame_abs - t_das0


# ---------------------------------------------------------------------------
# Bridge traversal estimate
# ---------------------------------------------------------------------------

def add_frame_end_bridge(
    df: pd.DataFrame,
    estimate_speed: float,
    side_length_ft: float,
    bridge_length_m: float,
    fps_video: float,
    speed_unit: str = "mph",
    frame_col: str = "frame_start",
    out_col: str = "frame_end_bridge",
) -> pd.DataFrame:
    """Add frame_end_bridge = frame when vehicle exits the far end of the bridge."""
    unit = speed_unit.lower()
    if unit == "mph":
        speed_mps = estimate_speed * 0.44704
    elif unit == "kmh":
        speed_mps = estimate_speed * (1000 / 3600)
    elif unit in ("mps", "m/s"):
        speed_mps = float(estimate_speed)
    else:
        raise ValueError("speed_unit must be one of {mph, kmh, mps}")

    side_m = side_length_ft * 0.3048
    total_m = 2 * side_m + bridge_length_m
    delta_frames = (total_m / speed_mps) * fps_video

    df[out_col] = df[frame_col].astype(float) + float(delta_frames)
    return df


# ---------------------------------------------------------------------------
# Window generation
# ---------------------------------------------------------------------------

def prepare_das_windows_and_labels(
    das_array: np.ndarray,
    labels_csv_or_df,
    window_length_s: float,
    stride_s: float,
    fs_das: int = 2000,
    fps_video: float = 29.98762733,
    frame_ref: int = 115,
    sys_time_ref_hms: str = "10:27:09",
    das_start_hms: str = "10:24:47",
    out_dir: str = "data/raw",
    csv_out: str = "../labels.csv",
    start_frame_col: str = "frame_start",
    end_frame_col: str = "frame_end_bridge",
    multi_token: str = "mixed",
    none_token: Optional[str] = None,
) -> tuple:
    """
    Slide a window over das_array, save each window as .npy, and write labels.csv.

    Columns written:
      sample_id, data_path, count, start_frame, end_frame, vehicle_type, signal_rects

    signal_rects: JSON list of [frame_start, frame_end] pairs (global video frames,
    clipped to the window's frame range) marking the ~1-second signal rectangle
    for each vehicle in the window.  [] for background windows.
    """
    if isinstance(labels_csv_or_df, (str, Path)):
        df_lbl = pd.read_csv(labels_csv_or_df)
    else:
        df_lbl = labels_csv_or_df.copy()

    required = {start_frame_col, end_frame_col}
    assert required.issubset(df_lbl.columns), f"labels must contain: {required}"

    # Convert frames → DAS seconds
    def to_das_s(f):
        return _frame_to_das_seconds(f, fps_video, frame_ref, sys_time_ref_hms, das_start_hms)

    df_lbl["start_s"] = df_lbl[start_frame_col].astype(float).map(to_das_s)
    df_lbl["end_s"] = df_lbl[end_frame_col].astype(float).map(to_das_s)
    df_lbl["start_s"], df_lbl["end_s"] = (
        np.minimum(df_lbl["start_s"], df_lbl["end_s"]),
        np.maximum(df_lbl["start_s"], df_lbl["end_s"]),
    )

    t_min, t_max = df_lbl["start_s"].min(), df_lbl["end_s"].max()
    i_min = max(0, int(np.ceil(t_min * fs_das)))
    i_max = min(das_array.shape[1], int(np.floor(t_max * fs_das)))

    win_len = int(round(window_length_s * fs_das))
    stride = int(round(stride_s * fs_das))

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    events_np = df_lbl[["start_s", "end_s"]].to_numpy()
    veh_frames = df_lbl[start_frame_col].to_numpy(dtype=float)

    # Constants for frame back-map
    das0 = _hms_to_seconds(das_start_hms)
    sys_ref = _hms_to_seconds(sys_time_ref_hms)

    rows = []
    sample_id = 0
    start_idx = i_min

    while start_idx <= i_max - win_len:
        end_idx = start_idx + win_len
        x = das_array[:, start_idx:end_idx]

        ws_s = start_idx / fs_das
        we_s = end_idx / fs_das

        # Overlap mask: which vehicles are present in this window
        overlap = np.minimum(we_s, events_np[:, 1]) - np.maximum(ws_s, events_np[:, 0])
        mask = overlap > 0
        cnt = int(mask.sum())

        # Back-map window to video frame numbers
        win_start_frame = int(np.floor(frame_ref + (ws_s + das0 - sys_ref) * fps_video))
        win_end_frame = int(np.ceil(frame_ref + (we_s + das0 - sys_ref) * fps_video))

        # vehicle_type aggregation
        if cnt == 0:
            vtype = none_token if none_token is not None else np.nan
        elif cnt == 1:
            idx = int(np.where(mask)[0][0])
            vtype = df_lbl.iloc[idx].get("vehicle_type", none_token)
        else:
            vtype = multi_token

        # signal_rects: one rect per overlapping vehicle, clipped to window
        signal_rects = []
        for i in np.where(mask)[0]:
            vf = veh_frames[i]
            rect_s = int(vf - 3)
            rect_e = int(vf + 27)
            clipped_s = max(rect_s, win_start_frame)
            clipped_e = min(rect_e, win_end_frame)
            if clipped_s < clipped_e:
                signal_rects.append([clipped_s, clipped_e])

        fname = f"sample_{sample_id:06d}.npy"
        np.save(out_path / fname, x.astype(np.float32))

        rows.append({
            "sample_id": f"{sample_id:06d}",
            "data_path": str((out_path / fname).as_posix()),
            "count": cnt,
            "start_frame": win_start_frame,
            "end_frame": win_end_frame,
            "vehicle_type": vtype,
            "signal_rects": json.dumps(signal_rects),
        })

        sample_id += 1
        start_idx += stride

    df_out = pd.DataFrame(rows)
    csv_path = out_path / csv_out
    df_out.to_csv(csv_path, index=False)
    print(f"Saved {len(df_out)} samples → {out_path.resolve()}")
    print(f"Labels → {csv_path.resolve()}")
    return df_out, out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Prepare DAS windows and labels.")
    parser.add_argument(
        "--config", default="configs/data_prep.yaml",
        help="Path to data_prep YAML config (default: configs/data_prep.yaml)",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    from config import DAS_RECORDING_DIR, LABELING_CSV

    # Load DAS recordings
    files = sorted(glob.glob(os.path.join(DAS_RECORDING_DIR, "*Newville*")))
    if not files:
        raise FileNotFoundError(f"No DAS files found in {DAS_RECORDING_DIR}")
    channels = np.arange(cfg["channel_start"], cfg["channel_end"])
    muldas = MulDAS(files, channels)
    das_array = muldas.data
    print(f"Loaded DAS array: {das_array.shape}")

    # Load and adjust vehicle labels
    vehicle_label = pd.read_csv(LABELING_CSV)
    vehicle_label_adjusted = add_frame_end_bridge(
        df=vehicle_label,
        estimate_speed=cfg["estimate_speed_mph"],
        side_length_ft=cfg["side_length_ft"],
        bridge_length_m=cfg["bridge_length_m"],
        fps_video=cfg["fps_video"],
        speed_unit="mph",
        frame_col="frame_start",
        out_col="frame_end_bridge",
    )

    prepare_das_windows_and_labels(
        das_array=das_array,
        labels_csv_or_df=vehicle_label_adjusted,
        window_length_s=cfg["window_length_s"],
        stride_s=cfg["stride_s"],
        fs_das=cfg["fs_das"],
        fps_video=cfg["fps_video"],
        frame_ref=cfg["frame_ref"],
        sys_time_ref_hms=cfg["sys_time_ref_hms"],
        das_start_hms=cfg["das_start_hms"],
        out_dir=cfg["out_dir"],
        csv_out=cfg["csv_out"],
        start_frame_col="frame_start",
        end_frame_col="frame_end_bridge",
        multi_token=cfg["multi_token"],
        none_token=cfg["none_token"],
    )


if __name__ == "__main__":
    main()
