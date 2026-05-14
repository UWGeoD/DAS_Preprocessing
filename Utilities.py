# Utilities.py (only the relevant bits shown)
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
from collections import OrderedDict
import re

epsilon = 1e-8
normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + epsilon)

_PUB_COLORS = {
    "raw":      "#d9534f",
    "pp":       "#5bc0de",
    "denoised": "#5cb85c",
}

_STEP_DISPLAY = {
    "detrend":               "detrend",
    "bandpass":              "bandpass",
    "fk_filter":             "f-k filter",
    "hilbert_transform":     "Hilbert",
    "curvelet_like_denoise": "curvelet",
}


def _fmt_step(step):
    """Format a step dict (or plain name str) into a human-readable label."""
    if isinstance(step, str):
        return _STEP_DISPLAY.get(step, step)
    name  = step.get("name", "?")
    label = _STEP_DISPLAY.get(name, name)
    if name == "bandpass":
        lo, hi = step.get("f_lo", "?"), step.get("f_hi", "?")
        label += f" [{lo}–{hi} Hz]"
    elif name == "fk_filter":
        vmin, vmax = step.get("vmin"), step.get("vmax")
        if vmin is not None and vmax is not None:
            label += f" [{vmin:.0f}–{vmax:.0f} m/s]"
    return label

def plot_das_comparison(
    raw, clean, denoised,
    snr_raw, snr_clean, snr_denoised,
    channels, dx, dt,
    pp_steps,
    signal_rects=None,
    win_start_frame=None,
    fps_video=None,
    show_signal_shade=False,
    show_signal_ticks=False,
    show_rect=False,
    sample_id=None,
    figsize=(14, 9),
):
    """
    Publication-quality DAS comparison figure.

    Left column (3 heatmaps, shared axes):
        Raw  ·  Preprocessed (with step labels)  ·  UNet Denoised
    Right column (SNR panel, spans all rows):
        Horizontal bar chart comparing SNR across the three stages,
        ordered top-to-bottom to match the heatmap stack.

    Parameters
    ----------
    raw, clean, denoised     : ndarray [channels, time]
    snr_raw, snr_clean,
      snr_denoised           : float   SNR in dB
    channels                 : 1-D array
    dx, dt                   : float   sample intervals (m, s)
    pp_steps                 : list of step dicts ({"name": ...}) from denoising.yaml
    signal_rects             : list of [frame_start, frame_end] pairs (optional)
    win_start_frame          : int     start_frame from labels.csv
    fps_video                : float
    sample_id                : str/int used in the figure title
    figsize                  : (width, height) in inches

    Returns
    -------
    fig, (ax_raw, ax_pp, ax_den, ax_snr)
    """
    from matplotlib.gridspec import GridSpec

    x = np.asarray(channels) * dx
    t = np.arange(raw.shape[1]) * dt
    extent = [x[0], x[-1], t[-1], t[0]]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize, facecolor="white")
    gs  = GridSpec(3, 2, figure=fig,
                   width_ratios=[5, 1],
                   hspace=0.30, wspace=0.28)
    ax_raw = fig.add_subplot(gs[0, 0])
    ax_pp  = fig.add_subplot(gs[1, 0], sharex=ax_raw, sharey=ax_raw)
    ax_den = fig.add_subplot(gs[2, 0], sharex=ax_raw, sharey=ax_raw)
    ax_snr = fig.add_subplot(gs[:, 1])

    _im_kw = dict(cmap="seismic", vmin=-1, vmax=1, aspect="auto",
                  extent=extent, interpolation="none", zorder=1)

    # ── Heatmap helper ────────────────────────────────────────────────────────
    def _draw_heatmap(ax, data, method_label, color,
                      sub_label=None, show_xlabel=False, show_ylabel=False):
        ax.imshow(normalize(data).T, **_im_kw)
        ax.set_xlim(x[0], x[-1])
        ax.set_ylim(t[-1], t[0])
        ax.tick_params(direction="in", which="both", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Time (s)" if show_ylabel else "", fontsize=10)
        if not show_xlabel:
            plt.setp(ax.get_xticklabels(), visible=False)
            ax.tick_params(bottom=False)
        else:
            ax.set_xlabel("Channel Position (m)", fontsize=10)
        # method label as colored title outside the heatmap (left-aligned)
        ax.set_title(method_label, color=color, fontsize=11,
                     fontweight="bold", loc="left", pad=5)
        # preprocessing step annotation — subtle dark badge at bottom-left inside heatmap
        if sub_label:
            ax.text(0.012, 0.03, sub_label,
                    transform=ax.transAxes, fontsize=7.5, color="white",
                    va="bottom", ha="left", zorder=5, style="italic",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="black",
                              alpha=0.45, edgecolor="none"))

    pp_sub = " → ".join(_fmt_step(s) for s in pp_steps)
    _draw_heatmap(ax_raw, raw,      "Raw",           _PUB_COLORS["raw"])
    _draw_heatmap(ax_pp,  clean,    "Preprocessed",  _PUB_COLORS["pp"],
                  sub_label=pp_sub, show_ylabel=True)
    _draw_heatmap(ax_den, denoised, "UNet Denoised", _PUB_COLORS["denoised"],
                  show_xlabel=True)

    # ── Signal region overlays (each independently optional) ──────────────────
    if signal_rects and win_start_frame is not None and fps_video is not None:
        for ax in (ax_raw, ax_pp, ax_den):
            for r_s, r_e in signal_rects:
                t_s = (r_s - win_start_frame) / fps_video
                t_e = (r_e - win_start_frame) / fps_video
                if show_signal_shade:
                    ax.axhspan(t_s, t_e, alpha=0.18, color="#FFA500",
                               zorder=2, linewidth=0)
                if show_signal_ticks:
                    for t_m in (t_s, t_e):
                        ax.axhline(t_m, xmin=0, xmax=0.025,
                                   color="#E06600", linewidth=2.0, zorder=3)
            if show_rect:
                draw_signal_rects(ax, signal_rects, win_start_frame, fps_video,
                                  x[0], x[-1])

    # ── Figure title ──────────────────────────────────────────────────────────
    sid_str = f"Sample {sample_id}" if sample_id is not None else "DAS Comparison"
    fig.suptitle(sid_str, fontsize=13, fontweight="bold", y=0.99)

    # ── SNR panel ─────────────────────────────────────────────────────────────
    # y positions are flipped so the panel reads top-to-bottom like the heatmaps:
    #   y=2 → Raw (top)   y=1 → Preprocessed   y=0 → UNet Denoised (bottom)
    y_raw, y_pp, y_den = 2, 1, 0
    snr_arr   = [snr_raw,             snr_clean,             snr_denoised]
    color_arr = [_PUB_COLORS["raw"],  _PUB_COLORS["pp"],     _PUB_COLORS["denoised"]]
    y_arr     = [y_raw,               y_pp,                  y_den]

    for val, col, y in zip(snr_arr, color_arr, y_arr):
        ax_snr.barh(y, val, color=col, alpha=0.88, height=0.52, edgecolor="none")

    ax_snr.set_yticks([y_raw, y_pp, y_den])
    ax_snr.set_yticklabels(["Raw", "Preprocessed", "UNet\nDenoised"], fontsize=9.5)
    ax_snr.set_xlabel("SNR (dB)", fontsize=10)
    ax_snr.set_title("SNR Comparison", fontsize=10.5, fontweight="semibold", pad=7)
    ax_snr.tick_params(direction="in", labelsize=9)
    ax_snr.spines[["top", "right"]].set_visible(False)
    ax_snr.axvline(0, color="#aaaaaa", linewidth=0.8, linestyle="--", zorder=0)
    ax_snr.set_ylim(-0.65, 2.7)

    x_min_snr = min(0.0, min(snr_arr) - 0.3)
    x_max_snr = max(snr_arr) * 1.62
    ax_snr.set_xlim(left=x_min_snr, right=x_max_snr)
    gap = (x_max_snr - x_min_snr) * 0.03

    # value labels to the right of each bar
    for val, col, y in zip(snr_arr, color_arr, y_arr):
        ax_snr.text(max(val, 0) + gap, y,
                    f"{val:.1f} dB",
                    va="center", fontsize=9, fontweight="bold", color=col)

    # Δ gain vs Raw — placed BELOW each bar to avoid overlap
    for val, delta, col, y in [
        (snr_clean,    snr_clean    - snr_raw, _PUB_COLORS["pp"],       y_pp),
        (snr_denoised, snr_denoised - snr_raw, _PUB_COLORS["denoised"], y_den),
    ]:
        sign = "+" if delta >= 0 else ""
        ax_snr.text(max(val, 0) + gap, y - 0.30,
                    f"{sign}{delta:.1f} dB vs Raw",
                    va="top", fontsize=7.5, color=col, style="italic")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, (ax_raw, ax_pp, ax_den, ax_snr)


def plot_das_data(data, channels, dx, dt,
                  start_time=None, end_time=None,
                  title=None,
                  ax=None, fig=None, show=True):
    """
    If ax is provided, draw into that axes.
    Otherwise create a new fig, ax.

    Returns (fig, ax).
    """
    x = np.asarray(channels) * dx
    t = np.arange(data.shape[1]) * dt

    if start_time is not None and end_time is not None:
        mask = (t >= start_time) & (t <= end_time)
        data = data[:, mask]
        t = t[mask]

    # Create fig/ax if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    elif fig is None:
        fig = ax.figure

    ax.imshow(
        normalize(data).T,
        #data.T,
        cmap="seismic",
        #cmap="binary",
        vmin=-1,
        vmax=1,
        aspect="auto",
        extent=[x[0], x[-1], t[-1], t[0]],
        interpolation="none",
        animated=True,
        zorder=1,
    )
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(t[-1], t[0])
    ax.set_xlabel("Channel Position (m)")
    ax.set_ylabel("Time (s)")
    if title:
        ax.set_title(title)

    fig.tight_layout(pad=0.7)
    if show:
        plt.show()

    return fig, ax


def plot_single(
    data, channel_num, dx, dt,
    start_time=None, end_time=None,
    ax=None, show=True
):
    t = np.arange(data.shape[1]) * dt
    sig = data[channel_num]

    if start_time is not None and end_time is not None:
        m = (t >= start_time) & (t <= end_time)
        t = t[m]
        sig = sig[m]

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.plot(t, sig, color="black")
    ax.set_title(f"Channel {channel_num} (Position: {channel_num*dx:.2f} m)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True)

    if show:
        fig.tight_layout()
        plt.show()

    return fig, ax


def downsample_data(data, original_fs, target_fs):
    if target_fs >= original_fs:
        raise ValueError("Target sampling rate must be less than original sampling rate.")
    if original_fs % target_fs != 0:
        raise ValueError("Original fs must be divisible by target fs for integer decimation.")
    q = int(original_fs / target_fs)
    return signal.decimate(data, q, axis=1, zero_phase=True)


def draw_signal_rects(ax, signal_rects, win_start_frame, fps_video, x_min, x_max):
    """
    Overlay signal_rects as colored rectangles on a DAS heatmap axes.

    Each rect spans the full channel width and its vehicle-signal time interval.
    Multiple rects get distinct tab10 colors so overlapping vehicles are distinguishable.

    Parameters
    ----------
    ax              : matplotlib Axes (y-axis = time in seconds, as produced by plot_das_data)
    signal_rects    : list of [frame_start, frame_end] global video frame pairs (labels.csv)
    win_start_frame : start_frame value from labels.csv for this window
    fps_video       : video frame rate (Hz)
    x_min, x_max    : channel-position extent of the heatmap (m)
    """
    from matplotlib.patches import Rectangle

    colors = plt.cm.tab10.colors

    for i, (r_s, r_e) in enumerate(signal_rects):
        t_s = (r_s - win_start_frame) / fps_video
        t_e = (r_e - win_start_frame) / fps_video
        rgb = colors[i % len(colors)]
        ax.add_patch(Rectangle(
            (x_min, t_s), x_max - x_min, t_e - t_s,
            facecolor=(*rgb, 0.20),
            edgecolor=(*rgb, 1.0),
            linewidth=2,
            zorder=3,
        ))


def compute_snr(window, signal_rects, win_start_frame, fps_video, fs_das, min_samples=200):
    """
    Compute SNR (dB) for a 2D DAS window [channels, time].

    Parameters
    ----------
    window          : ndarray [channels, time]
    signal_rects    : list of [frame_start, frame_end] in global video frame numbers
                      (as stored in labels.csv, already clipped to window bounds).
                      Pass [] for background windows — returns NaN.
    win_start_frame : start_frame value from labels.csv for this window.
    fps_video       : video frame rate used during data prep.
    fs_das          : DAS sampling rate.
    min_samples     : minimum number of time samples required in both the signal and
                      noise regions; returns NaN if either is smaller (unreliable estimate).

    How it works
    ------------
    1. Per-channel mean is removed from the window before any power computation.
       Raw DAS carries large per-channel DC offsets (absolute fiber strain levels)
       that dominate mean(x^2) in both signal and noise regions equally, collapsing
       SNR to ~0 dB for all samples. Mean removal isolates the AC component where
       the vehicle signal actually lives.
    2. Each [frame_start, frame_end] rect is converted to within-window DAS sample
       indices: i = (frame - win_start_frame) / fps_video * fs_das.
    3. A boolean mask over the time axis is built by OR-ing all rect ranges.
       Overlapping rects are handled automatically — e.g. [1,3] and [2,4]
       both set their index ranges to True, giving a union of [1,4] with no
       sample counted twice.
    4. Signal region  = window[:, signal_mask]  (all channels, signal time)
       Noise region   = window[:, ~signal_mask] (all channels, remaining time)
    5. SNR = 10 * log10( mean(signal^2) / mean(noise^2) )
       Mean is taken over all elements (channels × time) in each region.
       Returns NaN if either region has fewer than min_samples time steps.
    """
    if not signal_rects:
        return float("nan")

    # Remove per-channel DC offset so power estimates reflect the AC signal only.
    chan_mean = window.mean(axis=1, keepdims=True)
    window = window - chan_mean

    n_time = window.shape[1]
    signal_mask = np.zeros(n_time, dtype=bool)

    for rect_s, rect_e in signal_rects:
        i_s = int((rect_s - win_start_frame) / fps_video * fs_das)
        i_e = int((rect_e - win_start_frame) / fps_video * fs_das)
        i_s = max(0, i_s)
        i_e = min(n_time, i_e)
        if i_s < i_e:
            signal_mask[i_s:i_e] = True  # union: overlapping ranges set same bits

    noise_mask = ~signal_mask

    if not signal_mask.any() or not noise_mask.any():
        return float("nan")

    if signal_mask.sum() < min_samples or noise_mask.sum() < min_samples:
        return float("nan")

    signal_power = np.mean(window[:, signal_mask] ** 2)
    noise_power = np.mean(window[:, noise_mask] ** 2)

    if noise_power < 1e-12:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)