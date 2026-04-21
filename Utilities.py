# Utilities.py (only the relevant bits shown)
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
from collections import OrderedDict
import re

epsilon = 1e-8
normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + epsilon)

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


def compute_snr(window, signal_rects, win_start_frame, fps_video, fs_das):
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

    How it works
    ------------
    1. Each [frame_start, frame_end] rect is converted to within-window DAS sample
       indices: i = (frame - win_start_frame) / fps_video * fs_das.
    2. A boolean mask over the time axis is built by OR-ing all rect ranges.
       Overlapping rects are handled automatically — e.g. [1,3] and [2,4]
       both set their index ranges to True, giving a union of [1,4] with no
       sample counted twice.
    3. Signal region  = window[:, signal_mask]  (all channels, signal time)
       Noise region   = window[:, ~signal_mask] (all channels, remaining time)
    4. SNR = 10 * log10( mean(signal^2) / mean(noise^2) )
       Mean is taken over all elements (channels × time) in each region.
    """
    if not signal_rects:
        return float("nan")

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

    signal_power = np.mean(window[:, signal_mask] ** 2)
    noise_power = np.mean(window[:, noise_mask] ** 2)

    if noise_power < 1e-12:
        return float("inf")

    return 10 * np.log10(signal_power / noise_power)