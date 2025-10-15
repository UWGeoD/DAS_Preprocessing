# Utilities.py (only the relevant bits shown)
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from datetime import datetime
from collections import OrderedDict
import re

epsilon = 1e-8
normalize = lambda x: (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + epsilon)

def plot_das_data(data, channels, dx, dt, start_time=None, end_time=None, title=None):
    x = np.asarray(channels) * dx
    t = np.arange(data.shape[1]) * dt
    if start_time is not None and end_time is not None:
        mask = (t >= start_time) & (t <= end_time)
        data = data[:, mask]; t = t[mask]
    fig, ax = plt.subplots(figsize=(10,5))
    ax.imshow(normalize(data).T, cmap="seismic", vmin=-1, vmax=1, aspect="auto",
              extent=[x[0], x[-1], t[-1], t[0]], interpolation="none", animated=True, zorder=1)
    ax.set_xlim(x[0], x[-1]); ax.set_ylim(t[-1], t[0])
    ax.set_xlabel("Channel Position (m)"); ax.set_ylabel("Time (s)")
    if title: ax.set_title(title)
    fig.tight_layout(pad=0.7); fig.show()
    return fig, ax

def plot_single(data, channel_num, dx, dt, start_time=None, end_time=None):
    t = np.arange(data.shape[1]) * dt
    sig = data[channel_num]
    if start_time is not None and end_time is not None:
        m = (t >= start_time) & (t <= end_time); t = t[m]; sig = sig[m]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4)); plt.plot(t, sig, color="black")
    plt.title(f"Channel {channel_num} (Position: {channel_num*dx:.2f} m)")
    plt.xlabel("Time (s)"); plt.ylabel("Amplitude"); plt.grid(True); plt.tight_layout(); plt.show()

def downsample_data(data, original_fs, target_fs):
    if target_fs >= original_fs:
        raise ValueError("Target sampling rate must be less than original sampling rate.")
    if original_fs % target_fs != 0:
        raise ValueError("Original fs must be divisible by target fs for integer decimation.")
    q = int(original_fs / target_fs)
    return signal.decimate(data, q, axis=1, zero_phase=True)

# ... (create_time_file_dict, parse_datetime_from_filename, etc., same as you already have)
