# preprocessing.py
"""
DAS preprocessing utilities.

Design goals
------------
- Clear, composable preprocessing "pipeline" API (like sklearn.Pipeline, but lightweight).
- Each step is a pure function: x -> x (2D arrays: [channels, time]).
- A single builder `make_preprocess(...)` returns a callable:
      preprocess(x, fs, **ctx) -> y
  so you can plug it directly into your Dataset / notebook workflows.

Conventions
-----------
- Input data shape: (n_channels, n_time)
- Default time axis is axis=1
- fs in Hz, dt = 1/fs (unless provided)
- dx in meters per channel (only needed for f-k filter)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import signal


Array2D = np.ndarray


# =============================================================================
# Validation helpers
# =============================================================================
def _as_2d_float(x: np.ndarray) -> Array2D:
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array [channels, time], got shape={x.shape}")
    # keep dtype unless it's not float-ish
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float32, copy=False)
    return x


def _require_positive(name: str, val: float) -> float:
    val = float(val)
    if val <= 0:
        raise ValueError(f"{name} must be > 0, got {val}")
    return val


# =============================================================================
# Core processing primitives (stateless functions)
# =============================================================================
def detrend_linear(x: Array2D, *, axis: int = 1) -> Array2D:
    """Linear detrend along `axis` (default time axis=1)."""
    x = _as_2d_float(x)
    return signal.detrend(x, type="linear", axis=axis)


def bandpass_sos(
    x: Array2D,
    *,
    fs: float,
    f_lo: float = 1.0,
    f_hi: float = 20.0,
    order: int = 5,
    axis: int = 1,
    zero_phase: bool = False,
) -> Array2D:
    """
    Butterworth bandpass via SOS.

    Parameters
    ----------
    zero_phase:
        If True, uses sosfiltfilt (zero-phase, more expensive).
        If False, uses sosfilt (causal, faster).
    """
    x = _as_2d_float(x)
    fs = _require_positive("fs", fs)
    f_lo = float(f_lo)
    f_hi = float(f_hi)
    if not (0 < f_lo < f_hi < fs / 2):
        raise ValueError(f"Require 0 < f_lo < f_hi < fs/2. Got f_lo={f_lo}, f_hi={f_hi}, fs={fs}")

    if order < 1:
        raise ValueError(f"order must be >= 1, got {order}")

    sos = signal.butter(order, [f_lo, f_hi], btype="bandpass", fs=fs, output="sos")

    if zero_phase:
        return signal.sosfiltfilt(sos, x, axis=axis)
    return signal.sosfilt(sos, x, axis=axis)


def detrend_then_bandpass(
    x: Array2D,
    *,
    fs: float,
    f_lo: float = 1.0,
    f_hi: float = 20.0,
    order: int = 5,
    axis: int = 1,
    zero_phase: bool = False,
) -> Array2D:
    """Convenience: detrend -> bandpass."""
    return bandpass_sos(
        detrend_linear(x, axis=axis),
        fs=fs,
        f_lo=f_lo,
        f_hi=f_hi,
        order=order,
        axis=axis,
        zero_phase=zero_phase,
    )


def fk_filter(
    x: Array2D,
    *,
    dx: float,
    dt: float,
    fmin: float = 0.0,
    fmax: float = None,
    vmin: float = None,
    vmax: float = None,
    taper: float = 0.0,
) -> Array2D:
    """
    Refined 2D f-k filter.
    
    Tapering strategy: "Inner Taper" (tapers INSIDE the defined bounds).
    """
    x = np.asarray(x, dtype=float)
    nch, nt = x.shape

    # 1. Use RFFT2 (Real FFT) - faster and saves memory for real input
    # Note: RFFT2 only computes positive frequencies for the time axis
    X = np.fft.rfft2(x)
    
    # Shift only the spatial axis (k), time axis (f) starts at 0 naturally in RFFT
    X = np.fft.fftshift(X, axes=0)

    # 2. Define Frequency Grids
    # kx: centered from -k_nyquist to +k_nyquist
    kx = np.fft.fftshift(np.fft.fftfreq(nch, d=dx))
    # f: positive frequencies only (0 to f_nyquist)
    f = np.fft.rfftfreq(nt, d=dt)
    
    KX, F = np.meshgrid(kx, f, indexing="ij")
    absK = np.abs(KX)
    absF = np.abs(F)

    # Avoid division by zero
    eps = 1e-12
    vapp = absF / np.maximum(absK, eps)

    # 3. Create Mask
    M = np.ones_like(X, dtype=float)

    # Helper for inner tapering
    def apply_taper(mask, values, low, high, width):
        if width <= 0:
            if low is not None: mask *= (values >= low)
            if high is not None: mask *= (values <= high)
            return mask
            
        # Taper Logic
        if low is not None:
            # Ramp from 0 at low to 1 at low*(1+w)
            # We must kill everything below low
            mask[values < low] = 0.0
            ramp_up = low * (1 + width)
            idx = (values >= low) & (values < ramp_up)
            # Hanning-like ramp
            mask[idx] *= 0.5 * (1 - np.cos(np.pi * (values[idx] - low) / (ramp_up - low)))

        if high is not None:
            # Ramp from 1 at high*(1-w) to 0 at high
            mask[values > high] = 0.0
            ramp_down = high * (1 - width)
            idx = (values <= high) & (values > ramp_down)
            mask[idx] *= 0.5 * (1 + np.cos(np.pi * (values[idx] - ramp_down) / (high - ramp_down)))
        
        return mask

    # Apply Filters
    M = apply_taper(M, absF, fmin, fmax, taper)
    
    # Velocity filter requires care with vmin/vmax
    # If vmin is set, we want to REMOVE slow stuff (v < vmin)
    # If vmax is set, we want to REMOVE fast stuff (v > vmax)
    M = apply_taper(M, vapp, vmin, vmax, taper)

    # 4. Handle DC cleanly
    # We always preserve true static DC (f=0, k=0) because it's the mean.
    # We DO NOT force the whole k=0 axis to 1 unless naturally included.
    # Note: RFFT means f=0 is at index 0 on axis 1.
    # X is shifted on axis 0, so k=0 is at index nch//2.
    
    # Ensure mean is preserved (optional, but usually good)
    center_k = nch // 2
    M[center_k, 0] = 1.0 

    # 5. Apply and Inverse
    Y = X * M
    
    # Unshift k-axis before inverse
    Y = np.fft.ifftshift(Y, axes=0)
    y = np.fft.irfft2(Y, s=x.shape) # s=x.shape ensures odd/even count is correct
    
    return y


def hilbert_transform(
    x: Array2D,
    *,
    axis: int = 1,
    mode: str = "envelope",
) -> Array2D:
    """
    Applies the Hilbert transform.

    Parameters
    ----------
    mode : {"envelope", "phase", "real"}
        "envelope" (default) returns the amplitude envelope of the signal.
        "phase" returns the instantaneous phase [-pi, pi].
        "real" returns the original signal.
    """
    x = _as_2d_float(x)
    # signal.hilbert computes the complex analytic signal
    analytic = signal.hilbert(x, axis=axis)
    
    if mode == "envelope":
        return np.abs(analytic).astype(x.dtype)
    elif mode == "phase":
        return np.angle(analytic).astype(x.dtype)
    elif mode == "real":
        return np.real(analytic).astype(x.dtype)
    else:
        raise ValueError(f"Unknown hilbert mode: '{mode}'. Use 'envelope' or 'phase'.")
        

def curvelet_like_denoise(
    x: Array2D,
    *,
    n_scales: int = 4,
    n_angles: int = 16,
    thresh: float = 3.5,
    keep_lowpass: bool = True,
    robust_mad: bool = True,
) -> Array2D:
    """
    Directional multiscale shrinkage in the frequency domain (no external deps).

    This mimics curvelet-like behavior (directional multiscale sparsity) without
    exact curvelet transforms.

    Pipeline:
      1) FFT2 of [channels,time]
      2) Split spectrum into log-radial bands + angular wedges (smooth masks)
      3) Soft-threshold complex coefficients per band/wedge
      4) Sum and iFFT
    """
    x = _as_2d_float(x)

    ny, nx = x.shape  # channels, time
    F = np.fft.fftshift(np.fft.fft2(x), axes=(0, 1))

    ky = np.fft.fftshift(np.fft.fftfreq(ny))  # normalized
    fx = np.fft.fftshift(np.fft.fftfreq(nx))  # normalized
    KY, FX = np.meshgrid(ky, fx, indexing="ij")

    R = np.sqrt(KY**2 + FX**2) + 1e-12
    TH = np.arctan2(FX, KY)  # [-pi, pi]

    n_scales = int(max(2, n_scales))
    n_angles = int(max(8, n_angles))

    # log-ish radial edges in (0, 1]
    r_edges = np.geomspace(1e-3, 1.0, num=n_scales + 1)

    def radial_mask(r: np.ndarray, lo: float, hi: float, width: float = 0.15) -> np.ndarray:
        w = np.zeros_like(r, float)
        lo1 = lo * (1 - width)
        hi1 = hi * (1 + width)

        inner = (r >= lo1) & (r < lo)
        w[inner] = 0.5 * (1 - np.cos(np.pi * (r[inner] - lo1) / (lo - lo1)))

        w[(r >= lo) & (r <= hi)] = 1.0

        outer = (r > hi) & (r <= hi1)
        w[outer] = 0.5 * (1 - np.cos(np.pi * (hi1 - r[outer]) / (hi1 - hi)))
        return w

    ang_centers = np.linspace(-np.pi, np.pi, num=n_angles, endpoint=False)

    def angular_mask(theta: np.ndarray, center: float, width: float) -> np.ndarray:
        d = (theta - center + np.pi) % (2 * np.pi) - np.pi
        w = np.zeros_like(theta, float)
        passband = np.abs(d) <= (0.5 * width)
        ramp = (np.abs(d) > (0.5 * width)) & (np.abs(d) <= width)
        w[passband] = 1.0
        w[ramp] = 0.5 * (1 + np.cos(np.pi * (np.abs(d[ramp]) - 0.5 * width) / (0.5 * width)))
        return w

    wedge_width = np.pi / n_angles
    F_sum = np.zeros_like(F, dtype=complex)
    lowpass_added = False

    for s in range(n_scales):
        lo, hi = float(r_edges[s]), float(r_edges[s + 1])
        Rmask = radial_mask(R, lo, hi)
        is_lowpass = (s == n_scales - 1)

        for ac in ang_centers:
            Amask = angular_mask(TH, float(ac), wedge_width)
            W = Rmask * Amask
            if not np.any(W):
                continue

            C = F * W  # masked complex coefficients

            # noise estimate and threshold
            if robust_mad:
                coeffs = np.abs(C[C != 0]).ravel()
                if coeffs.size == 0:
                    sigma = 0.0
                else:
                    med = float(np.median(coeffs))
                    sigma = (med / 0.6745) if med > 0 else float(np.std(coeffs))
            else:
                sigma = float(np.std(np.abs(C)))

            lam = float(thresh) * sigma

            if is_lowpass and keep_lowpass:
                C_thr = C
                lowpass_added = True
            else:
                mag = np.abs(C)
                # stable complex soft-threshold
                C_thr = np.where(mag > 0, np.maximum(0.0, mag - lam) * (C / (mag + 1e-12)), 0.0)

            F_sum += C_thr

    if not lowpass_added:
        # keep a tiny disk near DC
        F_sum += (F * (R < float(r_edges[1])))

    y = np.fft.ifft2(np.fft.ifftshift(F_sum, axes=(0, 1))).real
    return y


# =============================================================================
# Pipeline API (recommended)
# =============================================================================
@dataclass(frozen=True)
class Step:
    """A single preprocessing step."""
    name: str
    fn: Callable[..., Array2D]
    kwargs: Dict[str, Any]


def _normalize_spec(
    spec: Optional[Union[Sequence[Tuple[str, Mapping[str, Any]]], Mapping[str, Mapping[str, Any]]]]
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Accept either:
      - Ordered list: [("detrend", {...}), ("bandpass", {...}), ...]
      - Dict (py3.7+ preserves insertion order): {"detrend": {...}, "bandpass": {...}}
    Return a list of (name, kwargs) in execution order.
    """
    if spec is None:
        return []
    if isinstance(spec, Mapping):
        return [(k, dict(v)) for k, v in spec.items()]
    out: List[Tuple[str, Dict[str, Any]]] = []
    for name, kw in spec:
        out.append((name, dict(kw)))
    return out


_STEP_REGISTRY: Dict[str, Callable[..., Array2D]] = {
    "detrend": detrend_linear,
    "bandpass": bandpass_sos,
    "fk_filter": fk_filter,
    "curvelet": curvelet_like_denoise,
    "hilbert": hilbert_transform,  # <--- NEW
}


def make_preprocess(
    steps: Optional[
        Union[
            Sequence[Tuple[str, Mapping[str, Any]]],
            Mapping[str, Mapping[str, Any]],
        ]
    ] = None,
    *,
    axis: int = 1,
    dx: Optional[float] = None,
    dt: Optional[float] = None,
) -> Callable[[Array2D, float], Array2D]:
    """
    Build a preprocessing callable preprocess(x, fs, **ctx).

    Parameters
    ----------
    steps:
        Ordered pipeline specification. Recommended forms:

        1) List of tuples (explicit order):
           steps = [
             ("detrend", {"axis": 1}),
             ("bandpass", {"f_lo": 1, "f_hi": 20, "order": 5, "zero_phase": False}),
             ("fk_filter", {"fmin": 1, "fmax": 20, "vmin": 200, "vmax": 4000}),
             ("curvelet", {"n_scales": 4, "n_angles": 16, "thresh": 3.5}),
           ]

        2) Dict (insertion order preserved in Python 3.7+):
           steps = {"detrend": {...}, "bandpass": {...}, ...}

    axis:
        Default axis for steps that need it (detrend/bandpass).

    dx, dt:
        Optional defaults captured in the closure (useful for fk_filter).
        - If dt is not provided, it will be computed from fs (dt = 1/fs).
        - fk_filter requires dx and dt to be available either here or at call time.

    Returns
    -------
    preprocess(x, fs, **ctx) -> y

    Notes
    -----
    - `bandpass` step automatically receives `fs`.
    - `fk_filter` step receives `dx` and `dt` (dt derived from fs unless provided).
    - You can override dx/dt at call time: preprocess(x, fs, dx=..., dt=...)
    """
    spec = _normalize_spec(steps)

    # Build normalized step list
    pipeline: List[Step] = []
    for name, kw in spec:
        if name not in _STEP_REGISTRY:
            raise ValueError(f"Unknown step '{name}'. Valid: {sorted(_STEP_REGISTRY)}")

        # common defaults
        if name in ("detrend", "bandpass", "hilbert"):
            kw.setdefault("axis", axis)

        pipeline.append(Step(name=name, fn=_STEP_REGISTRY[name], kwargs=kw))

    def preprocess(x: Array2D, fs: float, **ctx: Any) -> Array2D:
        x2 = _as_2d_float(x)

        fs_ = _require_positive("fs", fs)
        dt_ = float(ctx.get("dt", dt if dt is not None else 1.0 / fs_))
        dx_ = ctx.get("dx", dx)

        y = x2
        for step in pipeline:
            if step.name == "detrend":
                y = step.fn(y, **step.kwargs)

            elif step.name == "bandpass":
                y = step.fn(y, fs=fs_, **step.kwargs)

            elif step.name == "fk_filter":
                if dx_ is None:
                    raise ValueError("fk_filter requires dx (meters). Provide dx in make_preprocess(dx=...) or at call time.")
                y = step.fn(y, dx=float(dx_), dt=float(dt_), **step.kwargs)

            elif step.name == "curvelet":
                y = step.fn(y, **step.kwargs)
                
            elif step.name == "hilbert":              # <--- NEW
                y = step.fn(y, **step.kwargs)         # <--- NEW

            else:
                # should never happen due to registry guard
                raise RuntimeError(f"Unhandled step: {step.name}")

        return y

    return preprocess


def load_and_preprocess(path: str, fs: float, preprocess: Optional[Callable[..., Array2D]] = None, **ctx: Any) -> Array2D:
    """
    Load a .npy sample and optionally apply preprocess.

    Parameters
    ----------
    path:
        Path to sample_XXXXX.npy (2D array: channels x time)
    fs:
        Sampling rate (Hz)
    preprocess:
        Callable returned by make_preprocess
    ctx:
        Optional context (dx, dt, etc.) forwarded to preprocess
    """
    x = np.load(path)
    x = _as_2d_float(x)
    if preprocess is not None:
        x = preprocess(x, fs, **ctx)
    return x
