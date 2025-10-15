# preprocessing.py
import numpy as np
from scipy import signal

# ----------------------------
# Basic time-domain filters
# ----------------------------
def detrend_linear(x, axis=1):
    """Linear detrend along time axis (default axis=1)."""
    return signal.detrend(x, type="linear", axis=axis)

def bandpass_sos(x, fs, f_lo=1.0, f_hi=20.0, order=5, axis=1):
    """Butterworth bandpass via SOS; length preserved."""
    sos = signal.butter(order, [f_lo, f_hi], btype="bandpass", fs=float(fs), output="sos")
    return signal.sosfilt(sos, x, axis=axis)

def detrend_then_bandpass(x, fs, f_lo=1.0, f_hi=20.0, order=5, axis=1):
    """Convenience: detrend -> bandpass."""
    return bandpass_sos(detrend_linear(x, axis=axis), fs, f_lo, f_hi, order, axis)

def make_preprocess(f_lo=1.0, f_hi=20.0, order=5):
    """Return a callable preprocess(data, fs) for DAS.plot(preprocess=...)."""
    def _pp(x, fs):
        return detrend_then_bandpass(x, fs, f_lo=f_lo, f_hi=f_hi, order=order, axis=1)
    return _pp


# ----------------------------
# f–k (frequency–wavenumber) filter
# ----------------------------
def fk_filter(
    data, dx, dt,
    fmin=None, fmax=None,
    vmin=None, vmax=None,
    taper=0.0,
    preserve_dc=True
):
    """
    2D f–k cone/box filter on [channels, time].
    Keeps bins satisfying:
      fmin <= |f| <= fmax  AND  vmin <= |f|/|kx| <= vmax,
    with f in Hz, kx in cycles/m. Any bound None => ignored.
    """
    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError("fk_filter expects [channels, time]")

    nch, nt = x.shape
    X = np.fft.fftshift(np.fft.fft2(x), axes=(0, 1))

    kx = np.fft.fftshift(np.fft.fftfreq(nch, d=dx))  # cycles/m
    f  = np.fft.fftshift(np.fft.fftfreq(nt,  d=dt))  # Hz
    KX, F = np.meshgrid(kx, f, indexing="ij")
    absF, absK = np.abs(F), np.abs(KX)
    eps = 1e-12

    M = np.ones_like(X, dtype=float)
    if fmin is not None: M *= (absF >= float(fmin))
    if fmax is not None: M *= (absF <= float(fmax))
    if (vmin is not None) or (vmax is not None):
        vapp = absF / np.maximum(absK, eps)
        if vmin is not None: M *= (vapp >= float(vmin))
        if vmax is not None: M *= (vapp <= float(vmax))

    if taper and taper > 0:
        def _soft(val, lo, hi, w):
            out = np.ones_like(val, float)
            if (lo is not None) and (lo > 0):
                lo2 = lo * (1 + w)
                ramp = (val > lo) & (val < lo2)
                out[val < lo] = 0.0
                out[ramp] *= 0.5 * (1 - np.cos(np.pi * (val[ramp] - lo) / (lo2 - lo)))
            if (hi is not None) and (hi > 0):
                hi2 = hi * (1 - w)
                ramp = (val < hi) & (val > hi2)
                out[val > hi] = 0.0
                out[ramp] *= 0.5 * (1 - np.cos(np.pi * (hi - val[ramp]) / (hi - hi2)))
            return out
        if (fmin is not None) or (fmax is not None):
            M *= _soft(absF, fmin, fmax, float(taper))
        if (vmin is not None) or (vmax is not None):
            vapp = absF / np.maximum(absK, eps)
            M *= _soft(vapp, vmin, vmax, float(taper))

    if preserve_dc:
        M[(absK < eps) | (absF < (1.0 / (nt * dt)))] = 1.0

    Y = np.fft.ifftshift(X * M, axes=(0, 1))
    y = np.fft.ifft2(Y).real
    return y


# --------------------------------------------------------
# Curvelet-like directional denoise (no external deps)
# --------------------------------------------------------
def curvelet_like_denoise(
    data,                      # ndarray [channels, time]
    n_scales=4,                # log-radial bands (>=2)
    n_angles=16,               # angular wedges (>=8)
    thresh=3.5,                # soft-threshold multiplier (3..5 typical)
    keep_lowpass=True,         # keep the coarsest (low-f) band unshrunk
    robust_mad=True            # estimate noise via MAD per wedge
):
    """
    Directional multiscale shrinkage in the frequency domain.
    Steps:
      1) 2D FFT of [channels, time]
      2) Split spectrum into log-radial bands and angular wedges (smooth masks)
      3) Soft-threshold complex coeffs per (scale, wedge) using robust sigma
      4) Sum shrunk coeffs and iFFT

    This mimics curvelet behavior (directional multiscale sparsity) without
    installing external packages. It’s not mathematically exact curvelets,
    but works very well for ridge-like DAS energy.
    """
    x = np.asarray(data)
    if x.ndim != 2:
        raise ValueError("curvelet_like_denoise expects [channels, time]")

    ny, nx = x.shape  # channels, time
    F = np.fft.fftshift(np.fft.fft2(x), axes=(0, 1))

    # frequency grids (normalized to [-1,1] approximately)
    ky = np.fft.fftshift(np.fft.fftfreq(ny))  # spatial freq (normalized)
    fx = np.fft.fftshift(np.fft.fftfreq(nx))  # temporal freq (normalized)
    KY, FX = np.meshgrid(ky, fx, indexing="ij")

    R = np.sqrt(KY**2 + FX**2) + 1e-12               # radius
    TH = np.arctan2(FX, KY)                          # angle, range [-pi, pi]

    # ----- radial bands (log-like spacing in (0, 1]) -----
    n_scales = int(max(2, n_scales))
    r_edges = np.geomspace(1e-3, 1.0, num=n_scales+1)   # [very small .. 1]
    # smooth raised-cosine windows per band
    def radial_mask(r, lo, hi, width=0.15):
        """Raised-cosine band between lo..hi with soft edges."""
        w = np.zeros_like(r, float)
        lo1 = lo * (1 - width)
        hi1 = hi * (1 + width)
        mid = (r >= lo1) & (r <= hi1)
        # 0 outside lo1..hi1; cosine ramps into lo..hi
        # inner ramp
        inner = (r >= lo1) & (r < lo)
        w[inner] = 0.5 * (1 - np.cos(np.pi * (r[inner] - lo1) / (lo - lo1)))
        # passband
        w[(r >= lo) & (r <= hi)] = 1.0
        # outer ramp
        outer = (r > hi) & (r <= hi1)
        w[outer] = 0.5 * (1 - np.cos(np.pi * (hi1 - r[outer]) / (hi1 - hi)))
        return w

    # ----- angular wedges (n_angles, smooth wraps) -----
    n_angles = int(max(8, n_angles))
    ang_centers = np.linspace(-np.pi, np.pi, num=n_angles, endpoint=False)

    def angular_mask(theta, center, width=np.pi/n_angles):
        """Raised-cosine wedge around 'center' with ±width support (wraps)."""
        # wrap angle diff to [-pi, pi]
        d = (theta - center + np.pi) % (2*np.pi) - np.pi
        w = np.zeros_like(theta, float)
        passband = np.abs(d) <= (0.5*width)
        ramp = (np.abs(d) > (0.5*width)) & (np.abs(d) <= width)
        w[passband] = 1.0
        w[ramp] = 0.5 * (1 + np.cos(np.pi * (np.abs(d[ramp]) - 0.5*width) / (0.5*width)))
        return w

    # accumulate shrunk spectrum
    F_sum = np.zeros_like(F, dtype=complex)

    # optional untouched coarse lowpass (biggest scale)
    lowpass_added = False

    for s in range(n_scales):
        lo, hi = r_edges[s], r_edges[s+1]
        Rmask = radial_mask(R, lo, hi)

        # the coarsest band (highest hi) acts as lowpass
        is_lowpass = (s == n_scales - 1)

        for ac in ang_centers:
            Amask = angular_mask(TH, ac)
            W = Rmask * Amask
            if not np.any(W):
                continue

            # masked coefficients
            C = F * W

            # estimate noise and threshold
            if robust_mad:
                coeffs = np.abs(C[C != 0]).ravel()
                if coeffs.size == 0:
                    sigma = 0.0
                else:
                    med = np.median(coeffs)
                    sigma = med / 0.6745 if med > 0 else np.std(coeffs)
            else:
                sigma = np.std(np.abs(C))

            lam = float(thresh) * float(sigma)

            if is_lowpass and keep_lowpass:
                C_thr = C  # keep coarse band intact
                lowpass_added = True
            else:
                mag = np.abs(C)
                ang = np.exp(1j * np.angle(C))
                C_thr = np.maximum(0.0, mag - lam) * ang

            F_sum += C_thr

    # If lowpass was not explicitly added (n_scales small), ensure DC retained
    if not lowpass_added:
        F_sum += (F * (R < r_edges[1]))  # tiny disk near DC

    y = np.fft.ifft2(np.fft.ifftshift(F_sum, axes=(0, 1))).real
    return y
