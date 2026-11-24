"""Deterministic parametric binaural spatialisation (no downloaded HRTF).

- ITD: Woodworth formula, configurable head radius (8.75 cm);
- ILD: Brown & Duda (1998) head shadow filter, one pole / one zero per ear;
- front/back: treble attenuation for sources located behind.
Azimuth positive to the left (rad), 0 = in front. Parameters vary per sample (no jump).
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

ALPHA_MIN = 0.1
THETA_MIN = math.radians(150.0)


def woodworth_itd(az: float | np.ndarray, head_radius: float = 0.0875, c: float = 343.0):
    """ITD (s), positive when the source is on the left (the left ear leads)."""
    az = np.asarray(az, float)
    lat = np.arcsin(np.clip(np.sin(az), -1.0, 1.0))  # lateral angle ∈ [−π/2, π/2]
    return head_radius / c * (np.abs(lat) + np.sin(np.abs(lat))) * np.sign(lat)


@njit(cache=True)
def _alpha(theta):
    return (1.0 + ALPHA_MIN / 2.0) + (1.0 - ALPHA_MIN / 2.0) * math.cos(theta / THETA_MIN * math.pi)


@njit(cache=True)
def _wrap_angle(x):
    return math.atan2(math.sin(x), math.cos(x))


@njit(cache=True)
def spatialize_kernel(mono, az, fs, head_radius, c, back_cutoff):
    """mono (N,), az (N,) → stereo (N,2)."""
    n = mono.shape[0]
    out = np.zeros((n, 2))
    K = 2.0 * fs
    beta = 2.0 * c / head_radius
    maxd = int(head_radius / c * (math.pi / 2 + 1.0) * fs) + 4
    # filter states
    xl1 = 0.0
    yl1 = 0.0
    xr1 = 0.0
    yr1 = 0.0
    lpl = 0.0
    lpr = 0.0
    g_lp = 1.0 - math.exp(-2.0 * math.pi * back_cutoff / fs)
    for i in range(n):
        a = az[i]
        s = math.sin(a)
        if s > 1.0:
            s = 1.0
        if s < -1.0:
            s = -1.0
        lat = math.asin(s)
        itd = head_radius / c * (abs(lat) + math.sin(abs(lat)))
        dl = 0.0
        dr = 0.0
        if lat > 0:
            dr = itd * fs
        else:
            dl = itd * fs
        # fractional delays (linear interpolation)
        xl = 0.0
        xr = 0.0
        for ch in range(2):
            d = dl if ch == 0 else dr
            p = i - d
            if p < 0:
                v = 0.0
            else:
                k = int(math.floor(p))
                f = p - k
                v0 = mono[k]
                v1 = mono[k + 1] if k + 1 <= i else mono[k]
                v = (1.0 - f) * v0 + f * v1
            if ch == 0:
                xl = v
            else:
                xr = v
        # head shadow: angle between the source and each ear (left at +90°, right at −90°)
        th_l = abs(_wrap_angle(a - math.pi / 2))
        th_r = abs(_wrap_angle(a + math.pi / 2))
        al = _alpha(th_l)
        ar = _alpha(th_r)
        b0l = (al * K + beta) / (K + beta)
        b1l = (beta - al * K) / (K + beta)
        b0r = (ar * K + beta) / (K + beta)
        b1r = (beta - ar * K) / (K + beta)
        a1 = (beta - K) / (K + beta)
        yl = b0l * xl + b1l * xl1 - a1 * yl1
        yr = b0r * xr + b1r * xr1 - a1 * yr1
        xl1 = xl
        yl1 = yl
        xr1 = xr
        yr1 = yr
        # front/back cue: low-pass mixed according to "rear-ness"
        back = -2.0 * math.cos(a)  # 0 at 90°, 1 from 120° (continuous)
        if back < 0.0:
            back = 0.0
        if back > 1.0:
            back = 1.0
        lpl += g_lp * (yl - lpl)
        lpr += g_lp * (yr - lpr)
        out[i, 0] = (1.0 - back) * yl + back * lpl
        out[i, 1] = (1.0 - back) * yr + back * lpr
    _ = maxd
    return out


def spatialize(
    mono: np.ndarray,
    az: float | np.ndarray,
    fs: int = 44100,
    head_radius: float = 0.0875,
    c: float = 343.0,
    back_cutoff: float = 2500.0,
) -> np.ndarray:
    """Spatialises a mono signal; `az` scalar or per sample (rad, left positive)."""
    mono = np.ascontiguousarray(mono, np.float64)
    azv = (
        np.full(mono.shape[0], float(az))
        if np.isscalar(az)
        else np.ascontiguousarray(az, np.float64)
    )
    return spatialize_kernel(mono, azv, float(fs), float(head_radius), float(c), float(back_cutoff))


def smooth_azimuth(az: np.ndarray, fs: int, tau: float) -> np.ndarray:
    """Exponential smoothing of a per-sample azimuth, on the circle (no jump at ±π)."""
    if tau <= 0:
        return az
    return _smooth_az(np.ascontiguousarray(az, np.float64), math.exp(-1.0 / (tau * fs)))


@njit(cache=True)
def _smooth_az(az, k):
    out = np.empty_like(az)
    cx = math.cos(az[0])
    sx = math.sin(az[0])
    for i in range(az.shape[0]):
        cx = k * cx + (1 - k) * math.cos(az[i])
        sx = k * sx + (1 - k) * math.sin(az[i])
        out[i] = math.atan2(sx, cx)
    return out
