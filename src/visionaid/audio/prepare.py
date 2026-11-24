"""Preparation of the guidance sounds: silence trimming, fades, mono, equalised perceived loudness.

The original files (`assets/sounds/VisionAid_Sounds*.mp3`) are never modified; the
prepared versions are written to `assets/sounds/processed/` with a `levels.json` report.

Perceived loudness: maximum BS.1770 loudness (K-weighting) over a 400 ms sliding window
("momentary loudness"), brought to `target_lufs`; peak limited to `peak_dbfs`.

    uv run python -m visionaid.audio.prepare
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from visionaid.utils.config import resolve_path

# guidance sounds (VisionAid_Sounds, "Mellow Bells") → prepared files
# - mono: converted to mono (the stereo pan is added at rendering);
# - segment: (start, end) in s kept from the original file (files 2 and 4 repeat the same
#   pattern twice: only one is kept, replayed at the desired rhythm);
# - loop: rendered seamlessly loopable (crossfade end → start), without end fades.
SOUNDS = {
    "chemin.wav": {"src": "VisionAid_Sounds 2-Mellow Bells.mp3", "mono": True, "segment": (0, 0.8)},
    "recherche.wav": {"src": "VisionAid_Sounds 4.mp3", "mono": True, "segment": (0, 1.2)},
    "arrivee.wav": {
        "src": "VisionAid_Sounds 3-Mellow Bells.mp3",
        "mono": True,
        "segment": (0, 1.0),
    },
    "obstacles.wav": {"src": "VisionAid_Sounds 1-Mellow Bells.mp3", "mono": False, "loop": True},
}
LOOP_XFADE_S = 0.08


def make_loop(x: np.ndarray, fs: int, xfade_s: float = LOOP_XFADE_S) -> np.ndarray:
    """Seamless loop: the end (fade-out) is mixed into the start (fade-in).

    The result, repeated end to end, is continuous (no click at the junction)."""
    n = min(int(xfade_s * fs), len(x) // 4)
    if n <= 0:
        return x
    w = (0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)).reshape(-1, *([1] * (x.ndim - 1)))
    y = x[: len(x) - n].copy()
    y[:n] = x[:n] * w + x[len(x) - n :] * (1 - w)
    return y


def _biquad_shelf(fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1 of K-weighting (+4 dB high shelf), BS.1770, for any sample rate."""
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    A = 10 ** (G / 40)
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2 * Q)
    c = math.cos(w0)
    b = np.array(
        [
            A * ((A + 1) + (A - 1) * c + 2 * math.sqrt(A) * alpha),
            -2 * A * ((A - 1) + (A + 1) * c),
            A * ((A + 1) + (A - 1) * c - 2 * math.sqrt(A) * alpha),
        ]
    )
    a = np.array(
        [
            (A + 1) - (A - 1) * c + 2 * math.sqrt(A) * alpha,
            2 * ((A - 1) - (A + 1) * c),
            (A + 1) - (A - 1) * c - 2 * math.sqrt(A) * alpha,
        ]
    )
    return b / a[0], a / a[0]


def _biquad_highpass(fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Stage 2 of K-weighting (~38 Hz high-pass)."""
    Q, fc = 0.5003270373238773, 38.13547087602444
    w0 = 2 * math.pi * fc / fs
    alpha = math.sin(w0) / (2 * Q)
    c = math.cos(w0)
    b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    a = np.array([1 + alpha, -2 * c, 1 - alpha])
    return b / a[0], a / a[0]


def momentary_loudness(x: np.ndarray, fs: int, window_s: float = 0.4) -> float:
    """Maximum momentary loudness (LUFS); `x` (N,) or (N, C)."""
    from scipy.signal import lfilter

    x = x.reshape(len(x), -1)
    b1, a1 = _biquad_shelf(fs)
    b2, a2 = _biquad_highpass(fs)
    y = lfilter(b2, a2, lfilter(b1, a1, x, axis=0), axis=0)
    n = max(1, int(window_s * fs))
    p = np.sum(y**2, axis=1)  # summed channels (weight 1, BS.1770)
    if len(p) < n:
        p = np.pad(p, (0, n - len(p)))
    c = np.concatenate([[0.0], np.cumsum(p)])
    ms = (c[n:] - c[:-n]) / n  # moving average (cumulative sum: O(N))
    return float(-0.691 + 10 * math.log10(max(ms.max(), 1e-12)))


def trim(x: np.ndarray, fs: int, thresh_db: float = -50.0, pre_s: float = 0.005) -> np.ndarray:
    """Removes the silence at the start and the end (threshold relative to the peak of the 10 ms envelope)."""
    m = np.abs(x.reshape(len(x), -1)).max(axis=1)
    n = max(1, int(0.01 * fs))
    env = np.sqrt(np.convolve(m**2, np.ones(n) / n, mode="same"))
    on = np.nonzero(env > env.max() * 10 ** (thresh_db / 20))[0]
    if len(on) == 0:
        return x
    a = max(0, on[0] - int(pre_s * fs))
    b = min(len(x), on[-1] + n)
    return x[a:b]


def fades(x: np.ndarray, fs: int, fade_in_s: float = 0.004, fade_out_s: float = 0.03):
    """Raised-cosine fades (no click at the start or the end)."""
    x = x.copy()
    for n, rev in ((int(fade_in_s * fs), False), (int(fade_out_s * fs), True)):
        n = min(n, len(x) // 2)
        if n <= 0:
            continue
        w = 0.5 - 0.5 * np.cos(np.pi * np.arange(n) / n)
        if rev:
            x[-n:] *= w[::-1].reshape(-1, *([1] * (x.ndim - 1)))
        else:
            x[:n] *= w.reshape(-1, *([1] * (x.ndim - 1)))
    return x


def prepare_one(
    x: np.ndarray,
    fs: int,
    mono: bool,
    target_lufs: float,
    peak_dbfs: float,
    loop: bool = False,
    fade_out_s: float = 0.03,
) -> tuple[np.ndarray, dict]:
    x = np.asarray(x, np.float64)
    if x.ndim == 1:
        x = x[:, None]
    before = {"lufs_m": momentary_loudness(x, fs), "dur_s": len(x) / fs}
    if mono:
        x = x.mean(axis=1, keepdims=True)
    x = make_loop(trim(x, fs, pre_s=0.0), fs) if loop else fades(trim(x, fs), fs, 0.004, fade_out_s)
    gain_db = target_lufs - momentary_loudness(x, fs)
    x = x * 10 ** (gain_db / 20)
    peak = float(np.abs(x).max())
    lim = 10 ** (peak_dbfs / 20)
    if peak > lim:  # peak too high: lower it (the sound will be slightly quieter than the target)
        x *= lim / peak
    after = {
        "lufs_m": momentary_loudness(x, fs),
        "peak_dbfs": float(20 * math.log10(np.abs(x).max() + 1e-12)),
        "dur_s": len(x) / fs,
        "channels": int(x.shape[1]),
    }
    return (x[:, 0] if mono else x), {"before": before, "after": after, "gain_db": gain_db}


def main(argv=None) -> dict:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="assets/sounds")
    ap.add_argument("--dst", default="assets/sounds/processed")
    ap.add_argument("--target-lufs", type=float, default=-20.0)
    ap.add_argument("--peak-dbfs", type=float, default=-1.0)
    a = ap.parse_args(argv)
    src, dst = resolve_path(a.src), resolve_path(a.dst)
    dst.mkdir(parents=True, exist_ok=True)
    report = {"target_lufs": a.target_lufs, "peak_dbfs": a.peak_dbfs, "sounds": {}}
    for name, o in SOUNDS.items():
        srcp = Path(src) / o["src"]
        if not srcp.exists():
            print(f"{name:16s} absent : {srcp}")
            continue
        x, fs = sf.read(str(srcp), always_2d=True)
        if "segment" in o:
            a0, a1 = o["segment"]
            x = x[int(a0 * fs) : int(a1 * fs)]
        y, info = prepare_one(
            x, fs, o["mono"], a.target_lufs, a.peak_dbfs, o.get("loop", False),
            0.2 if "segment" in o else 0.03,
        )  # fmt: skip
        sf.write(str(dst / name), y, fs, subtype="PCM_16")
        info["src"] = o["src"]
        report["sounds"][name] = info
        print(
            f"{name:16s} {info['before']['lufs_m']:6.1f} → {info['after']['lufs_m']:6.1f} LUFS  "
            f"peak {info['after']['peak_dbfs']:5.1f} dBFS  {info['after']['dur_s']:.2f} s"
        )
    (dst / "levels.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
