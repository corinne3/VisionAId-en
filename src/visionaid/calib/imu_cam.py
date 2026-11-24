"""IMU ↔ camera calibration from a session: gyroscope bias, time offset, rotation.

1. **Bias**: mean of the gyroscope over the still segments (stable gyro and acceleration).
2. **Visual rotation** between successive images (`VisualRotation`) → camera angular velocity.
3. **Time offset** δ: maximises the correlation between ‖ω_gyro‖ averaged over [t_i+δ, t_j+δ]
   and ‖ω_visual‖ (insensitive to the unknown rotation).
4. **Rotation** `R_cam_imu` by Kabsch on the (ω_gyro, ω_visual) pairs, and **scale
   factor** ‖ω_vis‖/‖R ω_gyro‖: ≈ 1 if the units are right, ≈ π/180 (or 180/π) otherwise.

Convention: an image timestamped t (phone clock) corresponds to the IMU instant t + δ.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

from visionaid.calib.rotation import VisualRotation, kabsch, rotation_angle_deg


@dataclass
class ImuCamCalibration:
    """Result of the IMU ↔ camera calibration."""

    R_cam_imu: np.ndarray
    gyro_bias: np.ndarray  # rad/s, IMU frame (after unit correction)
    time_offset_s: float  # δ : instant IMU = instant image + δ
    gyro_scale: float  # measured ‖ω_vis‖ / ‖R ω_gyro‖ (before correction)
    units_ok: bool
    units_fix: float  # factor to apply to the gyroscope (1, π/180 or 180/π)
    correlation: float
    n_pairs: int
    residual_deg_s: float  # median residual ‖ω_vis − R ω_gyro‖ (deg/s)
    prior_angle_deg: float  # difference from the prior rotation
    static_s: float  # still duration used for the bias
    notes: list[str] = field(default_factory=list)
    # accelerometer: prior axis convention, measured sign (Safari iOS inverts it)
    R_cam_acc: np.ndarray | None = None
    accel_sign: float = 1.0
    accel_tilt_deg: float = float("nan")  # measured "up" difference / device +y (portrait)

    def to_yaml(self, path: str | Path, extra: dict | None = None) -> None:
        d = asdict(self)
        d["R_cam_imu"] = np.asarray(self.R_cam_imu).round(6).tolist()
        Ra = self.R_cam_acc if self.R_cam_acc is not None else self.R_cam_imu
        d["R_cam_acc"] = np.asarray(Ra).round(6).tolist()
        d["accel_sign"] = float(self.accel_sign)
        d["accel_tilt_deg"] = float(self.accel_tilt_deg)
        d["gyro_bias"] = [float(x) for x in self.gyro_bias]
        for k in ("time_offset_s", "gyro_scale", "units_fix", "correlation"):
            d[k] = float(d[k])
        d["residual_deg_s"] = float(self.residual_deg_s)
        d["prior_angle_deg"] = float(self.prior_angle_deg)
        d["static_s"] = float(self.static_s)
        d["units_ok"] = bool(self.units_ok)
        d.update(extra or {})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))


def static_mask(
    t: np.ndarray,
    gyro: np.ndarray,
    accel: np.ndarray,
    win_s: float = 0.5,
    gyro_std: float = 0.03,
    acc_std: float = 0.15,
) -> np.ndarray:
    """Samples belonging to a still window (low standard deviations)."""
    n = len(t)
    mask = np.zeros(n, bool)
    if n < 5:
        return mask
    i = 0
    while i < n:
        j = int(np.searchsorted(t, t[i] + win_s))
        if j - i >= 5:
            if gyro[i:j].std(0).max() < gyro_std and accel[i:j].std(0).max() < acc_std:
                mask[i:j] = True
        i = max(j, i + 1) if j - i < 5 else i + max((j - i) // 2, 1)
    return mask


def mean_gyro(t: np.ndarray, gyro: np.ndarray, a: float, b: float) -> np.ndarray | None:
    """Time average of the gyroscope over [a, b] (trapezoidal integration by interpolation)."""
    if b <= a or a < t[0] or b > t[-1]:
        return None
    inside = (t > a) & (t < b)
    ts = np.r_[a, t[inside], b]
    g = np.stack([np.interp(ts, t, gyro[:, k]) for k in range(3)], 1)
    return np.trapezoid(g, ts, axis=0) / (b - a)


def visual_rates(images, times, K, every: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Visual angular velocities (rad/s, camera frame) between successive images.

    Returns (t_start, t_end, ω) for the successful pairs.
    """
    vr = VisualRotation()
    feats = []
    for img in images:
        feats.append(vr.features(img))
    t0s, t1s, ws = [], [], []
    for i in range(0, len(images) - every, every):
        j = i + every
        R, n = vr.estimate(feats[i], feats[j], K)
        dt = times[j] - times[i]
        if R is None or dt <= 0 or n < 15:
            continue
        from visionaid.calib.rotation import so3_log

        t0s.append(times[i])
        t1s.append(times[j])
        ws.append(-so3_log(R) / dt)
    return np.array(t0s), np.array(t1s), np.array(ws).reshape(-1, 3)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 5 or a.std() < 1e-9 or b.std() < 1e-9:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


def calibrate_imu_cam(
    imu_t: np.ndarray,
    imu_gyro: np.ndarray,
    imu_accel: np.ndarray,
    vis_t0: np.ndarray,
    vis_t1: np.ndarray,
    vis_w: np.ndarray,
    R_prior: np.ndarray | None = None,
    max_offset_s: float = 0.3,
    min_rate: float = 0.15,
) -> ImuCamCalibration:
    """Calibration from the IMU and the visual velocities (see the module docstring)."""
    notes: list[str] = []
    R_prior = np.eye(3) if R_prior is None else np.asarray(R_prior, float)
    gyro = np.asarray(imu_gyro, float)
    # 1. bias on the still segments
    sm = static_mask(imu_t, gyro, np.asarray(imu_accel, float))
    static_s = float(sm.sum() / max(len(imu_t), 1) * (imu_t[-1] - imu_t[0])) if len(imu_t) else 0.0
    if sm.sum() >= 10:
        bias = gyro[sm].mean(0)
    else:
        bias = np.zeros(3)
        notes.append("no still segment: bias assumed to be zero")
    g = gyro - bias
    vn = np.linalg.norm(vis_w, axis=1)

    # 2. time offset: correlation of the norms (coarse 5 ms then fine 0.5 ms)
    def corr_at(d: float) -> tuple[float, np.ndarray, np.ndarray]:
        gm, keep = [], []
        for k in range(len(vis_t0)):
            m = mean_gyro(imu_t, g, vis_t0[k] + d, vis_t1[k] + d)
            if m is not None:
                gm.append(m)
                keep.append(k)
        if len(keep) < 5:
            return -1.0, np.zeros((0, 3)), np.zeros(0, int)
        gm = np.array(gm)
        return _corr(np.linalg.norm(gm, axis=1), vn[keep]), gm, np.array(keep)

    grid = np.arange(-max_offset_s, max_offset_s + 1e-9, 0.005)
    scores = np.array([corr_at(d)[0] for d in grid])
    d0 = float(grid[int(np.argmax(scores))])
    fine = np.arange(d0 - 0.006, d0 + 0.006 + 1e-9, 0.0005)
    fscores = np.array([corr_at(d)[0] for d in fine])
    delta = float(fine[int(np.argmax(fscores))])
    corr, gm, keep = corr_at(delta)
    if corr < 0.5:
        notes.append(f"weak correlation ({corr:.2f}): not enough motion or blurred image")
    # 3. rotation and scale on the pairs with enough motion
    vw = vis_w[keep]
    moving = (np.linalg.norm(vw, axis=1) > min_rate) & (np.linalg.norm(gm, axis=1) > 1e-6)
    if moving.sum() < 5:
        notes.append("too little rotation to estimate R_cam_imu: prior kept")
        R = R_prior
        scale = 1.0
    else:
        a, b = gm[moving], vw[moving]
        R = kabsch(a, b)
        ra = a @ R.T
        scale = float(np.sum(ra * b) / np.sum(ra * ra))
    # 4. units
    units_fix = 1.0
    units_ok = abs(scale - 1.0) < 0.15
    if not units_ok:
        for cand in (math.pi / 180.0, 180.0 / math.pi):
            if abs(scale / cand - 1.0) < 0.15:
                units_fix = cand
                notes.append(
                    f"gyroscope scale {scale:.4f} ≈ {cand:.4f}: incorrect units "
                    "(deg/s ↔ rad/s), correction proposed"
                )
                break
        else:
            notes.append(f"unexpected gyroscope scale: {scale:.3f}")
    res = (
        np.degrees(np.median(np.linalg.norm(vw - scale * gm @ R.T, axis=1)))
        if len(vw)
        else math.nan
    )
    # 5. accelerometer: at rest, phone in portrait on the chest, up is device +y
    #    (W3C standard: the "gravity included" measurement points upward). Safari iOS inverts the
    #    sign: we measure it on the still segments. The axes follow the standard: the
    #    calibrated R_cam_imu (which absorbs the gyroscope axis permutation) does not apply to it.
    accel_sign, tilt = 1.0, math.nan
    acc = np.asarray(imu_accel, float)
    if sm.sum() >= 10:
        a_mean = acc[sm].mean(0)
        accel_sign = 1.0 if a_mean[1] >= 0 else -1.0
        up = accel_sign * a_mean / max(np.linalg.norm(a_mean), 1e-9)
        tilt = float(np.degrees(np.arccos(np.clip(up[1], -1, 1))))
        if accel_sign < 0:
            notes.append("accelerometer with inverted sign (Safari iOS): corrected (accel_sign = −1)")
        if tilt > 35:
            notes.append(f"phone tilted by {tilt:.0f}° at rest: sign to check")
    else:
        notes.append("no still segment: accelerometer sign assumed standard")
    return ImuCamCalibration(
        R_cam_imu=R,
        gyro_bias=bias * units_fix,
        time_offset_s=delta,
        gyro_scale=scale,
        units_ok=units_ok,
        units_fix=units_fix,
        correlation=float(corr),
        n_pairs=int(len(keep)),
        residual_deg_s=float(res),
        prior_angle_deg=rotation_angle_deg(R @ R_prior.T),
        static_s=static_s,
        notes=notes,
        R_cam_acc=R_prior.copy(),
        accel_sign=accel_sign,
        accel_tilt_deg=tilt,
    )
