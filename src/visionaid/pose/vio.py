"""Fused pose: RGB-D visual odometry + gyroscope + gravity (+ scale re-anchoring).

Composition around `RGBDOdometry` (phase 5), without modifying it:
- **rotation prior**: before each image, the pose is predicted by rotating the previous one
  by the integrated gyroscope rotation (translation unchanged). It is used as the PnP seed, and the
  VO jump check is done relative to it;
- **consistency check**: if the VO rotation differs by more than `max_disagree_deg` from that
  of the gyroscope, the VO is judged wrong and the prediction is kept;
- **fallback**: VO lost → orientation from the gyroscope alone, position frozen, quality `degraded`,
  then `lost` after `degraded_max_s`;
- **gravity**: the world frame is aligned on gravity from initialisation (z = up) and
  each pose carries `up` (up in the world frame), used by floor tracking;
- **scale**: `correct_depth` re-anchors the depth on the declared camera height.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from visionaid.bus.messages import Depth, Frame, ImuBatch, Pose
from visionaid.calib.device import ImuCalib, load_imu_calib
from visionaid.calib.rotation import rotation_angle_deg
from visionaid.pose.imu import GravityFilter, ImuBuffer, gravity_aligned_rotation
from visionaid.pose.scale import DepthScaleCorrector
from visionaid.pose.vo import Observation, RGBDOdometry


class VisualInertialOdometry:
    """VO + IMU PoseSource. `add_imu(batch)` then `update(frame, depth)` (or `step(obs, …)`)."""

    def __init__(self, cfg, calib: ImuCalib | None = None) -> None:
        v = cfg.vio
        self.vo = RGBDOdometry(cfg.vo)
        self.calib = calib if calib is not None else load_imu_calib(cfg)
        # tolerance for the image being ahead of the last gyroscope received: IMU batches arrive
        # every 50 ms, so an image may precede its batch (rotation extended ≤ 100 ms)
        self.imu = ImuBuffer(
            self.calib, extrapolate_s=float(getattr(v, "imu_extrapolate_s", 0.05) or 0.05)
        )
        self.grav = GravityFilter(float(v.gravity_gain), float(v.gravity_tolerance))
        self.gyro_seed = bool(v.gyro_seed)
        self.max_disagree = float(v.max_disagree_deg)
        self.degraded_max = float(v.degraded_max_s)
        self.grav_init_s = float(v.gravity_init_s)
        sc = getattr(v, "scale", None)
        self.scaler = (
            DepthScaleCorrector(
                float(cfg.body.camera_height_m),
                float(sc.ema),
                tuple(sc.bounds),
                int(sc.min_points),
            )
            if sc is not None and bool(sc.enabled)
            else None
        )
        self.prev_t: float | None = None
        self.degraded_since: float | None = None
        self.n_frames = 0
        self.n_lost = 0
        self.n_degraded = 0
        self.n_disagree = 0
        self.n_vo_lost = 0
        self.n_no_imu = 0

    # ------------------------------------------------------------------ inputs
    def add_imu(self, batch: ImuBatch | None) -> None:
        self.imu.add(batch)

    @property
    def has_imu(self) -> bool:
        return len(self.imu) > 0

    def correct_depth(self, frame: Frame, depth: Depth) -> Depth:
        """Depth re-anchored in scale (if gravity is known and re-anchoring is enabled)."""
        if self.scaler is None or not self.grav.ready:
            return depth
        return self.scaler(depth, frame.K, self.grav.up)

    @property
    def depth_scale(self) -> float:
        return self.scaler.scale if self.scaler is not None else 1.0

    # ------------------------------------------------------------------ pose
    def update(self, frame: Frame, depth: Depth | None) -> Pose:
        d = depth.depth if depth is not None else np.zeros(frame.image.shape[:2], np.float32)
        return self.step(self.vo.observe(frame.image, d), frame.K, frame.t, frame.frame_idx)

    def _init_gravity(self, t: float) -> None:
        acc, _ = self.imu.accel_mean(t - self.grav_init_s, t)
        if self.grav.init(acc) and self.vo.kf is None:
            T = np.eye(4)
            T[:3, :3] = gravity_aligned_rotation(self.grav.up)
            self.vo.T_world_cam = T  # world frame: z = up from the first keyframe

    def step(self, obs: Observation, K: np.ndarray, t: float = 0.0, frame_idx: int = 0) -> Pose:
        self.n_frames += 1
        vo = self.vo
        dR = self.imu.delta_R(self.prev_t, t) if self.prev_t is not None else None
        if self.prev_t is not None and dR is None:
            self.n_no_imu += 1
        if not self.grav.ready:
            self._init_gravity(t)
        else:
            self.grav.predict(dR)
            acc, sd = self.imu.accel_mean(self.prev_t if self.prev_t is not None else t - 0.1, t)
            self.grav.correct(acc, sd)
        T_before = vo.T_world_cam.copy()
        T_pred = T_before.copy()
        if dR is not None:
            T_pred[:3, :3] = T_before[:3, :3] @ dR
            if vo.kf is not None:
                vo.T_world_cam = T_pred.copy()
                if self.gyro_seed:  # PnP seed: predicted T_cur_kf
                    vo.T_kf_cur_prev = np.linalg.inv(T_pred) @ vo.kf.T_world_kf
        p = vo.step(obs, K, t, frame_idx)
        quality = p.quality
        T = p.T_world_cam
        if quality == "ok" and dR is not None:
            R_rel = T_before[:3, :3].T @ T[:3, :3]
            if rotation_angle_deg(R_rel.T @ dR) > self.max_disagree:
                self.n_disagree += 1
                quality = "degraded_vo"
                vo.T_world_cam = T_pred.copy()
                if p.is_keyframe and vo.kf is not None:
                    vo.kf.T_world_kf = T_pred.copy()
                    if vo.keyframes:
                        vo.keyframes[-1] = (vo.keyframes[-1][0], T_pred.copy())
                T = T_pred.copy()
        if quality == "lost":
            self.n_vo_lost += 1
        if quality in ("lost", "degraded_vo"):
            if dR is not None or quality == "degraded_vo":
                self.degraded_since = t if self.degraded_since is None else self.degraded_since
                quality = "degraded" if t - self.degraded_since <= self.degraded_max else "lost"
                T = vo.T_world_cam.copy()
            else:
                quality = "lost"
        else:
            self.degraded_since = None
        if quality == "degraded":
            self.n_degraded += 1
        elif quality == "lost":
            self.n_lost += 1
        self.prev_t = t
        up = T[:3, :3] @ self.grav.up if self.grav.ready else None
        return Pose(
            t,
            frame_idx,
            T.copy(),
            quality,
            p.n_inliers,
            p.reproj_err,
            p.is_keyframe and quality == "ok",
            up,
        )

    @property
    def lost_fraction(self) -> float:
        return self.n_lost / max(self.n_frames, 1)

    def stats(self) -> dict:
        return {
            "frames": self.n_frames,
            "lost_fraction": self.lost_fraction,
            "degraded_fraction": self.n_degraded / max(self.n_frames, 1),
            "vo_lost_fraction": self.n_vo_lost / max(self.n_frames, 1),
            "disagree": self.n_disagree,
            "no_imu_frames": self.n_no_imu,
            "depth_scale": self.depth_scale,
            "imu_calib": self.calib.source,
        }

    # Block interface of the bus
    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        self.add_imu(inputs.get("imu"))
        f, d = inputs.get("frame"), inputs.get("depth")
        if f is None or d is None or d.frame_idx != f.frame_idx:
            return None
        return {"pose": self.update(f, d)}
