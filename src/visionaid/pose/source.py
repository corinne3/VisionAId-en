"""Pose sources: visual odometry (implemented) and ARKit (stub)."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from visionaid.bus.messages import Depth, Frame, Pose
from visionaid.pose.vo import RGBDOdometry


@runtime_checkable
class PoseSource(Protocol):
    """Contract: (frame, depth) → timestamped Pose (T_world_cam, quality)."""

    def update(self, frame: Frame, depth: Depth | None) -> Pose: ...


class VisualOdometry:
    """PoseSource by RGB-D odometry (monocular metric depth as the D channel)."""

    def __init__(self, cfg) -> None:
        self.vo = RGBDOdometry(cfg.vo)

    def update(self, frame: Frame, depth: Depth | None) -> Pose:
        d = depth.depth if depth is not None else np.zeros(frame.image.shape[:2], np.float32)
        obs = self.vo.observe(frame.image, d)
        return self.vo.step(obs, frame.K, frame.t, frame.frame_idx)

    @property
    def lost_fraction(self) -> float:
        return self.vo.n_lost / max(self.vo.n_frames, 1)

    # Block interface of the bus
    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        f, d = inputs.get("frame"), inputs.get("depth")
        if f is None or d is None or d.frame_idx != f.frame_idx:
            return None
        return {"pose": self.update(f, d)}


class ArkitPoseSource:
    """Stub: 6DoF pose provided by ARKit on the phone (to come).

    Expected contract: for each frame, `T_world_cam` expressed in the ARKit world frame
    (y up in ARKit → to convert to z up, see docs/frames.md),
    timestamped with the same clock as the images (`t`), `quality` derived from
    `ARCamera.trackingState` (normal → ok, limited → ok with n_inliers=0, notAvailable → lost).
    """

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("ArkitPoseSource is not implemented yet")

    def update(self, frame: Frame, depth: Depth | None) -> Pose:  # pragma: no cover
        raise NotImplementedError


def make_pose_source(cfg, imu_available: bool = False):
    """PoseSource according to `pose.source`: `vio` if the IMU is available in `auto` mode, otherwise `vo`."""
    pc = getattr(cfg, "pose", None)
    mode = str(getattr(pc, "source", "auto") or "auto")
    if mode == "vio" or (mode == "auto" and imu_available):
        from visionaid.pose.vio import VisualInertialOdometry

        return VisualInertialOdometry(cfg)
    return VisualOdometry(cfg)


def merge_imu(x):
    """IMU input from the bus: a batch, a list of (accumulated) batches or None → a batch or None."""
    if x is None:
        return None
    if isinstance(x, list):
        from visionaid.phone.block import merge_batches

        return merge_batches(x)
    return x


def run_vo_video(cfg, frames=None, depth_est=None) -> tuple[list[Pose], VisualOdometry]:
    """Odometry over a sequence of frames (by default the whole video), depth cached."""
    from visionaid.io.sources import VideoFileSource
    from visionaid.perception.depth import video_depth_estimator

    depth_est = depth_est or video_depth_estimator(cfg)
    if frames is None:
        frames = VideoFileSource.from_config(cfg)
    vo = VisualOdometry(cfg)
    poses = [vo.update(f, depth_est(f)) for f in frames]
    return poses, vo
