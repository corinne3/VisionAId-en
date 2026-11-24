"""Deterministic single-process pipeline (offline mode), same blocks as the real-time graph."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from visionaid.bus.messages import Detections, Frame, GuidanceCue, Pose
from visionaid.guidance.cues import GuidanceEngine
from visionaid.mapping.world_model import WorldModel
from visionaid.planning.planner import Planner
from visionaid.pose.source import make_pose_source
from visionaid.utils.timing import LatencyStats, Timer


@dataclass
class FrameRecord:
    """Summary of a processed frame (for the report and the evaluation)."""

    frame_idx: int
    t: float
    state: str
    azimuth: float
    distance: float
    target_seen: bool
    pose_quality: str
    cam_xyz: np.ndarray | None
    obj_xyz: np.ndarray | None
    path_valid: bool
    head_yaw: float = float("nan")  # horizontal heading of the head (world, rad)


@dataclass
class OfflinePipeline:
    """Chains perception → depth → pose → map → planning (5 Hz) → guidance."""

    cfg: object
    perception: object  # callable(Frame) -> Detections, or None if detections are provided
    depth: object  # DepthEstimator
    stats: LatencyStats = field(default_factory=LatencyStats)
    imu_source: object = None  # SessionImu (phone session) or None

    def __post_init__(self) -> None:
        self.vo = make_pose_source(
            self.cfg, self.imu_source is not None and len(self.imu_source) > 0
        )
        self._imu_fed: float | None = None
        self.world = WorldModel(self.cfg)
        self.planner = Planner(self.cfg)
        self.engine = GuidanceEngine(self.cfg)
        self.plan_period = 1.0 / float(self.cfg.planning.replan_hz)
        self._t_plan = -math.inf
        self.path = None
        self.cues: list[GuidanceCue] = []
        self.records: list[FrameRecord] = []
        self._t_landmarks = -math.inf

    def step(self, frame: Frame, dets: Detections | None = None) -> dict:
        st = self.stats
        with Timer(st, "perception"):
            if dets is None and self.perception is not None:
                dets = self.perception(frame)
        with Timer(st, "depth"):
            depth = self.depth(frame)
        with Timer(st, "pose"):
            if self.imu_source is not None and hasattr(self.vo, "add_imu"):
                # IMU up to a little after the image (the offset δ may be positive)
                t_end = frame.t + 0.3
                self.vo.add_imu(self.imu_source.window(self._imu_fed, t_end))
                self._imu_fed = t_end
            if hasattr(self.vo, "correct_depth"):
                depth = self.vo.correct_depth(frame, depth)
            pose = self.vo.update(frame, depth)
        # furniture detected with the target (same YOLOE pass; stored on the detections)
        furniture = getattr(dets, "_furniture", None) if dets is not None else None
        with Timer(st, "mapping"):
            cm, obj = self.world.update(frame, depth, pose, dets, furniture)
        lm = self.world.landmarks
        if lm is not None and frame.t - self._t_landmarks >= 0.5:
            self._t_landmarks = frame.t
            self.engine.landmarks = lm.to_msg(frame.t)
        pose_w = None
        if self.world.T_world_cam is not None:
            pose_w = Pose(
                pose.t,
                pose.frame_idx,
                self.world.T_world_cam.copy(),
                pose.quality,
                pose.n_inliers,
                pose.reproj_err,
                pose.is_keyframe,
            )
        with Timer(st, "planning"):
            if (
                cm is not None
                and pose_w is not None
                and obj.state != "unknown"
                and frame.t - self._t_plan >= self.plan_period - 1e-9
            ):
                self.path = self.planner.plan(cm, pose_w.T_world_cam[:2, 3], obj)
                self._t_plan = frame.t
        with Timer(st, "guidance"):
            cue = self.engine.update(
                frame.t, frame.frame_idx, pose_w if pose_w is not None else pose, obj, self.path, cm
            )
        self.cues.append(cue)
        self.records.append(
            FrameRecord(
                frame.frame_idx,
                frame.t,
                cue.state,
                cue.azimuth,
                cue.distance,
                dets is not None and dets.target_idx >= 0,
                pose.quality,
                None if pose_w is None else pose_w.T_world_cam[:3, 3].copy(),
                obj.position.copy() if np.isfinite(obj.position).all() else None,
                self.path is not None and self.path.valid,
                head_yaw(pose_w.T_world_cam) if pose_w is not None else float("nan"),
            )
        )
        return {
            "frame": frame,
            "dets": dets,
            "depth": depth,
            "pose": pose,
            "pose_world": pose_w,
            "costmap": cm,
            "object": obj,
            "path": self.path,
            "cue": cue,
        }


def guidance_coherence(records: list[FrameRecord], fps: float, horizon_s: float = 1.0) -> dict:
    """Angle between the guidance azimuth and the real direction of movement in the next
    second, over the segments where the camera gets closer to the bottle (informative)."""
    k = int(round(horizon_s * fps))
    angles = []
    for i in range(len(records) - k):
        a, b = records[i], records[i + k]
        if (
            a.state not in ("NAVIGATING", "APPROACHING")
            or a.cam_xyz is None
            or b.cam_xyz is None
            or a.obj_xyz is None
        ):
            continue
        move = b.cam_xyz[:2] - a.cam_xyz[:2]
        if np.linalg.norm(move) < 0.15:  # quasi immobile
            continue
        d0 = np.linalg.norm(a.obj_xyz[:2] - a.cam_xyz[:2])
        d1 = np.linalg.norm(a.obj_xyz[:2] - b.cam_xyz[:2])
        if d1 >= d0:  # not getting closer
            continue
        if not math.isfinite(a.head_yaw):
            continue
        guide = a.head_yaw + a.azimuth  # guidance direction in the world
        az_move = math.atan2(move[1], move[0])
        diff = math.atan2(math.sin(az_move - guide), math.cos(az_move - guide))
        angles.append(abs(math.degrees(diff)))
    if not angles:
        return {"n": 0, "median_deg": None}
    return {
        "n": len(angles),
        "median_deg": float(np.median(angles)),
        "p75_deg": float(np.percentile(angles, 75)),
    }


def head_yaw(T_world_cam: np.ndarray) -> float:
    f = T_world_cam[:2, 2]
    return math.atan2(f[1], f[0])


def now() -> float:
    return time.perf_counter()
