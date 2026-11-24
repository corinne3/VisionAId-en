"""World model: floor + costmap + target anchoring, from (frame, depth, pose, detections)."""

from __future__ import annotations

from typing import Any

import numpy as np

from visionaid.bus.messages import Costmap, Depth, Detections, Frame, ObjectEstimate, Pose
from visionaid.mapping.costmap import OccupancyGrid
from visionaid.mapping.floor import FloorTracker
from visionaid.mapping.geometry import backproject, transform
from visionaid.mapping.object_anchor import ObjectAnchor


class WorldModel:
    """3D fusion. VO poses are re-expressed in the aligned world frame (z up, floor z=0)."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.floor = FloorTracker(cfg.floor)
        self.grid = OccupancyGrid(cfg.costmap)
        self.anchor = ObjectAnchor(cfg.anchor)
        # probable (unconfirmed) target: separate anchor, only published if it is consistent in 3D
        self.cand_anchor = ObjectAnchor(cfg.anchor)
        self.cand_min_meas = int(getattr(cfg.anchor, "candidate_min_meas", 4))
        self.cand_max_offset = float(getattr(cfg.anchor, "candidate_max_offset_m", 1.0))
        self.n_cand_wrong = 0
        self.t_first_candidate: float | None = None
        self.step = int(cfg.costmap.pixel_step)
        self.cam_heights: list[float] = []
        self.T_world_cam: np.ndarray | None = None
        self.pose_quality = "init"
        self._pending: list[tuple] = []  # target measurements received before floor alignment
        self._t_prev: float | None = None
        # furniture (description of where the object is); kept when the target changes
        from visionaid.mapping.landmarks import LandmarkMap

        lm = getattr(cfg, "landmarks", None)
        self.landmarks = LandmarkMap(lm) if lm is not None else None
        self.target_id = 0

    def reset_target(self, target_id: int) -> None:
        """New request: the target is forgotten, the map and the furniture are kept."""
        self.anchor = ObjectAnchor(self.cfg.anchor)
        self.cand_anchor = ObjectAnchor(self.cfg.anchor)
        self._pending = []
        self.t_first_candidate = None
        self.target_id = int(target_id)

    def T_aligned(self, T_vo_cam: np.ndarray) -> np.ndarray:
        return self.floor.T_world_vo() @ T_vo_cam

    def update(
        self,
        frame: Frame,
        depth: Depth,
        pose: Pose,
        dets: Detections | None,
        furniture: Detections | None = None,
    ) -> tuple[Costmap | None, ObjectEstimate]:
        t = frame.t
        if dets is not None and getattr(dets, "target_id", 0) != self.target_id:
            dets = None  # detections of a previous request (still in flight on the bus)
        self.pose_quality = pose.quality
        # "degraded" (IMU-only orientation, frozen position): pose published, but no
        # mapping or anchoring, since the position is wrong as soon as the user walks
        lost = pose.quality in ("lost", "degraded")
        if not lost:
            self.floor.update(depth.depth, frame.K, pose.T_world_cam, getattr(pose, "up", None))
        target = None
        if dets is not None and dets.target_idx >= 0:
            i = dets.target_idx
            mask = None
            if dets.masks is not None:
                from visionaid.perception.pipeline import unpack_masks

                mask = unpack_masks(dets, frame.image.shape[1])[i]
            conf = float(dets.reid_scores[i]) if dets.reid_scores is not None else 1.0
            target = (mask, dets.boxes[i], conf)
        cand = None
        if dets is not None and dets.target_idx < 0 and getattr(dets, "candidate_idx", -1) >= 0:
            i = dets.candidate_idx
            mask = None
            if dets.masks is not None:
                from visionaid.perception.pipeline import unpack_masks

                mask = unpack_masks(dets, frame.image.shape[1])[i]
            # reduced confidence: the measurement weighs less than a confirmed detection
            cand = (mask, dets.boxes[i], 0.5 * float(dets.reid_scores[i]))
        if not self.floor.ready:
            if target is not None and not lost:
                self._pending.append((t, pose.T_world_cam.copy(), depth.depth, frame.K, *target))
            self.anchor.predict(t)
            return None, self._estimate(t, frame.frame_idx)
        if self._pending:
            for tp, Tp, dp, Kp, m, b, c in self._pending:
                self.anchor.update(tp, self.T_aligned(Tp), dp, Kp, m, b, c)
            self._pending = []
        T = self.T_aligned(pose.T_world_cam)
        self.T_world_cam = T
        self.cam_heights.append(float(T[2, 3]))
        if not lost:
            pc, _ = backproject(
                depth.depth, frame.K, self.step, None, 0.2, float(self.cfg.costmap.max_range_m) + 1
            )
            dt = 1.0 / 30.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
            self.grid.update(transform(T, pc), T[:3, 3], dt)
            carve = float(getattr(self.cfg.costmap, "carve_radius_m", 0.0) or 0.0)
            if carve > 0:
                self.grid.carve(T[:2, 3], carve)
            self._t_prev = t
            if target is not None:
                m, b, c = target
                self.anchor.update(t, T, depth.depth, frame.K, m, b, c)
                self._check_candidate()
            else:
                self.anchor.predict(t)
            if cand is not None and self.anchor.x is None:
                m, b, c = cand
                self.cand_anchor.update(t, T, depth.depth, frame.K, m, b, c)
            if furniture is not None and self.landmarks is not None and len(furniture.boxes):
                from visionaid.perception.pipeline import unpack_masks

                fm = unpack_masks(furniture, frame.image.shape[1])
                self.landmarks.update(t, T, depth.depth, frame.K, furniture, fm)
        else:
            self.anchor.predict(t)
        return self.grid.to_msg(t, frame.frame_idx), self._estimate(t, frame.frame_idx)

    def _check_candidate(self) -> None:
        """Target confirmed: the candidate is dropped; if it was far from it, it was false."""
        if self.cand_anchor.x is not None and self.anchor.x is not None:
            if np.linalg.norm(self.cand_anchor.x[:2] - self.anchor.x[:2]) > self.cand_max_offset:
                self.n_cand_wrong += 1
            self.cand_anchor = ObjectAnchor(self.cfg.anchor)

    def _estimate(self, t: float, idx: int) -> ObjectEstimate:
        """Confirmed target if it exists; otherwise a consistent probable target ("candidate" state)."""
        est = self.anchor.estimate(t, idx)
        est.target_id = self.target_id
        if est.state != "unknown" or self.cand_anchor.n_meas < self.cand_min_meas:
            return est
        c = self.cand_anchor.estimate(t, idx)
        if c.state == "stale":
            return est
        if self.t_first_candidate is None:
            self.t_first_candidate = t
        return ObjectEstimate(
            c.t, c.frame_idx, c.position, c.cov, "candidate", c.confidence, self.target_id
        )

    # Block interface of the bus
    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        f, d, p = inputs.get("frame"), inputs.get("depth"), inputs.get("pose")
        if f is None or d is None or p is None or not (f.frame_idx == d.frame_idx == p.frame_idx):
            return None
        dets = inputs.get("detections")
        if dets is not None and dets.frame_idx != f.frame_idx:
            dets = None
        cm, obj = self.update(f, d, p, dets)
        out: dict[str, Any] = {"object": obj}
        if cm is not None:
            out["costmap"] = cm
            out["pose_world"] = Pose(
                p.t,
                p.frame_idx,
                self.T_world_cam,
                p.quality,
                p.n_inliers,
                p.reproj_err,
                p.is_keyframe,
            )
        return out


def run_mapping_video(cfg, dets_by_idx: dict | None = None):
    """Depth (cache) + VO + world model over the whole video. Returns (world, records)."""
    from visionaid.io.sources import VideoFileSource
    from visionaid.perception.depth import video_depth_estimator
    from visionaid.pose.source import VisualOdometry

    src = VideoFileSource.from_config(cfg)
    dest = video_depth_estimator(cfg)
    vo = VisualOdometry(cfg)
    wm = WorldModel(cfg)
    records = []
    while (f := src.read()) is not None:
        d = dest(f)
        p = vo.update(f, d)
        det = (dets_by_idx or {}).get(f.frame_idx)
        cm, obj = wm.update(f, d, p, det)
        records.append(
            {
                "frame_idx": f.frame_idx,
                "t": f.t,
                "pose_quality": p.quality,
                "T_world_cam": None if wm.T_world_cam is None else wm.T_world_cam.copy(),
                "object": obj,
                "target_seen": det is not None and det.target_idx >= 0,
            }
        )
    return wm, records


def export_real_map(wm: WorldModel, records: list[dict], path) -> None:
    """Saves the final costmap (simulator "real map" scenario)."""
    from pathlib import Path

    s = wm.grid.state()
    traj = [r["T_world_cam"][:3, 3] for r in records if r["T_world_cam"] is not None]
    obj = records[-1]["object"].position
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state=s,
        origin=wm.grid.origin,
        res=wm.grid.res,
        obj=obj,
        start=traj[0] if traj else np.zeros(3),
    )
