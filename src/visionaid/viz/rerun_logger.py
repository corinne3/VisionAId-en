"""Rerun logging of all streams (timeline = video time) to a .rrd file."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visionaid.bus.messages import (
    Costmap,
    Depth,
    Detections,
    Frame,
    GuidanceCue,
    ObjectEstimate,
    Pose,
)
from visionaid.bus.messages import Path as PathMsg
from visionaid.mapping.costmap import FREE, OCC
from visionaid.mapping.geometry import backproject, transform


class RerunLogger:
    """Records images, depth, point cloud, trajectory, costmap, path and target."""

    def __init__(
        self,
        path: str | Path,
        enabled: bool = True,
        image_every: int = 2,
        cloud_every: int = 10,
        cloud_step: int = 10,
        depth_every: int = 6,
        depth_down: int = 4,
    ) -> None:
        self.depth_every = depth_every
        self.depth_down = depth_down
        self.enabled = enabled
        self.image_every = image_every
        self.cloud_every = cloud_every
        self.cloud_step = cloud_step
        self.traj: list[np.ndarray] = []
        if not enabled:
            return
        import rerun as rr

        self.rr = rr
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rr.init("visionaid", spawn=False)
        rr.save(str(self.path))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    def log(
        self,
        frame: Frame,
        depth: Depth | None,
        pose_world: Pose | None,
        dets: Detections | None,
        cm: Costmap | None,
        obj: ObjectEstimate | None,
        path: PathMsg | None,
        cue: GuidanceCue | None,
    ) -> None:
        if not self.enabled:
            return
        rr = self.rr
        rr.set_time("video", duration=float(frame.t))
        rr.set_time("frame", sequence=int(frame.frame_idx))
        k = frame.frame_idx
        h, w = frame.image.shape[:2]
        if pose_world is not None:
            T = pose_world.T_world_cam
            rr.log("world/camera", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
            rr.log(
                "world/camera/image",
                rr.Pinhole(
                    image_from_camera=frame.K, width=w, height=h, camera_xyz=rr.ViewCoordinates.RDF
                ),
            )
            self.traj.append(T[:3, 3].copy())
            if len(self.traj) > 1:
                rr.log(
                    "world/trajectory",
                    rr.LineStrips3D([np.array(self.traj)], colors=[[255, 120, 0]]),
                )
            if depth is not None and k % self.cloud_every == 0:
                pc, uv = backproject(depth.depth, frame.K, self.cloud_step, None, 0.2, 6.0)
                cols = frame.image[uv[:, 1].astype(int), uv[:, 0].astype(int), ::-1]
                rr.log("world/points", rr.Points3D(transform(T, pc), colors=cols, radii=0.01))
        if k % self.image_every == 0:
            rr.log(
                "world/camera/image/rgb",
                rr.Image(frame.image[:, :, ::-1]).compress(jpeg_quality=80),
            )
        if depth is not None and k % self.depth_every == 0:
            dd = (
                depth.depth[:: self.depth_down, :: self.depth_down]
                .astype(np.float16)
                .astype(np.float32)
            )
            rr.log("depth", rr.DepthImage(dd, meter=1.0))
        if dets is not None and len(dets.boxes):
            labels = []
            colors = []
            for i in range(len(dets.boxes)):
                sc = float(dets.reid_scores[i]) if dets.reid_scores is not None else 0.0
                labels.append(
                    f"{dets.labels[i]} reid={sc:.2f}" + (" CIBLE" if i == dets.target_idx else "")
                )
                colors.append([255, 140, 0] if i == dets.target_idx else [160, 160, 160])
            rr.log(
                "world/camera/image/detections",
                rr.Boxes2D(
                    array=dets.boxes, array_format=rr.Box2DFormat.XYXY, labels=labels, colors=colors
                ),
            )
        if cm is not None and k % 10 == 0:
            img = np.full(cm.state.shape + (3,), 128, np.uint8)
            img[cm.state == FREE] = 255
            img[cm.state == OCC] = 0
            rr.log("costmap", rr.Image(img[::-1]))
        if path is not None and path.valid and len(path.points) > 1:
            p3 = np.c_[path.points, np.full(len(path.points), 0.02)]
            rr.log("world/path", rr.LineStrips3D([p3], colors=[[0, 200, 0]]))
            rr.log(
                "world/lookahead",
                rr.Points3D([[*path.lookahead, 0.05]], colors=[[0, 255, 0]], radii=0.06),
            )
        if obj is not None and np.isfinite(obj.position).all():
            rr.log(
                "world/target",
                rr.Points3D([obj.position], colors=[[255, 140, 0]], radii=0.08, labels=[obj.state]),
            )
        if cue is not None:
            rr.log("guidance/azimuth_deg", rr.Scalars(np.degrees(cue.azimuth)))
            if np.isfinite(cue.distance):
                rr.log("guidance/distance_m", rr.Scalars(cue.distance))
            if cue.message:
                rr.log("guidance/messages", rr.TextLog(f"[{cue.state}] {cue.message}"))

    def close(self) -> None:
        if self.enabled:
            self.rr.disconnect()
