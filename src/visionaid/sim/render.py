"""Video rendering of a simulated episode (ground truth + known map + path + agent)."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from visionaid.mapping.costmap import FREE, OCC
from visionaid.sim.world import SimWorld

STATE_COLORS = {
    "SEARCHING": (200, 200, 0),
    "NAVIGATING": (0, 180, 0),
    "APPROACHING": (0, 200, 255),
    "ARRIVED": (255, 0, 255),
    "OBSTACLE_ALERT": (0, 0, 255),
    "TRACKING_LOST": (80, 80, 80),
}


def render_episode(
    world: SimWorld,
    frames: list,
    path: str | Path,
    scale: int = 4,
    fps: int = 10,
    occ_final: np.ndarray | None = None,
    captions: list[str] | None = None,
) -> Path:
    """Writes an MP4: ground truth on the left, the map built by the agent on the right.

    `captions`: one line of text per image, shown at the bottom (e.g. the sound played)."""
    H, W = world.occ.shape
    gt_occ = world.occ if occ_final is None else occ_final

    def px(xy, origin, res):
        return int((xy[0] - origin[0]) / res * scale), int((xy[1] - origin[1]) / res * scale)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = None
    for fi, (t, x, y, hd, st, pts, state, origin) in enumerate(frames):
        gt = np.full((H, W, 3), 255, np.uint8)
        gt[gt_occ] = (40, 40, 40)
        gt = cv2.resize(gt, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        # known map, cropped to the extent of the world
        km = np.full((H, W, 3), 150, np.uint8)
        oi = int(round((world.origin[1] - origin[1]) / world.res))
        oj = int(round((world.origin[0] - origin[0]) / world.res))
        sub = state[max(oi, 0) : oi + H, max(oj, 0) : oj + W]
        km[: sub.shape[0], : sub.shape[1]][sub == FREE] = (255, 255, 255)
        km[: sub.shape[0], : sub.shape[1]][sub == OCC] = (40, 40, 40)
        km = cv2.resize(km, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
        for img in (gt, km):
            o = px(world.obj, world.origin, world.res)
            cv2.circle(img, o, 2 * scale, (0, 140, 255), -1)
            if pts is not None and len(pts) > 1:
                cv2.polylines(
                    img,
                    [np.array([px(p, world.origin, world.res) for p in pts], np.int32)],
                    False,
                    (0, 180, 0),
                    2,
                )
            a = px((x, y), world.origin, world.res)
            b = px((x + 0.4 * math.cos(hd), y + 0.4 * math.sin(hd)), world.origin, world.res)
            cv2.circle(img, a, int(0.2 / world.res * scale), STATE_COLORS.get(st, (0, 0, 0)), 2)
            cv2.line(img, a, b, (255, 0, 0), 2)
        frame = np.hstack(
            [cv2.flip(gt, 0), np.full((H * scale, 6, 3), 0, np.uint8), cv2.flip(km, 0)]
        )
        cv2.putText(
            frame,
            f"{world.name} t={t:5.1f}s {st}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 200),
            2,
        )
        if captions is not None and fi < len(captions) and captions[fi]:
            frame = np.vstack([frame, np.full((34, frame.shape[1], 3), 255, np.uint8)])
            cv2.putText(
                frame,
                captions[fi],
                (8, frame.shape[0] - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (30, 30, 30),
                1,
                cv2.LINE_AA,
            )
        if vw is None:
            vw = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame.shape[1], frame.shape[0])
            )
        vw.write(frame)
    if vw is not None:
        vw.release()
    return path
