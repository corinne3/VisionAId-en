"""PNG rendering of a costmap (top-down view, y up)."""

from __future__ import annotations

import cv2
import numpy as np

from visionaid.mapping.costmap import FREE, OCC


def render_costmap(
    state: np.ndarray,
    origin: np.ndarray,
    res: float,
    traj: np.ndarray | None = None,
    obj: np.ndarray | None = None,
    path: np.ndarray | None = None,
    cam: np.ndarray | None = None,
    scale: int = 3,
    cost: np.ndarray | None = None,
) -> np.ndarray:
    """BGR image: free white, occupied black, unknown grey, inflated zone pink, trajectory blue,
    path green, target orange."""
    img = np.full(state.shape + (3,), 150, np.uint8)
    img[state == FREE] = (255, 255, 255)
    if cost is not None:
        img[np.isinf(cost) & (state != OCC)] = (200, 190, 240)
    img[state == OCC] = (30, 30, 30)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    def px(xy):
        xy = np.atleast_2d(xy)
        j = (xy[:, 0] - origin[0]) / res * scale
        i = (xy[:, 1] - origin[1]) / res * scale
        return np.stack([j, i], 1).astype(np.int32)

    if traj is not None and len(traj) > 1:
        cv2.polylines(img, [px(traj[:, :2])], False, (255, 120, 0), 2)
    if path is not None and len(path) > 1:
        cv2.polylines(img, [px(path[:, :2])], False, (0, 180, 0), 2)
    if obj is not None and np.isfinite(obj).all():
        cv2.circle(img, tuple(px(obj[:2])[0]), 3 * scale, (0, 140, 255), -1)
    if cam is not None:
        cv2.circle(img, tuple(px(cam[:2])[0]), 2 * scale, (255, 0, 0), -1)
    return cv2.flip(img, 0)
