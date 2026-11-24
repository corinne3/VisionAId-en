"""Rendering of the annotated video: detections, target, state, heading arrow, mini-costmap."""

from __future__ import annotations

import math

import cv2
import numpy as np

from visionaid.bus.messages import Costmap, Detections, GuidanceCue, ObjectEstimate, Path, Pose
from visionaid.mapping.costmap import FREE, OCC
from visionaid.sim.render import STATE_COLORS


def minimap(
    cm: Costmap | None,
    pose: Pose | None,
    obj: ObjectEstimate | None,
    path: Path | None,
    size: int,
    span_m: float = 6.0,
) -> np.ndarray:
    """Costmap inset centred on the user, heading towards the top of the image."""
    out = np.full((size, size, 3), 90, np.uint8)
    if cm is None or pose is None:
        cv2.putText(
            out, "carte...", (10, size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1
        )
        return out
    T = pose.T_world_cam
    c = T[:2, 3]
    f = T[:2, 2] / max(np.linalg.norm(T[:2, 2]), 1e-6)  # cap
    left = np.array([-f[1], f[0]])
    s = size / span_m  # px / m
    # sampling of the costmap in the user frame
    u, v = np.meshgrid(np.arange(size), np.arange(size))
    fwd_m = (size / 2 - v) / s
    lat_m = (size / 2 - u) / s
    X = c[0] + fwd_m * f[0] + lat_m * left[0]
    Y = c[1] + fwd_m * f[1] + lat_m * left[1]
    jj = np.floor((X - cm.origin[0]) / cm.resolution).astype(int)
    ii = np.floor((Y - cm.origin[1]) / cm.resolution).astype(int)
    ok = (ii >= 0) & (ii < cm.state.shape[0]) & (jj >= 0) & (jj < cm.state.shape[1])
    st = np.full((size, size), -1, np.int8)
    st[ok] = cm.state[ii[ok], jj[ok]]
    out[st == FREE] = (235, 235, 235)
    out[st == OCC] = (20, 20, 20)

    def to_px(xy):
        d = np.asarray(xy, float)[..., :2] - c
        fw = d @ f
        lt = d @ left
        return np.stack([size / 2 - lt * s, size / 2 - fw * s], -1).astype(np.int32)

    if path is not None and path.valid and len(path.points) > 1:
        cv2.polylines(out, [to_px(path.points)], False, (0, 170, 0), 2)
        if np.isfinite(path.lookahead).all():
            cv2.circle(out, tuple(to_px(path.lookahead)), 4, (0, 255, 0), -1)
    if obj is not None and np.isfinite(obj.position).all():
        cv2.circle(out, tuple(to_px(obj.position)), 6, (0, 140, 255), -1)
    cv2.circle(out, (size // 2, size // 2), 5, (255, 80, 0), -1)
    cv2.line(out, (size // 2, size // 2), (size // 2, size // 2 - 14), (255, 80, 0), 2)
    cv2.rectangle(out, (0, 0), (size - 1, size - 1), (255, 255, 255), 1)
    return out


def draw_overlay(
    image: np.ndarray,
    dets: Detections | None,
    masks: np.ndarray | None,
    cue: GuidanceCue | None,
    cm: Costmap | None,
    pose: Pose | None,
    obj: ObjectEstimate | None,
    path: Path | None,
    minimap_px: int = 200,
    arrow_len: int = 80,
) -> np.ndarray:
    out = image.copy()
    h, w = out.shape[:2]
    if dets is not None:
        for i, b in enumerate(dets.boxes):
            x0, y0, x1, y1 = b.astype(int)
            if i == dets.target_idx:
                if masks is not None:
                    m = masks[i]
                    out[m] = (0.5 * out[m] + 0.5 * np.array([0, 140, 255])).astype(np.uint8)
                cv2.rectangle(out, (x0, y0), (x1, y1), (0, 140, 255), 3)
                sc = float(dets.reid_scores[i]) if dets.reid_scores is not None else 0
                cv2.putText(
                    out,
                    f"gourde {sc:.2f}",
                    (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 140, 255),
                    2,
                )
            else:
                cv2.rectangle(out, (x0, y0), (x1, y1), (150, 150, 150), 1)
    # state banner
    if cue is not None:
        col = STATE_COLORS.get(cue.state, (255, 255, 255))
        cv2.rectangle(out, (0, 0), (w, 44), (0, 0, 0), -1)
        dist = f"{cue.distance:.2f} m" if math.isfinite(cue.distance) else "-"
        cv2.putText(out, f"{cue.state}  {dist}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
        # heading arrow (top-down view: top = ahead, left = positive azimuth)
        cx, cy = w // 2, h - 60
        a = cue.azimuth
        tip = (int(cx - arrow_len * math.sin(a)), int(cy - arrow_len * math.cos(a)))
        cv2.circle(out, (cx, cy), arrow_len + 6, (0, 0, 0), 2)
        cv2.arrowedLine(out, (cx, cy), tip, col, 5, tipLength=0.3)
        if cue.message:
            cv2.putText(
                out, cue.message, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
    mm = minimap(cm, pose, obj, path, minimap_px)
    out[52 : 52 + minimap_px, w - minimap_px - 8 : w - 8] = mm
    return out
