"""Checkerboard camera calibration (OpenCV), from the calibration views of a session."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class CameraCalibration:
    """Result: K, distortion, image size, errors."""

    K: np.ndarray
    dist: np.ndarray
    width: int
    height: int
    rms_px: float
    n_views: int
    per_view_px: list[float] = field(default_factory=list)
    rejected: dict[str, str] = field(default_factory=dict)

    @property
    def hfov_long_side_deg(self) -> float:
        """Field of view along the long side (comparable to `intrinsics.hfov_deg`)."""
        long_side = max(self.width, self.height)
        f = self.K[0, 0] if self.width >= self.height else self.K[1, 1]
        return math.degrees(2 * math.atan(long_side / 2 / f))

    def to_yaml(self, path: str | Path, extra: dict | None = None) -> None:
        d = {
            "K": self.K.tolist(),
            "dist": self.dist.ravel().tolist(),
            "width": self.width,
            "height": self.height,
            "rms_px": float(self.rms_px),
            "n_views": self.n_views,
            "hfov_long_side_deg": round(self.hfov_long_side_deg, 2),
        }
        d.update(extra or {})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))


def square_size_from_notes(notes: str) -> float | None:
    """Reads « case = 22 mm », « case =2.2cm » or « case: 0,022 m » in the notes; returns
    metres. Without a unit, the value is read in mm if > 5, otherwise in cm."""
    m = re.search(
        r"case\s*[=:]?\s*([0-9]+(?:[.,][0-9]+)?)\s*(mm|cm|m)?\b", notes or "", re.IGNORECASE
    )
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    unit = (m.group(2) or ("mm" if v > 5 else "cm")).lower()
    return v * {"mm": 1e-3, "cm": 1e-2, "m": 1.0}[unit]


def sharpness(gray: np.ndarray) -> float:
    """Sharpness (variance of the Laplacian)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def find_corners(gray: np.ndarray, pattern: tuple[int, int]) -> np.ndarray | None:
    """Inner corners of the checkerboard (sub-pixel), or None."""
    flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
    ok, corners = cv2.findChessboardCornersSB(gray, pattern, flags=flags)
    if not ok:
        ok, corners = cv2.findChessboardCorners(gray, pattern, None)
        if not ok:
            return None
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
        )
    return corners.reshape(-1, 2).astype(np.float32)


def calibrate_images(
    images: list[np.ndarray] | list[Path],
    pattern: tuple[int, int] = (9, 6),
    square_m: float = 0.025,
    min_views: int = 6,
    min_sharpness: float = 30.0,
    reject_factor: float = 2.5,
) -> CameraCalibration:
    """Calibrates K and the distortion; rejects blurred views, views without checkerboard or outliers."""
    obj = np.zeros((pattern[0] * pattern[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2) * square_m
    pts, names, rejected = [], [], {}
    size = None
    for k, im in enumerate(images):
        name = str(im) if isinstance(im, Path) else f"vue_{k}"
        img = cv2.imread(str(im)) if isinstance(im, Path) else im
        if img is None:
            rejected[name] = "illisible"
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        if size is None:
            size = gray.shape[::-1]
        if gray.shape[::-1] != size:
            rejected[name] = "different size"
            continue
        if sharpness(gray) < min_sharpness:
            rejected[name] = "floue"
            continue
        c = find_corners(gray, pattern)
        if c is None:
            rejected[name] = "checkerboard not found"
            continue
        pts.append(c)
        names.append(name)
    if len(pts) < min_views:
        raise ValueError(
            f"{len(pts)} usable views (minimum {min_views}); rejected: {rejected}"
        )

    def run(sel):
        rms, K, dist, rv, tv = cv2.calibrateCamera(
            [obj] * len(sel), [pts[i] for i in sel], size, None, None
        )
        errs = []
        for j, i in enumerate(sel):
            proj, _ = cv2.projectPoints(obj, rv[j], tv[j], K, dist)
            errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - pts[i]) ** 2, 1)))))
        return rms, K, dist, errs

    sel = list(range(len(pts)))
    rms, K, dist, errs = run(sel)
    med = float(np.median(errs))
    keep = [i for i, e in zip(sel, errs, strict=True) if e <= reject_factor * max(med, 0.05)]
    if len(keep) < len(sel) and len(keep) >= min_views:
        for i, e in zip(sel, errs, strict=True):
            if i not in keep:
                rejected[names[i]] = f"aberrante ({e:.2f} px)"
        rms, K, dist, errs = run(keep)
        sel = keep
    return CameraCalibration(K, dist, size[0], size[1], float(rms), len(sel), errs, rejected)


def render_board_view(
    K: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    size: tuple[int, int],
    pattern: tuple[int, int] = (9, 6),
    square_m: float = 0.025,
    px_per_square: int = 60,
) -> np.ndarray:
    """Synthetic image of a checkerboard seen by a pinhole camera (tests)."""
    cols, rows = pattern[0] + 1, pattern[1] + 1
    margin = px_per_square
    bw, bh = cols * px_per_square + 2 * margin, rows * px_per_square + 2 * margin
    board = np.full((bh, bw), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                y, x = margin + r * px_per_square, margin + c * px_per_square
                board[y : y + px_per_square, x : x + px_per_square] = 0
    # checkerboard frame: origin at the first inner corner, units in metres
    s = square_m / px_per_square
    A = np.array(
        [[s, 0, -(margin + px_per_square) * s], [0, s, -(margin + px_per_square) * s], [0, 0, 1]]
    )
    R, _ = cv2.Rodrigues(rvec)
    Hm = K @ np.c_[R[:, :2], tvec.reshape(3)] @ A
    img = cv2.warpPerspective(board, Hm, size, flags=cv2.INTER_AREA, borderValue=128)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
