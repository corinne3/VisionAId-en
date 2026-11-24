"""Geometry: back-projection, frame changes, SE(3) utilities."""

from __future__ import annotations

import numpy as np


def backproject(
    depth: np.ndarray,
    K: np.ndarray,
    step: int = 1,
    mask: np.ndarray | None = None,
    min_d: float = 0.0,
    max_d: float = np.inf,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth → 3D points in the camera frame (N,3) and pixels (N,2) (u,v), subsampled."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[v, u]
    ok = np.isfinite(z) & (z > min_d) & (z < max_d)
    if mask is not None:
        ok &= mask[v, u]
    u, v, z = u[ok].astype(np.float64), v[ok].astype(np.float64), z[ok].astype(np.float64)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1), np.stack([u, v], 1)


def backproject_pixels(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels (N,2) + depths (N,) → camera points (N,3)."""
    x = (uv[:, 0] - K[0, 2]) * z / K[0, 0]
    y = (uv[:, 1] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def project(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera points (N,3) → pixels (N,2)."""
    z = pts[:, 2:3]
    return (pts[:, :2] / z) * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])


def transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Applies a 4×4 transformation to points (N,3)."""
    return pts @ T[:3, :3].T + T[:3, 3]


def inv_T(T: np.ndarray) -> np.ndarray:
    """Inverse of a rigid transformation."""
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def rot_angle_deg(R: np.ndarray) -> float:
    """Angle (degrees) of a rotation."""
    c = (np.trace(R[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rodrigues(rvec: np.ndarray) -> np.ndarray:
    """Rotation vector → matrix (without OpenCV dependency)."""
    th = float(np.linalg.norm(rvec))
    if th < 1e-12:
        return np.eye(3)
    k = np.ravel(rvec) / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def rot_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Minimal rotation bringing the unit vector a onto b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        p = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        ax = np.cross(a, p)
        return rodrigues(ax / np.linalg.norm(ax) * np.pi)
    Kx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + Kx + Kx @ Kx * (1.0 / (1.0 + c))
