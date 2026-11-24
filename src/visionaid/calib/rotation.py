"""Rotation tools: SO(3) exponential/logarithm, Kabsch, visual rotation between two images."""

from __future__ import annotations

import math

import cv2
import numpy as np


def skew(w: np.ndarray) -> np.ndarray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], float)


def so3_exp(w: np.ndarray) -> np.ndarray:
    """Rotation matrix of the rotation vector `w` (rad)."""
    R, _ = cv2.Rodrigues(np.asarray(w, float).reshape(3, 1))
    return R


def so3_log(R: np.ndarray) -> np.ndarray:
    """Rotation vector (rad) of a rotation matrix."""
    w, _ = cv2.Rodrigues(np.asarray(R, float))
    return w.ravel()


def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle (degrees) of a rotation."""
    return float(np.degrees(np.linalg.norm(so3_log(R))))


def kabsch(a: np.ndarray, b: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Rotation R minimisant Σ w‖b − R a‖² (a, b : (N,3))."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.ones(len(a)) if w is None else np.asarray(w, float)
    H = (a * w[:, None]).T @ b
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T)) or 1.0
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def bearings(pts: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels (N,2) → unit vectors in the camera frame."""
    p = np.c_[pts, np.ones(len(pts))] @ np.linalg.inv(K).T
    return p / np.linalg.norm(p, axis=1, keepdims=True)


class VisualRotation:
    """Camera rotation between two images (pure rotation model, RANSAC on the directions).

    Returns R_c2_c1 such that b2 ≈ R b1 for a distant point: the camera turned by
    R_c1_c2 = Rᵀ, i.e. a rotation vector ω·dt = −log(R) in the camera frame. Robust to a
    small translation (loose angular threshold), enough to compare with the gyroscope.
    """

    def __init__(self, n_features: int = 1000, thresh_deg: float = 0.6, iters: int = 200) -> None:
        self.orb = cv2.ORB_create(n_features)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.thresh = math.radians(thresh_deg)
        self.iters = iters
        self.rng = np.random.default_rng(0)

    def features(self, img: np.ndarray):
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return self.orb.detectAndCompute(g, None)

    def estimate(self, f1, f2, K: np.ndarray) -> tuple[np.ndarray | None, int]:
        """Rotation between two sets of features (kp, desc); (None, 0) on failure."""
        (k1, d1), (k2, d2) = f1, f2
        if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
            return None, 0
        pairs = []
        for m in self.bf.knnMatch(d1, d2, k=2):
            if len(m) == 2 and m[0].distance < 0.75 * m[1].distance:
                pairs.append((m[0].queryIdx, m[0].trainIdx))
        if len(pairs) < 8:
            return None, 0
        p1 = np.array([k1[i].pt for i, _ in pairs])
        p2 = np.array([k2[j].pt for _, j in pairs])
        b1, b2 = bearings(p1, K), bearings(p2, K)
        best = None
        n = len(b1)
        cos_t = math.cos(self.thresh)
        for _ in range(self.iters):
            idx = self.rng.choice(n, 2, replace=False)
            R = kabsch(b1[idx], b2[idx])
            inl = np.einsum("ij,ij->i", b1 @ R.T, b2) > cos_t
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < 8:
            return None, 0
        R = kabsch(b1[best], b2[best])
        inl = np.einsum("ij,ij->i", b1 @ R.T, b2) > cos_t
        if inl.sum() >= 8:
            R = kabsch(b1[inl], b2[inl])
        return R, int(inl.sum())

    def angular_velocity(self, img1, img2, K, dt: float) -> tuple[np.ndarray | None, int]:
        """Mean angular velocity (rad/s, camera frame) between two images."""
        R, n = self.estimate(self.features(img1), self.features(img2), K)
        if R is None or dt <= 0:
            return None, 0
        return -so3_log(R) / dt, n
