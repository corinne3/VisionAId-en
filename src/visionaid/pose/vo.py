"""RGB-D visual odometry: ORB + matching + RANSAC PnP on keyframes, Open3D fallback."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from visionaid.bus.messages import Pose
from visionaid.mapping.geometry import backproject_pixels, inv_T, make_T, rot_angle_deg
from visionaid.utils.logging import get_logger

log = get_logger("vo")


@dataclass
class Observation:
    """Keypoints of an image: pixels, ORB descriptors, sampled depth."""

    uv: np.ndarray  # (N,2) float
    desc: np.ndarray  # (N,32) uint8
    depth: np.ndarray  # (N,) float (0 if invalid)
    image: np.ndarray | None = None  # grey levels (Open3D fallback)
    depth_map: np.ndarray | None = None


@dataclass
class Keyframe:
    obs: Observation
    pts3d: np.ndarray  # (N,3) keyframe camera frame (nan if invalid depth)
    T_world_kf: np.ndarray
    frame_idx: int


class RGBDOdometry:
    """Keyframe odometry. `step(obs, K, t, idx)` → Pose (T_world_cam)."""

    def __init__(self, cfg_vo) -> None:
        c = cfg_vo
        self.method = str(c.method)
        self.n_features = int(c.n_features)
        self.ratio = float(c.ratio_test)
        self.min_inliers = int(c.min_inliers)
        self.reproj = float(c.reproj_err_px)
        self.iters = int(c.ransac_iters)
        self.kf_trans = float(c.kf_trans_m)
        self.kf_rot = float(c.kf_rot_deg)
        self.kf_min_inliers = int(c.kf_min_inliers)
        self.min_d, self.max_d = float(c.min_depth_m), float(c.max_depth_m)
        self.max_jump_m = float(c.max_jump_m)
        self.max_jump_deg = float(c.max_jump_deg)
        self.min_kp = int(c.min_texture_keypoints)
        self.symmetric = bool(getattr(c, "symmetric_refine", True))
        self.orb = cv2.ORB_create(nfeatures=self.n_features, fastThreshold=10)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.kf: Keyframe | None = None
        self.T_world_cam = np.eye(4)
        self.T_kf_cur_prev = np.eye(4)  # last relative pose (PnP seed)
        self.lost_streak = 0
        self.n_lost = 0
        self.n_frames = 0
        self.keyframes: list[tuple[int, np.ndarray]] = []

    # ------------------------------------------------------------------ images
    def observe(self, image: np.ndarray, depth: np.ndarray) -> Observation:
        """Detects ORB features and samples the depth."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        kps, desc = self.orb.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            return Observation(
                np.zeros((0, 2)), np.zeros((0, 32), np.uint8), np.zeros(0), gray, depth
            )
        uv = np.array([k.pt for k in kps], np.float64)
        ui = np.clip(np.round(uv[:, 0]).astype(int), 0, depth.shape[1] - 1)
        vi = np.clip(np.round(uv[:, 1]).astype(int), 0, depth.shape[0] - 1)
        return Observation(uv, desc, depth[vi, ui].astype(np.float64), gray, depth)

    # ------------------------------------------------------------- PnP core
    def _kf_from(self, obs: Observation, K: np.ndarray, T_world: np.ndarray, idx: int) -> Keyframe:
        pts = np.full((len(obs.uv), 3), np.nan)
        ok = (obs.depth > self.min_d) & (obs.depth < self.max_d)
        if ok.any():
            pts[ok] = backproject_pixels(obs.uv[ok], obs.depth[ok], K)
        self.keyframes.append((idx, T_world.copy()))
        return Keyframe(obs, pts, T_world.copy(), idx)

    def _match(self, a: np.ndarray, b: np.ndarray) -> list[tuple[int, int]]:
        if len(a) < 2 or len(b) < 2:
            return []
        knn = self.matcher.knnMatch(a, b, k=2)
        out = []
        for m in knn:
            if len(m) == 2 and m[0].distance < self.ratio * m[1].distance:
                out.append((m[0].queryIdx, m[0].trainIdx))
        return out

    def _pnp(
        self, kf: Keyframe, obs: Observation, K: np.ndarray
    ) -> tuple[np.ndarray | None, int, float]:
        """T_cur_kf by RANSAC PnP + LM refinement. Returns (T, n_inliers, median reprojection error)."""
        matches = self._match(kf.obs.desc, obs.desc)
        if len(matches) < self.min_inliers:
            return None, len(matches), np.inf
        qi = np.array([m[0] for m in matches])
        ti = np.array([m[1] for m in matches])
        P = kf.pts3d[qi]
        ok = np.isfinite(P).all(1)
        if ok.sum() < self.min_inliers:
            return None, int(ok.sum()), np.inf
        P, uv = P[ok], obs.uv[ti[ok]]
        # minimal-sample RANSAC (AP3P, 4 points): robust to ~50% wrong matches
        good, rvec, tvec, inl = cv2.solvePnPRansac(
            P,
            uv,
            K,
            None,
            iterationsCount=self.iters,
            reprojectionError=self.reproj,
            confidence=0.999,
            flags=cv2.SOLVEPNP_AP3P,
        )
        if not good or inl is None or len(inl) < self.min_inliers:
            T0 = self.T_kf_cur_prev
            rvec0, _ = cv2.Rodrigues(T0[:3, :3])
            good, rvec, tvec, inl = cv2.solvePnPRansac(
                P,
                uv,
                K,
                None,
                rvec0.copy(),
                T0[:3, 3].reshape(3, 1).copy(),
                useExtrinsicGuess=True,
                iterationsCount=self.iters,
                reprojectionError=self.reproj,
                confidence=0.999,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not good or inl is None or len(inl) < self.min_inliers:
                return None, 0 if inl is None else len(inl), np.inf
        # re-estimation of the inlier set with the refined pose
        rvec, tvec = cv2.solvePnPRefineLM(P[inl.ravel()], uv[inl.ravel()], K, None, rvec, tvec)
        proj, _ = cv2.projectPoints(P, rvec, tvec, K, None)
        inl = np.nonzero(np.linalg.norm(proj.reshape(-1, 2) - uv, axis=1) < self.reproj)[0]
        if len(inl) < self.min_inliers:
            return None, len(inl), np.inf
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(P[inl], uv[inl], K, None, rvec, tvec)
        if self.symmetric:
            uv_kf = kf.obs.uv[qi[ok]][inl]
            z_cur = obs.depth[ti[ok]][inl]
            rvec, tvec = self._refine_symmetric(P[inl], uv[inl], uv_kf, z_cur, K, rvec, tvec)
        proj, _ = cv2.projectPoints(P[inl], rvec, tvec, K, None)
        err = float(np.median(np.linalg.norm(proj.reshape(-1, 2) - uv[inl], axis=1)))
        R, _ = cv2.Rodrigues(rvec)
        return make_T(R, tvec), len(inl), err

    def _refine_symmetric(self, P_kf, uv_cur, uv_kf, z_cur, K, rvec, tvec):
        """Symmetric refinement: reprojection kf→current and current→kf (current depth).

        Uses the information of both depth maps; Huber loss (1 px).
        """
        from scipy.optimize import least_squares

        okc = (z_cur > self.min_d) & (z_cur < self.max_d)
        if okc.sum() < self.min_inliers:
            return rvec, tvec
        P_cur = backproject_pixels(uv_cur[okc], z_cur[okc], K)
        uvk = uv_kf[okc]
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        def proj(X):
            return np.stack([fx * X[:, 0] / X[:, 2] + cx, fy * X[:, 1] / X[:, 2] + cy], 1)

        def res(x):
            R, _ = cv2.Rodrigues(x[:3])
            t = x[3:]
            A = P_kf @ R.T + t
            r1 = proj(A) - uv_cur
            B = (P_cur - t) @ R  # R^T (P - t)
            r2 = proj(B) - uvk
            bad = (A[:, 2] < 0.05).any() or (B[:, 2] < 0.05).any()
            if bad:
                return np.full(2 * (len(A) + len(B)), 1e3)
            return np.concatenate([r1.ravel(), r2.ravel()])

        x0 = np.concatenate([np.ravel(rvec), np.ravel(tvec)])
        sol = least_squares(
            res, x0, loss="huber", f_scale=1.0, max_nfev=30, method="trf", x_scale="jac"
        )
        if not sol.success or not np.isfinite(sol.x).all():
            return rvec, tvec
        return sol.x[:3].reshape(3, 1), sol.x[3:].reshape(3, 1)

    def _open3d(
        self, kf: Keyframe, obs: Observation, K: np.ndarray
    ) -> tuple[np.ndarray | None, int, float]:
        """Fallback: Open3D dense RGB-D odometry (T_cur_kf)."""
        if (
            kf.obs.image is None
            or obs.image is None
            or kf.obs.depth_map is None
            or obs.depth_map is None
        ):
            return None, 0, np.inf
        try:
            import open3d as o3d
        except ImportError:
            return None, 0, np.inf
        h, w = obs.image.shape[:2]
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, K[0, 0], K[1, 1], K[0, 2], K[1, 2])

        def rgbd(img, d):
            return o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(img)),
                o3d.geometry.Image(np.ascontiguousarray(d.astype(np.float32))),
                depth_scale=1.0,
                depth_trunc=self.max_d,
                convert_rgb_to_intensity=True,
            )

        opt = o3d.pipelines.odometry.OdometryOption()
        ok, T_kf_cur, info = o3d.pipelines.odometry.compute_rgbd_odometry(
            rgbd(obs.image, obs.depth_map),
            rgbd(kf.obs.image, kf.obs.depth_map),
            intr,
            inv_T(self.T_kf_cur_prev),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(),
            opt,
        )
        if not ok:
            return None, 0, np.inf
        # compute_rgbd_odometry(source=cur, target=kf) gives T bringing cur → kf: T_kf_cur
        return inv_T(np.asarray(T_kf_cur)), self.min_inliers, 0.0

    # ----------------------------------------------------------------- step
    def step(self, obs: Observation, K: np.ndarray, t: float = 0.0, frame_idx: int = 0) -> Pose:
        self.n_frames += 1
        if self.kf is None:
            if len(obs.uv) < self.min_kp or (obs.depth > self.min_d).sum() < self.min_inliers:
                self.n_lost += 1
                return Pose(t, frame_idx, self.T_world_cam.copy(), "lost")
            self.kf = self._kf_from(obs, K, self.T_world_cam, frame_idx)
            return Pose(t, frame_idx, self.T_world_cam.copy(), "init", len(obs.uv), 0.0, True)
        if len(obs.uv) < self.min_kp:
            return self._lost(t, frame_idx, obs, K)
        if self.method == "open3d":
            T_cur_kf, n_in, err = self._open3d(self.kf, obs, K)
        else:
            T_cur_kf, n_in, err = self._pnp(self.kf, obs, K)
            if T_cur_kf is None:
                T_cur_kf, n_in, err = self._open3d(self.kf, obs, K)
        if T_cur_kf is None:
            return self._lost(t, frame_idx, obs, K)
        T_world_cur = self.kf.T_world_kf @ inv_T(T_cur_kf)
        jump = inv_T(self.T_world_cam) @ T_world_cur
        if self.lost_streak == 0 and (
            np.linalg.norm(jump[:3, 3]) > self.max_jump_m
            or rot_angle_deg(jump[:3, :3]) > self.max_jump_deg
        ):
            return self._lost(t, frame_idx, obs, K)
        self.T_world_cam = T_world_cur
        self.T_kf_cur_prev = T_cur_kf
        self.lost_streak = 0
        rel = T_cur_kf
        is_kf = (
            np.linalg.norm(rel[:3, 3]) > self.kf_trans
            or rot_angle_deg(rel[:3, :3]) > self.kf_rot
            or n_in < self.kf_min_inliers
        )
        if is_kf:
            self.kf = self._kf_from(obs, K, self.T_world_cam, frame_idx)
            self.T_kf_cur_prev = np.eye(4)
        return Pose(t, frame_idx, self.T_world_cam.copy(), "ok", int(n_in), float(err), bool(is_kf))

    def _lost(self, t: float, idx: int, obs: Observation, K: np.ndarray) -> Pose:
        """Tracking loss: pose kept; after 2 failures, reset on a new keyframe."""
        self.lost_streak += 1
        self.n_lost += 1
        if (
            self.lost_streak >= 2
            and len(obs.uv) >= self.min_kp
            and (obs.depth > self.min_d).sum() >= self.min_inliers
        ):
            self.kf = self._kf_from(obs, K, self.T_world_cam, idx)
            self.T_kf_cur_prev = np.eye(4)
        return Pose(t, idx, self.T_world_cam.copy(), "lost")
