"""Monocular metric depth: interchangeable backends, temporal smoothing, disk cache."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from visionaid.bus.messages import Depth, Frame
from visionaid.utils.config import resolve_path
from visionaid.utils.device import select_device
from visionaid.utils.logging import get_logger

log = get_logger("depth")

FALLBACK_ORDER = ["da3metric", "da3small", "dav2_indoor"]


class DepthBackend(Protocol):
    """Contract: BGR image → depth (m) (H,W) float32 and optional confidence."""

    name: str

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]: ...


class HFDepthBackend:
    """`transformers` AutoModelForDepthEstimation model (Depth Anything V2 Metric)."""

    def __init__(self, model_id: str, device: str, input_width: int = 518) -> None:
        import torch
        from transformers import AutoModelForDepthEstimation

        self.torch = torch
        self.name = model_id
        self.device = device
        self.input_width = int(input_width)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        torch = self.torch
        h, w = image.shape[:2]
        iw = max(14, int(round(self.input_width / 14)) * 14)
        ih = max(14, int(round(h * iw / w / 14)) * 14)
        x = cv2.resize(image, (iw, ih), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
        t = (
            torch.from_numpy(np.ascontiguousarray(x)).to(self.device).permute(2, 0, 1)[None].float()
            / 255.0
        )
        t = (t - self.mean) / self.std
        with torch.inference_mode():
            d = self.model(pixel_values=t).predicted_depth
            d = torch.nn.functional.interpolate(
                d[:, None], size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]
        return d.float().cpu().numpy(), None


class StubDepthBackend:
    """Fake depth: function (H,W) → depth, by default floor + wall at 3 m."""

    name = "stub"

    def __init__(self, fn=None) -> None:
        self.fn = fn

    def infer(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        h, w = image.shape[:2]
        if self.fn is not None:
            return self.fn(h, w).astype(np.float32), None
        d = np.full((h, w), 3.0, np.float32)
        return d, np.ones((h, w), np.float32)


def make_backend(cfg, backend: str | None = None) -> DepthBackend:
    """Backend according to the config, with a fallback chain (DA3 → DA3-Small → DAv2 Indoor)."""
    dcfg = cfg.depth
    wanted = backend or dcfg.backend
    if wanted == "stub":
        return StubDepthBackend()
    device = select_device(cfg.device)
    order = (
        FALLBACK_ORDER
        if wanted == "auto"
        else [wanted] + [b for b in FALLBACK_ORDER if b != wanted]
    )
    errors = {}
    for b in order:
        try:
            if b in ("da3metric", "da3small"):
                # Depth Anything 3 is not supported by transformers; the
                # `depth-anything-3` package requires numpy<2 (incompatible with rerun): see DECISIONS.md.
                import depth_anything_3  # noqa: F401

                raise NotImplementedError("DA3 integration not done")
            mid = getattr(dcfg.models, b)
            be = HFDepthBackend(mid, device, dcfg.input_width)
            log.info("Depth: %s on %s", mid, device)
            return be
        except Exception as e:  # noqa: BLE001
            errors[b] = repr(e)[:200]
            log.info("Depth backend %s unavailable: %s", b, errors[b])
    raise RuntimeError(f"No depth backend: {errors}")


class DepthEstimator:
    """Estimation + scale factor + optional temporal smoothing + disk cache."""

    def __init__(
        self, cfg, backend: DepthBackend | None = None, cache_key: str | None = None
    ) -> None:
        d = cfg.depth
        self.backend = backend or make_backend(cfg)
        self.scale = float(d.scale_factor)
        self.min_m, self.max_m = float(d.min_m), float(d.max_m)
        self.smoothing = bool(d.temporal_smoothing)
        self.alpha = float(d.smoothing_alpha)
        self.cache_dir: Path | None = None
        if cache_key:
            self.cache_dir = resolve_path(cfg.paths.cache) / "depth" / cache_key
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._prev: np.ndarray | None = None
        self._prev_T: np.ndarray | None = None
        self._K: np.ndarray | None = None
        self.times: list[float] = []
        self.cache_hits = 0

    def cache_path(self, frame_idx: int) -> Path | None:
        return None if self.cache_dir is None else self.cache_dir / f"{frame_idx:06d}.npz"

    def raw(self, frame: Frame) -> np.ndarray:
        """Raw depth from the model (before scaling), with cache."""
        p = self.cache_path(frame.frame_idx)
        if p is not None and p.exists():
            self.cache_hits += 1
            return np.load(p)["d"].astype(np.float32)
        t0 = time.perf_counter()
        d, _conf = self.backend.infer(frame.image)
        self.times.append(time.perf_counter() - t0)
        if p is not None:
            np.savez_compressed(p, d=d.astype(np.float16))
        return d.astype(np.float32)

    def __call__(self, frame: Frame, T_world_cam: np.ndarray | None = None) -> Depth:
        d = np.clip(self.raw(frame) * self.scale, self.min_m, self.max_m)
        if self.smoothing:
            d = self._smooth(d, frame.K, T_world_cam)
        return Depth(
            t=frame.t, frame_idx=frame.frame_idx, depth=d.astype(np.float32), confidence=None
        )

    def _smooth(self, d: np.ndarray, K: np.ndarray, T: np.ndarray | None) -> np.ndarray:
        """Exponential average; if the pose is provided, the previous depth is reprojected."""
        prev = self._prev
        if prev is not None and T is not None and self._prev_T is not None:
            prev = warp_depth(prev, K, np.linalg.inv(T) @ self._prev_T)
        out = d
        if prev is not None and prev.shape == d.shape:
            valid = (prev > 0) & (np.abs(prev - d) < 0.2 * d)
            out = d.copy()
            out[valid] = self.alpha * d[valid] + (1 - self.alpha) * prev[valid]
        self._prev, self._prev_T = out, T
        return out

    # Block interface of the bus
    def process(self, inputs):
        f = inputs.get("frame")
        if f is None:
            return None
        pose = inputs.get("pose")
        return {"depth": self(f, None if pose is None else pose.T_world_cam)}


def warp_depth(
    depth: np.ndarray, K: np.ndarray, T_cur_prev: np.ndarray, step: int = 2
) -> np.ndarray:
    """Reprojects a depth map into another view (simple z-buffer, subsampled)."""
    h, w = depth.shape
    v, u = np.mgrid[0:h:step, 0:w:step]
    z = depth[v, u]
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    P = np.stack([x.ravel(), y.ravel(), z.ravel(), np.ones(z.size)])
    Q = (T_cur_prev @ P)[:3]
    ok = Q[2] > 0.05
    uu = np.round(K[0, 0] * Q[0, ok] / Q[2, ok] + K[0, 2]).astype(int)
    vv = np.round(K[1, 1] * Q[1, ok] / Q[2, ok] + K[1, 2]).astype(int)
    zz = Q[2, ok]
    inside = (uu >= 0) & (uu < w) & (vv >= 0) & (vv < h)
    out = np.full((h, w), np.inf, np.float32)
    np.minimum.at(out, (vv[inside], uu[inside]), zz[inside])
    out[~np.isfinite(out)] = 0
    if step > 1:
        out = cv2.dilate(out, np.ones((step, step), np.uint8))
    return out


def object_height_m(mask_or_box, depth: np.ndarray, K: np.ndarray) -> float | None:
    """Approximate 3D height of an object: vertical extent (px) × median depth / fy."""
    if isinstance(mask_or_box, np.ndarray) and mask_or_box.ndim == 2:
        ys, xs = np.nonzero(mask_or_box)
        if len(ys) < 20:
            return None
        y0, y1 = np.percentile(ys, [1, 99])
        z = np.median(depth[ys, xs])
    else:
        x0, y0, x1, y1 = [int(v) for v in mask_or_box]
        sub = depth[y0:y1, x0:x1]
        if sub.size == 0:
            return None
        z = np.median(sub)
    return float((y1 - y0) * z / K[1, 1])


def video_depth_estimator(
    cfg, video: str | Path | None = None, backend: DepthBackend | None = None
) -> DepthEstimator:
    """Estimator with a disk cache specific to (video, model, resolution)."""
    from visionaid.io.session import media_hash

    video = resolve_path(video or cfg.paths.video)
    be = backend or make_backend(cfg)
    key = media_hash(video, f"{be.name}|{cfg.depth.input_width}|{cfg.video.width}")
    return DepthEstimator(cfg, backend=be, cache_key=f"{video.stem}_{key}")


def make_depth_block(cfg_raw: dict) -> DepthEstimator:
    from visionaid.utils.config import _to_dataclass

    return DepthEstimator(_to_dataclass("config", cfg_raw))
