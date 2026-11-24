"""Target re-identification: embedding (DINOv2) + colour score (grey / orange)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import yaml

from visionaid.utils.config import resolve_path
from visionaid.utils.device import select_device
from visionaid.utils.logging import get_logger

log = get_logger("reid")

BG_COLOR = (104, 116, 124)  # ImageNet mean BGR, neutral background of the masked crops


class Embedder(Protocol):
    """Contract: list of BGR crops → normalised embeddings (N, D)."""

    def embed(self, crops: list[np.ndarray]) -> np.ndarray: ...


class DinoEmbedder:
    """DINOv2 (transformers): CLS token + mean of the patches, normalised."""

    def __init__(self, model_name: str, device: str, size: int = 224) -> None:
        import torch
        from transformers import AutoModel

        self.torch = torch
        self.device = device
        self.size = size
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 768), np.float32)
        torch = self.torch
        arr = np.stack(
            [
                cv2.resize(c, (self.size, self.size), interpolation=cv2.INTER_AREA)[:, :, ::-1]
                for c in crops
            ]
        )
        x = (
            torch.from_numpy(np.ascontiguousarray(arr)).to(self.device).permute(0, 3, 1, 2).float()
            / 255.0
        )
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            h = self.model(pixel_values=x).last_hidden_state
        feat = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=1)
        feat = torch.nn.functional.normalize(feat, dim=1)
        return feat.cpu().numpy().astype(np.float32)


class ClipEmbedder:
    """Fallback: CLIP image encoder (transformers)."""

    def __init__(self, model_name: str, device: str, size: int = 224) -> None:
        import torch
        from transformers import CLIPModel

        self.torch = torch
        self.device = device
        self.size = size
        self.model = CLIPModel.from_pretrained(model_name).to(device).eval()
        self.mean = torch.tensor([0.4815, 0.4578, 0.4082], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.2686, 0.2613, 0.2758], device=device).view(1, 3, 1, 1)

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        torch = self.torch
        arr = np.stack([cv2.resize(c, (self.size, self.size))[:, :, ::-1] for c in crops])
        x = (
            torch.from_numpy(np.ascontiguousarray(arr)).to(self.device).permute(0, 3, 1, 2).float()
            / 255.0
        )
        x = (x - self.mean) / self.std
        with torch.inference_mode():
            f = self.model.get_image_features(pixel_values=x)
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output
        return torch.nn.functional.normalize(f, dim=1).cpu().numpy().astype(np.float32)


class StubEmbedder:
    """Deterministic fake embedder: coarse colour histogram."""

    def embed(self, crops: list[np.ndarray]) -> np.ndarray:
        out = []
        for c in crops:
            small = cv2.resize(c, (8, 8)).astype(np.float32).ravel() + 1e-3
            out.append(small / np.linalg.norm(small))
        return np.asarray(out, np.float32).reshape(len(crops), -1)


def make_embedder(cfg) -> tuple[Embedder, str]:
    """Builds the embedder according to the config, with fallback DINOv2 → CLIP → stub."""
    rcfg = cfg.reid
    device = select_device(cfg.device)
    order = ["dinov2", "clip"] if rcfg.backend == "auto" else [rcfg.backend]
    for b in order:
        try:
            if b == "dinov2":
                return DinoEmbedder(rcfg.model, device, rcfg.crop_size), b
            if b == "clip":
                return ClipEmbedder(rcfg.clip_model, device, rcfg.crop_size), b
            if b == "stub":
                return StubEmbedder(), b
        except Exception as e:  # noqa: BLE001
            log.warning("Embedder %s unavailable: %r", b, e)
    log.warning("Falling back to the stub embedder")
    return StubEmbedder(), "stub"


def masked_crop(
    image: np.ndarray, box: np.ndarray, mask: np.ndarray | None, margin: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Crop of the box (relative margin) with a neutral background outside the mask. Returns (crop, crop mask)."""
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [float(v) for v in box]
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = int(max(0, np.floor(x0 - mx))), int(max(0, np.floor(y0 - my)))
    x1, y1 = int(min(w, np.ceil(x1 + mx))), int(min(h, np.ceil(y1 + my)))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), bool)
    crop = image[y0:y1, x0:x1].copy()
    if mask is not None and mask.shape[:2] == (h, w):
        m = mask[y0:y1, x0:x1].astype(bool)
        if m.sum() > 16:
            crop[~m] = BG_COLOR
        else:
            m = np.ones(crop.shape[:2], bool)
    else:
        m = np.ones(crop.shape[:2], bool)
    return crop, m


@dataclass
class ColorModel:
    """HSV parameters of the target colours."""

    orange_h: tuple[int, int]
    orange_s_min: int
    orange_v_min: int
    gray_s_max: int
    gray_v_min: int
    orange_frac_ref: float
    gray_frac_ref: float

    @classmethod
    def from_cfg(cls, rcfg) -> ColorModel:
        return cls(
            tuple(rcfg.orange_h),
            rcfg.orange_s_min,
            rcfg.orange_v_min,
            rcfg.gray_s_max,
            rcfg.gray_v_min,
            rcfg.orange_frac_ref,
            rcfg.gray_frac_ref,
        )

    def fractions(self, crop: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        """Fractions of orange and grey pixels in the mask."""
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hh, ss, vv = hsv[..., 0][mask], hsv[..., 1][mask], hsv[..., 2][mask]
        n = max(int(mask.sum()), 1)
        orange = (
            (hh >= self.orange_h[0])
            & (hh <= self.orange_h[1])
            & (ss >= self.orange_s_min)
            & (vv >= self.orange_v_min)
        )
        gray = (ss <= self.gray_s_max) & (vv >= self.gray_v_min)
        return float(orange.sum()) / n, float(gray.sum()) / n

    def proportion_score(self, crop: np.ndarray, mask: np.ndarray) -> float:
        """Score ∈ [0,1]: joint presence of orange and grey."""
        fo, fg = self.fractions(crop, mask)
        so = min(fo / self.orange_frac_ref, 1.0)
        sg = min(fg / self.gray_frac_ref, 1.0)
        return float(np.sqrt(so * sg))


def make_color_model(rcfg) -> ColorModel | None:
    """Colour model of the target card (`reid.color_model`); None if `none`."""
    kind = str(getattr(rcfg, "color_model", "orange_gray") or "orange_gray")
    if kind == "none":
        return None
    if kind != "orange_gray":
        raise ValueError(f"unknown colour model: {kind}")
    return ColorModel.from_cfg(rcfg)


class TextOnlyReID:
    """Target without a reference photo: decision on the detector confidence alone (text prompt).

    The score of a detection is its confidence; the threshold is `reid.textonly_conf`. Accepted limit:
    no distinction between two objects of the same category (recorded in the report).
    """

    text_only = True
    backend = "text-only"

    def __init__(self, cfg) -> None:
        self.threshold = float(getattr(cfg.reid, "textonly_conf", 0.35))
        self.gallery = None

    def score_detections(self, image, boxes, masks, scores=None) -> np.ndarray:
        if scores is None:
            return np.zeros(len(boxes), np.float32)
        return np.asarray(scores, np.float32)


def hsv_hist(crop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalised H-S histogram (16×8) under the mask."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], mask.astype(np.uint8), [16, 8], [0, 180, 0, 256]).ravel()
    return h / max(h.sum(), 1e-9)


def hist_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1 − Bhattacharyya distance."""
    return float(np.sum(np.sqrt(a * b)))


def augment(img: np.ndarray, rng: np.random.Generator, occlude: bool = True) -> np.ndarray:
    """Augmentation: rotation, scale, blur, brightness, partial crop, 30% occlusion."""
    h, w = img.shape[:2]
    ang = rng.uniform(-15, 15)
    sc = rng.uniform(0.6, 1.15)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=BG_COLOR)
    if rng.random() < 0.6:
        k = int(rng.choice([3, 5, 7]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    gain = rng.uniform(0.65, 1.35)
    out = np.clip(out.astype(np.float32) * gain + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
    if rng.random() < 0.4:  # recadrage partiel
        fy0, fy1 = rng.uniform(0, 0.15), rng.uniform(0.85, 1.0)
        fx0, fx1 = rng.uniform(0, 0.15), rng.uniform(0.85, 1.0)
        out = out[int(fy0 * h) : int(fy1 * h), int(fx0 * w) : int(fx1 * w)]
    if occlude and rng.random() < 0.5:  # occlusion of about 30% of the area
        hh, ww = out.shape[:2]
        oh, ow = int(hh * rng.uniform(0.4, 0.6)), int(ww * rng.uniform(0.5, 0.7))
        oh = min(oh, hh)
        ow = min(int(0.3 * hh * ww / max(oh, 1)), ww)
        y0, x0 = rng.integers(0, max(hh - oh, 1)), rng.integers(0, max(ww - ow, 1))
        out = out.copy()
        out[y0 : y0 + oh, x0 : x0 + ow] = rng.integers(0, 255, 3)
    return out


def hue_shift(img: np.ndarray, shift: int) -> np.ndarray:
    """Shifts the hue (OpenCV H ∈ [0,180))."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = ((hsv[..., 0].astype(int) + shift) % 180).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


class TargetReID:
    """Re-identification score of a crop against the reference gallery."""

    def __init__(self, cfg, embedder: Embedder | None = None, backend: str | None = None) -> None:
        self.cfg = cfg
        r = cfg.reid
        if embedder is None:
            embedder, backend = make_embedder(cfg)
        self.embedder = embedder
        self.backend = backend or type(embedder).__name__
        self.colors = make_color_model(r)  # None: target without a distinctive colour
        self.w_embed = float(r.w_embed)
        self.w_color = float(r.w_color) if self.colors is not None else 0.0
        self.last_color: np.ndarray | None = None  # colour share of the last call
        self.gallery: np.ndarray | None = None
        self.gallery_hists: list[np.ndarray] = []
        self.ref_crops: list[tuple[np.ndarray, np.ndarray]] = []
        self.threshold = load_threshold(cfg)

    # --- galerie -------------------------------------------------------------
    def build_gallery(
        self, crops: list[tuple[np.ndarray, np.ndarray]], n_aug: int | None = None, seed: int = 0
    ) -> None:
        """Gallery = reference crops + augmentations (without occlusion)."""
        rng = np.random.default_rng(seed)
        n_aug = int(self.cfg.reid.n_augment_gallery if n_aug is None else n_aug)
        imgs = []
        self.ref_crops = list(crops)
        self.gallery_hists = []
        for crop, mask in crops:
            imgs.append(crop)
            self.gallery_hists.append(hsv_hist(crop, mask))
            for _ in range(n_aug):
                imgs.append(augment(crop, rng, occlude=False))
        self.gallery = self.embedder.embed(imgs)

    # --- scores --------------------------------------------------------------
    def embed_scores(self, crops: list[np.ndarray]) -> np.ndarray:
        if self.gallery is None:
            raise RuntimeError("Gallery not built")
        if not crops:
            return np.zeros(0, np.float32)
        e = np.concatenate(
            [self.embedder.embed(crops[i : i + 64]) for i in range(0, len(crops), 64)]
        )
        # numpy 2.x + Accelerate (macOS) emits spurious RuntimeWarnings on float32 matmul
        with np.errstate(all="ignore"):
            sims = e.astype(np.float64) @ self.gallery.T.astype(np.float64)
        return sims.max(axis=1).astype(np.float32)

    def color_score(self, crop: np.ndarray, mask: np.ndarray) -> float:
        if self.colors is None:
            return 1.0  # without a colour model: neutral (w_color = 0)
        prop = self.colors.proportion_score(crop, mask)
        h = hsv_hist(crop, mask)
        hs = max((hist_similarity(h, g) for g in self.gallery_hists), default=0.0)
        return 0.5 * prop + 0.5 * hs

    def score_crops(self, items: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Combined score ∈ [0,1] for (crop, mask) pairs."""
        if not items:
            self.last_color = np.zeros(0, np.float32) if self.w_color > 0 else None
            return np.zeros(0, np.float32)
        emb = self.embed_scores([c for c, _ in items])
        emb01 = np.clip(emb, 0.0, 1.0)
        self.last_color = None
        if self.w_color <= 0:
            return emb01.astype(np.float32)
        col = np.array([self.color_score(c, m) for c, m in items], np.float32)
        self.last_color = col  # colour share, read by the tracker (probable target)
        return combine_scores(emb01, col, self.w_embed, self.w_color)

    def score_detections(
        self, image: np.ndarray, boxes: np.ndarray, masks: np.ndarray | None
    ) -> np.ndarray:
        items = [
            masked_crop(image, b, None if masks is None else masks[i]) for i, b in enumerate(boxes)
        ]
        return self.score_crops(items)


def combine_scores(
    emb: np.ndarray, color: np.ndarray, w_e: float, w_c: float, eps: float = 0.05
) -> np.ndarray:
    """Weighted geometric mean: a missing colour cancels the score, whatever the embedding.

    (DINOv2 is not very sensitive to hue: an arithmetic mean let a blue bottle through.)
    """
    w = w_e + w_c
    c = (np.asarray(color) + eps) / (1.0 + eps)
    return (np.power(np.clip(emb, 1e-6, 1.0), w_e / w) * np.power(c, w_c / w)).astype(np.float32)


def load_threshold(cfg) -> float:
    p = resolve_path(cfg.reid.threshold_file)
    if p.exists():
        d = yaml.safe_load(p.read_text()) or {}
        if "threshold" in d:
            return float(d["threshold"])
    return float(cfg.reid.default_threshold)


def reference_crops(
    cfg, detector, ref_paths: list[Path], work_width: int = 1280
) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """Locates the target in each reference photo (detection + colour) and returns the masked crop.

    The crop is extracted at full resolution then reduced; detection is done on an image
    of width `work_width` (the bottle may be small in the scene).
    """
    colors = make_color_model(cfg.reid)
    out = []
    for p in ref_paths:
        full = cv2.imread(str(p))
        if full is None:
            continue
        s = work_width / full.shape[1]
        small = cv2.resize(
            full, (work_width, int(round(full.shape[0] * s))), interpolation=cv2.INTER_AREA
        )
        d = detector.detect(small)
        best, best_sc = None, -1.0
        for i, b in enumerate(d.boxes):
            crop, m = masked_crop(small, b, None if d.masks is None else d.masks[i])
            # with a colour model: colour decides; otherwise the most confident detection
            sc = (colors.proportion_score(crop, m) if colors is not None else 0.0) + 0.2 * float(
                d.scores[i]
            )
            if sc > best_sc:
                best, best_sc = i, sc
        if best is None:
            log.warning("Target not found in %s", p)
            continue
        crop, m = masked_crop(small, d.boxes[best], None if d.masks is None else d.masks[best])
        out.append(
            (
                crop,
                m,
                {
                    "path": str(p),
                    "box": d.boxes[best].tolist(),
                    "score": float(d.scores[best]),
                    "color": best_sc,
                },
            )
        )
    return out
