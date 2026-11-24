"""Perception block (detection + ReID + tracking), shared by the offline and real-time modes."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import time
from typing import Any

import numpy as np

from visionaid.bus.messages import Detections, Frame
from visionaid.utils.config import load_config, resolve_path, to_dict
from visionaid.utils.logging import get_logger

log = get_logger("perception")


def subset(d: Detections, idx) -> Detections:
    """Detections at indices `idx` (boxes, scores, labels, masks)."""
    idx = np.asarray(idx, int)
    return Detections(
        t=d.t,
        frame_idx=d.frame_idx,
        boxes=np.asarray(d.boxes, np.float32).reshape(-1, 4)[idx],
        scores=np.asarray(d.scores, np.float32)[idx],
        labels=[d.labels[i] for i in idx] if d.labels else [],
        masks=None if d.masks is None else d.masks[idx],
    )


def _empty(t: float, idx: int) -> Detections:
    return Detections(t, idx, np.zeros((0, 4), np.float32), np.zeros(0, np.float32), [])


class PerceptionBlock:
    """Image → Detections with the "target" decision (confirmed track).

    Furniture (`landmarks.prompts`): detected in the same YOLOE pass as the target, separated from
    the target detections (no ReID) and stored in `last_furniture` (and on the bus, topic
    `furniture`), to describe where the object is.

    Target can change along the way (`set_target` / `clear_target`, voice command): without
    a target, only furniture is detected.
    """

    def __init__(self, cfg, detector=None, reid=None, active: bool = True, reid_factory=None):
        from visionaid.perception.tracker import TargetTracker

        self.cfg = cfg
        lm = getattr(cfg, "landmarks", None)
        on = lm is not None and bool(getattr(lm, "enabled", False))
        self.furniture_prompts = [str(p) for p in lm.prompts] if on else []
        self.furniture_conf = float(getattr(lm, "min_conf", 0.35)) if on else 1.0
        self.furniture_hz = float(getattr(lm, "publish_hz", 3.0)) if on else 0.0
        self._t_furniture = -1e9
        self.make_reid = reid_factory or self._build_reid
        self.target_id = 0
        self._targets: dict[str, tuple] = {}
        self.last_furniture: Detections | None = None
        self.times: list[float] = []
        if active:
            if detector is None or reid is None:
                from visionaid.apps.calibrate_reid import build_reid

                reid, _rc, detector = build_reid(cfg, detector)
            self.detector = detector
            self.reid = reid
            self.tracker = TargetTracker(cfg, reid.threshold)
            self.target_prompts = list(
                getattr(detector, "prompts", None) or cfg.detector.text_prompts
            )
            name = str(getattr(cfg.target, "name", "cible"))
            self._targets[name] = (cfg, reid, self.target_prompts)
        else:
            if detector is None:
                from visionaid.perception.detector import YoloDetector

                detector = YoloDetector(cfg, prompts=self.furniture_prompts or None)
            self.detector = detector
            self.reid = None
            self.tracker = None
            self.target_prompts = []
        self._apply_prompts()

    # ------------------------------------------------------------------ target
    @staticmethod
    def _build_reid(cfg, detector):
        from visionaid.apps.calibrate_reid import build_reid

        return build_reid(cfg, detector)[0]

    def _apply_prompts(self) -> None:
        """Detector vocabulary: target + furniture (if the model allows it)."""
        fur = [p for p in self.furniture_prompts if p not in self.target_prompts]
        ok = False
        if hasattr(self.detector, "set_prompts") and (self.target_prompts or fur):
            ok = bool(self.detector.set_prompts(self.target_prompts + fur))
        stub = not hasattr(self.detector, "set_prompts")  # test detector: labels provided
        self.furniture_set = set(fur) if (ok or stub) else set()

    def set_target(self, cfg, target_id: int) -> None:
        """New target (config with the card applied). The ReID of each target is kept."""
        from visionaid.perception.tracker import TargetTracker

        name = str(getattr(cfg.target, "name", "cible"))
        det = self.detector
        if name not in self._targets:
            prompts = list(cfg.detector.text_prompts)
            if hasattr(det, "set_prompts"):
                det.set_prompts(prompts)  # gallery built without the furniture
            if hasattr(det, "conf"):
                det.conf = float(cfg.detector.conf)
            reid = self.make_reid(cfg, det)
            self._targets[name] = (cfg, reid, prompts)
        cfg_t, reid, prompts = self._targets[name]
        if hasattr(det, "conf"):
            det.conf = float(cfg_t.detector.conf)
        self.cfg = cfg_t
        self.reid = reid
        self.target_prompts = list(prompts)
        self.tracker = TargetTracker(cfg_t, reid.threshold)
        self.target_id = int(target_id)
        self._apply_prompts()

    def clear_target(self, target_id: int) -> None:
        """No more target: only furniture is detected."""
        self.reid = None
        self.tracker = None
        self.target_prompts = []
        self.target_id = int(target_id)
        self._apply_prompts()

    # ------------------------------------------------------------------ image
    def __call__(self, frame: Frame) -> Detections:
        t0 = time.perf_counter()
        if self.reid is None and not self.furniture_set:
            d = _empty(frame.t, frame.frame_idx)
        else:
            d = self.detector.detect(frame.image, frame.t, frame.frame_idx)
        fur_idx, tgt_idx = [], []
        for i in range(len(d.boxes)):
            lab = d.labels[i] if i < len(d.labels) else ""
            if lab in self.furniture_set:
                if float(d.scores[i]) >= self.furniture_conf:
                    fur_idx.append(i)
            else:
                tgt_idx.append(i)
        fur = subset(d, fur_idx) if self.furniture_set else None
        if fur is not None:
            fur.target_id = self.target_id
        self.last_furniture = fur
        if fur_idx or len(tgt_idx) != len(d.boxes):
            d = subset(d, tgt_idx)
        if self.reid is None:
            d = _empty(frame.t, frame.frame_idx) if len(d.boxes) else d
        else:
            self.reid.last_color = None
            if not len(d.boxes):
                scores = np.zeros(0, np.float32)
            elif getattr(self.reid, "text_only", False):
                scores = self.reid.score_detections(frame.image, d.boxes, d.masks, d.scores)
            else:
                scores = self.reid.score_detections(frame.image, d.boxes, d.masks)
            colors = getattr(self.reid, "last_color", None) if len(d.boxes) else None
            d = self.tracker.update(d, scores, colors)
        d.target_id = self.target_id
        # furniture also stored on the object (offline mode cache); outside the bus message fields
        d._furniture = fur
        self.times.append(time.perf_counter() - t0)
        return d

    # Block interface of the bus
    def process(self, inputs: dict[str, Any]) -> dict[str, Any] | None:
        f = inputs.get("frame")
        if f is None:
            return None
        d = self(f)
        if d.masks is not None and d.masks.dtype == bool:  # compressed masks for the bus
            d.masks = np.packbits(d.masks, axis=-1)
        out: dict[str, Any] = {"detections": d}
        fur = self.last_furniture
        if (
            fur is not None
            and len(fur.boxes)
            and f.t - self._t_furniture >= 1.0 / max(self.furniture_hz, 1e-6)
        ):
            self._t_furniture = f.t
            if fur.masks is not None and fur.masks.dtype == bool:
                fur.masks = np.packbits(fur.masks, axis=-1)
            out["furniture"] = fur
        return out


def make_perception_block(cfg_raw: dict) -> PerceptionBlock:
    from visionaid.utils.config import _to_dataclass

    return PerceptionBlock(_to_dataclass("config", cfg_raw))


def run_video_perception(
    cfg=None, video=None, use_cache: bool = True
) -> tuple[list[Detections], dict]:
    """Perception over the whole video, with a disk cache (.cache/perception/<hash>.pkl)."""

    cfg = cfg or load_config()
    video = resolve_path(video or cfg.paths.video)
    key_cfg = json.dumps(
        {
            k: to_dict(getattr(cfg, k))
            for k in ("detector", "reid", "tracker", "video", "intrinsics", "landmarks")
            if hasattr(cfg, k)
        },
        sort_keys=True,
    )
    th_file = resolve_path(cfg.reid.threshold_file)
    key_cfg += th_file.read_text() if th_file.exists() else ""
    # the tracking and ReID code is part of the key: otherwise a code change
    # re-read old detections from the cache (identical "before/after" measurements)
    here = Path(__file__).parent
    key_cfg += "".join((here / f).read_text() for f in ("tracker.py", "reid.py", "pipeline.py"))
    from visionaid.io.session import media_hash, open_media

    key = media_hash(video, key_cfg)
    cpath = resolve_path(cfg.paths.cache) / "perception" / f"{video.stem}_{key}.pkl"
    if use_cache and cpath.exists():
        data = pickle.loads(cpath.read_bytes())
        return data["dets"], data["info"]
    block = PerceptionBlock(cfg)
    src = open_media(cfg, video)  # video or phone session
    dets = []
    while (f := src.read()) is not None:
        d = block(f)
        if d.masks is not None:  # mask compression (packbits along the width)
            d.masks = np.packbits(d.masks, axis=-1)
        fur = getattr(d, "_furniture", None)
        if fur is not None and fur.masks is not None and fur.masks.dtype == bool:
            fur.masks = np.packbits(fur.masks, axis=-1)
        dets.append(d)
    times = np.asarray(block.times[5:] or block.times)
    info = {
        "detector_backend": block.detector.backend,
        "detector_weights": block.detector.weights,
        "reid_backend": block.reid.backend,
        "threshold": block.reid.threshold,
        "n_frames": len(dets),
        "perception_fps": float(1.0 / times.mean()) if len(times) else 0.0,
        "perception_ms_p50": float(np.median(times) * 1e3) if len(times) else 0.0,
        "image_shape": [src.out_h, src.out_w],
        "device": block.detector.device,
    }
    cpath.parent.mkdir(parents=True, exist_ok=True)
    cpath.write_bytes(pickle.dumps({"dets": dets, "info": info}))
    return dets, info


def unpack_masks(d: Detections, width: int) -> np.ndarray | None:
    """Decompresses the masks stored in the cache (packbits)."""
    if d.masks is None:
        return None
    if d.masks.dtype == bool:
        return d.masks
    return np.unpackbits(d.masks, axis=-1, count=width).astype(bool)
