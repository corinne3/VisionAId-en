"""Open-vocabulary detection (YOLOE) with a fallback chain, and stub detector for tests."""

from __future__ import annotations

import contextlib
import os
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from visionaid.bus.messages import Detections
from visionaid.utils.config import resolve_path
from visionaid.utils.device import select_device
from visionaid.utils.logging import get_logger

log = get_logger("detector")

FALLBACK_ORDER = ["yoloe26", "yoloe11", "yoloworld", "coco"]


class Detector(Protocol):
    """Contract: BGR image → Detections (boxes, scores, labels, masks)."""

    def detect(self, image: np.ndarray, t: float = 0.0, frame_idx: int = 0) -> Detections: ...


@contextlib.contextmanager
def _in_dir(path: Path):
    """Ultralytics downloads land in the current directory."""
    path.mkdir(parents=True, exist_ok=True)
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


class YoloDetector:
    """YOLOE (text prompt) with fallback YOLOE-11 → YOLO-World v2 → COCO YOLO "bottle"."""

    def __init__(self, cfg, prompts: list[str] | None = None, backend: str | None = None) -> None:
        self.cfg = cfg
        dcfg = cfg.detector
        self.device = select_device(cfg.device)
        self.prompts = list(prompts or dcfg.text_prompts)
        self.conf = float(dcfg.conf)
        self.iou = float(dcfg.iou)
        self.imgsz = int(dcfg.imgsz)
        self.weights_dir = resolve_path(cfg.paths.cache) / "weights"
        wanted = backend or dcfg.backend
        order = (
            FALLBACK_ORDER
            if wanted == "auto"
            else [wanted] + [b for b in FALLBACK_ORDER if b != wanted]
        )
        self.backend = None
        self.errors: dict[str, str] = {}
        for b in order:
            try:
                self._load(b)
                self.backend = b
                break
            except Exception as e:  # noqa: BLE001 - we log it and move on to the next fallback
                self.errors[b] = repr(e)[:300]
                log.warning("Detector backend %s unavailable: %s", b, self.errors[b])
        if self.backend is None:
            raise RuntimeError(f"No detector available: {self.errors}")
        log.info(
            "Detector: %s (%s) on %s, prompts=%s",
            self.backend,
            self.weights,
            self.device,
            self.labels,
        )

    def _load(self, b: str) -> None:
        self.weights = getattr(self.cfg.detector.weights, b)
        with _in_dir(self.weights_dir):
            if b in ("yoloe26", "yoloe11"):
                from ultralytics import YOLOE

                from visionaid.perception.clip_shim import install

                install()
                m = YOLOE(self.weights)
                m.set_classes(self.prompts)
                self.labels = list(self.prompts)
                self.class_filter = None
            elif b == "yoloworld":
                from ultralytics import YOLOWorld

                from visionaid.perception.clip_shim import install

                install()
                m = YOLOWorld(self.weights)
                m.set_classes(self.prompts)
                self.labels = list(self.prompts)
                self.class_filter = None
            else:
                from ultralytics import YOLO

                m = YOLO(self.weights)
                names = m.names
                self.labels = [names[i] for i in sorted(names)]
                wanted = (
                    {"bottle", "cup", "vase"}
                    if "bottle" in " ".join(self.prompts)
                    else set(self.prompts)
                )
                self.class_filter = [i for i, n in names.items() if n in wanted]
            # warm-up
            m.predict(
                np.zeros((self.imgsz, self.imgsz, 3), np.uint8), device=self.device, verbose=False
            )
        self.model = m

    def set_prompts(self, prompts: list[str]) -> bool:
        """Changes the vocabulary (YOLOE / YOLO-World); False if the model does not allow it.

        The score of each class only depends on its own text and duplicate suppression
        is done class by class: adding classes (furniture) does not change the
        detections of the classes already present (checked on the reference photos).
        """
        if self.backend not in ("yoloe26", "yoloe11", "yoloworld"):
            return False
        prompts = list(prompts)
        if prompts == self.labels:
            return True
        with _in_dir(self.weights_dir):
            if self.backend in ("yoloe26", "yoloe11"):
                # text embeddings kept per prompt: changing target does not recompute them
                import torch

                cache = self.__dict__.setdefault("_pe", {})
                missing = [p for p in prompts if p not in cache]
                if missing:
                    pe = self.model.get_text_pe(missing)
                    for k, p in enumerate(missing):
                        cache[p] = pe[:, k : k + 1]
                self.model.set_classes(prompts, torch.cat([cache[p] for p in prompts], dim=1))
            else:
                self.model.set_classes(prompts)
        self.prompts = list(prompts)
        self.labels = list(prompts)
        return True

    def detect(self, image: np.ndarray, t: float = 0.0, frame_idx: int = 0) -> Detections:
        r = self.model.predict(
            image,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
            retina_masks=True,
            classes=self.class_filter,
        )[0]
        boxes = (
            r.boxes.xyxy.cpu().numpy().astype(np.float32)
            if r.boxes is not None
            else np.zeros((0, 4), np.float32)
        )
        scores = (
            r.boxes.conf.cpu().numpy().astype(np.float32)
            if r.boxes is not None
            else np.zeros(0, np.float32)
        )
        cls = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else np.zeros(0, int)
        masks = None
        if r.masks is not None and len(boxes):
            masks = r.masks.data.cpu().numpy() > 0.5
            if masks.shape[1:] != image.shape[:2]:
                import cv2

                masks = np.stack(
                    [
                        cv2.resize(
                            m.astype(np.uint8),
                            (image.shape[1], image.shape[0]),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        > 0
                        for m in masks
                    ]
                )
        labels = [self.labels[c] if c < len(self.labels) else str(c) for c in cls]
        return Detections(
            t=t, frame_idx=frame_idx, boxes=boxes, scores=scores, labels=labels, masks=masks
        )

    def benchmark(self, image: np.ndarray, n: int = 20) -> float:
        """Throughput (images/s) on a given image."""
        self.detect(image)
        t0 = time.perf_counter()
        for _ in range(n):
            self.detect(image)
        return n / (time.perf_counter() - t0)


class StubDetector:
    """Fake detector: returns detections programmed by frame index."""

    def __init__(
        self, script: dict[int, list[tuple[np.ndarray, float, str]]] | None = None, fn=None
    ) -> None:
        self.script = script or {}
        self.fn = fn

    def detect(self, image: np.ndarray, t: float = 0.0, frame_idx: int = 0) -> Detections:
        items = self.fn(image, frame_idx) if self.fn is not None else self.script.get(frame_idx, [])
        h, w = image.shape[:2]
        boxes = np.array([np.asarray(b, np.float32) for b, _, _ in items], np.float32).reshape(
            -1, 4
        )
        masks = np.zeros((len(items), h, w), bool)
        for k, b in enumerate(boxes):
            x0, y0, x1, y1 = np.clip(b, 0, [w, h, w, h]).astype(int)
            masks[k, y0:y1, x0:x1] = True
        return Detections(
            t=t,
            frame_idx=frame_idx,
            boxes=boxes,
            scores=np.array([s for _, s, _ in items], np.float32),
            labels=[lab for _, _, lab in items],
            masks=masks,
        )
