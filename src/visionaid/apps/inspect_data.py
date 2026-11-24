"""Inspects the input data: video metadata, reference photos, sampled frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from visionaid.utils.config import load_config, resolve_path
from visionaid.utils.logging import get_logger

log = get_logger("inspect")


def rotate_frame(img: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Applies a rotation that is a multiple of 90° (clockwise for display)."""
    r = int(rotation_deg) % 360
    if r == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if r == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if r == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def video_metadata(path: Path) -> dict:
    """Resolution, fps, number of frames, duration and rotation of a video."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Cannot open {path}")
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    meta = {
        "path": str(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": fps,
        "frame_count": n,
        "duration_s": n / fps if fps > 0 else 0.0,
        "rotation_deg": int(cap.get(cv2.CAP_PROP_ORIENTATION_META)),
    }
    cap.release()
    return meta


def sample_frames(path: Path, n: int, rotation: int) -> list[np.ndarray]:
    """Samples n evenly spaced frames (rotation applied)."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = set(np.linspace(0, max(total - 1, 0), n).astype(int).tolist())
    out = []
    i = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in idxs:
            out.append(rotate_frame(img, rotation))
        i += 1
    cap.release()
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    out_dir = resolve_path(cfg.paths.outputs) / "inspect"
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"video": None, "refs": []}
    vpath = resolve_path(cfg.paths.video)
    if vpath.exists():
        meta = video_metadata(vpath)
        report["video"] = meta
        log.info("Video: %s", meta)
        for k, img in enumerate(sample_frames(vpath, args.n, meta["rotation_deg"])):
            cv2.imwrite(str(out_dir / f"frame_{k:02d}.jpg"), img)
    else:
        log.warning("Video missing: %s", vpath)
    for p in sorted(resolve_path(cfg.paths.ref_dir).glob(cfg.paths.ref_glob)):
        img = cv2.imread(str(p))
        info = {"path": str(p), "readable": img is not None}
        if img is not None:
            info.update(width=img.shape[1], height=img.shape[0])
        report["refs"].append(info)
        log.info("Reference: %s", info)
    (out_dir / "inspect.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
