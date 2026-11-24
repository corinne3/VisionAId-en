"""Shared fixtures: data paths, synthetic video, config."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from visionaid.utils.config import ROOT, load_config

VIDEO = ROOT / "data" / "video" / "appart.mp4"
REF_DIR = ROOT / "data" / "ref"


def ref_paths() -> list[Path]:
    return sorted(REF_DIR.glob("gourde_*.jpg"))


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def video_path() -> Path:
    if not VIDEO.exists():
        pytest.skip(f"Video missing: {VIDEO}")
    return VIDEO


@pytest.fixture(scope="session")
def ref_images() -> list[np.ndarray]:
    paths = ref_paths()
    if not paths:
        pytest.skip(f"No reference photo in {REF_DIR}")
    return [cv2.imread(str(p)) for p in paths]


def make_synthetic_video(
    path: Path, seconds: float = 3.0, fps: int = 30, w: int = 320, h: int = 240
) -> int:
    """Writes a synthetic video: a moving square + frame number as a pattern."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n = int(round(seconds * fps))
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        img[:, : w // 4] = (0, 0, 255)  # red band on the left (rotation marker)
        x = int(i * (w - 40) / max(n - 1, 1))
        cv2.rectangle(img, (x, h // 2 - 20), (x + 40, h // 2 + 20), (255, 255, 255), -1)
        vw.write(img)
    vw.release()
    return n


@pytest.fixture()
def synthetic_video(tmp_path: Path) -> tuple[Path, int]:
    p = tmp_path / "synth.mp4"
    n = make_synthetic_video(p)
    return p, n


@pytest.fixture(scope="session")
def reid_bundle():
    """(reid, ref_crops, detector) with the real models (slow tests)."""
    if not ref_paths():
        pytest.skip("No reference photo")
    from visionaid.apps.calibrate_reid import build_reid

    return build_reid(load_config())


@pytest.fixture(scope="session")
def video_perception(video_path):
    """Full perception of the video (cached on disk)."""
    from visionaid.perception.pipeline import run_video_perception

    if not ref_paths():
        pytest.skip("No reference photo")
    return run_video_perception(load_config())
