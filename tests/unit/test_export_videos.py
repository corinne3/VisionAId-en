"""Video export: rendering of the three views on a synthetic image, and writing of an MP4."""

import numpy as np
import pytest

from visionaid.apps.export_videos import Mp4Writer, World3DRenderer, render_camera, render_map

pytestmark = pytest.mark.unit


def _frame():
    st = np.full((60, 60), -1, np.int8)
    st[10:50, 10:50] = 0
    st[30:34, 20:40] = 1
    T = np.eye(4)
    T[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).T
    T[:3, 3] = [0.5, 0.5, 1.4]
    return {
        "t": 1.0,
        "image": np.zeros((640, 480, 3), np.uint8),
        "boxes": np.array([[100, 100, 150, 220]], float),
        "scores": np.array([0.7]),
        "target_idx": 0,
        "cand_idx": -1,
        "target_mask": None,
        "T": T,
        "cm": (st, st == 1, np.array([0.0, 0.0]), 0.05),
        "obj": np.array([2.0, 1.0, 0.8]),
        "obj_state": "seen",
        "path": np.array([[0.5, 0.5], [1.5, 0.8], [1.9, 1.0]]),
        "la": np.array([1.5, 0.8]),
        "state": "ARRIVED",
        "dist": 0.5,
        "cloud": np.random.default_rng(0).random((200, 3)).astype(np.float32) * 2,
    }


def test_three_views_and_mp4(tmp_path):
    fr = _frame()
    cam = render_camera(fr)
    assert cam.shape[0] == 720 and cam.shape[1] % 2 == 0
    traj = np.array([[0.2, 0.2], [0.5, 0.5]])
    mp = render_map(fr, traj, (-0.5, 3.5, -0.5, 3.5))
    assert mp.shape == (720, 720, 3)
    w3 = World3DRenderer((-0.5, 3.5, -0.5, 3.5)).render(fr, traj)
    assert w3.shape[:2] == (720, 1280)
    w = Mp4Writer(tmp_path / "v.mp4", 10.0)
    for _ in range(5):
        w.write(mp)
    w.close()
    assert (tmp_path / "v.mp4").stat().st_size > 1000
