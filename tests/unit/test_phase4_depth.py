import numpy as np
import pytest

from visionaid.bus.messages import Frame
from visionaid.io.intrinsics import from_hfov
from visionaid.perception.depth import DepthEstimator, StubDepthBackend, object_height_m, warp_depth
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit


def _frame(i=0, h=120, w=160):
    K = from_hfov(w, h, 69).K
    return Frame(t=i / 30, frame_idx=i, image=np.zeros((h, w, 3), np.uint8), K=K)


def test_stub_shape_dtype_positive(tmp_path):
    cfg = load_config(overrides={"paths.cache": str(tmp_path)})
    est = DepthEstimator(cfg, backend=StubDepthBackend(), cache_key="unit")
    d = est(_frame())
    assert d.depth.shape == (120, 160) and d.depth.dtype == np.float32
    assert (d.depth > 0).all()


def test_cache_path_and_hit(tmp_path):
    cfg = load_config(overrides={"paths.cache": str(tmp_path), "depth.scale_factor": 1.0})
    calls = []

    def fn(h, w):
        calls.append(1)
        return np.full((h, w), 2.0)

    est = DepthEstimator(cfg, backend=StubDepthBackend(fn), cache_key="vid_abc")
    p = est.cache_path(7)
    assert p == tmp_path / "depth" / "vid_abc" / "000007.npz"
    est(_frame(7))
    assert p.exists()
    est2 = DepthEstimator(cfg, backend=StubDepthBackend(fn), cache_key="vid_abc")
    d = est2(_frame(7))
    assert len(calls) == 1 and est2.cache_hits == 1
    assert np.allclose(d.depth, 2.0, atol=1e-2)


def test_scale_and_clip(tmp_path):
    cfg = load_config(overrides={"depth.scale_factor": 2.0, "depth.max_m": 5.0})
    est = DepthEstimator(cfg, backend=StubDepthBackend(lambda h, w: np.full((h, w), 3.0)))
    assert np.allclose(est(_frame()).depth, 5.0)


def test_warp_identity():
    K = from_hfov(160, 120, 69).K
    d = np.full((120, 160), 2.0, np.float32)
    w = warp_depth(d, K, np.eye(4), step=1)
    assert np.allclose(w[5:-5, 5:-5], 2.0)


def test_temporal_smoothing(tmp_path):
    cfg = load_config(
        overrides={
            "depth.temporal_smoothing": True,
            "depth.smoothing_alpha": 0.5,
            "depth.scale_factor": 1.0,
        }
    )
    vals = iter([2.0, 2.2])
    est = DepthEstimator(cfg, backend=StubDepthBackend(lambda h, w: np.full((h, w), next(vals))))
    est(_frame(0))
    d = est(_frame(1)).depth
    assert np.allclose(d, 2.1, atol=1e-5)


def test_object_height():
    K = np.array([[500, 0, 80], [0, 500, 60], [0, 0, 1]], float)
    depth = np.full((120, 160), 2.0)
    mask = np.zeros((120, 160), bool)
    mask[30:90, 70:90] = True  # 60 px at 2 m, f=500 → 0.24 m
    assert object_height_m(mask, depth, K) == pytest.approx(0.24, rel=0.05)
