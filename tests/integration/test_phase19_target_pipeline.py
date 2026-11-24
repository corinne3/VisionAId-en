"""Phase 19 (integration): full pipeline with the "keys" card, without a reference photo (stubs)."""

import pytest

from visionaid.perception.depth import DepthEstimator, StubDepthBackend
from visionaid.perception.detector import StubDetector
from visionaid.perception.pipeline import PerceptionBlock
from visionaid.perception.reid import TextOnlyReID
from visionaid.perception.target import load_config_for_target

pytestmark = [pytest.mark.integration, pytest.mark.unit]


def test_keys_text_only_pipeline(tmp_path):
    import importlib.util
    from pathlib import Path

    from visionaid.apps.run_offline import run
    from visionaid.io.intrinsics import from_hfov

    spec = importlib.util.spec_from_file_location(
        "sp", Path(__file__).with_name("test_phase16_session_pipeline.py")
    )
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    sess = tmp_path / "sess"
    sp.make_session(sess)
    cfg = load_config_for_target(
        "où sont mes clés",
        overrides={
            "video.width": sp.W,
            "paths.cache": str(tmp_path / "cache"),
            "depth.scale_factor": 1.0,
            "phone.calib_dir": str(tmp_path / "calib"),  # without the real iPhone calibration
        },
    )
    K = from_hfov(sp.W, sp.H, cfg.intrinsics.hfov_deg).K
    det = StubDetector(fn=lambda img, k: [(sp.BOX, 0.6, "keys")])
    det.backend, det.weights, det.device = "stub", "stub", "cpu"
    perception = PerceptionBlock(cfg, detector=det, reid=TextOnlyReID(cfg))
    depth = DepthEstimator(cfg, backend=StubDepthBackend(lambda h, w: sp.scene_depth(h, w, K)))
    summ = run(
        cfg,
        sess,
        tmp_path / "out",
        perception=perception,
        depth_est=depth,
        rerun=False,
        report_section=None,
    )
    assert summ["confirmed_fraction"] > 0.5 and summ["n_plans"] > 0
