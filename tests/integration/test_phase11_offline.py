"""Full pipeline on the real video (real models, caches)."""

import json

import pytest

from visionaid.utils.config import load_raw, resolve_path
from visionaid.utils.report import report_path, update_report

pytestmark = [pytest.mark.integration, pytest.mark.data, pytest.mark.slow]


@pytest.fixture(scope="module")
def offline(video_path, video_perception):
    from visionaid.apps.run_offline import run

    return run()


def test_offline_completes(offline):
    assert offline["n_frames"] > 0


def test_outputs_exist(offline, cfg):
    out = resolve_path(cfg.paths.outputs)
    for name in ("annotated.mp4", "session.rrd", "guidance.wav", "report.json"):
        p = out / name
        assert p.exists() and p.stat().st_size > 0, name


def test_target_confirmed(offline):
    assert offline["confirmed_fraction"] > 0


def test_approach_if_close_at_end(offline):
    """APPROACHING/ARRIVED reached if the video ends near the bottle (< 1.2 m); informative otherwise."""
    d = offline["final_distance_m"]
    update_report("offline", {"final_distance_check": d})
    if d is None or d >= 1.2:
        pytest.skip(f"final distance {d} ≥ 1.2 m: informative")
    assert offline["reached_approach"]


@pytest.mark.timeout(400)
def test_realtime_30s():
    """Multi-process graph for 30 s in replay: no crash, clean stop, latencies recorded."""
    from visionaid.apps.run_realtime import run

    raw = load_raw()
    s = run(30.0, raw)
    update_report("realtime", s)
    assert s["nodes_alive_at_end"] == s["n_nodes"]
    assert s["stop_time_s"] < 15
    assert s["e2e_latency"].get("n_cues", 0) > 0
    rep = json.loads(report_path().read_text())
    assert "realtime" in rep
