"""Phase 20 (slow): fake phone replaying the real video, real models; real session."""

import socket

import pytest

from visionaid.apps.session_info import list_sessions
from visionaid.utils.config import load_config, load_raw, resolve_path
from visionaid.utils.report import update_report

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.data
@pytest.mark.timeout(900)
def test_fake_phone_video_real_models(video_path, tmp_path):
    from visionaid.apps.run_live import run

    raw = load_raw(overrides={"phone.certs_dir": str(tmp_path / "certs")})
    raw["_phone_https_port"] = free_port()
    raw["_phone_http_port"] = 0
    s = run(30.0, raw, fake=True, fake_video=str(video_path), rerun="file", warmup_s=30.0)
    update_report("live_fake_phone_video", s)
    print(s["path"], s["geometry"])
    assert s["nodes_alive_at_end"] == s["n_nodes"]
    assert s["stop_time_s"] < 10
    p = s["path"]
    assert p["n_valid_paths"] > 0, "no valid path (bottle never confirmed?)"
    assert p["valid_path_hz"] is not None and p["valid_path_hz"] >= 2.0
    # latency target recorded, not blocking (see DECISIONS / report)
    update_report(
        "live_latency_check",
        {"p95_ms": p["e2e_capture_to_path_ms_p95"], "target_ms": 700},
    )


@pytest.mark.data
@pytest.mark.timeout(900)
def test_session_gourde_replay_live(tmp_path):
    from visionaid.apps.run_live import run

    ss = [p for p in list_sessions(resolve_path(load_config().session.dir)) if "gourde" in p.name]
    if not ss:
        pytest.skip("No \"gourde_piece\" session (see docs/checkpoint_enregistrements.md)")
    raw = load_raw(overrides={"target.query": "où est ma gourde"})
    s = run(60.0, raw, session=str(ss[-1]), rerun="file", warmup_s=30.0)
    update_report("live_session_gourde", s)
    assert s["nodes_alive_at_end"] == s["n_nodes"]
    assert s["path"]["object_states"].get("seen", 0) > 0, "bottle never confirmed"
    assert s["path"]["n_valid_paths"] > 0
