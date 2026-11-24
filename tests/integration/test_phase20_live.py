"""Phase 20: real-time graph phone → path (fake phone, stub models)."""

import socket

import pytest

from visionaid.utils.config import load_raw

pytestmark = [pytest.mark.integration, pytest.mark.unit]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.timeout(240)
def test_fake_phone_to_path_with_reconnection(tmp_path):
    from visionaid.apps.run_live import run

    raw = load_raw(
        overrides={
            "video.width": 320,
            "depth.scale_factor": 1.0,
            "paths.outputs": str(tmp_path / "out"),
            "paths.cache": str(tmp_path / "cache"),
            "phone.certs_dir": str(tmp_path / "certs"),
            "bus.base_port": free_port() + 20,
            "phone.calib_dir": str(tmp_path / "calib"),  # without the real iPhone calibration
        }
    )
    raw["_phone_https_port"] = free_port()
    raw["_phone_http_port"] = 0
    # 12 s connected, 3 s dropout, 8 s reconnected
    s = run(
        raw=raw,
        fake=True,
        fake_schedule=[(12.0, 3.0), (8.0, 0.0)],
        rerun="off",
        warmup_s=4.0,
        stub_models=True,
    )
    print(s["path"], s["geometry"], s["fake_phone"])
    assert s["nodes_alive_at_end"] == s["n_nodes"], "a node died"
    assert s["stop_time_s"] < 10
    p = s["path"]
    assert p["first_valid_after_s"] is not None and p["first_valid_after_s"] < 60
    assert p["n_valid_paths"] > 10
    # dropout: path marked invalid, then valid again after reconnection
    assert p["n_invalid_paths"] >= 1
    inv = p["invalid_wall_times"]
    assert any(t > inv[0] + 1 for t in p["valid_wall_times"]), "no recovery after dropout"
    assert p["e2e_capture_to_path_ms_p50"] is not None
    g = s["geometry"]
    assert g["imu_calib"] in ("a priori", "calibration") and g["no_imu_frames"] < g["frames"] * 0.2
    assert len(s["fake_phone"]) == 2 and all(f["sent"] > 20 for f in s["fake_phone"])


@pytest.mark.timeout(120)
def test_live_records_session(tmp_path):
    """`run_live --record`: live mode also records the session (replayable afterwards)."""
    from visionaid.apps.run_live import run
    from visionaid.io.session import load_session

    raw = load_raw(
        overrides={
            "video.width": 320,
            "paths.outputs": str(tmp_path / "out"),
            "paths.cache": str(tmp_path / "cache"),
            "phone.certs_dir": str(tmp_path / "certs"),
            "phone.calib_dir": str(tmp_path / "calib"),
            "session.dir": str(tmp_path / "sessions"),
            "bus.base_port": free_port() + 40,
        }
    )
    raw["_phone_https_port"] = free_port()
    raw["_phone_http_port"] = 0
    raw["_phone_record"] = "essai_direct"
    raw["_phone_meta"] = {"target": "où est ma gourde", "notes": ""}
    run(raw=raw, fake=True, fake_schedule=[(4.0, 0.0)], rerun="off", warmup_s=3.0, stub_models=True)
    dirs = list((tmp_path / "sessions").glob("*_essai_direct"))
    assert len(dirs) == 1
    s = load_session(dirs[0])
    assert len(s.frames) > 20 and len(s.imu_t) > 100
    assert s.meta["target"] == "où est ma gourde"


@pytest.mark.timeout(120)
def test_rerun_file_mode_writes_recording(tmp_path):
    """`--rerun file` (and `both`) does write `live_session.rrd` (regression: Rerun 0.23 without
    `set_sinks`, the old `both` mode wrote no file)."""
    from visionaid.apps.live_blocks import RerunLiveBlock

    raw = load_raw(overrides={"paths.outputs": str(tmp_path)})
    raw["_rerun"] = "file"
    blk = RerunLiveBlock(raw)
    import numpy as np

    from visionaid.bus.messages import Frame, Pose

    for k in range(5):
        img = np.full((48, 64, 3), 100 + k, np.uint8)
        blk.process(
            {"frame": Frame(k * 0.1, k, img, np.eye(3)), "pose_world": Pose(k * 0.1, k, np.eye(4))}
        )
    blk.close()
    out = tmp_path / "live_session.rrd"
    assert out.exists() and out.stat().st_size > 1000
