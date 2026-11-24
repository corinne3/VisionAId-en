"""Phase 29: request spoken on the phone → search → announcements (fake phone, stub models)."""

import socket

import pytest

from visionaid.utils.config import load_raw

pytestmark = [pytest.mark.integration, pytest.mark.unit]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.timeout(240)
def test_spoken_request_then_announcements(tmp_path):
    from visionaid.apps.run_live import run

    ov = {
        "video.width": 320,
        "depth.scale_factor": 1.0,
        "paths.outputs": str(tmp_path / "out"),
        "paths.cache": str(tmp_path / "cache"),
        "phone.certs_dir": str(tmp_path / "certs"),
        "bus.base_port": free_port() + 20,
        "phone.calib_dir": str(tmp_path / "calib"),
        "command.mode": "stt",
        "command.stt": "phone",  # text sent by the phone: no Whisper in this test
    }
    raw = load_raw(overrides=ov)
    raw["_raw_base"] = load_raw(overrides=ov)
    raw["_phone_https_port"] = free_port()
    raw["_phone_http_port"] = 0
    s = run(
        raw=raw,
        fake=True,
        fake_schedule=[(22.0, 0.0)],
        rerun="off",
        warmup_s=4.0,
        stub_models=True,
        fake_commands=[(2.0, "je cherche mon téléphone"), (4.0, "je cherche ma gourde")],
    )
    fp = s["fake_phone"][0]
    print(fp["speech"], fp["heard"], s["guidance"])
    assert s["nodes_alive_at_end"] == s["n_nodes"]
    assert [h["action"] for h in fp["heard"]] == ["unknown", "search"]
    sp = fp["speech"]
    assert any(t.startswith("Je n'ai pas compris") for t in sp)
    i = next(k for k, t in enumerate(sp) if t.startswith("Je cherche la gourde"))
    assert any("gourde est" in t for t in sp[i + 1 :]), sp
    # before the request: silence (IDLE), then search then guidance
    modes = s["guidance"]["modes"]
    assert modes.get("silent", 0) > 0 and modes.get("guide", 0) > 0
    assert s["path"]["n_valid_paths"] > 5
