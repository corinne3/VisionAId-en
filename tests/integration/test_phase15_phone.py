"""Phase 15 (integration): fake phone → HTTPS/WebSocket server → callbacks / session."""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest

from visionaid.phone.certs import client_ssl_context
from visionaid.phone.fake_phone import FakePhone, run_fake_phone
from visionaid.phone.server import PhoneServer
from visionaid.utils.config import load_config

pytestmark = [pytest.mark.integration, pytest.mark.unit]


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def phone_cfg(tmp_path):
    return load_config(
        overrides={
            "phone.certs_dir": str(tmp_path / "certs"),
            "session.dir": str(tmp_path / "sessions"),
            "phone.host": "127.0.0.1",
        }
    )


class Collector:
    def __init__(self, delay: float = 0.0) -> None:
        self.frames = []
        self.imu = []
        self.clock = []
        self.delay = delay
        self.lock = threading.Lock()

    def on_frame(self, f):
        if self.delay:
            time.sleep(self.delay)
        with self.lock:
            self.frames.append(f)

    def on_imu(self, b):
        with self.lock:
            self.imu.append(b)

    def on_clock(self, c):
        with self.lock:
            self.clock.append(c)


def _server(cfg, col, **kw):
    return PhoneServer(
        cfg,
        on_frame=col.on_frame,
        on_imu=col.on_imu,
        on_clock=col.on_clock,
        https_port=free_port(),
        http_port=0,
        **kw,
    )


def _phone(srv, **kw):
    return FakePhone(f"https://localhost:{srv.https_port}/", client_ssl_context(srv.certs), **kw)


def test_fake_phone_stream_order_rate_and_clock(phone_cfg):
    col = Collector()
    with _server(phone_cfg, col) as srv:
        st = run_fake_phone(_phone(srv, fps=15, clock_offset=5.0), 6.0)
        time.sleep(0.2)
        summ = srv.summary()
    assert st.welcome.get("protocol") == 1 and not st.errors
    seqs = [f.frame_idx for f in col.frames]
    assert len(seqs) == len(set(seqs)), "duplicate images"
    assert seqs == sorted(seqs), "images out of order"
    assert len(col.frames) >= 0.9 * st.sent_frames
    assert st.sent_frames >= 0.9 * 15 * 6 * 0.9  # sent rate ≈ 15 fps (start-up margin)
    f = col.frames[-1]
    assert f.image.ndim == 3 and f.K.shape == (3, 3)
    assert f.t_phone is not None and f.t_recv is not None and f.t == f.t_phone
    # IMU: ~60 Hz, covering the duration
    n_imu = sum(len(b) for b in col.imu)
    assert n_imu >= 0.9 * st.imu_samples and n_imu > 6 * 50
    assert all(np.all(np.abs(b.gyro) < 0.1) for b in col.imu)  # deg/s → rad/s converti
    # clock: phone = computer + 5 s  →  offset ≈ −5 s
    assert col.clock, "no clock synchronisation"
    assert abs(col.clock[-1].offset + 5.0) < 0.02
    # image age (reception − capture, computer clock): local network, no accumulated delay
    assert summ["frame_age_ms_p95"] < 2 * 1000 / 15


def test_slow_consumer_backpressure_no_accumulation(phone_cfg):
    col = Collector(delay=0.15)  # ingestion lente (~6 img/s)
    with _server(phone_cfg, col) as srv:
        st = run_fake_phone(_phone(srv, fps=15), 4.0)
        summ = srv.summary()
    assert st.skipped_frames > st.sent_frames  # the phone waits for the acknowledgement
    rate = len(col.frames) / 4.0
    assert rate < 8
    # the received image is always fresh: age < 2 sending periods
    assert summ["frame_age_ms_p95"] < 2 * 1000 / 15


def test_reconnection_without_restart(phone_cfg):
    col = Collector()
    with _server(phone_cfg, col) as srv:
        run_fake_phone(_phone(srv), 1.5)
        n1 = len(col.frames)
        time.sleep(0.3)
        st2 = run_fake_phone(_phone(srv, seed=1), 1.5)
        n2 = len(col.frames) - n1
        assert srv.stats.n_connections == 2
    assert n1 > 10 and n2 > 10 and not st2.errors


def test_unknown_protocol_version_refused(phone_cfg):
    col = Collector()
    with _server(phone_cfg, col) as srv:
        st = run_fake_phone(_phone(srv, protocol=99), 1.0)
    assert st.errors and "version" in st.errors[0]
    assert not col.frames


def test_session_recording_complete(phone_cfg, tmp_path):
    col = Collector()
    with _server(phone_cfg, col, record="essai", meta={"target": "gourde"}) as srv:
        st = run_fake_phone(_phone(srv, fps=10), 3.0, calib_frames=3)
        time.sleep(0.2)
        d = srv.recorder.dir
    meta = json.loads((d / "meta.json").read_text())
    assert meta["ua"].startswith("FakePhone") and meta["target"] == "gourde"
    assert meta["protocol"] == 1 and "t_end_pc" in meta
    lines = [json.loads(x) for x in (d / "frames.jsonl").read_text().splitlines()]
    files = sorted((d / "frames").glob("*.jpg"))
    assert len(lines) == len(files) == meta["n_frames"] == st.sent_frames
    assert len(list((d / "calib").glob("*.jpg"))) == 3 == meta["n_calib"]
    assert all((d / ln["file"]).exists() for ln in lines)
    imu = [json.loads(x) for x in (d / "imu.jsonl").read_text().splitlines()]
    assert len(imu) == meta["n_imu"] > 100
    t_f = [ln["t_phone"] for ln in lines]
    t_i = [s["t"] for s in imu]
    assert t_i[0] <= t_f[0] + 0.2 and t_i[-1] >= t_f[-1] - 0.2  # the IMU covers the images
    assert (d / "clock.jsonl").read_text().strip()


def test_server_reachable_over_ipv6(tmp_path):
    """"auto" host: the server also answers over IPv6 (IPv6-only networks)."""
    import ssl
    import urllib.request

    cfg = load_config(
        overrides={
            "phone.certs_dir": str(tmp_path / "certs"),
            "session.dir": str(tmp_path / "sessions"),
            "phone.host": "auto",
        }
    )
    srv = PhoneServer(cfg, https_port=free_port(), http_port=0).start()
    try:
        ctx = ssl.create_default_context(cafile=str(srv.certs.ca_cert))
        ctx.check_hostname = False  # ::1 is not necessarily in the test certificate
        for host in ("127.0.0.1", "[::1]"):
            with urllib.request.urlopen(
                f"https://{host}:{srv.https_port}/", context=ctx, timeout=5
            ) as r:
                assert r.status == 200
        assert any(u.startswith("https://") for u in srv.urls())
    finally:
        srv.stop()
