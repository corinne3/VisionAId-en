"""Phase 15 (unit): certificates, protocol, clock, input buffer."""

import ssl
import threading
import time

import numpy as np
import pytest
from cryptography import x509

from visionaid.bus.messages import Frame, ImuBatch
from visionaid.phone.block import PhoneInbox, merge_batches
from visionaid.phone.certs import ensure_certs, server_cert_covers
from visionaid.phone.clock import ClockEstimator
from visionaid.phone.protocol import (
    ProtocolError,
    check_hello,
    imu_batch_from_json,
    pack_frame,
    unpack_frame,
)

pytestmark = pytest.mark.unit


def test_certs_cover_names_and_chain(tmp_path):
    paths = ensure_certs(tmp_path, ["localhost", "monpc.local"], ["127.0.0.1", "192.168.1.20"])
    cert = x509.load_pem_x509_certificate(paths.server_cert.read_bytes())
    san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    assert {"localhost", "monpc.local"} <= set(san.get_values_for_type(x509.DNSName))
    assert "192.168.1.20" in {str(i) for i in san.get_values_for_type(x509.IPAddress)}
    days = (cert.not_valid_after_utc - cert.not_valid_before_utc).days
    assert days <= 398  # limite iOS
    ca_bytes = paths.ca_cert.read_bytes()
    ca = x509.load_pem_x509_certificate(ca_bytes)
    cert.verify_directly_issued_by(ca)
    # idempotent as long as the host is covered; re-issued if the IP changes, with the same authority
    before = paths.server_cert.read_bytes()
    ensure_certs(tmp_path, ["localhost", "monpc.local"], ["127.0.0.1", "192.168.1.20"])
    assert paths.server_cert.read_bytes() == before
    assert not server_cert_covers(paths, ["localhost"], ["10.0.0.5"])
    ensure_certs(tmp_path, ["localhost"], ["10.0.0.5"])
    assert paths.server_cert.read_bytes() != before
    assert paths.ca_cert.read_bytes() == ca_bytes


def test_tls_client_trusting_local_ca_connects(tmp_path):
    import socket

    from visionaid.phone.certs import client_ssl_context, server_ssl_context

    paths = ensure_certs(tmp_path, ["localhost"], ["127.0.0.1"])
    srv_ctx = server_ssl_context(paths)
    lsock = socket.socket()
    lsock.bind(("127.0.0.1", 0))
    lsock.listen(1)
    port = lsock.getsockname()[1]
    got = {}

    def serve():
        conn, _ = lsock.accept()
        with srv_ctx.wrap_socket(conn, server_side=True) as s:
            got["data"] = s.recv(5)

    th = threading.Thread(target=serve, daemon=True)
    th.start()
    cli = client_ssl_context(paths)
    with socket.create_connection(("127.0.0.1", port), timeout=5) as raw:
        with cli.wrap_socket(raw, server_hostname="localhost") as s:
            s.sendall(b"hello")
    th.join(5)
    lsock.close()
    assert got["data"] == b"hello"
    # a client without the local authority refuses the certificate
    lsock2 = socket.socket()
    lsock2.bind(("127.0.0.1", 0))
    lsock2.listen(1)
    p2 = lsock2.getsockname()[1]
    th2 = threading.Thread(target=_accept_tls, args=(lsock2, srv_ctx), daemon=True)
    th2.start()
    try:
        with socket.create_connection(("127.0.0.1", p2), timeout=5) as raw:
            with pytest.raises(ssl.SSLCertVerificationError):
                ssl.create_default_context().wrap_socket(raw, server_hostname="localhost")
    finally:
        lsock2.close()
        th2.join(5)


def _accept_tls(lsock, ctx):
    try:
        conn, _ = lsock.accept()
        ctx.wrap_socket(conn, server_side=True).close()
    except (OSError, ssl.SSLError):
        pass


def test_frame_pack_unpack_and_errors():
    data = pack_frame(12, 1700000000.25, 480, 640, b"\xff\xd8jpeg", calib=True)
    p = unpack_frame(data)
    assert (p.seq, p.t_phone, p.w, p.h, p.calib, p.jpeg) == (
        12,
        1700000000.25,
        480,
        640,
        True,
        b"\xff\xd8jpeg",
    )
    for bad in (b"", b"\x00\x00", b"\x00\x00\x00\x10{}", b"\x00\x00\x00\x02{}jpeg"):
        with pytest.raises(ProtocolError):
            unpack_frame(bad)
    with pytest.raises(ProtocolError):
        unpack_frame(pack_frame(1, 0.0, 1, 1, b""))


def test_hello_version_check():
    check_hello({"type": "hello", "protocol": 1, "gyro_units": "deg/s"})
    with pytest.raises(ProtocolError, match="version"):
        check_hello({"type": "hello", "protocol": 99})
    with pytest.raises(ProtocolError):
        check_hello({"type": "imu"})
    with pytest.raises(ProtocolError, match="unit"):
        check_hello({"type": "hello", "protocol": 1, "gyro_units": "tours/min"})


def test_imu_json_conversion_and_filtering():
    rows = [
        [2.0, 90.0, 0.0, -180.0, 0.0, 9.81, 0.0],
        [1.0, 0.0, 0.0, 0.0, 0.0, 9.81, 0.0],
        [3.0, None, 0, 0, 0, 0, 0],  # sensor missing: ignored
        [4.0, 1.0],  # incomplete: ignored
    ]
    b = imu_batch_from_json(rows, "deg/s", 5.0)
    assert len(b) == 2 and b.t_recv == 5.0
    np.testing.assert_allclose(b.t, [1.0, 2.0])  # sorted
    np.testing.assert_allclose(b.gyro[1], [np.pi / 2, 0.0, -np.pi])
    assert len(imu_batch_from_json([], "deg/s")) == 0


def test_clock_estimator_recovers_offset():
    rng = np.random.default_rng(0)
    est = ClockEstimator()
    true_offset = -3.25  # t_pc = t_phone + offset
    prev = None
    for k in range(40):
        t_phone = 100.0 + k
        up = 0.004 + rng.exponential(0.01)
        down = 0.004 + rng.exponential(0.01)
        est.ping(k, t_phone, t_phone + true_offset + up)
        if prev is not None:
            est.rtt(*prev)  # the RTT of an exchange arrives with the next ping
        prev = (k, up + down)
    c = est.estimate()
    assert abs(c.offset - true_offset) <= c.uncertainty + 1e-9
    assert c.uncertainty < 0.02
    # RTT of an unknown or invalid ping: ignored
    est.rtt(999, 0.001)
    est.rtt(3, float("nan"))
    assert est.estimate().rtt == pytest.approx(c.rtt)
    # without RTT: bias = minimal outbound delay, default uncertainty
    e2 = ClockEstimator()
    e2.ping(1, 10.0, 10.0 + true_offset + 0.01)
    c2 = e2.estimate()
    assert c2.offset == pytest.approx(true_offset + 0.01)
    assert c2.uncertainty == pytest.approx(0.05)
    assert not np.isfinite(ClockEstimator().estimate().uncertainty)


def _frame(i):
    return Frame(float(i), i, np.zeros((2, 2, 3), np.uint8), np.eye(3), float(i), time.time())


def test_inbox_keeps_latest_frame_and_all_imu():
    box = PhoneInbox()
    for i in range(5):
        box.put_frame(_frame(i))
        box.put_imu(ImuBatch(np.array([i + 0.5]), np.zeros((1, 3)), np.ones((1, 3))))
    f, imu, c = box.get(0.01)
    assert f.frame_idx == 4 and box.n_overwritten == 4
    np.testing.assert_allclose(imu.t, [0.5, 1.5, 2.5, 3.5, 4.5])
    f2, imu2, _ = box.get(0.01)
    assert f2 is None and imu2 is None
    assert merge_batches([]) is None


def test_inbox_slow_consumer_gets_fresh_frames():
    box = PhoneInbox()
    period = 1 / 30
    stop = threading.Event()

    def producer():
        i = 0
        while not stop.is_set():
            box.put_frame(_frame(i))
            i += 1
            time.sleep(period)

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    ages = []
    t_end = time.time() + 1.5
    while time.time() < t_end:
        f, _, _ = box.get(0.2)
        if f is not None:
            ages.append(time.time() - f.t_recv)
        time.sleep(0.15)  # slow consumer (~6 Hz)
    stop.set()
    th.join(1)
    assert len(ages) >= 5
    # the consumed image is at most one period late: nothing piles up
    assert max(ages) < 2 * period
    assert box.n_overwritten > 10


def test_cert_reload_on_network_change(tmp_path):
    """IP change: certificate re-issued and reloadable in the existing TLS context."""
    from visionaid.phone.certs import server_ssl_context

    paths = ensure_certs(tmp_path, ["localhost"], ["127.0.0.1", "172.20.10.5"])
    ctx = server_ssl_context(paths)
    assert not server_cert_covers(paths, ["localhost"], ["127.0.0.1", "10.5.60.177"])
    paths = ensure_certs(tmp_path, ["localhost"], ["127.0.0.1", "10.5.60.177"])
    ctx.load_cert_chain(str(paths.server_cert), str(paths.server_key))
    assert server_cert_covers(paths, ["localhost"], ["127.0.0.1", "10.5.60.177"])


def test_ipv6_only_network_addresses(monkeypatch):
    """IPv6-only hotspot: 192.0.0.2 (translation) excluded, IPv6 offered in brackets."""
    from visionaid.phone import certs

    monkeypatch.setattr(
        certs, "local_ips", lambda: ["127.0.0.1", "192.0.0.2", "2a01:cb1a:5025:cb4c::1d39"]
    )
    assert certs.reachable_ips() == ["2a01:cb1a:5025:cb4c::1d39"]
    assert certs.url_host("2a01:cb1a:5025:cb4c::1d39") == "[2a01:cb1a:5025:cb4c::1d39]"
    assert certs.url_host("10.0.0.2") == "10.0.0.2"


def test_cert_covers_ipv6(tmp_path):
    paths = ensure_certs(tmp_path, ["localhost"], ["127.0.0.1", "::1", "2a01:cb1a:5025:cb4c::1d39"])
    assert server_cert_covers(paths, ["localhost"], ["2a01:cb1a:5025:cb4c::1d39"])
