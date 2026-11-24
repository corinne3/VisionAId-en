import dataclasses
import threading
import time

import numpy as np
import pytest
import zmq

from visionaid.bus import messages as M
from visionaid.bus.nodes import NodeSpec
from visionaid.bus.runner import GraphRunner
from visionaid.bus.transport import Publisher, Subscriber

pytestmark = pytest.mark.unit

_port = [6100]


def next_addr():
    _port[0] += 1
    return f"tcp://127.0.0.1:{_port[0]}"


def _examples():
    img = np.random.randint(0, 255, (48, 64, 3), np.uint8)
    return [
        M.Frame(1.0, 1, img, np.eye(3)),
        M.Pose(1.0, 1, np.eye(4), "ok", 50, 0.7, True),
        M.Depth(1.0, 1, np.random.rand(48, 64).astype(np.float32), None),
        M.Detections(
            1.0,
            1,
            np.random.rand(2, 4),
            np.array([0.5, 0.9]),
            ["a", "b"],
            np.zeros((2, 48, 64), bool),
            np.array([3, 4]),
            np.array([0.1, 0.8]),
            1,
            4,
        ),
        M.ObjectEstimate(1.0, 1, np.array([1.0, 2.0, 0.5]), np.eye(3) * 0.01, "seen", 0.8),
        M.Costmap(
            1.0,
            1,
            np.zeros((10, 12), np.int8),
            np.ones((10, 12), np.float32),
            np.array([-1.0, -2.0]),
            0.05,
        ),
        M.Path(1.0, 1, np.random.rand(5, 2), np.array([1.0, 1.0]), np.array([2.0, 2.0]), True, 3.2),
        M.GuidanceCue(1.0, 1, "NAVIGATING", 0.3, -0.1, 2.0, 0.0, "straight ahead"),
    ]


@pytest.mark.parametrize("msg", _examples(), ids=lambda m: type(m).__name__)
def test_roundtrip(msg):
    back = M.decode(M.encode(msg))
    assert type(back) is type(msg)
    for f in dataclasses.fields(msg):
        a, b = getattr(msg, f.name), getattr(back, f.name)
        if isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
        elif isinstance(a, float) and np.isnan(a):
            assert np.isnan(b)
        else:
            assert a == b


def _wait_connected(pub, sub, topic, probe):
    for _ in range(200):
        pub.send(topic, probe)
        if sub.recv(10) is not None:
            break
    while sub.recv(20) is not None:
        pass


def test_pubsub_order():
    addr = next_addr()
    pub = Publisher(addr)
    sub = Subscriber(addr, ["cue"], conflate=False)
    probe = M.GuidanceCue(0, -1, "PROBE")
    _wait_connected(pub, sub, "cue", probe)
    for i in range(1000):
        pub.send("cue", M.GuidanceCue(float(i), i, "S"))
    got = []
    t_end = time.time() + 5
    while len(got) < 1000 and time.time() < t_end:
        r = sub.recv(100)
        if r:
            got.append(r[2].frame_idx)
    pub.close()
    sub.close()
    assert got == list(range(1000))


def test_conflate_slow_subscriber():
    addr = next_addr()
    pub = Publisher(addr)
    sub = Subscriber(addr, ["frame"], conflate=True)
    img = np.zeros((48, 64, 3), np.uint8)
    _wait_connected(pub, sub, "frame", M.Frame(0, -1, img))
    period = 0.01
    stop = threading.Event()

    def producer():
        i = 0
        while not stop.is_set():
            pub.send("frame", M.Frame(time.time(), i, img))
            i += 1
            time.sleep(period)

    th = threading.Thread(target=producer)
    th.start()
    ages = []
    try:
        for _ in range(15):
            time.sleep(0.1)  # subscriber 10× slower
            r = sub.recv(200)
            if r:
                ages.append(time.time() - r[2].t)
    finally:
        stop.set()
        th.join()
        pub.close()
        sub.close()
    assert len(ages) >= 10
    assert np.median(ages) < 2 * period + 0.005  # scheduling tolerance
    assert max(ages) < 0.1  # never any accumulated backlog


def test_latency_640x480():
    addr = next_addr()
    pub = Publisher(addr)
    sub = Subscriber(addr, ["frame"], conflate=False)
    img = np.random.randint(0, 255, (480, 640, 3), np.uint8)
    _wait_connected(pub, sub, "frame", M.Frame(0, -1, img))
    lat = []
    for i in range(50):
        t0 = time.perf_counter()
        pub.send("frame", M.Frame(0.0, i, img))
        r = sub.recv(1000)
        lat.append(time.perf_counter() - t0)
        assert r is not None and r[2].frame_idx == i
    pub.close()
    sub.close()
    assert np.median(lat) < 0.005


def test_runner_start_stop():
    specs = [
        NodeSpec(
            "src", "visionaid.bus.testing_blocks:CounterSource", outputs=["frame"], rate_hz=50
        ),
        NodeSpec(
            "echo",
            "visionaid.bus.testing_blocks:SlowEcho",
            inputs=["frame"],
            outputs=["cue"],
            trigger="frame",
            heavy_inputs=["frame"],
        ),
        NodeSpec("sink", "visionaid.bus.testing_blocks:Collector", inputs=["cue"], trigger="cue"),
    ]
    r = GraphRunner(specs, {"test_delay": 0.03}, base_port=6300)
    r.start()
    time.sleep(3.0)
    assert len(r.alive()) == 3
    dt = r.stop()
    assert dt < 2.0
    assert r.alive() == []
    assert all(p.exitcode is not None for p in r.procs)
    assert r.stats["sink"]["_block"]["n"] > 10
    # the echo is slower than the source: conflation keeps fresh inputs
    assert r.stats["echo"]["age_in:frame"]["p50_ms"] < 60


def test_zmq_context_clean():
    assert zmq.Context.instance() is not None


def test_sequential_runner_same_blocks():
    from visionaid.bus.runner import SequentialRunner

    specs = [
        NodeSpec("src", "visionaid.bus.testing_blocks:CounterSource", outputs=["frame"]),
        NodeSpec(
            "echo",
            "visionaid.bus.testing_blocks:SlowEcho",
            inputs=["frame"],
            outputs=["cue"],
            trigger="frame",
        ),
        NodeSpec("sink", "visionaid.bus.testing_blocks:Collector", inputs=["cue"], trigger="cue"),
    ]
    r = SequentialRunner(specs, {"test_delay": 0.0})
    assert r.run(max_ticks=25) == 25
    assert r.blocks["sink"].n == 25
    assert r.latest["cue"].frame_idx == 24
