import math

import numpy as np
import pytest

from visionaid.bus.messages import Costmap, ObjectEstimate, Path, Pose
from visionaid.guidance.cues import GuidanceEngine, relative_direction
from visionaid.guidance.state_machine import (
    APPROACHING,
    ARRIVED,
    NAVIGATING,
    OBSTACLE_ALERT,
    SEARCHING,
    TRACKING_LOST,
    GuidanceInputs,
    GuidanceStateMachine,
)
from visionaid.mapping.costmap import FREE, OCC
from visionaid.mapping.geometry import make_T
from visionaid.utils.config import load_config

pytestmark = pytest.mark.unit
INF = math.inf


def sm():
    return GuidanceStateMachine(load_config().guidance)


def run(seq, dt=0.1):
    m = sm()
    return [m.step(GuidanceInputs(k * dt, *row)) for k, row in enumerate(seq)]


# (pose_lost, object_known, distance, obstacle_dist, path_valid) → expected state (after hold)
TABLE = [
    ("searching", [(False, False, INF, INF, False)] * 3, [SEARCHING] * 3),
    ("navigating", [(False, True, 4.0, INF, True)] * 5, [NAVIGATING] * 5),
    (
        "approach then arrive",
        [(False, True, d, INF, True) for d in (3.0, 2.0, 1.1, 1.0, 0.9, 0.6, 0.45, 0.4, 0.4)],
        [
            NAVIGATING,
            NAVIGATING,
            NAVIGATING,
            APPROACHING,
            APPROACHING,
            APPROACHING,
            ARRIVED,
            ARRIVED,
            ARRIVED,
        ],
    ),
    (
        "lost",
        [(False, True, 3.0, INF, True)] * 4 + [(True, True, 3.0, INF, True)] * 4,
        [NAVIGATING] * 4 + [TRACKING_LOST] * 4,
    ),
]


@pytest.mark.parametrize("name,seq,expected", TABLE, ids=[t[0] for t in TABLE])
def test_table(name, seq, expected):
    assert run(seq) == expected


@pytest.mark.parametrize("sigma", [0.05, 0.3])
def test_dwell_noisy_input(sigma):
    """Noisy inputs around the thresholds: never more than one state change per 0.3 s."""
    rng = np.random.default_rng(0)
    m = sm()
    dt = 1 / 30
    for k in range(600):  # 20 s
        m.step(
            GuidanceInputs(
                k * dt,
                rng.random() < 0.02,
                True,
                1.2 + rng.normal(0, sigma),
                0.6 + rng.normal(0, sigma),
                True,
            )
        )
    times = [t for t, _ in m.history[1:]]
    assert all(b - a >= 0.3 - 1e-9 for a, b in zip(times[:-1], times[1:], strict=True))


def test_hysteresis_limits_flicker():
    """Realistic noise (σ = 5 cm, filtered distance) around the approach threshold: few changes."""
    rng = np.random.default_rng(1)
    m = sm()
    for k in range(600):
        m.step(GuidanceInputs(k / 30, False, True, 1.2 + rng.normal(0, 0.05), INF, True))
    assert len(m.history) - 1 <= 4


def test_obstacle_priority_over_navigation():
    seq = (
        [(False, True, 3.0, INF, True)] * 4
        + [(False, True, 3.0, 0.4, True)] * 4
        + [(False, True, 3.0, 2.0, True)] * 4
    )
    out = run(seq)
    assert out[4] == OBSTACLE_ALERT
    assert out[-1] == NAVIGATING


def test_lost_has_priority_over_obstacle():
    seq = [(False, True, 3.0, 0.3, True)] * 4 + [(True, True, 3.0, 0.3, True)] * 4
    assert run(seq)[-1] == TRACKING_LOST


def test_azimuth_sign_left_positive():
    # head at (0,0,1.4), looking towards +x: OpenCV frame z_cam = world +x, x_cam = world −y, y_cam = −z
    R = np.stack([[0, -1.0, 0], [0, 0, -1.0], [1.0, 0, 0]], 1)
    T = make_T(R, [0, 0, 1.4])
    az_left, el, d = relative_direction(T, np.array([2.0, 1.0, 1.4]))
    az_right, _, _ = relative_direction(T, np.array([2.0, -1.0, 1.4]))
    assert az_left > 0 > az_right
    assert az_left == pytest.approx(math.atan2(1, 2))
    _, el_up, _ = relative_direction(T, np.array([2.0, 0.0, 2.4]))
    assert el_up > 0


def _engine_inputs(obj_xy, occ_cells=()):
    R = np.stack([[0, -1.0, 0], [0, 0, -1.0], [1.0, 0, 0]], 1)
    pose = Pose(0.0, 0, make_T(R, [0, 0, 1.4]), "ok")
    st = np.full((100, 100), FREE, np.int8)
    for i, j in occ_cells:
        st[i, j] = OCC
    cm = Costmap(0.0, 0, st, np.zeros(st.shape, np.float32), np.array([-2.5, -2.5]), 0.05)
    obj = ObjectEstimate(0.0, 0, np.array([*obj_xy, 0.8]), np.eye(3), "seen", 1.0)
    path = Path(
        0.0, 0, np.array([[0, 0], obj_xy]), np.array([1.0, 0.5]), np.array(obj_xy), True, 3.0
    )
    return pose, obj, path, cm


def test_engine_navigating_toward_lookahead():
    e = GuidanceEngine(load_config())
    pose, obj, path, cm = _engine_inputs([3.0, 1.0])
    cue = e.update(0.0, 0, pose, obj, path, cm)
    assert cue.state == NAVIGATING
    assert cue.azimuth == pytest.approx(math.atan2(0.5, 1.0))
    assert cue.message


def test_engine_obstacle_alert_from_costmap():
    e = GuidanceEngine(load_config())
    # obstacle 0.4 m ahead (x=+0.4 → j = (0.4+2.5)/0.05 = 58; y=0 → i = 50)
    pose, obj, path, cm = _engine_inputs([3.0, 1.0], [(50, 58)])
    cue = e.update(0.0, 0, pose, obj, path, cm)
    assert cue.state == OBSTACLE_ALERT and cue.alert_level > 0.2


def test_engine_obstacle_behind_ignored():
    e = GuidanceEngine(load_config())
    pose, obj, path, cm = _engine_inputs([3.0, 1.0], [(50, 42)])  # x = −0.4: behind
    assert e.update(0.0, 0, pose, obj, path, cm).state == NAVIGATING


def test_voice_rate_limit():
    e = GuidanceEngine(load_config())
    msgs = []
    states = [SEARCHING, NAVIGATING, APPROACHING, NAVIGATING, APPROACHING, ARRIVED]
    for k, s in enumerate(states * 5):
        msgs.append((k * 0.5, e.voice(k * 0.5, s)))
    spoken = [t for t, m in msgs if m]
    assert all(b - a >= 3.0 for a, b in zip(spoken[:-1], spoken[1:], strict=True))
    assert len(spoken) >= 3
