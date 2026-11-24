import math

import numpy as np
import pytest

from visionaid.sim.agent import head_pose, optimal_length, run_episode
from visionaid.sim.scenarios import SCENARIOS
from visionaid.sim.world import SimSensor
from visionaid.utils.config import load_config, resolve_path
from visionaid.utils.report import update_report

pytestmark = pytest.mark.unit


def test_head_pose_convention():
    T = head_pose(1.0, 2.0, math.pi / 2)  # looking towards +y
    assert np.allclose(T[:3, 2], [0, 1, 0], atol=1e-9)  # camera z = forward
    assert np.allclose(T[:3, 1], [0, 0, -1])  # camera y = down
    assert np.allclose(T[:3, 0], [1, 0, 0], atol=1e-9)  # camera x = right
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_scenarios_valid(name):
    for seed in range(5):
        w = SCENARIOS[name](seed)
        assert w.clearance(w.start[:2]) >= 0.45
        assert math.isfinite(optimal_length(w))


def test_sensor_sees_walls():
    w = SCENARIOS["room"](0)
    pts = SimSensor(load_config().sim, np.random.default_rng(0)).scan(w, *w.start)
    assert len(pts) > 100
    assert (pts[:, 2] > 0.2).any() and (np.abs(pts[:, 2]) < 0.1).any()


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_smoke_episodes(name):
    """Quick smoke test (5 seeds): the simulator loops and succeeds most of the time."""
    cfg = load_config()
    ok = [run_episode(cfg, SCENARIOS[name](s), s, max_time=60).success for s in range(5)]
    assert sum(ok) >= 3


@pytest.mark.slow
@pytest.mark.timeout(3000)
def test_sim_statistics_100_seeds():
    """100 seeds per scenario type: success ≥ 90%, 0 collisions in ≥ 95% of runs,
    travelled / optimal length ratio ≤ 1.6 (median), replanning ≥ 90%."""
    from visionaid.sim.evaluate import evaluate, render_examples

    cfg = load_config()
    real = resolve_path(cfg.paths.outputs) / "real_map.npz"
    scen = ("room", "corridor", "appearing") + (("real",) if real.exists() else ())
    summ, rows = evaluate(cfg, n_seeds=100, scenarios=scen, real_map_path=real)
    out = resolve_path(cfg.paths.outputs) / "sim"
    picks = [("room", 1), ("corridor", 2), ("appearing", 3)]
    vids = render_examples(cfg, out, picks, real)
    update_report("sim", {**summ, "videos": vids})
    for name in ("room", "corridor", "appearing"):
        s = summ[name]
        assert s["errors"] == 0
        assert s["success_rate"] >= 0.90, (name, s)
        assert s["collision_free_rate"] >= 0.95, (name, s)
        assert s["ratio_median"] <= 1.6, (name, s)
    # sudden obstacle scenario: successful replanning in ≥ 90% of the cases where it appeared
    assert summ["appearing"]["success_rate_with_event"] >= 0.90
    if "real" in summ:  # informative: noisy real map
        update_report("sim", {"real_note": "real map: informative"})
