"""Phase 26 (D): path diagnostic tool (indicators and before/after image)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from visionaid.apps.diag_path import compare, summarize

pytestmark = pytest.mark.unit


def _detail(t_first: float, jumpy: bool, blocked: bool):
    frames, plans = [], []
    for k in range(60):
        t = k * 0.1
        frames.append(
            {
                "t": t,
                "state": "SEARCHING" if t < t_first else "NAVIGATING",
                "confirmed": t >= t_first,
                "obj_state": "seen" if t >= t_first else "unknown",
                "user": [0.0, 0.0],
            }
        )
        if k % 2 == 0 and t >= t_first:
            ang = np.radians(90 if (jumpy and k % 4 == 0) else 0)
            plans.append(
                {
                    "t": t,
                    "valid": True,
                    "length": 2.0,
                    "start_blocked": blocked,
                    "user": [0.0, 0.0],
                    "la": [float(np.cos(ang)), float(np.sin(ang))],
                    "points": [[0, 0], [2, 0]],
                    "obj": [2.0, 0.0],
                    "occ": 0.1,
                    "free": 0.2,
                }
            )
    return frames, plans


def test_summary_indicators():
    fr, pl = _detail(2.0, jumpy=True, blocked=True)
    s = summarize("x_test", fr, pl)
    assert s["first_valid_path_s"] == pytest.approx(2.0)
    assert s["searching_frac"] == pytest.approx(20 / 60, abs=0.01)
    assert s["start_blocked_frac"] == 1.0
    assert s["aim_jumps_over_30"] >= 10 and s["aim_jump_deg_p90"] == pytest.approx(90, abs=1)
    fr2, pl2 = _detail(1.0, jumpy=False, blocked=False)
    s2 = summarize("x_test", fr2, pl2)
    assert s2["aim_jumps_over_30"] == 0 and s2["start_blocked_frac"] == 0.0


def test_compare_writes_image(tmp_path):
    for sub, (t0, j, b) in (("avant", (2.0, True, True)), ("apres", (1.0, False, False))):
        d = tmp_path / sub
        d.mkdir()
        fr, pl = _detail(t0, j, b)
        (d / "s1_test_summary.json").write_text(json.dumps(summarize("s1_test", fr, pl)))
        (d / "s1_test_frames.json").write_text(json.dumps({"frames": fr, "plans": pl}))
    sim = {
        k: {"success_rate": 0.95, "collision_free_rate": 0.96}
        for k in ("room", "corridor", "appearing")
    }
    png = compare(tmp_path / "avant", tmp_path / "apres", tmp_path / "c.png", (sim, sim))
    assert png.exists() and png.stat().st_size > 20_000
