"""Phase 18 (data): VIO vs VO on the real sessions (informative + safeguards)."""

import numpy as np
import pytest

from visionaid.apps.session_info import list_sessions
from visionaid.utils.config import load_config, resolve_path
from visionaid.utils.report import update_report

pytestmark = [pytest.mark.data, pytest.mark.slow]


def _find(tag):
    ss = [p for p in list_sessions(resolve_path(load_config().session.dir)) if tag in p.name]
    if not ss:
        pytest.skip(f"No \"{tag}\" session (see docs/checkpoint_enregistrements.md)")
    return ss[-1]


@pytest.mark.timeout(1800)
@pytest.mark.parametrize("tag", ["rotations_rapides", "gourde_piece"])
def test_vio_vs_vo_on_session(tag, tmp_path):
    from visionaid.apps.run_offline import run

    sess = _find(tag)
    res = {}
    for mode in ("vo", "vio"):
        cfg = load_config(overrides={"pose.source": mode, "viz.rerun": False})
        s = run(cfg, sess, tmp_path / mode, rerun=False, report_section=None)
        res[mode] = {
            "lost": s["lost_fraction"],
            "degraded": s.get("degraded_fraction", 0.0),
            "pose_stats": s.get("pose_stats"),
            "confirmed": s["confirmed_fraction"],
        }
    update_report("vio_sessions", {tag: res})
    print(res)
    assert res["vio"]["lost"] <= res["vo"]["lost"] + 0.02
    scale = (res["vio"]["pose_stats"] or {}).get("depth_scale", 1.0)
    assert 0.5 < scale < 2.0 and np.isfinite(scale)
